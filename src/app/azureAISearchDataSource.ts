import { AzureKeyCredential, SearchClient, SearchOptions } from "@azure/search-documents";
import { AzureOpenAI } from "openai";

/**
 * Document shape matching your Azure AI Search index fields.
 */
export interface OneDriveChunkDocument {
  chunk_id?: string;             // key
  parent_id?: string | null;
  title?: string | null;         // doc/page title
  chunk?: string | null;         // text content
  text_vector?: number[] | null; // vector field (dim 1536)
}

export interface AzureAISearchDataSourceOptions {
  name: string;
  indexName: string;

  // Azure OpenAI (for query embedding)
  azureOpenAIApiKey: string;
  azureOpenAIEndpoint: string;
  azureOpenAIEmbeddingDeploymentName: string;

  // Azure AI Search
  azureAISearchApiKey: string;
  azureAISearchEndpoint: string;

  // Retrieval knobs (optional)
  kNearestNeighborsCount?: number;   // default 5
  topPerVariant?: number;            // default 3 (used for paging even in single-query mode)
  maxTotalAfterFusion?: number;      // default 5
  maxPerParent?: number;             // default 2
  minTopScore?: number;              // default 0.3
  minScoreGap?: number;              // default 0.0

  // token budget for the entire <context> payload (rough estimate)
  maxContextTokens?: number;         // default 1800 (~4 chars/token heuristic)
}

export class AzureAISearchDataSource {
  public readonly name: string;
  private readonly options: AzureAISearchDataSourceOptions;
  private readonly searchClient: SearchClient<OneDriveChunkDocument>;

  public constructor(options: AzureAISearchDataSourceOptions) {
    this.name = options.name;
    this.options = {
      kNearestNeighborsCount: 16,
      topPerVariant: 12,              // broaden initial candidate pool
      maxTotalAfterFusion: 6,        // keep final context tight
      maxPerParent: 4,               // allow more parents -> more diversity
      minTopScore: 0.12,             // allow weaker matches when needed
      minScoreGap: 0.01,             // relaxed gap
      maxContextTokens: 1200,        // smaller budget
      ...options
    };

    this.searchClient = new SearchClient<OneDriveChunkDocument>(
      this.options.azureAISearchEndpoint,
      this.options.indexName,
      new AzureKeyCredential(this.options.azureAISearchApiKey),
      {}
    );
  }

  /**
   * MAIN: Build grounded context or return "" if confidence is low.
   * (Single-query path; no reformulation, no fusion)
   */
  public async renderContext(query: string): Promise<string> {
    if (!query) return "";

    // Embed the raw user query
    const qVec = await this.getEmbeddingVector(query);

    // One hybrid search: text + vector (no Semantic Ranker required)
    const ranked = await this.searchCollect(query, qVec, this.options.topPerVariant!);

    // Confidence gate — use native Azure Search scores (NOT any fused score)
    if (this.shouldClarify(ranked)) return "";

    // Enforce diversity & total limits on the ORIGINAL ranked order
    const picked = this.limitByParentAndTotal(
      ranked,
      this.options.maxPerParent!,
      this.options.maxTotalAfterFusion!
    );

    // Group by parent and merge chunks -> compact & dedupe within each group
    const grouped = this.groupByParent(picked);
    const contexts: string[] = [];
    for (const group of grouped) {
      const title = group.title ?? "(untitled)";

      // 1) normalize+compact each chunk to strip whitespace bloat
      const normalizedChunks = group.chunks
        .map((c) => this.compactWhitespace(c))
        .filter((c) => c.length > 0);

      // 2) de-dupe near-identical normalized chunks (by normalized hash)
      const deduped = this.dedupeByHash(normalizedChunks);

      // 3) **Make smaller, denser**: keep just first ~3–4 sentences max
      const cut = this.firstNSentences(deduped.join(" "), 4);

      // 4) format per parent
      const piece = this.formatDocument(cut, title);
      if (piece.trim().length > 0) contexts.push(piece);
    }

    // 5) enforce a global token budget across all <context> blocks
    const budget = this.options.maxContextTokens!;
    const { stitched } = this.stitchWithTokenBudget(contexts, budget);

    return stitched;
  }

  /** Collect search results into an array (preserving Azure Search's native ranking). */
  private async searchCollect(query: string, vector: number[], top: number) {
    const res = await this.searchOnce(query, vector, top);
    const arr: Array<{ document: OneDriveChunkDocument; score?: number }> = [];
    for await (const r of (res as any).results ?? res) {
      // r has shape: { document, score, ... }
      arr.push(r);
    }
    return arr;
  }

  /** Single hybrid search (text + vector). No Semantic Ranker. */
  private async searchOnce(query: string, vector: number[], top: number) {
    const opts: SearchOptions<OneDriveChunkDocument> = {
      top,
      searchFields: ["title", "chunk"],
      select: ["chunk_id", "parent_id", "title", "chunk"] as any,
      // hybrid: supply the text 'query' AND vector queries
      queryType: "simple",
      searchMode: "all",
      vectorSearchOptions: {
        queries: [
          {
            kind: "vector",
            fields: ["text_vector"],
            kNearestNeighborsCount: this.options.kNearestNeighborsCount!,
            vector
          }
        ]
      }
    };

    return this.searchClient.search(query, opts);
  }

  /** Enforce diversity & total limit while preserving original ranking. */
  private limitByParentAndTotal<T extends { document: OneDriveChunkDocument }>(
    items: T[],
    maxPerParent: number,
    maxTotal: number
  ): T[] {
    const perParent = new Map<string, number>();
    const out: T[] = [];
    for (const it of items) {
      const parent = it.document.parent_id ?? it.document.chunk_id ?? "_";
      const n = perParent.get(parent) ?? 0;
      if (n < maxPerParent) {
        out.push(it);
        perParent.set(parent, n + 1);
        if (out.length >= maxTotal) break;
      }
    }
    return out;
  }

  /** Confidence / answerability gate using Azure Search's native scores. */
  private shouldClarify(items: Array<{ score?: number; document: OneDriveChunkDocument }>): boolean {
    if (!items || items.length === 0) return true;
    if (items.length >= 3) return false;

    const s1 = items[0]?.score ?? 0;
    const s2 = items[1]?.score ?? 0;

    const lowTop = s1 < (this.options.minTopScore ?? 0.12);
    const smallGap = (s1 - s2) < (this.options.minScoreGap ?? 0.01);

    const distinctParents = new Set(
      items.map((r) => r.document.parent_id ?? r.document.chunk_id)
    ).size;

    console.debug(
      "[RAG] topScore:", s1,
      "second:", s2,
      "parents:", distinctParents,
      "lowTop?", lowTop,
      "smallGap?", smallGap
    );

    // block only when obviously weak
    return lowTop && smallGap;
  }

  /** Group by parent and aggregate raw text chunks */
  private groupByParent<T extends { document: OneDriveChunkDocument }>(items: T[]) {
    const map = new Map<string, { title?: string | null; chunks: string[] }>();
    for (const it of items) {
      const parent = it.document.parent_id ?? it.document.chunk_id ?? "_";
      const g = map.get(parent) ?? { title: it.document.title, chunks: [] };
      if (it.document.chunk) g.chunks.push(it.document.chunk);
      if (!g.title && it.document.title) g.title = it.document.title;
      map.set(parent, g);
    }
    // Deterministic order
    return Array.from(map.values()).map((g) => ({
      title: g.title,
      chunks: g.chunks
    }));
  }

  private formatDocument(content: string, citation: string): string {
    const clean = this.compactWhitespace(content);
    if (!clean) return "";
    return `<context source="${this.escapeAttr(citation)}">\n${clean}\n</context>`;
  }

  /** Embeddings */
  private async getEmbeddingVector(text: string): Promise<number[]> {
    const client = new AzureOpenAI({
      apiKey: this.options.azureOpenAIApiKey,
      endpoint: this.options.azureOpenAIEndpoint,
      apiVersion: "2024-02-01"
    });
    const result = await client.embeddings.create({
      input: text,
      model: this.options.azureOpenAIEmbeddingDeploymentName
    });
    if (!result.data?.length) {
      throw new Error(`Failed to generate embeddings for text.`);
    }
    return result.data[0].embedding;
  }

  // -------------------------
  // Whitespace & size helpers
  // -------------------------

  /** Very light token estimator (swap with tiktoken later if desired). */
  private estimateTokens(text: string): number {
    // ~4 chars/token heuristic
    return Math.ceil((text?.length ?? 0) / 4);
  }

  /** Stitch multiple <context> pieces under a global token budget; trims the last piece to fit. */
  private stitchWithTokenBudget(pieces: string[], maxTokens: number) {
    const out: string[] = [];
    let used = 0;
    for (const p of pieces) {
      const t = this.estimateTokens(p);
      if (used + t <= maxTokens) {
        out.push(p);
        used += t;
      } else {
        const remaining = Math.max(0, maxTokens - used);
        if (remaining <= 0) break;

        const approxChars = remaining * 4;
        // Preserve <context> wrapper if possible
        const openIdx = p.indexOf(">");
        const closeIdx = p.lastIndexOf("</context>");
        if (openIdx !== -1 && closeIdx !== -1) {
          const head = p.slice(0, openIdx + 1);
          const body = p.slice(openIdx + 1, closeIdx);
          const tail = p.slice(closeIdx);
          const trimmedBody = this.safeTrimToChars(body, approxChars);
          out.push(`${head}${trimmedBody}${tail}`);
        } else {
          out.push(this.safeTrimToChars(p, approxChars));
        }
        used = maxTokens;
        break;
      }
    }
    return { stitched: out.join("\n") };
  }

  /** Trim to a clean boundary without chopping mid-token too harshly. */
  private safeTrimToChars(s: string, maxChars: number): string {
    if (s.length <= maxChars) return s;
    const slice = s.slice(0, Math.max(0, maxChars - 1));
    const lastBreak = Math.max(slice.lastIndexOf("\n"), slice.lastIndexOf(" "), slice.lastIndexOf("\t"));
    const cut = lastBreak > maxChars * 0.6 ? slice.slice(0, lastBreak) : slice;
    return cut.replace(/[ \t\r\n]*$/g, "") + "…";
  }

  /** Remove zero-width & control chars, collapse spaces/blank lines, trim lines. */
  private compactWhitespace(s?: string | null): string {
    if (!s) return "";
    let out = s;

    // Normalize Unicode whitespace and remove zero-width chars
    out = out.replace(/[\u200B-\u200D\uFEFF\u2028\u2029]/g, "");

    // Normalize line endings
    out = out.replace(/\r\n?/g, "\n");

    // Trim each line and remove trailing spaces
    out = out
      .split("\n")
      .map((line) => line.replace(/[ \t]+/g, " ").trim())
      .join("\n");

    // Collapse multiple blank lines
    out = out.replace(/\n{3,}/g, "\n\n");

    return out.trim();
  }

  /** Deduplicate strings by a simple normalized hash (post-compaction). */
  private dedupeByHash(chunks: string[]): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const c of chunks) {
      const key = this.hashString(c);
      if (!seen.has(key)) {
        seen.add(key);
        out.push(c);
      }
    }
    return out;
  }

  /** Simple fast non-crypto hash for dedupe */
  private hashString(s: string): string {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = (h * 31 + s.charCodeAt(i)) | 0;
    }
    return h.toString(16);
  }

  private escapeAttr(v: string): string {
    return String(v).replace(/"/g, "&quot;");
  }

  /** Keep first N sentences (very small & dense) */
  private firstNSentences(text: string, n = 4): string {
    const parts = (text || "").split(/(?<=[.!?])\s+/).filter(Boolean);
    return parts.slice(0, n).join(" ");
  }
}
