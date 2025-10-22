import { AzureKeyCredential, SearchClient, SearchOptions } from "@azure/search-documents";
import { AzureOpenAI } from "openai";

/** Document shape matching your Azure AI Search index fields. */
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

  // Minimal knobs (vector-only)
  kNearestNeighborsCount?: number; // default 20
  top?: number;                    // default 20 (fetch window; we’ll slice to 5)
  maxReturn?: number;              // default 5 (exact # sent back)
}

export class AzureAISearchDataSource {
  public readonly name: string;
  private readonly options: Required<Pick<AzureAISearchDataSourceOptions,
    "kNearestNeighborsCount" | "top" | "maxReturn"
  >> & Omit<AzureAISearchDataSourceOptions, "kNearestNeighborsCount" | "top" | "maxReturn">;

  private readonly searchClient: SearchClient<OneDriveChunkDocument>;
  private readonly embedder: AzureOpenAI;

  public constructor(options: AzureAISearchDataSourceOptions) {
    this.name = options.name;
    this.options = {
      kNearestNeighborsCount: options.kNearestNeighborsCount ?? 20,
      top: options.top ?? 20,
      maxReturn: options.maxReturn ?? 5,
      ...options
    };

    this.searchClient = new SearchClient<OneDriveChunkDocument>(
      this.options.azureAISearchEndpoint,
      this.options.indexName,
      new AzureKeyCredential(this.options.azureAISearchApiKey)
    );

    this.embedder = new AzureOpenAI({
      apiKey: this.options.azureOpenAIApiKey,
      endpoint: this.options.azureOpenAIEndpoint,
      apiVersion: "2024-02-01"
    });
  }

  /**
   * VECTOR-ONLY: Build up to 5 <context> blocks from the top-scoring vector results.
   * Pass either the user's natural-language question, or your focus terms, or both.
   */
  public async renderContext(query: string): Promise<string> {
    if (!query || !query.trim()) return "";

    // 1) Create a single embedding from whatever string you pass in (user input and/or focus terms)
    const qVec = await this.getEmbeddingVector(query);

    // 2) Vector-only search (no lexical query, no semantic ranker)
    const results = await this.searchVectorOnly(qVec, this.options.top, this.options.kNearestNeighborsCount);

    // 3) Sort by score (SDK usually returns in-score order, but we enforce)
    results.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    // 4) Take the top N (default 5)
    const topN = results.slice(0, this.options.maxReturn);

    // 5) Format into <context> blocks (no grouping/compaction)
    const contexts: string[] = [];
    for (const r of topN) {
      const title = r.document.title ?? "(untitled)";
      let text = r.document.chunk ?? "";
      text = this.cleanContent(text);
      if (!text.trim()) continue;
      contexts.push(this.formatDocument(text, title));
    }

    // Remove redundant whitespace across the entire stitched string
    const stitched = contexts.join("\n").replace(/\s+/g, " ").trim();

    console.log(`[Context from Azure AI Search]: ${stitched}`);
    return stitched;
  }

  /** Cleans up text content from Azure AI Search results. */
  private cleanContent(text: string): string {
    let clean = text;

    // 1. Remove filenames or lines that look like image/media files
    clean = clean.replace(/^\s*[\w,\-]+\.(jpg|jpeg|png|svg|gif|pdf|webp)\s*$/gim, "");

    // 2. Remove CSS or style fragments
    clean = clean.replace(/\.[\w\-]+_?[A-Za-z0-9\-]*\s*\{[^}]*\}/gim, "");

    // 3. Strip HTML tags if any remain
    clean = clean.replace(/<[^>]+>/g, "");

    // 4. Collapse whitespace and newlines
    clean = clean.replace(/\s+/g, " ").trim();

    // 5. Filter out non-informative residuals like "fill:#..."
    clean = clean.replace(/fill\s*:\s*#[0-9A-Fa-f]{3,6};?/g, "");

    return clean;
  }

  /** Performs a single vector-only search against the embedding field. */
  private async searchVectorOnly(vector: number[], top: number, kNearestNeighborsCount: number) {
    const opts: SearchOptions<OneDriveChunkDocument> = {
      top,
      select: ["chunk_id", "parent_id", "title", "chunk"] as any,
      vectorSearchOptions: {
        queries: [
          {
            kind: "vector",
            fields: ["text_vector"],
            kNearestNeighborsCount,
            vector
          }
        ]
      }
    };

    const out: Array<{ document: OneDriveChunkDocument; score?: number }> = [];
    const res: any = await this.searchClient.search("", opts);

    // Preferred: iterate the async iterable of results
    if (res?.results && Symbol.asyncIterator in Object(res.results)) {
      for await (const r of res.results) {
        out.push({ document: r.document as OneDriveChunkDocument, score: r.score as number | undefined });
      }
      return out;
    }

    // Some SDK shapes allow iterating the top-level result
    if (res && Symbol.asyncIterator in Object(res)) {
      for await (const r of res) {
        out.push({ document: r.document as OneDriveChunkDocument, score: r.score as number | undefined });
      }
      return out;
    }

    // Fallback: page already materialized (rare/older shapes)
    if (Array.isArray(res?.value)) {
      for (const r of res.value) {
        out.push({ document: r.document as OneDriveChunkDocument, score: r.score as number | undefined });
      }
      return out;
    }

    throw new TypeError("Unexpected search() return shape; not iterable.");
  }


  /** Simple wrapper for Azure OpenAI embeddings. */
  private async getEmbeddingVector(text: string): Promise<number[]> {
    const res = await this.embedder.embeddings.create({
      model: this.options.azureOpenAIEmbeddingDeploymentName,
      input: text
    });
    const v = res.data?.[0]?.embedding;
    if (!v) throw new Error("Failed to generate embeddings.");
    return v;
  }

  /** Minimal formatter; no whitespace compaction/grouping. */
  private formatDocument(content: string, citation: string): string {
    return `<context source="${this.escapeAttr(citation)}">\n${content}\n</context>`;
  }

  private escapeAttr(v: string): string {
    return String(v).replace(/"/g, "&quot;");
  }
}
