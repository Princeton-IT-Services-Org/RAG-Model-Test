import { App } from "@microsoft/teams.apps";
import { ChatPrompt, Message } from "@microsoft/teams.ai";
import { LocalStorage } from "@microsoft/teams.common";
import { OpenAIChatModel } from "@microsoft/teams.openai";
import { MessageActivity, TokenCredentials } from "@microsoft/teams.api";
import { ManagedIdentityCredential } from "@azure/identity";
import * as fs from "fs";
import * as path from "path";
import config from "../config";
import { AzureAISearchDataSource } from "./azureAISearchDataSource";

// NEW: raw OpenAI SDK for the tiny router
import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

const MAX_MESSAGES = 10;

const storage = new LocalStorage();
const dataSource = new AzureAISearchDataSource({
  name: "azure-ai-search",
  indexName: "onedrivebusiness-1760031907411-index",
  azureAISearchApiKey: config.azureSearchKey!,
  azureAISearchEndpoint: config.azureSearchEndpoint!,
  azureOpenAIApiKey: config.azureOpenAIKey!,
  azureOpenAIEndpoint: config.azureOpenAIEndpoint!,
  azureOpenAIEmbeddingDeploymentName: config.azureOpenAIEmbeddingDeploymentName!,
  // Retrieval tuning (no semantic ranker):
  kNearestNeighborsCount: 16,
  topPerVariant: 12,
  maxPerParent: 4,
  maxTotalAfterFusion: 6,
  minTopScore: 0.12,
  minScoreGap: 0.01,
  maxContextTokens: 1200
});

const instructions = loadInstructions();

const createTokenFactory = () => {
  return async (scope: string | string[], tenantId?: string): Promise<string> => {
    const managedIdentityCredential = new ManagedIdentityCredential({
      clientId: process.env.CLIENT_ID
    });
    const scopes = Array.isArray(scope) ? scope : [scope];
    const tokenResponse = await managedIdentityCredential.getToken(scopes, { tenantId });
    return tokenResponse.token!;
  };
};

const tokenCredentials: TokenCredentials = {
  clientId: process.env.CLIENT_ID || "",
  token: createTokenFactory()
};

const credentialOptions = config.MicrosoftAppType === "UserAssignedMsi" ? { ...tokenCredentials } : undefined;

const app = new App({
  ...credentialOptions,
  storage
});

// Load instructions once
function loadInstructions(): string {
  const instructionPath = path.join(__dirname, "instructions.txt");
  return fs.readFileSync(instructionPath, "utf-8").trim();
}

/**
 * Keep only the first system message and the last N-1 other turns.
 */
function trimToLastN<T extends Message>(messages: T[], max = MAX_MESSAGES): T[] {
  if (!Array.isArray(messages) || max <= 0) return [];
  const firstSystemIdx = messages.findIndex((m) => m?.role === "system");
  const headSystem = firstSystemIdx >= 0 ? messages[firstSystemIdx] : undefined;

  const nonSystem = messages.filter((m) => m?.role !== "system");

  const tailBudget = headSystem ? Math.max(0, max - 1) : max;
  const trimmedTail = nonSystem.slice(-tailBudget);
  return headSystem ? [headSystem, ...trimmedTail] : trimmedTail;
}

/**
 * Purge older conversations for the same user (keeps storage lean).
 */
function purgeOlderConversationsForUser(storage: LocalStorage, userId: string, currentConversationKey: string) {
  const allKeys: string[] = storage.keys;
  if (!Array.isArray(allKeys) || allKeys.length === 0) return;

  const suffix = `/${userId}`;
  for (const key of allKeys) {
    if (typeof key !== "string") continue;
    if (!key.endsWith(suffix)) continue;
    if (key === currentConversationKey) continue;
    storage.delete(key);
  }
}

// ---------- TINY LLM KEYWORD ROUTER (heuristic removed) ----------

type SearchDecision = { shouldSearch: boolean; keywords: string[] };

/**
 * Tiny LLM router: returns { shouldSearch, keywords }
 */
async function shouldSearchLLMKeywords(query: string): Promise<SearchDecision> {
  const deployment = config.azureOpenAIDeploymentName;
  const client = new OpenAI({
    apiKey: config.azureOpenAIKey,
    baseURL: `${config.azureOpenAIEndpoint}/openai/deployments/${deployment}`,
    defaultQuery: { "api-version": "2024-10-21" }
  });

  const sys = `
Decide if the user's query requires external knowledge retrieval (search over company documents).
If retrieval is needed, output concise keywords.
Respond ONLY in strict JSON:
{"shouldSearch": true/false, "keywords": ["word1","word2",...]}
Rules:
- If retrieval is not needed, use {"shouldSearch": false, "keywords": []}.
- Use 1–8 short keyword tokens/phrases.
- Avoid filler words, punctuation, or duplicates.
`.trim();

  const msg = [
    { role: "system", content: sys },
    { role: "user", content: query }
  ] satisfies ChatCompletionMessageParam[];

  const res = await client.chat.completions.create({
    model: config.azureOpenAIDeploymentName,
    messages: msg,
    max_tokens: 48,
    temperature: 0
  });

  const raw = (res.choices[0]?.message?.content ?? "").trim();

  try {
    const parsed = JSON.parse(raw) as SearchDecision;
    if (typeof parsed?.shouldSearch === "boolean" && Array.isArray(parsed?.keywords)) return parsed;
  } catch {
    const m = raw.match(/\{[\s\S]*\}/);
    if (m) {
      try {
        const parsed = JSON.parse(m[0]) as SearchDecision;
        if (typeof parsed?.shouldSearch === "boolean" && Array.isArray(parsed?.keywords)) return parsed;
      } catch { }
    }
  }

  return { shouldSearch: false, keywords: [] };
}

// ------------------------ MESSAGE HANDLER ------------------------

app.on("message", async ({ send, activity }) => {
  const conversationKey = `${activity.conversation.id}/${activity.from.id}`;
  purgeOlderConversationsForUser(storage, activity.from.id, conversationKey);

  let messages: Message[] = storage.get(conversationKey) || [];

  try {
    if (messages.length === 0) {
      messages.push({ role: "system", content: instructions });
      // console.log(messages.slice(-1));
    }

    messages = trimToLastN(messages, MAX_MESSAGES);

    // Heuristic removed: rely solely on LLM router
    const decision = await shouldSearchLLMKeywords(activity.text);
    const shouldSearch = decision.shouldSearch;
    const keywords = decision.keywords ?? [];

    // 🔍 Debug log line
    console.log(
      `[Search Router] shouldSearch=${shouldSearch} | keywords=${keywords.length ? keywords.join(", ") : "(none)"}`
    );

    let turnMessages: Message[] = [...messages];
    let contextData = "";

    if (shouldSearch) {
      const augmentedQuery =
        keywords.length > 0
          ? `${activity.text}\n\n[focus terms: ${keywords.join(", ")}]`
          : activity.text;

      contextData = (await dataSource.renderContext(augmentedQuery)).trim();
      if (contextData) {
        turnMessages.push({
          role: "system",
          content: `Context from Azure AI Search:\n${contextData}`
        });
      }
    }

    const prompt = new ChatPrompt({
      messages: [...turnMessages],
      model: new OpenAIChatModel({
        model: config.azureOpenAIDeploymentName,
        apiKey: config.azureOpenAIKey,
        endpoint: config.azureOpenAIEndpoint,
        apiVersion: "2024-10-21"
      })
    });

    const response = await prompt.send(activity.text);

    messages.push({ role: "user", content: activity.text });
    messages.push({ role: "model", content: response.content });
    // console.log(messages.slice(-2));

    let result = null;
    try {
      result = JSON.parse(response.content);
    } catch (error) {
      console.error(`Response is not valid json, using raw text. error: ${error}`);
      await send(response.content);
      storage.set(conversationKey, messages);
      return;
    }

    let position = 1;
    let content = "";
    const footnotes: string[] = [];

    if (result?.results?.length) {
      for (const item of result.results) {
        if (item.citationTitle) {
          content += `${item.answer}[${position}]  \n`;
          const note =
            `**[${position}]** ${item.citationTitle}` +
            (item.citationContent ? ` — ${item.citationContent}` : "");
          footnotes.push(note);
          position++;
        } else {
          content += `${item.answer}  \n`;
        }
      }
    }

    if (footnotes.length) {
      content += `\n---\n**Sources**\n` + footnotes.map((f) => `- ${f}`).join("\n");
    }

    await send(content);
    storage.set(conversationKey, messages);
  } catch (error) {
    console.error("Error processing message:", error);
    await send("Sorry, I encountered an error while processing your message.");
  }
});

export default app;
