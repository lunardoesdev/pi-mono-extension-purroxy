import { GoogleGenAI, FunctionCallingConfigMode, ThinkingLevel } from "@google/genai";
import type { ExtensionAPI, ProviderConfig, ProviderModelConfig } from "@mariozechner/pi-coding-agent";
import { calculateCost, createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";

const DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1";
const DEFAULT_DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"];

type OverrideSpec = {
  provider: string;
  baseUrlKeys?: string[];
  apiKeyKeys?: string[];
  normalizeBaseUrl?: (value: string) => string;
  api?: ProviderConfig["api"];
  streamSimple?: ProviderConfig["streamSimple"];
};

type GoogleThinkingConfig = {
  enabled: boolean;
  level?: "MINIMAL" | "LOW" | "MEDIUM" | "HIGH";
  budgetTokens?: number;
};

type GoogleVertexProxyOptions = SimpleStreamOptions & {
  thinking?: GoogleThinkingConfig;
  project?: string;
  location?: string;
  toolChoice?: "auto" | "none" | "required";
};

export default function register(pi: ExtensionAPI) {
  registerOverride(pi, {
    provider: "anthropic",
    baseUrlKeys: ["ANTHROPIC_BASE_URL"],
    normalizeBaseUrl: normalizeAnthropicBaseUrl
  });

  registerOverride(pi, {
    provider: "openai",
    baseUrlKeys: ["OPENAI_BASE_URL"],
    normalizeBaseUrl: normalizeOpenAICompatibleBaseUrl
  });

  registerOverride(pi, {
    provider: "google",
    baseUrlKeys: ["GEMINI_BASE_URL", "GOOGLE_BASE_URL"],
    normalizeBaseUrl: normalizeGoogleBaseUrl
  });

  registerOverride(pi, {
    provider: "xai",
    baseUrlKeys: ["XAI_BASE_URL", "GROK_BASE_URL"],
    apiKeyKeys: ["GROK_API_KEY", "XAI_API_KEY"],
    normalizeBaseUrl: normalizeOpenAICompatibleBaseUrl
  });

  registerOverride(pi, {
    provider: "zai",
    baseUrlKeys: ["GLM_BASE_URL", "ZAI_BASE_URL", "ZHIPU_BASE_URL"],
    apiKeyKeys: ["GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY"],
    normalizeBaseUrl: normalizeOpenAICompatibleBaseUrl
  });

  // Override the Vertex API handler so model.baseUrl is honored.
  registerOverride(pi, {
    provider: "google-vertex",
    baseUrlKeys: ["VERTEX_BASE_URL", "GOOGLE_VERTEX_BASE_URL"],
    apiKeyKeys: ["VERTEX_API_KEY", "GOOGLE_VERTEX_API_KEY", "GOOGLE_CLOUD_API_KEY"],
    normalizeBaseUrl: normalizeVertexBaseUrl,
    api: "google-vertex",
    streamSimple: streamSimpleGoogleVertexWithBaseUrl
  });

  const deepseekProvider = buildDeepSeekProvider();
  if (deepseekProvider) {
    pi.registerProvider("deepseek", deepseekProvider);
  }
}

function registerOverride(pi: ExtensionAPI, spec: OverrideSpec) {
  const baseUrl = readFirstEnvValue(spec.baseUrlKeys);
  const apiKeyKey = readFirstPresentEnvKey(spec.apiKeyKeys);
  const shouldRegister = !!baseUrl || !!apiKeyKey || !!spec.streamSimple;

  if (!shouldRegister) {
    return;
  }

  const config: ProviderConfig = {};

  if (baseUrl) {
    config.baseUrl = spec.normalizeBaseUrl ? spec.normalizeBaseUrl(baseUrl) : normalizeUrl(baseUrl);
  }

  if (apiKeyKey) {
    config.apiKey = apiKeyKey;
  }

  if (spec.api) {
    config.api = spec.api;
  }

  if (spec.streamSimple) {
    config.streamSimple = spec.streamSimple;
  }

  pi.registerProvider(spec.provider, config);
}

function buildDeepSeekProvider(): ProviderConfig | undefined {
  const apiKeyKey = readFirstPresentEnvKey(["DEEPSEEK_API_KEY"]);
  const baseUrl =
    normalizeOpenAICompatibleBaseUrl(
      readFirstEnvValue(["DEEPSEEK_BASE_URL"]) ?? (apiKeyKey ? DEFAULT_DEEPSEEK_BASE_URL : "")
    ) || undefined;

  if (!baseUrl || !apiKeyKey) {
    return undefined;
  }

  return {
    baseUrl,
    apiKey: apiKeyKey,
    api: "openai-completions",
    models: buildDeepSeekModels(),
  };
}

function buildDeepSeekModels(): ProviderModelConfig[] {
  const modelIds = parseCsvEnv("DEEPSEEK_MODELS", DEFAULT_DEEPSEEK_MODELS);

  return modelIds.map((id) => {
    const reasoning = id.toLowerCase().includes("reasoner") || id.toLowerCase().includes("r1");
    return {
      id,
      name: id,
      reasoning,
      input: ["text"],
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0
      },
      contextWindow: 128000,
      maxTokens: reasoning ? 64000 : 8192,
      compat: {
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: true,
        supportsUsageInStreaming: true,
        maxTokensField: "max_completion_tokens",
        requiresToolResultName: false,
        requiresAssistantAfterToolResult: false,
        requiresThinkingAsText: false,
        thinkingFormat: "openai",
        openRouterRouting: {},
        vercelGatewayRouting: {},
        supportsStrictMode: true
      }
    };
  });
}

function readFirstEnvValue(keys?: string[]) {
  if (!keys) {
    return undefined;
  }

  for (const key of keys) {
    const value = process.env[key]?.trim();
    if (value) {
      return value;
    }
  }

  return undefined;
}

function readFirstPresentEnvKey(keys?: string[]) {
  if (!keys) {
    return undefined;
  }

  for (const key of keys) {
    const value = process.env[key];
    if (typeof value === "string" && value.trim()) {
      return key;
    }
  }

  return undefined;
}

function parseCsvEnv(key: string, fallback: string[]) {
  const value = process.env[key]?.trim();
  if (!value) {
    return fallback;
  }

  const items = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  return items.length > 0 ? items : fallback;
}

function normalizeUrl(value: string) {
  return value.trim().replace(/\/+$/, "");
}

function stripEndpointSuffix(value: string, suffixes: string[]) {
  const normalized = normalizeUrl(value);
  const lower = normalized.toLowerCase();

  for (const suffix of suffixes) {
    if (lower.endsWith(suffix.toLowerCase())) {
      return normalized.slice(0, normalized.length - suffix.length).replace(/\/+$/, "");
    }
  }

  return normalized;
}

function normalizeAnthropicBaseUrl(value: string) {
  return stripEndpointSuffix(value, ["/v1/messages", "/messages", "/v1"]);
}

function normalizeOpenAICompatibleBaseUrl(value: string) {
  return stripEndpointSuffix(value, ["/chat/completions", "/responses", "/completions"]);
}

function normalizeGoogleBaseUrl(value: string) {
  return stripEndpointSuffix(value, ["/generateContent", "/streamGenerateContent"]);
}

function normalizeVertexBaseUrl(value: string) {
  return stripEndpointSuffix(value, ["/streamGenerateContent", "/generateContent", "/v1"]);
}

function sanitizeSurrogates(text: string) {
  return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

function buildBaseOptions(
  model: Model<any>,
  options: SimpleStreamOptions | undefined,
  apiKey: string | undefined
): GoogleVertexProxyOptions {
  return {
    temperature: options?.temperature,
    maxTokens: options?.maxTokens || Math.min(model.maxTokens, 32000),
    signal: options?.signal,
    apiKey: apiKey || options?.apiKey,
    cacheRetention: options?.cacheRetention,
    sessionId: options?.sessionId,
    headers: options?.headers,
    onPayload: options?.onPayload,
    maxRetryDelayMs: options?.maxRetryDelayMs,
    metadata: options?.metadata
  };
}

function clampReasoning(effort: SimpleStreamOptions["reasoning"]) {
  return effort === "xhigh" ? "high" : effort;
}

function streamSimpleGoogleVertexWithBaseUrl(
  model: Model<any>,
  context: Context,
  options?: SimpleStreamOptions
) {
  const base = buildBaseOptions(model, options, undefined);

  if (!options?.reasoning) {
    return streamGoogleVertexWithBaseUrl(model, context, {
      ...base,
      thinking: { enabled: false }
    });
  }

  const effort = clampReasoning(options.reasoning);
  if (!effort) {
    return streamGoogleVertexWithBaseUrl(model, context, {
      ...base,
      thinking: { enabled: false }
    });
  }

  if (isGemini3ProModel(model) || isGemini3FlashModel(model)) {
    return streamGoogleVertexWithBaseUrl(model, context, {
      ...base,
      thinking: {
        enabled: true,
        level: getGemini3ThinkingLevel(effort, model)
      }
    });
  }

  return streamGoogleVertexWithBaseUrl(model, context, {
    ...base,
    thinking: {
      enabled: true,
      budgetTokens: getGoogleBudget(model, effort, options.thinkingBudgets)
    }
  });
}

function streamGoogleVertexWithBaseUrl(
  model: Model<any>,
  context: Context,
  options: GoogleVertexProxyOptions
) {
  const stream = createAssistantMessageEventStream();

  void (async () => {
    const output: any = {
      role: "assistant",
      content: [],
      api: "google-vertex",
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
      },
      stopReason: "stop",
      timestamp: Date.now()
    };

    try {
      const apiKey = resolveVertexApiKey(options);
      const client = apiKey
        ? createVertexClientWithApiKey(model, apiKey, options.headers)
        : createVertexClient(
            model,
            resolveVertexProject(options),
            resolveVertexLocation(options),
            options.headers
          );

      let params = buildVertexParams(model, context, options);
      const nextParams = await options.onPayload?.(params, model);
      if (nextParams !== undefined) {
        params = nextParams as typeof params;
      }

      const googleStream = await client.models.generateContentStream(params);

      stream.push({ type: "start", partial: output });

      let currentBlock: any = null;
      const blocks = output.content;
      const blockIndex = () => blocks.length - 1;

      for await (const chunk of googleStream as any) {
        output.responseId ||= chunk.responseId;

        const candidate = chunk.candidates?.[0];
        if (candidate?.content?.parts) {
          for (const part of candidate.content.parts) {
            if (part.text !== undefined) {
              const thinking = isThinkingPart(part);

              if (
                !currentBlock ||
                (thinking && currentBlock.type !== "thinking") ||
                (!thinking && currentBlock.type !== "text")
              ) {
                if (currentBlock) {
                  if (currentBlock.type === "text") {
                    stream.push({
                      type: "text_end",
                      contentIndex: blockIndex(),
                      content: currentBlock.text,
                      partial: output
                    });
                  } else {
                    stream.push({
                      type: "thinking_end",
                      contentIndex: blockIndex(),
                      content: currentBlock.thinking,
                      partial: output
                    });
                  }
                }

                if (thinking) {
                  currentBlock = {
                    type: "thinking",
                    thinking: "",
                    thinkingSignature: undefined
                  };
                  output.content.push(currentBlock);
                  stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
                } else {
                  currentBlock = { type: "text", text: "" };
                  output.content.push(currentBlock);
                  stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
                }
              }

              if (currentBlock.type === "thinking") {
                currentBlock.thinking += part.text;
                currentBlock.thinkingSignature = retainThoughtSignature(
                  currentBlock.thinkingSignature,
                  part.thoughtSignature
                );
                stream.push({
                  type: "thinking_delta",
                  contentIndex: blockIndex(),
                  delta: part.text,
                  partial: output
                });
              } else {
                currentBlock.text += part.text;
                currentBlock.textSignature = retainThoughtSignature(
                  currentBlock.textSignature,
                  part.thoughtSignature
                );
                stream.push({
                  type: "text_delta",
                  contentIndex: blockIndex(),
                  delta: part.text,
                  partial: output
                });
              }
            }

            if (part.functionCall) {
              if (currentBlock) {
                if (currentBlock.type === "text") {
                  stream.push({
                    type: "text_end",
                    contentIndex: blockIndex(),
                    content: currentBlock.text,
                    partial: output
                  });
                } else {
                  stream.push({
                    type: "thinking_end",
                    contentIndex: blockIndex(),
                    content: currentBlock.thinking,
                    partial: output
                  });
                }
                currentBlock = null;
              }

              const providedId = part.functionCall.id;
              const needsNewId =
                !providedId || output.content.some((block: any) => block.type === "toolCall" && block.id === providedId);
              const toolCallId = needsNewId
                ? `${part.functionCall.name}_${Date.now()}_${output.content.length + 1}`
                : providedId;

              const toolCall: any = {
                type: "toolCall" as const,
                id: toolCallId,
                name: part.functionCall.name || "",
                arguments: part.functionCall.args ?? {},
                ...(part.thoughtSignature && { thoughtSignature: part.thoughtSignature })
              };

              output.content.push(toolCall);
              stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
              stream.push({
                type: "toolcall_delta",
                contentIndex: blockIndex(),
                delta: JSON.stringify(toolCall.arguments),
                partial: output
              });
              stream.push({
                type: "toolcall_end",
                contentIndex: blockIndex(),
                toolCall,
                partial: output
              });
            }
          }
        }

        if (candidate?.finishReason) {
          output.stopReason = mapStopReason(candidate.finishReason);
          if (output.content.some((block: any) => block.type === "toolCall")) {
            output.stopReason = "toolUse";
          }
        }

        if (chunk.usageMetadata) {
          output.usage = {
            input:
              (chunk.usageMetadata.promptTokenCount || 0) -
              (chunk.usageMetadata.cachedContentTokenCount || 0),
            output:
              (chunk.usageMetadata.candidatesTokenCount || 0) +
              (chunk.usageMetadata.thoughtsTokenCount || 0),
            cacheRead: chunk.usageMetadata.cachedContentTokenCount || 0,
            cacheWrite: 0,
            totalTokens: chunk.usageMetadata.totalTokenCount || 0,
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              total: 0
            }
          };
          calculateCost(model, output.usage);
        }
      }

      if (currentBlock) {
        if (currentBlock.type === "text") {
          stream.push({
            type: "text_end",
            contentIndex: blockIndex(),
            content: currentBlock.text,
            partial: output
          });
        } else {
          stream.push({
            type: "thinking_end",
            contentIndex: blockIndex(),
            content: currentBlock.thinking,
            partial: output
          });
        }
      }

      if (options.signal?.aborted) {
        throw new Error("Request was aborted");
      }

      if (output.stopReason === "aborted" || output.stopReason === "error") {
        throw new Error("An unknown error occurred");
      }

      stream.push({ type: "done", reason: output.stopReason, message: output });
      stream.end(output);
    } catch (error) {
      for (const block of output.content) {
        if ("index" in block) {
          delete block.index;
        }
      }

      output.stopReason = options.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end(output);
    }
  })();

  return stream;
}

function createVertexClient(
  model: Model<any>,
  project: string,
  location: string,
  headers?: Record<string, string>
) {
  const httpOptions: Record<string, unknown> = {};

  if (model.baseUrl) {
    httpOptions.baseUrl = model.baseUrl;
  }

  if (model.headers || headers) {
    httpOptions.headers = { ...model.headers, ...headers };
  }

  const hasHttpOptions = Object.keys(httpOptions).length > 0;

  return new GoogleGenAI({
    vertexai: true,
    project,
    location,
    apiVersion: "v1",
    httpOptions: hasHttpOptions ? httpOptions : undefined
  });
}

function createVertexClientWithApiKey(
  model: Model<any>,
  apiKey: string,
  headers?: Record<string, string>
) {
  const httpOptions: Record<string, unknown> = {};

  if (model.baseUrl) {
    httpOptions.baseUrl = model.baseUrl;
  }

  if (model.headers || headers) {
    httpOptions.headers = { ...model.headers, ...headers };
  }

  const hasHttpOptions = Object.keys(httpOptions).length > 0;

  return new GoogleGenAI({
    vertexai: true,
    apiKey,
    apiVersion: "v1",
    httpOptions: hasHttpOptions ? httpOptions : undefined
  });
}

function resolveVertexApiKey(options?: GoogleVertexProxyOptions) {
  const apiKey =
    options?.apiKey?.trim() ||
    process.env.VERTEX_API_KEY?.trim() ||
    process.env.GOOGLE_VERTEX_API_KEY?.trim() ||
    process.env.GOOGLE_CLOUD_API_KEY?.trim();

  if (!apiKey || /^<[^>]+>$/.test(apiKey)) {
    return undefined;
  }

  return apiKey;
}

function resolveVertexProject(options?: GoogleVertexProxyOptions) {
  const project =
    options?.project || process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT;

  if (!project) {
    throw new Error(
      "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or provide it through the environment."
    );
  }

  return project;
}

function resolveVertexLocation(options?: GoogleVertexProxyOptions) {
  const location = options?.location || process.env.GOOGLE_CLOUD_LOCATION;

  if (!location) {
    throw new Error(
      "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or provide it through the environment."
    );
  }

  return location;
}

function buildVertexParams(
  model: Model<any>,
  context: Context,
  options: GoogleVertexProxyOptions = {}
) {
  const contents = convertMessages(model, context);
  const generationConfig: Record<string, unknown> = {};

  if (options.temperature !== undefined) {
    generationConfig.temperature = options.temperature;
  }

  if (options.maxTokens !== undefined) {
    generationConfig.maxOutputTokens = options.maxTokens;
  }

  const config: Record<string, unknown> = {
    ...(Object.keys(generationConfig).length > 0 ? generationConfig : {}),
    ...(context.systemPrompt ? { systemInstruction: sanitizeSurrogates(context.systemPrompt) } : {}),
    ...(context.tools && context.tools.length > 0 ? { tools: convertTools(context.tools) } : {})
  };

  if (context.tools && context.tools.length > 0 && options.toolChoice) {
    config.toolConfig = {
      functionCallingConfig: {
        mode: mapToolChoice(options.toolChoice)
      }
    };
  }

  if (options.thinking?.enabled && model.reasoning) {
    const thinkingConfig: Record<string, unknown> = { includeThoughts: true };
    if (options.thinking.level !== undefined) {
      thinkingConfig.thinkingLevel = THINKING_LEVEL_MAP[options.thinking.level];
    } else if (options.thinking.budgetTokens !== undefined) {
      thinkingConfig.thinkingBudget = options.thinking.budgetTokens;
    }
    config.thinkingConfig = thinkingConfig;
  } else if (model.reasoning && options.thinking && !options.thinking.enabled) {
    config.thinkingConfig = getDisabledThinkingConfig(model);
  }

  if (options.signal) {
    if (options.signal.aborted) {
      throw new Error("Request aborted");
    }
    config.abortSignal = options.signal;
  }

  return {
    model: model.id,
    contents,
    config
  };
}

const THINKING_LEVEL_MAP = {
  MINIMAL: ThinkingLevel.MINIMAL,
  LOW: ThinkingLevel.LOW,
  MEDIUM: ThinkingLevel.MEDIUM,
  HIGH: ThinkingLevel.HIGH
};

function isThinkingPart(part: any) {
  return part.thought === true;
}

function retainThoughtSignature(existing: string | undefined, incoming: string | undefined) {
  if (typeof incoming === "string" && incoming.length > 0) {
    return incoming;
  }
  return existing;
}

function mapToolChoice(choice: string) {
  switch (choice) {
    case "auto":
      return FunctionCallingConfigMode.AUTO;
    case "none":
      return FunctionCallingConfigMode.NONE;
    case "required":
      return FunctionCallingConfigMode.ANY;
    default:
      return FunctionCallingConfigMode.AUTO;
  }
}

function mapStopReason(reason: string) {
  switch (reason) {
    case "STOP":
      return "stop";
    case "MAX_TOKENS":
      return "length";
    case "MALFORMED_FUNCTION_CALL":
    case "UNEXPECTED_TOOL_CALL":
    case "TOOL_USE":
      return "toolUse";
    default:
      return "error";
  }
}

function requiresToolCallId(modelId: string) {
  return modelId.startsWith("claude-") || modelId.startsWith("gpt-oss-");
}

function convertMessages(model: Model<any>, context: Context) {
  const contents: any[] = [];
  const normalizeToolCallId = (id: string) => {
    if (!requiresToolCallId(model.id)) {
      return id;
    }
    return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  };

  const transformedMessages = transformMessages(context.messages as any[], model, normalizeToolCallId);

  for (const msg of transformedMessages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        contents.push({
          role: "user",
          parts: [{ text: sanitizeSurrogates(msg.content) }]
        });
      } else {
        const parts = msg.content.map((item: any) =>
          item.type === "text"
            ? { text: sanitizeSurrogates(item.text) }
            : {
                inlineData: {
                  mimeType: item.mimeType,
                  data: item.data
                }
              }
        );

        const filteredParts = !model.input.includes("image")
          ? parts.filter((part: any) => part.text !== undefined)
          : parts;

        if (filteredParts.length === 0) {
          continue;
        }

        contents.push({
          role: "user",
          parts: filteredParts
        });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const parts: any[] = [];
      const isSameProviderAndModel = msg.provider === model.provider && msg.model === model.id;

      for (const block of msg.content) {
        if (block.type === "text") {
          if (!block.text || block.text.trim() === "") {
            continue;
          }

          parts.push({
            text: sanitizeSurrogates(block.text),
            ...(resolveThoughtSignature(isSameProviderAndModel, block.textSignature)
              ? { thoughtSignature: block.textSignature }
              : {})
          });
          continue;
        }

        if (block.type === "thinking") {
          if (!block.thinking || block.thinking.trim() === "") {
            continue;
          }

          if (isSameProviderAndModel) {
            parts.push({
              thought: true,
              text: sanitizeSurrogates(block.thinking),
              ...(resolveThoughtSignature(isSameProviderAndModel, block.thinkingSignature)
                ? { thoughtSignature: block.thinkingSignature }
                : {})
            });
          } else {
            parts.push({
              text: sanitizeSurrogates(block.thinking)
            });
          }
          continue;
        }

        if (block.type === "toolCall") {
          const thoughtSignature = resolveThoughtSignature(isSameProviderAndModel, block.thoughtSignature);
          const isGemini3 = model.id.toLowerCase().includes("gemini-3");
          const effectiveSignature =
            thoughtSignature || (isGemini3 ? "skip_thought_signature_validator" : undefined);

          parts.push({
            functionCall: {
              name: block.name,
              args: block.arguments ?? {},
              ...(requiresToolCallId(model.id) ? { id: block.id } : {})
            },
            ...(effectiveSignature ? { thoughtSignature: effectiveSignature } : {})
          });
        }
      }

      if (parts.length > 0) {
        contents.push({
          role: "model",
          parts
        });
      }
      continue;
    }

    if (msg.role === "toolResult") {
      const textResult = msg.content
        .filter((item: any) => item.type === "text")
        .map((item: any) => item.text)
        .join("\n");

      const imageContent = model.input.includes("image")
        ? msg.content.filter((item: any) => item.type === "image")
        : [];

      const hasText = textResult.length > 0;
      const hasImages = imageContent.length > 0;
      const supportsMultimodal = supportsMultimodalFunctionResponse(model.id);
      const responseValue = hasText ? sanitizeSurrogates(textResult) : hasImages ? "(see attached image)" : "";
      const imageParts = imageContent.map((imageBlock: any) => ({
        inlineData: {
          mimeType: imageBlock.mimeType,
          data: imageBlock.data
        }
      }));

      const functionResponsePart = {
        functionResponse: {
          name: msg.toolName,
          response: msg.isError ? { error: responseValue } : { output: responseValue },
          ...(hasImages && supportsMultimodal ? { parts: imageParts } : {}),
          ...(requiresToolCallId(model.id) ? { id: msg.toolCallId } : {})
        }
      };

      const lastContent = contents[contents.length - 1];
      if (lastContent?.role === "user" && lastContent.parts?.some((part: any) => part.functionResponse)) {
        lastContent.parts.push(functionResponsePart);
      } else {
        contents.push({
          role: "user",
          parts: [functionResponsePart]
        });
      }

      if (hasImages && !supportsMultimodal) {
        contents.push({
          role: "user",
          parts: [{ text: "Tool result image:" }, ...imageParts]
        });
      }
    }
  }

  return contents;
}

function convertTools(tools: any[]) {
  if (tools.length === 0) {
    return undefined;
  }

  return [
    {
      functionDeclarations: tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        parametersJsonSchema: tool.parameters
      }))
    }
  ];
}

function supportsMultimodalFunctionResponse(modelId: string) {
  const match = modelId.toLowerCase().match(/^gemini(?:-live)?-(\d+)/);
  if (!match) {
    return true;
  }
  return Number.parseInt(match[1], 10) >= 3;
}

const BASE64_SIGNATURE = /^[A-Za-z0-9+/]+={0,2}$/;

function resolveThoughtSignature(isSameProviderAndModel: boolean, signature?: string) {
  if (!isSameProviderAndModel || !signature) {
    return undefined;
  }

  if (signature.length % 4 !== 0 || !BASE64_SIGNATURE.test(signature)) {
    return undefined;
  }

  return signature;
}

function transformMessages(messages: any[], model: Model<any>, normalizeToolCallId?: (id: string) => string) {
  const toolCallIdMap = new Map<string, string>();

  const transformed = messages.map((msg) => {
    if (msg.role === "user") {
      return msg;
    }

    if (msg.role === "toolResult") {
      const normalizedId = toolCallIdMap.get(msg.toolCallId);
      return normalizedId && normalizedId !== msg.toolCallId
        ? { ...msg, toolCallId: normalizedId }
        : msg;
    }

    if (msg.role !== "assistant") {
      return msg;
    }

    const isSameModel =
      msg.provider === model.provider && msg.api === model.api && msg.model === model.id;

    const transformedContent = msg.content.flatMap((block: any) => {
      if (block.type === "thinking") {
        if (block.redacted) {
          return isSameModel ? block : [];
        }
        if (isSameModel && block.thinkingSignature) {
          return block;
        }
        if (!block.thinking || block.thinking.trim() === "") {
          return [];
        }
        return isSameModel ? block : { type: "text", text: block.thinking };
      }

      if (block.type === "text") {
        return isSameModel ? block : { type: "text", text: block.text };
      }

      if (block.type === "toolCall") {
        let normalizedToolCall = !isSameModel && block.thoughtSignature
          ? { ...block, thoughtSignature: undefined }
          : block;

        if (!isSameModel && normalizeToolCallId) {
          const normalizedId = normalizeToolCallId(block.id);
          if (normalizedId !== block.id) {
            toolCallIdMap.set(block.id, normalizedId);
            normalizedToolCall = { ...normalizedToolCall, id: normalizedId };
          }
        }

        return normalizedToolCall;
      }

      return block;
    });

    return { ...msg, content: transformedContent };
  });

  const result: any[] = [];
  let pendingToolCalls: any[] = [];
  let existingToolResultIds = new Set<string>();

  for (const msg of transformed) {
    if (msg.role === "assistant") {
      if (pendingToolCalls.length > 0) {
        for (const toolCall of pendingToolCalls) {
          if (!existingToolResultIds.has(toolCall.id)) {
            result.push({
              role: "toolResult",
              toolCallId: toolCall.id,
              toolName: toolCall.name,
              content: [{ type: "text", text: "No result provided" }],
              isError: true,
              timestamp: Date.now()
            });
          }
        }
        pendingToolCalls = [];
        existingToolResultIds = new Set<string>();
      }

      if (msg.stopReason === "error" || msg.stopReason === "aborted") {
        continue;
      }

      const toolCalls = msg.content.filter((block: any) => block.type === "toolCall");
      if (toolCalls.length > 0) {
        pendingToolCalls = toolCalls;
        existingToolResultIds = new Set<string>();
      }

      result.push(msg);
      continue;
    }

    if (msg.role === "toolResult") {
      existingToolResultIds.add(msg.toolCallId);
      result.push(msg);
      continue;
    }

    if (msg.role === "user") {
      if (pendingToolCalls.length > 0) {
        for (const toolCall of pendingToolCalls) {
          if (!existingToolResultIds.has(toolCall.id)) {
            result.push({
              role: "toolResult",
              toolCallId: toolCall.id,
              toolName: toolCall.name,
              content: [{ type: "text", text: "No result provided" }],
              isError: true,
              timestamp: Date.now()
            });
          }
        }
        pendingToolCalls = [];
        existingToolResultIds = new Set<string>();
      }
      result.push(msg);
      continue;
    }

    result.push(msg);
  }

  return result;
}

function isGemini3ProModel(model: Model<any>) {
  return /gemini-3(?:\.\d+)?-pro/.test(model.id.toLowerCase());
}

function isGemini3FlashModel(model: Model<any>) {
  return /gemini-3(?:\.\d+)?-flash/.test(model.id.toLowerCase());
}

function getDisabledThinkingConfig(model: Model<any>) {
  if (isGemini3ProModel(model)) {
    return { thinkingLevel: ThinkingLevel.LOW };
  }

  if (isGemini3FlashModel(model)) {
    return { thinkingLevel: ThinkingLevel.MINIMAL };
  }

  return { thinkingBudget: 0 };
}

function getGemini3ThinkingLevel(
  effort: Exclude<ReturnType<typeof clampReasoning>, undefined>,
  model: Model<any>
) {
  if (isGemini3ProModel(model)) {
    switch (effort) {
      case "minimal":
      case "low":
        return "LOW";
      case "medium":
      case "high":
        return "HIGH";
    }
  }

  switch (effort) {
    case "minimal":
      return "MINIMAL";
    case "low":
      return "LOW";
    case "medium":
      return "MEDIUM";
    case "high":
      return "HIGH";
  }
}

function getGoogleBudget(
  model: Model<any>,
  effort: Exclude<ReturnType<typeof clampReasoning>, undefined>,
  customBudgets?: Partial<Record<"minimal" | "low" | "medium" | "high", number>>
) {
  if (customBudgets?.[effort] !== undefined) {
    return customBudgets[effort];
  }

  if (model.id.includes("2.5-pro")) {
    return {
      minimal: 128,
      low: 2048,
      medium: 8192,
      high: 32768
    }[effort];
  }

  if (model.id.includes("2.5-flash")) {
    return {
      minimal: 128,
      low: 2048,
      medium: 8192,
      high: 24576
    }[effort];
  }

  return -1;
}
