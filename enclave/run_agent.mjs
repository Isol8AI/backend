#!/usr/bin/env node
/**
 * Bridge between Python (bedrock_server.py) and OpenClaw (runEmbeddedPiAgent).
 *
 * Protocol:
 *   stdin:  Single JSON object with request parameters
 *   stdout: NDJSON events, one per line (consumed by Python bridge)
 *   stderr: Diagnostic logs (not parsed by Python)
 *   exit 0: Success (errors are sent as NDJSON events, not exit codes)
 *   exit 1: Bridge-level failure (couldn't parse input, couldn't load OpenClaw)
 *
 * Required stdin fields:
 *   - stateDir:   Absolute path to extracted agent tarball (tmpfs)
 *   - agentName:  Agent identifier (matches agents/{name}/ directory)
 *   - message:    User message text
 *
 * Optional stdin fields:
 *   - model:      LLM model ID (default: auto-discovered from Bedrock)
 *   - provider:   LLM provider name (default: "amazon-bedrock")
 *   - timeoutMs:  Max execution time in ms (default: 90000)
 *   - sessionId:  Session ID for conversation continuity (default: auto-generated)
 *
 * Environment variables:
 *   - OPENCLAW_PATH: Path to OpenClaw dist directory (default: /opt/openclaw)
 *   - AWS_PROFILE:   Set to "default" to enable IMDS credential chain
 *   - AWS_REGION:    AWS region for Bedrock (default: us-east-1)
 */

// ---------------------------------------------------------------------------
// 0. Nitro Enclave networking (handled by CJS preload)
// ---------------------------------------------------------------------------
// The Nitro Enclave has NO network stack. HTTPS proxying is enabled via
// --require proxy_bootstrap.cjs (a CJS preload) which uses global-agent to
// route all https.request() calls through the vsock TCP bridge on 127.0.0.1:3128.

import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import * as readline from "node:readline";
import { pathToFileURL } from "node:url";

// ---------------------------------------------------------------------------
// 1. Read JSON request from stdin
// ---------------------------------------------------------------------------
const rl = readline.createInterface({ input: process.stdin });
const lines = [];
for await (const line of rl) {
  lines.push(line);
}

let request;
try {
  request = JSON.parse(lines.join("\n"));
} catch (err) {
  process.stderr.write(`[Bridge] Failed to parse stdin JSON: ${err.message}\n`);
  process.exit(1);
}

const { stateDir, agentName, message, model, provider, timeoutMs, sessionId } =
  request;

if (!stateDir || !agentName || !message) {
  process.stderr.write(
    "[Bridge] Missing required fields: stateDir, agentName, message\n",
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 2. Dynamically import runEmbeddedPiAgent from OpenClaw
// ---------------------------------------------------------------------------
const openclawPath =
  process.env.OPENCLAW_PATH || "/opt/openclaw";
const importPath = `${openclawPath}/dist/agents/pi-embedded-runner.js`;

// Verify the import target exists before attempting dynamic import
if (!fs.existsSync(importPath)) {
  process.stderr.write(
    `[Bridge] OpenClaw not found at ${importPath}. Set OPENCLAW_PATH env var.\n`,
  );
  process.exit(1);
}

let runEmbeddedPiAgent;
try {
  const mod = await import(pathToFileURL(importPath).href);
  runEmbeddedPiAgent = mod.runEmbeddedPiAgent;
  if (typeof runEmbeddedPiAgent !== "function") {
    throw new Error("runEmbeddedPiAgent is not exported as a function");
  }
} catch (err) {
  process.stderr.write(`[Bridge] Failed to import OpenClaw: ${err.message}\n`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 3. Helper: emit a single NDJSON event to stdout
// ---------------------------------------------------------------------------
function emit(event) {
  process.stdout.write(JSON.stringify(event) + "\n");
}

// ---------------------------------------------------------------------------
// 4. Load and override openclaw.json config for enclave safety
// ---------------------------------------------------------------------------
const configPath = `${stateDir}/openclaw.json`;
let config = {};
try {
  if (fs.existsSync(configPath)) {
    config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
  }
} catch {
  // Proceed with empty config if malformed or missing
}

// Enclave safety overrides:
//   - exec/bash: ENABLED (tmpfs sandbox, files persist in tarball)
//   - web search: ENABLED when BRAVE_API_KEY is set (vsock allowlist permits api.search.brave.com)
//   - web fetch/media/browser: DISABLED (unnecessary attack surface)
const braveKey = process.env.BRAVE_API_KEY || "";
config.tools = {
  ...(config.tools || {}),
  web: {
    search: {
      enabled: !!braveKey,
      provider: "brave",
    },
    fetch: { enabled: false },
  },
  media: { enabled: false },
  browser: { enabled: false },
};

// Configure Bedrock provider if not already set
if (!config.models) {
  config.models = {};
}
if (!config.models.providers) {
  config.models.providers = {};
}
if (!config.models.providers["amazon-bedrock"]) {
  const region = process.env.AWS_REGION || "us-east-1";
  config.models.providers["amazon-bedrock"] = {
    baseUrl: `https://bedrock-runtime.${region}.amazonaws.com`,
    api: "bedrock-converse-stream",
    auth: "aws-sdk",
    models: [
      {
        id: "us.anthropic.claude-opus-4-5-20251101-v1:0",
        name: "Claude Opus 4.5",
        contextWindow: 200000,
        maxTokens: 16384,
        reasoning: false,
        input: ["text", "image"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
    ],
  };
}

// No discovery needed — model is defined explicitly above
config.models.bedrockDiscovery = { enabled: false };

// Configure Bedrock embeddings for vector memory (Nova 2 via AWS SDK).
// Uses IAM credentials passed from bedrock_server.py via env vars.
// No local model loading = no cold start latency.
//
// IMPORTANT: Set OPENCLAW_STATE_DIR so resolveStateDir() points inside the
// extracted tarball. Without this, the SQLite vector DB ends up at
// ~/.openclaw/memory/ which is NOT packed back into the tarball, causing
// vector memory to be lost between invocations.
process.env.OPENCLAW_STATE_DIR = stateDir;

if (!config.agents) {
  config.agents = {};
}
if (!config.agents.defaults) {
  config.agents.defaults = {};
}
if (!config.agents.defaults.memorySearch) {
  config.agents.defaults.memorySearch = {
    provider: "bedrock",
    model: "amazon.nova-2-multimodal-embeddings-v1:0",
    store: {
      driver: "sqlite",
      path: `${stateDir}/memory/${agentName}.sqlite`,
      vector: { enabled: true },
    },
    query: {
      maxResults: 20,
      hybrid: { enabled: true, vectorWeight: 0.7, textWeight: 0.3 },
    },
  };
}

// ---------------------------------------------------------------------------
// 5. Resolve session file and workspace
// ---------------------------------------------------------------------------
const workspaceDir = `${stateDir}/agents/${agentName}`;
const sessionsDir = `${workspaceDir}/sessions`;

// Ensure directories exist
fs.mkdirSync(sessionsDir, { recursive: true });

const resolvedSessionId = sessionId || randomUUID();

// Find existing session or create new path
let sessionFile;
if (sessionId) {
  // Explicit session ID — use it directly
  sessionFile = `${sessionsDir}/${sessionId}.jsonl`;
} else {
  // Find most recent existing session
  const existing = fs
    .readdirSync(sessionsDir)
    .filter((f) => f.endsWith(".jsonl"))
    .sort();
  if (existing.length > 0) {
    sessionFile = `${sessionsDir}/${existing[existing.length - 1]}`;
  } else {
    sessionFile = `${sessionsDir}/${resolvedSessionId}.jsonl`;
  }
}

// ---------------------------------------------------------------------------
// 6. Resolve model
// ---------------------------------------------------------------------------
// Default model — can be overridden by the request or openclaw.json
const resolvedModel =
  model || "us.anthropic.claude-opus-4-5-20251101-v1:0";
const resolvedProvider = provider || "amazon-bedrock";

process.stderr.write(
  `[Bridge] agent=${agentName} model=${resolvedModel} session=${sessionFile}\n`,
);
process.stderr.write(
  `[Bridge] config.models.providers keys=${Object.keys(config.models?.providers || {}).join(",")}\n`,
);
const bedrockCfg = config.models?.providers?.["amazon-bedrock"];
if (bedrockCfg) {
  process.stderr.write(
    `[Bridge] bedrock provider: models=${(bedrockCfg.models || []).length}, api=${bedrockCfg.api}, auth=${bedrockCfg.auth}\n`,
  );
  for (const m of bedrockCfg.models || []) {
    process.stderr.write(
      `[Bridge]   model: id=${m.id}, maxTokens=${m.maxTokens}, contextWindow=${m.contextWindow}, reasoning=${m.reasoning}\n`,
    );
  }
}
process.stderr.write(
  `[Bridge] bedrockDiscovery: enabled=${config.models?.bedrockDiscovery?.enabled}, region=${config.models?.bedrockDiscovery?.region}, filter=${JSON.stringify(config.models?.bedrockDiscovery?.providerFilter)}\n`,
);
process.stderr.write(
  `[Bridge] env: AWS_ACCESS_KEY_ID=${process.env.AWS_ACCESS_KEY_ID ? "set" : "unset"}, AWS_REGION=${process.env.AWS_REGION}, HTTP_PROXY=${process.env.HTTP_PROXY}, AWS_PROFILE=${process.env.AWS_PROFILE || "unset"}\n`,
);

// ---------------------------------------------------------------------------
// 7. Run the agent
// ---------------------------------------------------------------------------

// agentDir controls where models.json is written/read by OpenClaw.
// Without this, OpenClaw falls back to ~/.openclaw/agents/{id}/agent which
// is disconnected from our workspace. Pass workspaceDir so everything is colocated.
const agentDir = workspaceDir;

// Diagnostic paths for models.json
const modelsJsonPath = `${agentDir}/models.json`;
// Also check the default fallback path in case agentDir param isn't respected
const defaultModelsJsonPath = `${process.env.HOME || "/root"}/.openclaw/agents/default/agent/models.json`;

// Track accumulated text so we can emit only deltas for streaming
let lastPartialText = "";

try {
  const result = await runEmbeddedPiAgent({
    // Required
    sessionId: resolvedSessionId,
    sessionFile,
    workspaceDir,
    agentDir,
    prompt: message,
    timeoutMs: timeoutMs || 90_000,
    runId: randomUUID(),

    // Model
    model: resolvedModel,
    provider: resolvedProvider,

    // Config
    config,
    disableTools: false,

    // Streaming callbacks → NDJSON events
    onPartialReply: (payload) => {
      process.stderr.write(`[Bridge] onPartialReply called, text_len=${(payload.text || "").length}, keys=${Object.keys(payload).join(",")}\n`);
      if (payload.text) {
        // OpenClaw emits accumulated text (full response so far), not deltas.
        // Extract only the new text since last emit to avoid duplication.
        const delta = payload.text.slice(lastPartialText.length);
        lastPartialText = payload.text;
        if (delta) {
          emit({ type: "partial", text: delta });
        }
      } else {
        emit({ type: "partial_empty", keys: Object.keys(payload) });
      }
      if (payload.mediaUrls?.length) {
        emit({ type: "media", urls: payload.mediaUrls });
      }
    },

    onBlockReply: (payload) => {
      process.stderr.write(`[Bridge] onBlockReply called, text_len=${(payload.text || "").length}, keys=${Object.keys(payload).join(",")}\n`);
      if (payload.text) {
        emit({ type: "block", text: payload.text });
      } else {
        emit({ type: "block_empty", keys: Object.keys(payload) });
      }
    },

    onToolResult: (payload) => {
      process.stderr.write(`[Bridge] onToolResult called, text_len=${(payload.text || "").length}\n`);
      if (payload.text) {
        emit({ type: "tool_result", text: payload.text });
      }
    },

    onReasoningStream: (payload) => {
      process.stderr.write(`[Bridge] onReasoningStream called\n`);
      if (payload.text) {
        emit({ type: "reasoning", text: payload.text });
      }
    },

    onAssistantMessageStart: () => {
      process.stderr.write("[Bridge] onAssistantMessageStart called\n");
      emit({ type: "assistant_start" });
    },

    onAgentEvent: (evt) => {
      process.stderr.write(`[Bridge] onAgentEvent: stream=${evt.stream}, data_keys=${evt.data ? Object.keys(evt.data).join(",") : "null"}\n`);
      // Forward low-level agent lifecycle events for diagnostics
      emit({ type: "agent_event", stream: evt.stream, data: evt.data });
    },
  });

  // Emit the full result object for diagnostics
  process.stderr.write(`[Bridge] result keys=${Object.keys(result).join(",")}, meta keys=${result.meta ? Object.keys(result.meta).join(",") : "null"}\n`);
  process.stderr.write(`[Bridge] result.text length=${(result.text || "").length}, stopReason=${result.meta?.stopReason}, error=${JSON.stringify(result.meta?.error)}\n`);
  process.stderr.write(`[Bridge] result.didSendViaMessagingTool=${result.didSendViaMessagingTool}\n`);

  // Extract text from payloads if result.text is missing
  // EmbeddedPiRunResult.payloads is Array<{ text?: string; isError?: boolean; ... }>
  let responseText = result.text || "";
  if (!responseText && Array.isArray(result.payloads)) {
    process.stderr.write(`[Bridge] payloads count=${result.payloads.length}\n`);
    for (const [i, p] of result.payloads.entries()) {
      process.stderr.write(`[Bridge] payload[${i}] keys=${Object.keys(p).join(",")}, text_len=${(p.text || "").length}, isError=${p.isError}\n`);
      if (p.text && !p.isError) {
        process.stderr.write(`[Bridge] payload[${i}] text preview=${p.text.slice(0, 200)}\n`);
        responseText = p.text;
      }
    }
  }
  if (responseText) {
    process.stderr.write(`[Bridge] Final responseText length=${responseText.length}, preview=${responseText.slice(0, 100)}\n`);
  } else {
    process.stderr.write(`[Bridge] WARNING: No response text found anywhere in result!\n`);
    process.stderr.write(`[Bridge] Full result JSON: ${JSON.stringify(result).slice(0, 2000)}\n`);
  }

  // Emit completion with metadata + extracted response text
  emit({
    type: "done",
    meta: {
      durationMs: result.meta.durationMs,
      agentMeta: result.meta.agentMeta,
      error: result.meta.error,
      stopReason: result.meta.stopReason,
    },
    resultText: responseText,
    resultKeys: Object.keys(result),
  });
} catch (err) {
  emit({ type: "error", message: err.message || String(err) });
  // Exit 0 even on agent errors — the error is communicated via NDJSON
}

// Diagnostic: dump models.json and check for both known failure modes
try {
  if (fs.existsSync(modelsJsonPath)) {
    const raw = fs.readFileSync(modelsJsonPath, "utf-8");
    process.stderr.write(`[Bridge] models.json (${raw.length} bytes): ${raw.slice(0, 3000)}\n`);

    // Failure mode 1: Discovery returned 0 models
    // If models.json has "amazon-bedrock" with models:[] or no models key,
    // it means discoverBedrockModels() returned empty (API error or filter mismatch).
    // resolveModel() then falls to fallback with DEFAULT_CONTEXT_TOKENS=200000 → error.
    try {
      const parsed = JSON.parse(raw);
      const bedrockProvider = parsed?.providers?.["amazon-bedrock"];
      const modelCount = Array.isArray(bedrockProvider?.models) ? bedrockProvider.models.length : 0;
      process.stderr.write(`[Bridge] DIAG: amazon-bedrock models count=${modelCount}\n`);
      if (modelCount === 0) {
        process.stderr.write(`[Bridge] DIAG: *** DISCOVERY RETURNED 0 MODELS — resolveModel will use fallback with maxTokens=200000 ***\n`);
      }

      // Failure mode 2: Model ID mismatch
      // Discovery may register models with base IDs (anthropic.claude-...) instead of
      // inference profile IDs (us.anthropic.claude-...). If run_agent.mjs requests a
      // profile ID but the registry only has the base ID, resolveModel fails.
      if (modelCount > 0) {
        const modelIds = bedrockProvider.models.map((m) => m.id);
        const hasRequestedModel = modelIds.includes(resolvedModel);
        process.stderr.write(`[Bridge] DIAG: requested model="${resolvedModel}", found in models.json=${hasRequestedModel}\n`);
        if (!hasRequestedModel) {
          process.stderr.write(`[Bridge] DIAG: *** MODEL ID MISMATCH — available IDs: ${modelIds.join(", ")} ***\n`);
        }
      }
    } catch (parseErr) {
      process.stderr.write(`[Bridge] DIAG: Failed to parse models.json: ${parseErr.message}\n`);
    }
  } else {
    process.stderr.write(`[Bridge] models.json NOT FOUND at ${modelsJsonPath}\n`);
    process.stderr.write(`[Bridge] DIAG: *** NO models.json — ensureOpenClawModelsJson may have failed or written to a different path ***\n`);
  }
} catch (err) {
  process.stderr.write(`[Bridge] models.json read error: ${err.message}\n`);
}

// Also check the default fallback path (in case agentDir param wasn't respected)
if (defaultModelsJsonPath !== modelsJsonPath) {
  try {
    if (fs.existsSync(defaultModelsJsonPath)) {
      const raw = fs.readFileSync(defaultModelsJsonPath, "utf-8");
      process.stderr.write(`[Bridge] DIAG: default models.json EXISTS at ${defaultModelsJsonPath} (${raw.length} bytes)\n`);
      process.stderr.write(`[Bridge] DIAG: This means OpenClaw ignored our agentDir and used the default path!\n`);
    }
  } catch {}
}

process.stderr.write("[Bridge] Done\n");
