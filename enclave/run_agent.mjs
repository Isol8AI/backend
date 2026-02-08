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
// The Nitro Enclave has NO network stack.  net.connect monkey-patching is done
// via --require net_vsock_patch.cjs (a CJS preload script) so it takes effect
// BEFORE the ESM loader freezes module namespaces.  See net_vsock_patch.cjs.

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
//   - web/media/browser: DISABLED (vsock whitelist blocks outbound)
config.tools = {
  ...(config.tools || {}),
  web: { enabled: false },
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
    models: [],
  };
}

// Enable Bedrock auto-discovery
if (!config.models.bedrockDiscovery) {
  config.models.bedrockDiscovery = {
    enabled: true,
    region: process.env.AWS_REGION || "us-east-1",
    providerFilter: ["anthropic"],
    refreshInterval: 3600,
  };
}

// Configure local embeddings for vector memory (no external API needed)
if (!config.agents) {
  config.agents = {};
}
if (!config.agents.defaults) {
  config.agents.defaults = {};
}
if (!config.agents.defaults.memorySearch) {
  config.agents.defaults.memorySearch = {
    provider: "local",
    local: {
      modelPath:
        "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
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
  model || "us.anthropic.claude-3-5-sonnet-20241022-v2:0";
const resolvedProvider = provider || "amazon-bedrock";

process.stderr.write(
  `[Bridge] agent=${agentName} model=${resolvedModel} session=${sessionFile}\n`,
);

// ---------------------------------------------------------------------------
// 7. Run the agent
// ---------------------------------------------------------------------------
try {
  const result = await runEmbeddedPiAgent({
    // Required
    sessionId: resolvedSessionId,
    sessionFile,
    workspaceDir,
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
        emit({ type: "partial", text: payload.text });
      } else {
        // Emit even with empty text so we know the callback fired
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

process.stderr.write("[Bridge] Done\n");
