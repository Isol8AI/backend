/**
 * CJS preload: enable HTTP proxy for Nitro Enclave networking.
 *
 * MUST be loaded via `node --require ./proxy_bootstrap.cjs` BEFORE any ESM
 * modules. Sets up TWO proxy mechanisms:
 *
 * 1. global-agent: patches http.request/https.request (HTTP/1.1 traffic)
 * 2. Standard HTTP_PROXY/HTTPS_PROXY env vars: detected by the AWS SDK's
 *    pi-ai Bedrock provider, which switches from NodeHttp2Handler to
 *    NodeHttpHandler + proxy-agent (HTTP/2 doesn't support proxy agents)
 *
 * Inside the enclave, vsock_tcp_bridge.py listens on 127.0.0.1:BRIDGE_PORT
 * and tunnels CONNECT requests through vsock to the parent's proxy, which
 * makes the real outbound connection. TLS terminates inside the enclave.
 *
 * Environment:
 *   VSOCK_BRIDGE_PORT  TCP port for bridge (default 3128, 0 = disabled)
 */

"use strict";

var BRIDGE_PORT = parseInt(process.env.VSOCK_BRIDGE_PORT || "3128", 10);

if (BRIDGE_PORT > 0) {
  var proxyUrl = "http://127.0.0.1:" + BRIDGE_PORT;

  // Standard env vars: detected by proxy-agent, AWS SDK, and other libraries
  process.env.HTTP_PROXY = proxyUrl;
  process.env.HTTPS_PROXY = proxyUrl;
  process.env.http_proxy = proxyUrl;
  process.env.https_proxy = proxyUrl;

  // global-agent env var: triggers http/https module patching
  process.env.GLOBAL_AGENT_HTTPS_PROXY = proxyUrl;
  require("/opt/openclaw/node_modules/global-agent/bootstrap");

  process.stderr.write(
    "[proxy-bootstrap] HTTPS proxy enabled: " + proxyUrl + "\n"
  );
} else {
  process.stderr.write("[proxy-bootstrap] Disabled (VSOCK_BRIDGE_PORT=0)\n");
}
