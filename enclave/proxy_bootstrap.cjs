/**
 * CJS preload: enable HTTP proxy for Nitro Enclave networking.
 *
 * MUST be loaded via `node --require ./proxy_bootstrap.cjs` BEFORE any ESM
 * modules. Sets standard HTTP_PROXY/HTTPS_PROXY env vars which are detected
 * by OpenClaw's pi-ai Bedrock provider (amazon-bedrock.js). When present,
 * pi-ai switches from NodeHttp2Handler to NodeHttpHandler + proxy-agent,
 * which handles HTTP CONNECT tunneling with correct TLS hostname verification.
 *
 * NOTE: We do NOT use global-agent. global-agent patches https.request with
 * forceGlobalAgent=true, overriding proxy-agent and breaking TLS hostname
 * verification (sets servername to 'localhost' instead of the target host,
 * causing ERR_TLS_CERT_ALTNAME_INVALID).
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

  process.stderr.write(
    "[proxy-bootstrap] HTTPS proxy enabled: " + proxyUrl + "\n"
  );
} else {
  process.stderr.write("[proxy-bootstrap] Disabled (VSOCK_BRIDGE_PORT=0)\n");
}
