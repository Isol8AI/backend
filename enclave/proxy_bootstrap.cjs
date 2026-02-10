/**
 * CJS preload: enable HTTP proxy for Nitro Enclave networking.
 *
 * MUST be loaded via `node --require ./proxy_bootstrap.cjs` BEFORE any ESM
 * modules. The global-agent library patches http.request/https.request to
 * route HTTPS traffic through an HTTP CONNECT proxy.
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
  process.env.GLOBAL_AGENT_HTTPS_PROXY = "http://127.0.0.1:" + BRIDGE_PORT;
  require("/opt/openclaw/node_modules/global-agent/bootstrap");
  process.stderr.write(
    "[proxy-bootstrap] HTTPS proxy enabled: http://127.0.0.1:" + BRIDGE_PORT + "\n"
  );
} else {
  process.stderr.write("[proxy-bootstrap] Disabled (VSOCK_BRIDGE_PORT=0)\n");
}
