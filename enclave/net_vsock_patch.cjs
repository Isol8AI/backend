/**
 * CJS preload: monkey-patch net.connect for Nitro Enclave networking.
 *
 * MUST be loaded via `node --require ./net_vsock_patch.cjs` BEFORE any ESM
 * modules.  When the CJS patch runs first, the ESM loader's live bindings
 * for `net.connect` / `net.createConnection` reflect the patched versions.
 *
 * Why CJS?  ESM `import * as net` produces a frozen Module Namespace object
 * whose properties cannot be reassigned.  CJS `require("net")` returns the
 * mutable exports object, so direct assignment works.
 *
 * The Nitro Enclave has NO network stack — no DNS, no outbound TCP.
 * All external traffic must go through the vsock-proxy on the parent
 * instance.  This patch intercepts connections to *.amazonaws.com and
 * routes them through a local TCP-to-vsock bridge (vsock_tcp_bridge.py)
 * on 127.0.0.1:BRIDGE_PORT.  The bridge forwards to the parent's
 * vsock-proxy via HTTP CONNECT, which establishes the real connection.
 *
 * Environment:
 *   VSOCK_BRIDGE_PORT  TCP port for bridge (default 3128, 0 = disabled)
 */

"use strict";

const net = require("net");

const BRIDGE_PORT = parseInt(process.env.VSOCK_BRIDGE_PORT || "3128", 10);
const BRIDGE_ENABLED = BRIDGE_PORT > 0;

if (BRIDGE_ENABLED) {
  const _origConnect = net.connect;
  const _origCreateConnection = net.createConnection;

  /**
   * Wrap a connection attempt: if the target host matches *.amazonaws.com,
   * connect to the local vsock bridge instead and send an HTTP CONNECT
   * request.  Otherwise fall through to the original implementation.
   */
  function patchedConnect(...args) {
    // Normalize arguments: net.connect(options), net.connect(port, host), etc.
    let options = {};
    let cb;
    if (typeof args[0] === "object" && args[0] !== null && !Array.isArray(args[0])) {
      options = args[0];
      cb = typeof args[1] === "function" ? args[1] : undefined;
    } else if (typeof args[0] === "number") {
      options = { port: args[0], host: args[1] };
      cb = typeof args[2] === "function" ? args[2] : undefined;
    } else {
      // Fall through for anything unexpected (path sockets, etc.)
      return _origConnect.apply(net, args);
    }

    const host = options.host || "localhost";
    const port = options.port || 443;

    // Only intercept *.amazonaws.com connections
    if (!host.endsWith(".amazonaws.com")) {
      return _origConnect.apply(net, args);
    }

    process.stderr.write(
      "[net-vsock-patch] Intercepting " + host + ":" + port +
      " -> 127.0.0.1:" + BRIDGE_PORT + "\n"
    );

    // Connect to local TCP-to-vsock bridge instead
    var sock = _origConnect.call(net, { host: "127.0.0.1", port: BRIDGE_PORT }, function () {
      // Send HTTP CONNECT to the bridge (forwarded to parent vsock-proxy)
      var connectReq =
        "CONNECT " + host + ":" + port + " HTTP/1.1\r\n" +
        "Host: " + host + "\r\n" +
        "\r\n";
      sock.write(connectReq);
    });

    // Wait for the 200 response from the bridge before signalling "connected"
    var handshakeDone = false;
    var buffered = Buffer.alloc(0);

    function onHandshake(chunk) {
      if (handshakeDone) return;

      buffered = Buffer.concat([buffered, chunk]);
      var text = buffered.toString("utf-8");

      // Look for end of HTTP response headers
      var headerEnd = text.indexOf("\r\n\r\n");
      if (headerEnd === -1) return; // Need more data

      handshakeDone = true;
      sock.removeListener("data", onHandshake);

      if (!text.startsWith("HTTP/1.1 200")) {
        process.stderr.write(
          "[net-vsock-patch] CONNECT failed: " + text.slice(0, 100) + "\n"
        );
        sock.destroy(new Error("CONNECT to " + host + ":" + port + " failed: " + text.slice(0, 100)));
        return;
      }

      process.stderr.write(
        "[net-vsock-patch] Tunnel established to " + host + ":" + port + "\n"
      );

      // If there are bytes after the headers, push them back
      var remainder = buffered.subarray(headerEnd + 4);
      if (remainder.length > 0) {
        sock.unshift(remainder);
      }

      // Signal to the caller that the connection is ready
      if (cb) cb();
      sock.emit("connect");
    }

    sock.on("data", onHandshake);

    return sock;
  }

  net.connect = patchedConnect;
  net.createConnection = patchedConnect;

  process.stderr.write(
    "[net-vsock-patch] Enabled: *.amazonaws.com -> 127.0.0.1:" + BRIDGE_PORT + "\n"
  );
}
