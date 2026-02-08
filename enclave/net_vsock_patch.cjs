/**
 * CJS preload: monkey-patch networking for Nitro Enclave.
 *
 * MUST be loaded via `node --require ./net_vsock_patch.cjs` BEFORE any ESM
 * modules.  When the CJS patch runs first, the ESM loader's live bindings
 * reflect the patched versions.
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
 * Two layers are patched:
 *
 *   1. net.Socket.prototype.connect — catches ALL socket connections
 *      including those from tls.connect() and http2.connect().
 *      This is the critical patch because:
 *        - HTTP/1.1: https.Agent.createConnection → tls.connect → socket.connect
 *        - HTTP/2:   http2.connect → tls.connect → socket.connect
 *      Both paths end at socket.connect(), so patching here covers everything.
 *
 *      The technique: when socket.connect() is called for *.amazonaws.com,
 *      we redirect the TCP connection to the bridge (127.0.0.1:BRIDGE_PORT),
 *      intercept the 'connect' event, perform the HTTP CONNECT handshake,
 *      and only THEN re-emit 'connect' so TLS (or any caller) starts its
 *      handshake over the established tunnel.
 *
 *   2. net.connect / net.createConnection — catches explicit raw TCP
 *      connections (belt-and-suspenders for any code calling module-level
 *      net.connect directly).
 *
 * Environment:
 *   VSOCK_BRIDGE_PORT  TCP port for bridge (default 3128, 0 = disabled)
 */

"use strict";

var net = require("net");

var BRIDGE_PORT = parseInt(process.env.VSOCK_BRIDGE_PORT || "3128", 10);
var BRIDGE_ENABLED = BRIDGE_PORT > 0;

if (BRIDGE_ENABLED) {
  var _origConnect = net.connect;
  var _origCreateConnection = net.createConnection;
  var _origSocketConnect = net.Socket.prototype.connect;

  /**
   * Check if a host should be tunneled through the vsock bridge.
   */
  function shouldIntercept(host) {
    return typeof host === "string" && host.endsWith(".amazonaws.com");
  }

  // =========================================================================
  // Core: net.Socket.prototype.connect patch
  //
  // This is the critical patch. When tls.connect() or http2.connect() create
  // a socket and call socket.connect(options, callback), this patch:
  //
  //   1. Redirects the TCP connection to the bridge (127.0.0.1:BRIDGE_PORT)
  //   2. Intercepts the 'connect' event (suppresses it)
  //   3. Performs HTTP CONNECT handshake with the bridge
  //   4. Re-emits 'connect' after tunnel is established
  //
  // For TLS: tls.connect() registers socket.once('connect', _start) where
  // _start begins the TLS handshake. By suppressing the TCP 'connect' event
  // and re-emitting it after the CONNECT handshake, we ensure TLS negotiation
  // happens over the tunnel, not directly with the bridge.
  //
  // Before _start() is called, the TLSSocket's handle is still a raw TCP
  // handle (TLS wrapping happens inside _start), so we can safely read/write
  // raw HTTP CONNECT data on the socket during the handshake phase.
  // =========================================================================

  function patchedSocketConnect(/* ...args */) {
    // Normalize arguments to extract host and port
    var args = arguments;
    var options, host, port;

    if (typeof args[0] === "object" && args[0] !== null && !Array.isArray(args[0])) {
      options = args[0];
      host = options.host || options.hostname || "localhost";
      port = options.port || 443;
    } else if (typeof args[0] === "number") {
      port = args[0];
      host = (typeof args[1] === "string") ? args[1] : "localhost";
    } else {
      // Unix socket path or unexpected form — pass through
      return _origSocketConnect.apply(this, args);
    }

    if (!shouldIntercept(host)) {
      return _origSocketConnect.apply(this, args);
    }

    process.stderr.write(
      "[net-vsock-patch] socket.connect intercepting " + host + ":" + port +
      " -> 127.0.0.1:" + BRIDGE_PORT + "\n"
    );

    var self = this;
    var targetHost = host;
    var targetPort = port;

    // Build redirect args: connect to bridge instead of target
    var redirectArgs = [{ host: "127.0.0.1", port: BRIDGE_PORT }];

    // Preserve the callback (last argument if it's a function).
    // The original socket.connect registers it via once('connect', cb).
    // Our interceptor will suppress the first 'connect' event and re-emit
    // it after the CONNECT handshake, so the callback fires at the right time.
    var lastArg = args[args.length - 1];
    if (typeof lastArg === "function") {
      redirectArgs.push(lastArg);
    }

    // Intercept the 'connect' event to insert HTTP CONNECT handshake
    var origEmit = self.emit;
    var connectIntercepted = false;

    self.emit = function (event) {
      if (event === "connect" && !connectIntercepted) {
        connectIntercepted = true;

        // TCP connected to bridge. Now do HTTP CONNECT handshake.
        process.stderr.write(
          "[net-vsock-patch] CONNECT " + targetHost + ":" + targetPort + "\n"
        );

        self.write(
          "CONNECT " + targetHost + ":" + targetPort + " HTTP/1.1\r\n" +
          "Host: " + targetHost + "\r\n" +
          "\r\n"
        );

        var handshakeDone = false;
        var buffered = Buffer.alloc(0);

        function onHandshake(chunk) {
          if (handshakeDone) return;

          buffered = Buffer.concat([buffered, chunk]);
          var text = buffered.toString("utf-8");

          // Wait for complete HTTP response headers
          var headerEnd = text.indexOf("\r\n\r\n");
          if (headerEnd === -1) return; // Need more data

          handshakeDone = true;
          self.removeListener("data", onHandshake);

          if (!text.startsWith("HTTP/1.1 200")) {
            process.stderr.write(
              "[net-vsock-patch] CONNECT failed: " + text.slice(0, 100) + "\n"
            );
            self.destroy(
              new Error("CONNECT to " + targetHost + ":" + targetPort +
                " failed: " + text.slice(0, 100))
            );
            return;
          }

          process.stderr.write(
            "[net-vsock-patch] Tunnel established to " +
            targetHost + ":" + targetPort + "\n"
          );

          // Push back any bytes that arrived after the CONNECT response.
          // These would be TLS handshake bytes from the target server
          // (if the bridge pipelined them).
          var remainder = buffered.subarray(headerEnd + 4);
          if (remainder.length > 0) {
            self.unshift(remainder);
          }

          // Restore original emit and fire the real 'connect' event.
          // For TLS: this triggers _start() which wraps the handle with
          // TLS and begins the TLS handshake over the tunnel.
          // For plain TCP callers: this triggers their connect callback.
          self.emit = origEmit;
          origEmit.call(self, "connect");
        }

        self.on("data", onHandshake);
        return false; // Suppress the TCP-level connect event
      }

      // Pass all other events through (error, close, data, etc.)
      return origEmit.apply(self, arguments);
    };

    // Connect to bridge (not to the original target)
    return _origSocketConnect.apply(self, redirectArgs);
  }

  // =========================================================================
  // Belt-and-suspenders: net.connect / net.createConnection module-level patch
  //
  // For any code that calls net.connect() directly (not via tls/http2).
  // In practice, the Socket.prototype.connect patch above handles these too,
  // but keeping this as explicit documentation of intent.
  // =========================================================================

  function patchedConnect(/* ...args */) {
    var args = arguments;
    var options, cb;

    if (typeof args[0] === "object" && args[0] !== null && !Array.isArray(args[0])) {
      options = args[0];
      cb = typeof args[1] === "function" ? args[1] : undefined;
    } else if (typeof args[0] === "number") {
      options = { port: args[0], host: args[1] };
      cb = typeof args[2] === "function" ? args[2] : undefined;
    } else {
      return _origConnect.apply(net, args);
    }

    var host = options.host || "localhost";
    var port = options.port || 443;

    if (!shouldIntercept(host)) {
      return _origConnect.apply(net, args);
    }

    process.stderr.write(
      "[net-vsock-patch] net.connect intercepting " + host + ":" + port +
      " -> 127.0.0.1:" + BRIDGE_PORT + "\n"
    );

    // Connect to bridge; the Socket.prototype.connect patch will NOT
    // fire for 127.0.0.1 (doesn't match *.amazonaws.com).
    var sock = _origConnect.call(net, { host: "127.0.0.1", port: BRIDGE_PORT }, function () {
      sock.write(
        "CONNECT " + host + ":" + port + " HTTP/1.1\r\n" +
        "Host: " + host + "\r\n" +
        "\r\n"
      );
    });

    sock.on("error", function (err) {
      process.stderr.write(
        "[net-vsock-patch] Bridge connection error: " + err.message + "\n"
      );
    });

    var handshakeDone = false;
    var buffered = Buffer.alloc(0);

    function onHandshake(chunk) {
      if (handshakeDone) return;

      buffered = Buffer.concat([buffered, chunk]);
      var text = buffered.toString("utf-8");

      var headerEnd = text.indexOf("\r\n\r\n");
      if (headerEnd === -1) return;

      handshakeDone = true;
      sock.removeListener("data", onHandshake);

      if (!text.startsWith("HTTP/1.1 200")) {
        process.stderr.write(
          "[net-vsock-patch] CONNECT failed: " + text.slice(0, 100) + "\n"
        );
        sock.destroy(new Error("CONNECT to " + host + ":" + port +
          " failed: " + text.slice(0, 100)));
        return;
      }

      process.stderr.write(
        "[net-vsock-patch] Tunnel established to " + host + ":" + port + "\n"
      );

      var remainder = buffered.subarray(headerEnd + 4);
      if (remainder.length > 0) {
        sock.unshift(remainder);
      }

      if (cb) cb();
      sock.emit("connect");
    }

    sock.on("data", onHandshake);

    return sock;
  }

  // Apply patches
  net.connect = patchedConnect;
  net.createConnection = patchedConnect;
  net.Socket.prototype.connect = patchedSocketConnect;

  process.stderr.write(
    "[net-vsock-patch] Enabled: *.amazonaws.com -> 127.0.0.1:" + BRIDGE_PORT +
    " (Socket.prototype.connect + net.connect patched)\n"
  );
}
