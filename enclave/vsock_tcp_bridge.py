#!/usr/bin/env python3
"""
TCP-to-vsock bridge for Nitro Enclave.

Runs INSIDE the enclave on 127.0.0.1:3128. Accepts plain TCP connections
from Node.js (which has no vsock support), forwards the bytes to the
parent's vsock-proxy on CID 3, port 8443.

The parent's vsock-proxy expects an HTTP CONNECT request, then tunnels
bytes bidirectionally to the target host. TLS termination happens inside
the enclave (Node.js → this bridge → vsock → parent → target), so the
parent only sees encrypted TLS bytes after the CONNECT handshake.

Usage (started automatically by bedrock_server.py):
    python3 vsock_tcp_bridge.py

Node.js monkey-patch in run_agent.mjs intercepts net.connect() for
*.amazonaws.com and routes through this bridge.
"""

import socket
import select
import threading
import sys

# Bridge configuration
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 3128

# Parent vsock-proxy
AF_VSOCK = 40
PARENT_CID = 3
VSOCK_PROXY_PORT = 8443


def tunnel_bidirectional(sock_a: socket.socket, sock_b: socket.socket):
    """Bidirectional byte tunnel between two sockets."""
    sockets = [sock_a, sock_b]
    try:
        while True:
            readable, _, exceptional = select.select(sockets, [], sockets, 60.0)
            if exceptional:
                break
            if not readable:
                continue
            for sock in readable:
                try:
                    data = sock.recv(65536)
                    if not data:
                        return
                    other = sock_b if sock is sock_a else sock_a
                    other.sendall(data)
                except Exception:
                    return
    except Exception:
        pass


def handle_client(client: socket.socket, addr: tuple):
    """
    Handle a TCP connection from Node.js.

    Node.js sends an HTTP CONNECT request (e.g., CONNECT host:443 HTTP/1.1).
    We forward it to the parent's vsock-proxy, which establishes the TCP
    connection and replies with 200. We relay that 200 back to Node.js,
    then tunnel bytes bidirectionally.
    """
    vsock = None
    try:
        # Read the HTTP CONNECT request from Node.js
        data = client.recv(4096)
        if not data:
            return

        # Open vsock to parent's proxy
        vsock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        vsock.settimeout(30.0)
        vsock.connect((PARENT_CID, VSOCK_PROXY_PORT))
        vsock.settimeout(None)

        # Forward the CONNECT request to parent's vsock-proxy
        vsock.sendall(data)

        # Read response from vsock-proxy (200 Connection Established or error)
        response = vsock.recv(4096)
        if not response:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return

        # Forward vsock-proxy's response to Node.js
        client.sendall(response)

        # If it was a success, tunnel bytes bidirectionally
        if b"200" in response:
            tunnel_bidirectional(client, vsock)

    except Exception as e:
        print(f"[vsock-tcp-bridge] Error: {e}", flush=True)
        try:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        except Exception:
            pass
    finally:
        client.close()
        if vsock:
            vsock.close()


def main():
    print(f"[vsock-tcp-bridge] Starting on {LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    print(f"[vsock-tcp-bridge] Forwarding to vsock CID={PARENT_CID} port={VSOCK_PROXY_PORT}", flush=True)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((LISTEN_HOST, LISTEN_PORT))
    listener.listen(20)

    print("[vsock-tcp-bridge] Ready", flush=True)

    try:
        while True:
            client, addr = listener.accept()
            thread = threading.Thread(target=handle_client, args=(client, addr), daemon=True)
            thread.start()
    except KeyboardInterrupt:
        print("[vsock-tcp-bridge] Shutting down", flush=True)
    finally:
        listener.close()


if __name__ == "__main__":
    main()
