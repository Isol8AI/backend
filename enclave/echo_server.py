#!/usr/bin/env python3
"""
M2: Echo Server for Nitro Enclave
=================================
Listens on vsock port 5000 and echoes messages back to the parent.

vsock addressing:
- CID 3 = Parent instance (always)
- CID assigned to enclave at runtime (e.g., 16)
- Port 5000 = Our chosen application port
"""

import socket
import sys
import json

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = 40  # Address family for vsock


def create_vsock_listener(port: int) -> socket.socket:
    """Create a vsock listener socket."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    # CID_ANY (0xFFFFFFFF or -1) means listen on all CIDs
    sock.bind((socket.VMADDR_CID_ANY, port))
    sock.listen(5)
    return sock


def handle_client(conn: socket.socket, addr: tuple):
    """Handle a single client connection."""
    cid, port = addr
    print(f"[Enclave] Connection from CID={cid}, port={port}", flush=True)

    try:
        while True:
            # Receive data (max 4KB)
            data = conn.recv(4096)
            if not data:
                print("[Enclave] Client disconnected", flush=True)
                break

            message = data.decode("utf-8")
            print(f"[Enclave] Received: {message}", flush=True)

            # Echo back with enclave tag
            response = json.dumps({"status": "success", "source": "nitro-enclave", "echo": message})
            conn.sendall(response.encode("utf-8"))
            print("[Enclave] Sent response", flush=True)

    except Exception as e:
        print(f"[Enclave] Error handling client: {e}", flush=True)
    finally:
        conn.close()


def main():
    print("=" * 50, flush=True)
    print("NITRO ENCLAVE ECHO SERVER", flush=True)
    print("=" * 50, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Listening on vsock port {VSOCK_PORT}...", flush=True)

    try:
        listener = create_vsock_listener(VSOCK_PORT)
        print("[Enclave] Server ready, waiting for connections...", flush=True)

        while True:
            conn, addr = listener.accept()
            handle_client(conn, addr)

    except Exception as e:
        print(f"[Enclave] Fatal error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
