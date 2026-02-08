#!/usr/bin/env python3
"""
M4: vsock-proxy for Nitro Enclave
==================================

This proxy runs on the PARENT EC2 instance and allows the enclave to make
outbound HTTPS connections. The enclave connects via vsock, sends an HTTP
CONNECT request, and the proxy tunnels the TLS connection.

Key security property: TLS termination happens INSIDE the enclave.
The parent only sees encrypted bytes and cannot read the plaintext.

Usage (on parent instance):
    python3 vsock_proxy.py

The proxy listens on:
    - vsock port 8443 (for enclave connections)

Enclave connects to CID 3 (parent), port 8443, sends:
    CONNECT bedrock-runtime.us-east-1.amazonaws.com:443 HTTP/1.1
    Host: bedrock-runtime.us-east-1.amazonaws.com

Proxy establishes TCP connection and tunnels bytes bidirectionally.
"""

import socket
import select
import threading
import re

# vsock constants
AF_VSOCK = 40
VMADDR_CID_HOST = 3  # Parent's CID from enclave perspective
VSOCK_PROXY_PORT = 8443

# Allowed destination hosts (whitelist for security)
ALLOWED_HOSTS = [
    "bedrock-runtime.us-east-1.amazonaws.com",
    "bedrock-runtime.us-west-2.amazonaws.com",
    "bedrock.us-east-1.amazonaws.com",
    "sts.amazonaws.com",
    "sts.us-east-1.amazonaws.com",
]


def parse_connect_request(data: bytes) -> tuple:
    """
    Parse HTTP CONNECT request.

    Returns (host, port) or (None, None) if invalid.
    """
    try:
        text = data.decode("utf-8")
        # Match: CONNECT host:port HTTP/1.x
        match = re.match(r"CONNECT\s+([^:]+):(\d+)\s+HTTP/1\.\d", text)
        if match:
            return match.group(1), int(match.group(2))
    except Exception:
        pass
    return None, None


def tunnel_bidirectional(client_sock: socket.socket, target_sock: socket.socket):
    """
    Bidirectional tunnel between two sockets.
    Runs until either socket closes or errors.
    """
    sockets = [client_sock, target_sock]

    try:
        while True:
            readable, _, exceptional = select.select(sockets, [], sockets, 30.0)

            if exceptional:
                break

            if not readable:
                # Timeout - send a keepalive or just continue
                continue

            for sock in readable:
                try:
                    data = sock.recv(65536)
                    if not data:
                        return  # Connection closed

                    # Forward to the other socket
                    other = target_sock if sock == client_sock else client_sock
                    other.sendall(data)
                except Exception:
                    return
    except Exception:
        pass


def handle_enclave_connection(conn: socket.socket, addr: tuple):
    """
    Handle a connection from the enclave.

    1. Read CONNECT request
    2. Parse target host:port
    3. Validate against whitelist
    4. Connect to target
    5. Send 200 OK to enclave
    6. Tunnel bidirectionally
    """
    cid, port = addr
    print(f"[vsock-proxy] Connection from enclave CID={cid}, port={port}", flush=True)

    target_sock = None

    try:
        # Read CONNECT request (up to 4KB should be plenty)
        data = conn.recv(4096)
        if not data:
            print("[vsock-proxy] Empty request from enclave", flush=True)
            return

        host, target_port = parse_connect_request(data)

        if not host:
            error_response = b"HTTP/1.1 400 Bad Request\r\n\r\nInvalid CONNECT request"
            conn.sendall(error_response)
            print(f"[vsock-proxy] Invalid CONNECT request: {data[:100]}", flush=True)
            return

        # Security: Check whitelist
        if host not in ALLOWED_HOSTS:
            error_response = b"HTTP/1.1 403 Forbidden\r\n\r\nHost not in whitelist"
            conn.sendall(error_response)
            print(f"[vsock-proxy] BLOCKED: {host} not in whitelist", flush=True)
            return

        print(f"[vsock-proxy] CONNECT to {host}:{target_port}", flush=True)

        # Connect to target
        target_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_sock.settimeout(30.0)
        target_sock.connect((host, target_port))
        target_sock.settimeout(None)

        # Send success response to enclave
        success_response = b"HTTP/1.1 200 Connection Established\r\n\r\n"
        conn.sendall(success_response)

        print(f"[vsock-proxy] Tunnel established to {host}:{target_port}", flush=True)

        # Tunnel bidirectionally (TLS happens inside enclave)
        tunnel_bidirectional(conn, target_sock)

        print(f"[vsock-proxy] Tunnel closed for {host}:{target_port}", flush=True)

    except socket.timeout:
        print("[vsock-proxy] Connection timeout", flush=True)
    except ConnectionRefusedError:
        error_response = b"HTTP/1.1 502 Bad Gateway\r\n\r\nConnection refused"
        try:
            conn.sendall(error_response)
        except Exception:
            pass
        print("[vsock-proxy] Connection refused to target", flush=True)
    except Exception as e:
        print(f"[vsock-proxy] Error: {e}", flush=True)
    finally:
        conn.close()
        if target_sock:
            target_sock.close()


def main():
    print("=" * 60, flush=True)
    print("VSOCK PROXY FOR NITRO ENCLAVE (M4)", flush=True)
    print("=" * 60, flush=True)
    print(f"Listening on vsock port {VSOCK_PROXY_PORT}...", flush=True)
    print(f"Allowed hosts: {ALLOWED_HOSTS}", flush=True)

    # Create vsock listener
    listener = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((socket.VMADDR_CID_ANY, VSOCK_PROXY_PORT))
    listener.listen(10)

    print("[vsock-proxy] Ready, waiting for enclave connections...", flush=True)

    try:
        while True:
            conn, addr = listener.accept()
            # Handle each connection in a thread
            thread = threading.Thread(target=handle_enclave_connection, args=(conn, addr), daemon=True)
            thread.start()
    except KeyboardInterrupt:
        print("\n[vsock-proxy] Shutting down...", flush=True)
    finally:
        listener.close()


if __name__ == "__main__":
    main()
