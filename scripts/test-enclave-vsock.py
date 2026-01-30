#!/usr/bin/env python3
"""
M2: Test Client for Enclave vsock Communication
================================================
Run this from the parent EC2 instance to test communication with the enclave.

Usage:
    python3 test-enclave-vsock.py <enclave-cid> [message]

Example:
    python3 test-enclave-vsock.py 16 "Hello from parent!"
"""

import socket
import sys
import json

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = 40  # Address family for vsock


def send_message(cid: int, message: str) -> dict:
    """Send a message to the enclave and receive the response."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)

    try:
        print(f"[Parent] Connecting to enclave CID={cid}, port={VSOCK_PORT}...")
        sock.connect((cid, VSOCK_PORT))
        print(f"[Parent] Connected!")

        # Send message
        print(f"[Parent] Sending: {message}")
        sock.sendall(message.encode('utf-8'))

        # Receive response
        response_data = sock.recv(4096)
        response = json.loads(response_data.decode('utf-8'))
        print(f"[Parent] Received: {json.dumps(response, indent=2)}")

        return response

    finally:
        sock.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test-enclave-vsock.py <enclave-cid> [message]")
        print("Example: python3 test-enclave-vsock.py 16 'Hello from parent!'")
        sys.exit(1)

    cid = int(sys.argv[1])
    message = sys.argv[2] if len(sys.argv) > 2 else "Hello from parent instance!"

    print("=" * 50)
    print("ENCLAVE VSOCK TEST CLIENT")
    print("=" * 50)

    try:
        response = send_message(cid, message)

        if response.get("source") == "nitro-enclave":
            print("\n[SUCCESS] Received response from Nitro Enclave!")
            print(f"Echo: {response.get('echo')}")
        else:
            print("\n[WARNING] Response doesn't appear to be from enclave")

    except ConnectionRefusedError:
        print(f"\n[ERROR] Connection refused. Is the enclave running?")
        print("Start the enclave with: nitro-cli run-enclave ...")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
