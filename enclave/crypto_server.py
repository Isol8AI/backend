#!/usr/bin/env python3
"""
M3: Crypto Server for Nitro Enclave
===================================

A vsock server that demonstrates cryptographic operations:
1. Generates an X25519 keypair on startup (enclave transport key)
2. Accepts encrypted messages from the parent
3. Decrypts, processes, and re-encrypts responses

Protocol:
- GET_PUBLIC_KEY: Returns enclave's public key
- ENCRYPT_TEST: Accepts plaintext, returns encrypted payload
- DECRYPT_TEST: Accepts encrypted payload, returns decrypted plaintext
- PROCESS_MESSAGE: Full flow - decrypt input, process, re-encrypt output
- RUN_TESTS: Execute crypto test vectors and return results

All messages are JSON over vsock.
"""

import socket
import sys
import json

from crypto_primitives import (
    generate_x25519_keypair,
    encrypt_to_public_key,
    decrypt_with_private_key,
    EncryptedPayload,
    KeyPair,
    bytes_to_hex,
    hex_to_bytes,
)

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = 40


class CryptoServer:
    """Crypto server with persistent enclave keypair."""

    def __init__(self):
        """Initialize with a fresh keypair."""
        self.keypair: KeyPair = generate_x25519_keypair()
        print("[Enclave] Generated transport keypair", flush=True)
        print(f"[Enclave] Public key: {bytes_to_hex(self.keypair.public_key)}", flush=True)

    def handle_get_public_key(self) -> dict:
        """Return the enclave's public key."""
        return {
            "status": "success",
            "command": "GET_PUBLIC_KEY",
            "public_key": bytes_to_hex(self.keypair.public_key),
        }

    def handle_encrypt_test(self, data: dict) -> dict:
        """Encrypt plaintext to a specified public key."""
        try:
            plaintext = data.get("plaintext", "")
            recipient_public_key = hex_to_bytes(data["recipient_public_key"])
            context = data.get("context", "test-context")

            payload = encrypt_to_public_key(recipient_public_key, plaintext.encode("utf-8"), context)

            return {
                "status": "success",
                "command": "ENCRYPT_TEST",
                "encrypted": payload.to_dict(),
            }
        except Exception as e:
            return {
                "status": "error",
                "command": "ENCRYPT_TEST",
                "error": str(e),
            }

    def handle_decrypt_test(self, data: dict) -> dict:
        """Decrypt a payload encrypted to the enclave's public key."""
        try:
            payload = EncryptedPayload.from_dict(data["encrypted"])
            context = data.get("context", "client-to-enclave-transport")

            plaintext = decrypt_with_private_key(self.keypair.private_key, payload, context)

            return {
                "status": "success",
                "command": "DECRYPT_TEST",
                "plaintext": plaintext.decode("utf-8"),
            }
        except Exception as e:
            return {
                "status": "error",
                "command": "DECRYPT_TEST",
                "error": str(e),
            }

    def handle_process_message(self, data: dict) -> dict:
        """
        Full message processing flow:
        1. Decrypt incoming message (encrypted to enclave key)
        2. Process the message (echo with transformation)
        3. Re-encrypt response to user's storage key
        """
        try:
            # Get user's public key for response encryption
            user_public_key = hex_to_bytes(data["user_public_key"])

            # Decrypt incoming message
            incoming_payload = EncryptedPayload.from_dict(data["encrypted_message"])
            plaintext = decrypt_with_private_key(
                self.keypair.private_key,
                incoming_payload,
                "client-to-enclave-transport",
            )

            message = plaintext.decode("utf-8")
            print(f"[Enclave] Decrypted message: {message}", flush=True)

            # Process the message (simple transformation for M3)
            processed = f"[Processed by Enclave] {message}"

            # Re-encrypt for storage (to user's key)
            response_payload = encrypt_to_public_key(
                user_public_key,
                processed.encode("utf-8"),
                "assistant-message-storage",
            )

            return {
                "status": "success",
                "command": "PROCESS_MESSAGE",
                "encrypted_response": response_payload.to_dict(),
                "debug": {
                    "received_length": len(message),
                    "response_length": len(processed),
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "command": "PROCESS_MESSAGE",
                "error": str(e),
            }

    def handle_run_tests(self) -> dict:
        """Run crypto test vectors and return results."""
        try:
            # Import and run test module
            import test_crypto_vectors

            # Capture test results
            results = {
                "ecdh_tests": [],
                "aes_gcm_tests": [],
            }

            # Test ECDH vectors
            for vector in test_crypto_vectors.TEST_VECTORS["ecdh_derivation"]:
                from crypto_primitives import derive_key_from_ecdh

                derived_key, _ = derive_key_from_ecdh(
                    hex_to_bytes(vector["private_key_hex"]),
                    hex_to_bytes(vector["public_key_hex"]),
                    vector["context"],
                    hex_to_bytes(vector["salt_hex"]),
                )
                passed = bytes_to_hex(derived_key) == vector["expected_key_hex"]
                results["ecdh_tests"].append(
                    {
                        "description": vector["description"],
                        "passed": passed,
                    }
                )

            # Test AES-GCM vectors
            for vector in test_crypto_vectors.TEST_VECTORS["aes_gcm"]:
                from crypto_primitives import decrypt_aes_gcm

                aad = hex_to_bytes(vector["aad_hex"]) if "aad_hex" in vector else None
                plaintext = decrypt_aes_gcm(
                    hex_to_bytes(vector["key_hex"]),
                    hex_to_bytes(vector["iv_hex"]),
                    hex_to_bytes(vector["ciphertext_hex"]),
                    hex_to_bytes(vector["auth_tag_hex"]),
                    aad,
                )
                passed = bytes_to_hex(plaintext) == vector["plaintext_hex"]
                results["aes_gcm_tests"].append(
                    {
                        "description": vector["description"],
                        "passed": passed,
                    }
                )

            # Count results
            ecdh_passed = sum(1 for t in results["ecdh_tests"] if t["passed"])
            aes_passed = sum(1 for t in results["aes_gcm_tests"] if t["passed"])
            total_passed = ecdh_passed + aes_passed
            total_tests = len(results["ecdh_tests"]) + len(results["aes_gcm_tests"])

            return {
                "status": "success",
                "command": "RUN_TESTS",
                "results": results,
                "summary": {
                    "ecdh_passed": ecdh_passed,
                    "ecdh_total": len(results["ecdh_tests"]),
                    "aes_gcm_passed": aes_passed,
                    "aes_gcm_total": len(results["aes_gcm_tests"]),
                    "total_passed": total_passed,
                    "total_tests": total_tests,
                    "all_passed": total_passed == total_tests,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "command": "RUN_TESTS",
                "error": str(e),
            }

    def handle_request(self, request: dict) -> dict:
        """Route request to appropriate handler."""
        command = request.get("command", "").upper()

        handlers = {
            "GET_PUBLIC_KEY": self.handle_get_public_key,
            "ENCRYPT_TEST": lambda: self.handle_encrypt_test(request),
            "DECRYPT_TEST": lambda: self.handle_decrypt_test(request),
            "PROCESS_MESSAGE": lambda: self.handle_process_message(request),
            "RUN_TESTS": self.handle_run_tests,
        }

        handler = handlers.get(command)
        if handler:
            return handler()
        else:
            return {
                "status": "error",
                "error": f"Unknown command: {command}",
                "available_commands": list(handlers.keys()),
            }


def create_vsock_listener(port: int) -> socket.socket:
    """Create a vsock listener socket."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((socket.VMADDR_CID_ANY, port))
    sock.listen(5)
    return sock


def handle_client(server: CryptoServer, conn: socket.socket, addr: tuple):
    """Handle a single client connection."""
    cid, port = addr
    print(f"[Enclave] Connection from CID={cid}, port={port}", flush=True)

    try:
        while True:
            # Receive data (max 64KB for larger encrypted payloads)
            data = conn.recv(65536)
            if not data:
                print("[Enclave] Client disconnected", flush=True)
                break

            try:
                request = json.loads(data.decode("utf-8"))
                print(
                    f"[Enclave] Received command: {request.get('command', 'unknown')}",
                    flush=True,
                )

                response = server.handle_request(request)
                response["source"] = "nitro-enclave-crypto"

            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "source": "nitro-enclave-crypto",
                    "error": f"Invalid JSON: {e}",
                }

            conn.sendall(json.dumps(response).encode("utf-8"))
            print("[Enclave] Sent response", flush=True)

    except Exception as e:
        print(f"[Enclave] Error handling client: {e}", flush=True)
    finally:
        conn.close()


def main():
    print("=" * 60, flush=True)
    print("NITRO ENCLAVE CRYPTO SERVER (M3)", flush=True)
    print("=" * 60, flush=True)
    print(f"Python version: {sys.version}", flush=True)

    # Initialize crypto server with keypair
    server = CryptoServer()

    print(f"Listening on vsock port {VSOCK_PORT}...", flush=True)

    try:
        listener = create_vsock_listener(VSOCK_PORT)
        print("[Enclave] Server ready, waiting for connections...", flush=True)

        while True:
            conn, addr = listener.accept()
            handle_client(server, conn, addr)

    except Exception as e:
        print(f"[Enclave] Fatal error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
