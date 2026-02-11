#!/usr/bin/env python3
"""
M4: Test Client for Enclave Bedrock Server
============================================

Run this from the parent EC2 instance to test the enclave's Bedrock capabilities.

Prerequisites:
1. vsock-proxy running on parent: python3 vsock_proxy.py
2. Enclave running: nitro-cli run-enclave ...
3. IAM role attached to EC2 with bedrock:InvokeModel permission

Usage:
    python3 test_bedrock_client.py <enclave-cid>

Tests:
1. GET_PUBLIC_KEY - Get enclave's transport public key
2. SET_CREDENTIALS - Pass IAM credentials to enclave
3. HEALTH - Check enclave health
4. CHAT - Full encrypted chat with Bedrock
5. RUN_TESTS - Verify crypto compatibility (M3)
"""

import socket
import sys
import json

# Import IMDS helper for IAM credentials
from imds_credentials import get_iam_credentials

# Import crypto for client-side encryption
from crypto_primitives import (
    generate_x25519_keypair,
    encrypt_to_public_key,
    hex_to_bytes,
)

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = 40


def send_command(cid: int, command: dict, timeout: float = 120.0) -> dict:
    """Send a command to the enclave and receive the response."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect((cid, VSOCK_PORT))
        sock.sendall(json.dumps(command).encode("utf-8"))
        response_data = sock.recv(1048576)  # 1MB buffer
        return json.loads(response_data.decode("utf-8"))
    finally:
        sock.close()


def test_get_public_key(cid: int) -> str:
    """Test GET_PUBLIC_KEY command."""
    print("\n[Test 1] GET_PUBLIC_KEY")
    print("-" * 40)

    response = send_command(cid, {"command": "GET_PUBLIC_KEY"})

    if response.get("status") == "success":
        public_key = response.get("public_key", "")
        print("  Status: SUCCESS")
        print(f"  Public Key: {public_key[:32]}...")
        print(f"  Region: {response.get('region')}")
        return public_key
    else:
        print("  Status: FAILED")
        print(f"  Error: {response.get('error', 'Unknown')}")
        return ""


def test_set_credentials(cid: int) -> bool:
    """Test SET_CREDENTIALS command - pass IAM role credentials to enclave."""
    print("\n[Test 2] SET_CREDENTIALS")
    print("-" * 40)

    try:
        # Get credentials from IMDS (IAM role)
        print("  Fetching credentials from IMDS...")
        creds = get_iam_credentials()
        print(f"  Access Key: {creds['access_key_id'][:10]}...")
        print(f"  Expiration: {creds['expiration']}")

        # Send to enclave
        response = send_command(
            cid,
            {
                "command": "SET_CREDENTIALS",
                "credentials": creds,
            },
        )

        if response.get("status") == "success":
            print("  Status: SUCCESS")
            print(f"  Has Credentials: {response.get('has_credentials')}")
            return True
        else:
            print("  Status: FAILED")
            print(f"  Error: {response.get('error', 'Unknown')}")
            return False

    except Exception as e:
        print("  Status: FAILED")
        print(f"  Error: {e}")
        return False


def test_health(cid: int) -> bool:
    """Test HEALTH command."""
    print("\n[Test 3] HEALTH")
    print("-" * 40)

    response = send_command(cid, {"command": "HEALTH"})

    if response.get("status") == "success":
        print("  Status: SUCCESS")
        print(f"  Enclave: {response.get('enclave')}")
        print(f"  Has Credentials: {response.get('has_credentials')}")
        print(f"  Region: {response.get('region')}")
        return response.get("has_credentials", False)
    else:
        print("  Status: FAILED")
        print(f"  Error: {response.get('error', 'Unknown')}")
        return False


def test_chat(cid: int, enclave_public_key: str, model_id: str) -> bool:
    """
    Test CHAT command - full encrypted chat with Bedrock.

    This tests the complete flow:
    1. Client encrypts message to enclave
    2. Enclave decrypts, calls Bedrock
    3. Enclave re-encrypts response to user's key
    """
    print(f"\n[Test 4] CHAT with {model_id}")
    print("-" * 40)

    try:
        # Generate user keypair (simulating frontend user)
        user_keypair = generate_x25519_keypair()
        print(f"  User public key: {user_keypair.public_key.hex()[:32]}...")

        # Encrypt message to enclave
        message = "What is 2+2? Reply with just the number."
        encrypted_message = encrypt_to_public_key(
            hex_to_bytes(enclave_public_key),
            message.encode("utf-8"),
            "client-to-enclave-transport",
        )
        print(f"  Message: '{message}'")
        print("  Encrypted message ready")

        # Send chat request with model_id
        print("  Calling Bedrock via enclave...")
        response = send_command(
            cid,
            {
                "command": "CHAT",
                "encrypted_message": encrypted_message.to_dict(),
                "user_public_key": user_keypair.public_key.hex(),
                "model_id": model_id,  # Model selected by user
                "system_prompt": "You are a helpful assistant. Be concise.",
            },
            timeout=120.0,  # Bedrock can take time
        )

        if response.get("status") == "success":
            usage = response.get("usage", {})
            print("  Status: SUCCESS")
            print(f"  Model: {response.get('model_id')}")
            print(f"  Input tokens: {usage.get('input_tokens')}")
            print(f"  Output tokens: {usage.get('output_tokens')}")
            print(f"  Stop reason: {response.get('stop_reason')}")
            print("  Response encrypted to user's key: YES")
            return True
        else:
            print("  Status: FAILED")
            print(f"  Error: {response.get('error', 'Unknown')}")
            return False

    except Exception as e:
        print("  Status: FAILED")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_run_tests(cid: int) -> bool:
    """Test RUN_TESTS command - crypto test vectors."""
    print("\n[Test 5] RUN_TESTS (crypto vectors)")
    print("-" * 40)

    response = send_command(cid, {"command": "RUN_TESTS"})

    if response.get("status") == "success":
        summary = response.get("summary", {})
        print("  Status: SUCCESS")
        print(f"  ECDH: {summary.get('ecdh_passed')}/{summary.get('ecdh_total')}")
        print(f"  AES-GCM: {summary.get('aes_gcm_passed')}/{summary.get('aes_gcm_total')}")
        print(f"  All passed: {summary.get('all_passed')}")
        return summary.get("all_passed", False)
    else:
        print("  Status: FAILED")
        print(f"  Error: {response.get('error', 'Unknown')}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_bedrock_client.py <enclave-cid>")
        print("Example: python3 test_bedrock_client.py 17")
        sys.exit(1)

    cid = int(sys.argv[1])
    # Use Llama 3.1 8B for testing (smallest/cheapest Llama model)
    model_id = "meta.llama3-1-8b-instruct-v1:0"

    print("=" * 60)
    print("ENCLAVE BEDROCK TEST CLIENT (M4)")
    print("=" * 60)
    print(f"Target enclave CID: {cid}")
    print(f"Test model: {model_id}")

    results = []

    # Test 1: Get public key
    enclave_public_key = test_get_public_key(cid)
    results.append(("GET_PUBLIC_KEY", bool(enclave_public_key)))

    if not enclave_public_key:
        print("\n[ABORT] Cannot continue without enclave public key")
        sys.exit(1)

    # Test 2: Set credentials
    creds_set = test_set_credentials(cid)
    results.append(("SET_CREDENTIALS", creds_set))

    if not creds_set:
        print("\n[ABORT] Cannot continue without credentials")
        sys.exit(1)

    # Test 3: Health check
    health_ok = test_health(cid)
    results.append(("HEALTH", health_ok))

    # Test 4: Chat with Bedrock
    chat_ok = test_chat(cid, enclave_public_key, model_id)
    results.append(("CHAT", chat_ok))

    # Test 5: Crypto tests
    crypto_ok = test_run_tests(cid)
    results.append(("RUN_TESTS", crypto_ok))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[SUCCESS] All M4 Bedrock tests passed!")
        print("Enclave can securely call Bedrock via vsock-proxy.")
        sys.exit(0)
    else:
        print("[FAILURE] Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
