#!/usr/bin/env python3
"""
M3: Test Client for Enclave Crypto Server
==========================================

Run this from the parent EC2 instance to test the enclave's crypto capabilities.

Usage:
    python3 test_crypto_client.py <enclave-cid>

Tests:
    1. GET_PUBLIC_KEY - Get enclave's transport public key
    2. RUN_TESTS - Run crypto test vectors inside enclave
    3. ENCRYPT_TEST - Test encryption
    4. DECRYPT_TEST - Test decryption (requires client-side crypto)
    5. PROCESS_MESSAGE - Full encrypt-decrypt-reencrypt flow
"""

import socket
import sys
import json

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = 40


def send_command(cid: int, command: dict) -> dict:
    """Send a command to the enclave and receive the response."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)

    try:
        sock.connect((cid, VSOCK_PORT))
        sock.sendall(json.dumps(command).encode("utf-8"))
        response_data = sock.recv(65536)
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
        print(f"  Key Length: {len(public_key) // 2} bytes")
        return public_key
    else:
        print("  Status: FAILED")
        print(f"  Error: {response.get('error', 'Unknown')}")
        return ""


def test_run_tests(cid: int) -> bool:
    """Test RUN_TESTS command - runs crypto test vectors inside enclave."""
    print("\n[Test 2] RUN_TESTS (crypto test vectors)")
    print("-" * 40)

    response = send_command(cid, {"command": "RUN_TESTS"})

    if response.get("status") == "success":
        summary = response.get("summary", {})
        print("  Status: SUCCESS")
        print(f"  ECDH Tests: {summary.get('ecdh_passed')}/{summary.get('ecdh_total')}")
        print(f"  AES-GCM Tests: {summary.get('aes_gcm_passed')}/{summary.get('aes_gcm_total')}")
        print(f"  Total: {summary.get('total_passed')}/{summary.get('total_tests')} passed")

        all_passed = summary.get("all_passed", False)
        if all_passed:
            print("  [PASS] All crypto test vectors passed!")
        else:
            print("  [FAIL] Some test vectors failed!")
            # Print individual failures
            for test in response.get("results", {}).get("ecdh_tests", []):
                if not test["passed"]:
                    print(f"    - ECDH FAIL: {test['description']}")
            for test in response.get("results", {}).get("aes_gcm_tests", []):
                if not test["passed"]:
                    print(f"    - AES FAIL: {test['description']}")

        return all_passed
    else:
        print("  Status: FAILED")
        print(f"  Error: {response.get('error', 'Unknown')}")
        return False


def test_encrypt(cid: int, enclave_public_key: str) -> dict:
    """Test ENCRYPT_TEST command."""
    print("\n[Test 3] ENCRYPT_TEST")
    print("-" * 40)

    # Generate a test keypair for the "user"
    # In real usage, this would be the user's storage key
    # For this test, we use a fixed test key
    test_public_key = "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"

    response = send_command(
        cid,
        {
            "command": "ENCRYPT_TEST",
            "plaintext": "Hello from parent!",
            "recipient_public_key": test_public_key,
            "context": "test-context",
        },
    )

    if response.get("status") == "success":
        encrypted = response.get("encrypted", {})
        print("  Status: SUCCESS")
        print(f"  Ephemeral PK: {encrypted.get('ephemeral_public_key', '')[:32]}...")
        print(f"  Ciphertext: {encrypted.get('ciphertext', '')[:32]}...")
        print(f"  IV Length: {len(encrypted.get('iv', '')) // 2} bytes")
        print(f"  Auth Tag Length: {len(encrypted.get('auth_tag', '')) // 2} bytes")
        return encrypted
    else:
        print("  Status: FAILED")
        print(f"  Error: {response.get('error', 'Unknown')}")
        return {}


def test_decrypt(cid: int, enclave_public_key: str) -> bool:
    """
    Test DECRYPT_TEST command.

    This requires encrypting a message TO the enclave's public key,
    then asking the enclave to decrypt it.
    """
    print("\n[Test 4] DECRYPT_TEST (full round-trip)")
    print("-" * 40)

    # To test decryption, we need to encrypt something to the enclave's key
    # We'll use the ENCRYPT_TEST to create a payload, then DECRYPT_TEST
    # This simulates what the client would do

    # First, ask enclave to encrypt TO its own key (for testing)
    encrypt_response = send_command(
        cid,
        {
            "command": "ENCRYPT_TEST",
            "plaintext": "Secret message for decryption test",
            "recipient_public_key": enclave_public_key,
            "context": "client-to-enclave-transport",
        },
    )

    if encrypt_response.get("status") != "success":
        print(f"  Encrypt step failed: {encrypt_response.get('error')}")
        return False

    encrypted = encrypt_response.get("encrypted", {})

    # Now ask enclave to decrypt it
    decrypt_response = send_command(
        cid,
        {
            "command": "DECRYPT_TEST",
            "encrypted": encrypted,
            "context": "client-to-enclave-transport",
        },
    )

    if decrypt_response.get("status") == "success":
        plaintext = decrypt_response.get("plaintext", "")
        expected = "Secret message for decryption test"
        if plaintext == expected:
            print("  Status: SUCCESS")
            print(f"  Decrypted: {plaintext}")
            print("  [PASS] Round-trip encryption/decryption works!")
            return True
        else:
            print("  Status: MISMATCH")
            print(f"  Expected: {expected}")
            print(f"  Got: {plaintext}")
            return False
    else:
        print("  Status: FAILED")
        print(f"  Error: {decrypt_response.get('error', 'Unknown')}")
        return False


def test_process_message(cid: int, enclave_public_key: str) -> bool:
    """
    Test PROCESS_MESSAGE command - full message processing flow.

    This simulates the real use case:
    1. Client encrypts message to enclave's transport key
    2. Enclave decrypts, processes, re-encrypts to user's storage key
    3. Response is returned encrypted
    """
    print("\n[Test 5] PROCESS_MESSAGE (full flow)")
    print("-" * 40)

    # User's storage key (for response encryption)
    # Using a fixed test key for verification
    user_public_key = "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"

    # First encrypt a message TO the enclave
    encrypt_response = send_command(
        cid,
        {
            "command": "ENCRYPT_TEST",
            "plaintext": "What is 2+2?",
            "recipient_public_key": enclave_public_key,
            "context": "client-to-enclave-transport",
        },
    )

    if encrypt_response.get("status") != "success":
        print(f"  Encrypt step failed: {encrypt_response.get('error')}")
        return False

    encrypted_message = encrypt_response.get("encrypted", {})

    # Now send for full processing
    process_response = send_command(
        cid,
        {
            "command": "PROCESS_MESSAGE",
            "encrypted_message": encrypted_message,
            "user_public_key": user_public_key,
        },
    )

    if process_response.get("status") == "success":
        encrypted_response = process_response.get("encrypted_response", {})
        debug = process_response.get("debug", {})

        print("  Status: SUCCESS")
        print(f"  Input length: {debug.get('received_length')} chars")
        print(f"  Output length: {debug.get('response_length')} chars")
        print("  Response encrypted to user's key")
        print(f"  Ciphertext: {encrypted_response.get('ciphertext', '')[:32]}...")
        print("  [PASS] Full message processing flow works!")
        return True
    else:
        print("  Status: FAILED")
        print(f"  Error: {process_response.get('error', 'Unknown')}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_crypto_client.py <enclave-cid>")
        print("Example: python3 test_crypto_client.py 17")
        sys.exit(1)

    cid = int(sys.argv[1])

    print("=" * 60)
    print("ENCLAVE CRYPTO TEST CLIENT (M3)")
    print("=" * 60)
    print(f"Target enclave CID: {cid}")

    results = []

    # Test 1: Get public key
    enclave_public_key = test_get_public_key(cid)
    results.append(("GET_PUBLIC_KEY", bool(enclave_public_key)))

    if not enclave_public_key:
        print("\n[ABORT] Cannot continue without enclave public key")
        sys.exit(1)

    # Test 2: Run crypto test vectors
    tests_passed = test_run_tests(cid)
    results.append(("RUN_TESTS", tests_passed))

    # Test 3: Test encryption
    encrypted = test_encrypt(cid, enclave_public_key)
    results.append(("ENCRYPT_TEST", bool(encrypted)))

    # Test 4: Test decryption round-trip
    decrypt_passed = test_decrypt(cid, enclave_public_key)
    results.append(("DECRYPT_TEST", decrypt_passed))

    # Test 5: Test full message processing
    process_passed = test_process_message(cid, enclave_public_key)
    results.append(("PROCESS_MESSAGE", process_passed))

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
        print("[SUCCESS] All M3 crypto tests passed!")
        print("Enclave crypto is working correctly.")
        sys.exit(0)
    else:
        print("[FAILURE] Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
