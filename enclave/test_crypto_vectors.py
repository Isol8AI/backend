#!/usr/bin/env python3
"""
M3: Crypto Test Vector Verification for Nitro Enclave
======================================================

This script verifies that the Python crypto primitives produce
identical outputs to the TypeScript frontend implementation.

Run this inside the enclave to prove cryptographic compatibility.

Usage:
    python3 test_crypto_vectors.py

Exit codes:
    0 = All tests passed
    1 = One or more tests failed
"""

import sys

from crypto_primitives import (
    derive_key_from_ecdh,
    decrypt_aes_gcm,
    encrypt_to_public_key,
    decrypt_with_private_key,
    generate_x25519_keypair,
    hex_to_bytes,
    bytes_to_hex,
)

# =============================================================================
# Test Vectors (copied from frontend/tests/fixtures/crypto_test_vectors.json)
# =============================================================================

TEST_VECTORS = {
    "version": "1.0",
    "description": "Cross-platform crypto test vectors",
    "ecdh_derivation": [
        {
            "description": "Basic ECDH",
            "private_key_hex": "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
            "public_key_hex": "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
            "context": "test-context",
            "salt_hex": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "expected_key_hex": "01fb75e5729e6de652f1e68381f46fc832ee8b26fc0b5521cce72de25b635109",
        },
        {
            "description": "ECDH with transport context",
            "private_key_hex": "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
            "public_key_hex": "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
            "context": "client-to-enclave-transport",
            "salt_hex": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "expected_key_hex": "1e0306eafbcd33da1f613d27240f73fa0611a19be823d58e123d780549e5d772",
        },
        {
            "description": "ECDH with storage context",
            "private_key_hex": "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
            "public_key_hex": "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
            "context": "user-message-storage",
            "salt_hex": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "expected_key_hex": "979cab9e1b0ebc242ff4673251b96b7f1ee32fa5164539edb7600fee692b06dc",
        },
        {
            "description": "ECDH with org-key-distribution context",
            "private_key_hex": "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
            "public_key_hex": "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
            "context": "org-key-distribution",
            "salt_hex": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "expected_key_hex": "6b269f827a8b2b2c7702aa808f8135f88060a5eeac18729bd1df177e0a98476e",
        },
    ],
    "aes_gcm": [
        {
            "description": "Simple encryption",
            "key_hex": "0000000000000000000000000000000000000000000000000000000000000000",
            "iv_hex": "11111111111111111111111111111111",
            "plaintext_hex": "48656c6c6f2c20576f726c6421",
            "ciphertext_hex": "9b2ee55288f111f35e648a71bb",
            "auth_tag_hex": "b60febd361c3d05ee4ac80ec98dc5232",
        },
        {
            "description": "Encryption with AAD",
            "key_hex": "0000000000000000000000000000000000000000000000000000000000000000",
            "iv_hex": "11111111111111111111111111111111",
            "plaintext_hex": "48656c6c6f2c20576f726c6421",
            "ciphertext_hex": "9b2ee55288f111f35e648a71bb",
            "auth_tag_hex": "45beff23d7f2c9eebfc3e91e671ca37e",
            "aad_hex": "61757468656e746963617465642064617461",
        },
        {
            "description": "Longer message",
            "key_hex": "0000000000000000000000000000000000000000000000000000000000000000",
            "iv_hex": "11111111111111111111111111111111",
            "plaintext_hex": "546869732069732061206c6f6e676572206d65737361676520666f722074657374696e67204145532d47434d20656e6372797074696f6e2077697468206d6f726520646174612e",
            "ciphertext_hex": "8723e04dc7b4428450368a7af42a0cec327cf599e18f0a80ff629606302b011f798947bd3744856aede4f5244eaa3d237b0f8931d795f7fa44146990e868c91e6224689ee5eb8f",
            "auth_tag_hex": "30a2b4a027c328bb6366620b437fc5fc",
        },
    ],
}


def test_ecdh_derivation() -> tuple[int, int]:
    """Test ECDH key derivation against vectors."""
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("ECDH Key Derivation Tests (X25519 + HKDF-SHA512)")
    print("=" * 60)

    for vector in TEST_VECTORS["ecdh_derivation"]:
        desc = vector["description"]
        try:
            derived_key, _ = derive_key_from_ecdh(
                hex_to_bytes(vector["private_key_hex"]),
                hex_to_bytes(vector["public_key_hex"]),
                vector["context"],
                hex_to_bytes(vector["salt_hex"]),
            )
            result = bytes_to_hex(derived_key)
            expected = vector["expected_key_hex"]

            if result == expected:
                print(f"  [PASS] {desc}")
                passed += 1
            else:
                print(f"  [FAIL] {desc}")
                print(f"         Expected: {expected}")
                print(f"         Got:      {result}")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {desc}")
            print(f"         Error: {e}")
            failed += 1

    return passed, failed


def test_aes_gcm_decryption() -> tuple[int, int]:
    """Test AES-GCM decryption against vectors."""
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("AES-256-GCM Decryption Tests")
    print("=" * 60)

    for vector in TEST_VECTORS["aes_gcm"]:
        desc = vector["description"]
        try:
            aad = hex_to_bytes(vector["aad_hex"]) if "aad_hex" in vector else None
            plaintext = decrypt_aes_gcm(
                hex_to_bytes(vector["key_hex"]),
                hex_to_bytes(vector["iv_hex"]),
                hex_to_bytes(vector["ciphertext_hex"]),
                hex_to_bytes(vector["auth_tag_hex"]),
                aad,
            )
            result = bytes_to_hex(plaintext)
            expected = vector["plaintext_hex"]

            if result == expected:
                print(f"  [PASS] {desc}")
                passed += 1
            else:
                print(f"  [FAIL] {desc}")
                print(f"         Expected: {expected}")
                print(f"         Got:      {result}")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {desc}")
            print(f"         Error: {e}")
            failed += 1

    return passed, failed


def test_encrypt_decrypt_roundtrip() -> tuple[int, int]:
    """Test encrypt/decrypt round-trip with various contexts."""
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Encrypt/Decrypt Round-Trip Tests")
    print("=" * 60)

    contexts = [
        "client-to-enclave-transport",
        "enclave-to-client-transport",
        "user-message-storage",
        "assistant-message-storage",
        "org-key-distribution",
    ]

    messages = [
        b"Hello, World!",
        b"Short",
        b"A" * 1000,  # 1KB message
        b'{"type":"message","content":"JSON payload"}',
        bytes(range(256)),  # All byte values
    ]

    for context in contexts:
        for i, message in enumerate(messages):
            desc = f"{context} - message {i + 1} ({len(message)} bytes)"
            try:
                # Generate recipient keypair
                recipient = generate_x25519_keypair()

                # Encrypt
                payload = encrypt_to_public_key(recipient.public_key, message, context)

                # Decrypt
                decrypted = decrypt_with_private_key(recipient.private_key, payload, context)

                if decrypted == message:
                    print(f"  [PASS] {desc}")
                    passed += 1
                else:
                    print(f"  [FAIL] {desc}")
                    print("         Decrypted doesn't match original")
                    failed += 1
            except Exception as e:
                print(f"  [FAIL] {desc}")
                print(f"         Error: {e}")
                failed += 1

    return passed, failed


def test_cross_keypair_ecdh() -> tuple[int, int]:
    """Test that ECDH produces same shared secret from both sides."""
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Cross-Keypair ECDH Symmetry Test")
    print("=" * 60)

    # Generate two keypairs
    alice = generate_x25519_keypair()
    bob = generate_x25519_keypair()

    context = "test-symmetry"

    try:
        # Alice derives key using her private + Bob's public
        key_alice, salt = derive_key_from_ecdh(alice.private_key, bob.public_key, context)

        # Bob derives key using his private + Alice's public (with same salt)
        key_bob, _ = derive_key_from_ecdh(bob.private_key, alice.public_key, context, salt)

        if key_alice == key_bob:
            print("  [PASS] Alice and Bob derive identical keys")
            passed += 1
        else:
            print("  [FAIL] Keys don't match!")
            print(f"         Alice: {bytes_to_hex(key_alice)}")
            print(f"         Bob:   {bytes_to_hex(key_bob)}")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        failed += 1

    return passed, failed


def test_wrong_context_fails() -> tuple[int, int]:
    """Test that decryption with wrong context fails."""
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Wrong Context Rejection Test")
    print("=" * 60)

    try:
        recipient = generate_x25519_keypair()
        message = b"Secret message"

        # Encrypt with one context
        payload = encrypt_to_public_key(recipient.public_key, message, "client-to-enclave-transport")

        # Try to decrypt with different context (should fail)
        try:
            decrypt_with_private_key(recipient.private_key, payload, "user-message-storage")
            print("  [FAIL] Should have rejected wrong context")
            failed += 1
        except Exception:
            print("  [PASS] Correctly rejected wrong context")
            passed += 1

    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        failed += 1

    return passed, failed


def test_payload_serialization() -> tuple[int, int]:
    """Test that payloads can be serialized/deserialized."""
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Payload Serialization Test")
    print("=" * 60)

    try:
        recipient = generate_x25519_keypair()
        message = b"Test serialization"
        context = "test-context"

        # Encrypt
        payload = encrypt_to_public_key(recipient.public_key, message, context)

        # Serialize to JSON
        json_str = payload.to_json()

        # Deserialize
        payload2 = payload.from_json(json_str)

        # Decrypt with deserialized payload
        decrypted = decrypt_with_private_key(recipient.private_key, payload2, context)

        if decrypted == message:
            print("  [PASS] Payload survives JSON round-trip")
            passed += 1
        else:
            print("  [FAIL] Decryption failed after serialization")
            failed += 1

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        failed += 1

    return passed, failed


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("NITRO ENCLAVE CRYPTO TEST VECTORS")
    print("=" * 60)
    print(f"Vector version: {TEST_VECTORS['version']}")
    print(f"Python version: {sys.version}")

    total_passed = 0
    total_failed = 0

    # Run all test suites
    for test_func in [
        test_ecdh_derivation,
        test_aes_gcm_decryption,
        test_encrypt_decrypt_roundtrip,
        test_cross_keypair_ecdh,
        test_wrong_context_fails,
        test_payload_serialization,
    ]:
        passed, failed = test_func()
        total_passed += passed
        total_failed += failed

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {total_passed + total_failed}")

    if total_failed == 0:
        print("\n[SUCCESS] All crypto tests passed!")
        print("Enclave crypto is compatible with frontend.")
        return 0
    else:
        print(f"\n[FAILURE] {total_failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
