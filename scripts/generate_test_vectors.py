#!/usr/bin/env python3
"""
Generate cross-platform crypto test vectors.
Python is the source of truth.
"""
import json
import sys
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from core.crypto.primitives import derive_key_from_ecdh
from tests.utils.crypto_test_utils import derive_key_from_passcode
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def hex_to_bytes(h: str) -> bytes:
    return bytes.fromhex(h)


# Fast params for testing
TEST_ARGON_PARAMS = {"time_cost": 1, "memory_cost": 16384, "parallelism": 1}


def generate_vectors():
    vectors = {
        "version": "1.0",
        "description": "Cross-platform crypto test vectors",
        "passcode_derivation": [],
        "ecdh_derivation": [],
        "aes_gcm": [],
    }

    # Passcode vectors
    salt1 = bytes(31) + b'\x01'  # 32 bytes ending with 01
    key1 = derive_key_from_passcode("123456", salt1, **TEST_ARGON_PARAMS)
    vectors["passcode_derivation"].append({
        "description": "6-digit passcode",
        "passcode": "123456",
        "salt_hex": salt1.hex(),
        **TEST_ARGON_PARAMS,
        "expected_key_hex": key1.hex(),
    })

    # Different passcode
    key2 = derive_key_from_passcode("654321", salt1, **TEST_ARGON_PARAMS)
    vectors["passcode_derivation"].append({
        "description": "Different passcode same salt",
        "passcode": "654321",
        "salt_hex": salt1.hex(),
        **TEST_ARGON_PARAMS,
        "expected_key_hex": key2.hex(),
    })

    # Alphanumeric passcode
    key3 = derive_key_from_passcode("admin123", salt1, **TEST_ARGON_PARAMS)
    vectors["passcode_derivation"].append({
        "description": "Alphanumeric passcode",
        "passcode": "admin123",
        "salt_hex": salt1.hex(),
        **TEST_ARGON_PARAMS,
        "expected_key_hex": key3.hex(),
    })

    # ECDH vectors (using RFC 7748 test keys)
    alice_priv = hex_to_bytes("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
    bob_pub = hex_to_bytes("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
    hkdf_salt = bytes.fromhex("bb" * 32)

    ecdh_key, _ = derive_key_from_ecdh(alice_priv, bob_pub, "test-context", hkdf_salt)
    vectors["ecdh_derivation"].append({
        "description": "Basic ECDH",
        "private_key_hex": alice_priv.hex(),
        "public_key_hex": bob_pub.hex(),
        "context": "test-context",
        "salt_hex": hkdf_salt.hex(),
        "expected_key_hex": ecdh_key.hex(),
    })

    # Different context
    ecdh_key2, _ = derive_key_from_ecdh(alice_priv, bob_pub, "client-to-enclave-transport", hkdf_salt)
    vectors["ecdh_derivation"].append({
        "description": "ECDH with transport context",
        "private_key_hex": alice_priv.hex(),
        "public_key_hex": bob_pub.hex(),
        "context": "client-to-enclave-transport",
        "salt_hex": hkdf_salt.hex(),
        "expected_key_hex": ecdh_key2.hex(),
    })

    # Storage context
    ecdh_key3, _ = derive_key_from_ecdh(alice_priv, bob_pub, "user-message-storage", hkdf_salt)
    vectors["ecdh_derivation"].append({
        "description": "ECDH with storage context",
        "private_key_hex": alice_priv.hex(),
        "public_key_hex": bob_pub.hex(),
        "context": "user-message-storage",
        "salt_hex": hkdf_salt.hex(),
        "expected_key_hex": ecdh_key3.hex(),
    })

    # Org key distribution context
    ecdh_key4, _ = derive_key_from_ecdh(alice_priv, bob_pub, "org-key-distribution", hkdf_salt)
    vectors["ecdh_derivation"].append({
        "description": "ECDH with org-key-distribution context",
        "private_key_hex": alice_priv.hex(),
        "public_key_hex": bob_pub.hex(),
        "context": "org-key-distribution",
        "salt_hex": hkdf_salt.hex(),
        "expected_key_hex": ecdh_key4.hex(),
    })

    # AES-GCM vectors
    aes_key = bytes(32)
    aes_iv = bytes.fromhex("11" * 16)
    plaintext = b"Hello, World!"

    aesgcm = AESGCM(aes_key)
    ct_with_tag = aesgcm.encrypt(aes_iv, plaintext, None)

    vectors["aes_gcm"].append({
        "description": "Simple encryption",
        "key_hex": aes_key.hex(),
        "iv_hex": aes_iv.hex(),
        "plaintext_hex": plaintext.hex(),
        "ciphertext_hex": ct_with_tag[:-16].hex(),
        "auth_tag_hex": ct_with_tag[-16:].hex(),
    })

    # With AAD
    aad = b"authenticated data"
    ct_with_aad = aesgcm.encrypt(aes_iv, plaintext, aad)
    vectors["aes_gcm"].append({
        "description": "Encryption with AAD",
        "key_hex": aes_key.hex(),
        "iv_hex": aes_iv.hex(),
        "plaintext_hex": plaintext.hex(),
        "ciphertext_hex": ct_with_aad[:-16].hex(),
        "auth_tag_hex": ct_with_aad[-16:].hex(),
        "aad_hex": aad.hex(),
    })

    # Longer message
    long_plaintext = b"This is a longer message for testing AES-GCM encryption with more data."
    ct_long = aesgcm.encrypt(aes_iv, long_plaintext, None)
    vectors["aes_gcm"].append({
        "description": "Longer message",
        "key_hex": aes_key.hex(),
        "iv_hex": aes_iv.hex(),
        "plaintext_hex": long_plaintext.hex(),
        "ciphertext_hex": ct_long[:-16].hex(),
        "auth_tag_hex": ct_long[-16:].hex(),
    })

    # Write vectors to both backend and frontend test directories
    backend_output = Path(__file__).parent.parent / "tests" / "fixtures" / "crypto_test_vectors.json"
    frontend_output = Path(__file__).parent.parent.parent / "frontend" / "tests" / "fixtures" / "crypto_test_vectors.json"

    backend_output.parent.mkdir(parents=True, exist_ok=True)
    frontend_output.parent.mkdir(parents=True, exist_ok=True)

    backend_output.write_text(json.dumps(vectors, indent=2))
    frontend_output.write_text(json.dumps(vectors, indent=2))

    print(f"Generated: {backend_output}")
    print(f"Generated: {frontend_output}")


if __name__ == "__main__":
    generate_vectors()
