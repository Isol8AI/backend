#!/usr/bin/env python3
"""
Cross-platform crypto test: Python encrypt → verify round-trip.
This tests the exact serialization chain used by the agent state flow.
"""
import json
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'enclave')

from enclave.crypto_primitives import (
    generate_x25519_keypair,
    encrypt_to_public_key,
    decrypt_with_private_key,
    EncryptedPayload as EnclaveEncryptedPayload,
)
from core.crypto.primitives import EncryptedPayload as CoreEncryptedPayload


def test_round_trip():
    """Test the exact serialization chain used by agent state flow."""

    # 1. Generate a user keypair (simulates what the frontend does)
    user_keypair = generate_x25519_keypair()
    print(f"User public key:  {user_keypair.public_key.hex()}")
    print(f"User private key: {user_keypair.private_key.hex()}")

    # 2. Encrypt data to user's public key (simulates enclave encrypt_to_public_key)
    plaintext = b"Hello, this is test agent state data"
    encrypted = encrypt_to_public_key(
        user_keypair.public_key,
        plaintext,
        "agent-state-storage",
    )
    print(f"\n--- Enclave encrypt_to_public_key output ---")
    print(f"ephemeral_public_key: {encrypted.ephemeral_public_key.hex()}")
    print(f"iv:                   {encrypted.iv.hex()}")
    print(f"ciphertext:           {encrypted.ciphertext.hex()}")
    print(f"auth_tag:             {encrypted.auth_tag.hex()}")
    print(f"hkdf_salt:            {encrypted.hkdf_salt.hex()}")

    # 3. Python round-trip: encrypt → to_dict → JSON → from_dict → decrypt
    # This is the EXACT path: enclave → vsock → nitro_client → websocket_chat → DB → agents.py

    # Step 3a: enclave calls .to_dict() (hex strings)
    enclave_dict = encrypted.to_dict()
    print(f"\n--- Step 3a: enclave to_dict ---")
    print(json.dumps(enclave_dict, indent=2)[:200])

    # Step 3b: sent via vsock as JSON, received by NitroEnclaveClient
    # NitroEnclaveClient calls CoreEncryptedPayload.from_dict()
    vsock_json = json.dumps({"encrypted_state": enclave_dict})
    vsock_event = json.loads(vsock_json)
    core_payload = CoreEncryptedPayload.from_dict(vsock_event["encrypted_state"])

    # Step 3c: websocket_chat.py calls .to_dict() → json.dumps → DB
    state_dict = core_payload.to_dict()
    state_json = json.dumps(state_dict).encode("utf-8")

    # Step 3d: agents.py reads from DB → _deserialize_encrypted_payload
    obj = json.loads(state_json.decode())
    deserialized = CoreEncryptedPayload(
        ephemeral_public_key=bytes.fromhex(obj["ephemeral_public_key"]),
        iv=bytes.fromhex(obj["iv"]),
        ciphertext=bytes.fromhex(obj["ciphertext"]),
        auth_tag=bytes.fromhex(obj["auth_tag"]),
        hkdf_salt=bytes.fromhex(obj["hkdf_salt"]),
    )

    # Step 3e: from_crypto_payload → API JSON (hex strings)
    api_response = {
        "ephemeral_public_key": deserialized.ephemeral_public_key.hex(),
        "iv": deserialized.iv.hex(),
        "ciphertext": deserialized.ciphertext.hex(),
        "auth_tag": deserialized.auth_tag.hex(),
        "hkdf_salt": deserialized.hkdf_salt.hex(),
    }

    print(f"\n--- Step 3e: API response (what frontend receives) ---")
    print(json.dumps(api_response, indent=2)[:200])

    # Step 3f: Verify data is identical
    assert enclave_dict == api_response, "DATA MISMATCH between enclave output and API response!"
    print("\n✓ Data integrity check passed: enclave output == API response")

    # 4. Python decrypt (verify Python→Python works)
    decrypted = decrypt_with_private_key(
        user_keypair.private_key,
        deserialized,  # Use the deserialized version (same as what API would return as bytes)
        "agent-state-storage",
    )
    assert decrypted == plaintext, f"Python decrypt failed! Got: {decrypted}"
    print("✓ Python encrypt → Python decrypt: PASSED")

    # 5. Output test vector for JS
    test_vector = {
        "private_key": user_keypair.private_key.hex(),
        "public_key": user_keypair.public_key.hex(),
        "plaintext_hex": plaintext.hex(),
        "encrypted_state": api_response,
        "context": "agent-state-storage",
    }

    with open("test_vector.json", "w") as f:
        json.dump(test_vector, f, indent=2)

    print(f"\n✓ Test vector written to test_vector.json")
    print(f"  Run the JS test to verify cross-platform decryption")


if __name__ == "__main__":
    test_round_trip()
