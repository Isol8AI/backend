"""
KMS envelope encryption for agent state storage (Enclave version).

This module provides KMS-based envelope encryption for background mode agents.
In background mode, the enclave encrypts agent state using a data encryption key (DEK),
then encrypts the DEK with AWS KMS. This allows the enclave to autonomously decrypt
and re-encrypt agent state without requiring the user's key.

Security model:
- DEK is randomly generated per encryption operation
- DEK is used to encrypt the data with AES-256-GCM
- DEK is encrypted with KMS (envelope encryption)
- KMS key policy restricts access to authenticated Nitro Enclaves only (via attestation)

In production, the KMS key has an attestation-based policy that only allows
decryption from the specific enclave PCR measurements, ensuring the server
cannot decrypt the data outside the enclave.
"""

import os
from typing import Dict

import boto3
from botocore.exceptions import ClientError

from crypto_primitives import encrypt_aes_gcm, decrypt_aes_gcm


def encrypt_with_kms(plaintext_data: bytes, kms_key_id: str) -> Dict[str, bytes]:
    """
    Encrypt data using KMS envelope encryption.

    Process:
    1. Generate random 32-byte data encryption key (DEK)
    2. Encrypt plaintext data with DEK using AES-256-GCM
    3. Encrypt DEK with AWS KMS
    4. Return envelope containing encrypted DEK and encrypted data

    Args:
        plaintext_data: Data to encrypt (e.g., agent state tarball)
        kms_key_id: KMS key ID or ARN

    Returns:
        Envelope dict containing:
        - encrypted_dek: KMS-encrypted data encryption key
        - iv: AES-GCM initialization vector
        - ciphertext: Encrypted data
        - auth_tag: AES-GCM authentication tag

    Raises:
        ValueError: If KMS key ID is invalid
        ClientError: If KMS encryption fails
    """
    if not kms_key_id:
        raise ValueError("KMS key ID is required for background mode encryption")

    # Step 1: Generate random 32-byte DEK
    dek = os.urandom(32)

    # Step 2: Encrypt data with DEK using AES-256-GCM
    iv, ciphertext, auth_tag = encrypt_aes_gcm(dek, plaintext_data)

    # Step 3: Encrypt DEK with KMS
    kms_client = boto3.client("kms")
    try:
        response = kms_client.encrypt(
            KeyId=kms_key_id,
            Plaintext=dek,
        )
        encrypted_dek = response["CiphertextBlob"]
    except ClientError as e:
        print(f"[Enclave] KMS encryption failed: {e}", flush=True)
        raise

    # Step 4: Return envelope
    envelope = {
        "encrypted_dek": encrypted_dek,
        "iv": iv,
        "ciphertext": ciphertext,
        "auth_tag": auth_tag,
    }

    print(
        f"[Enclave] Encrypted {len(plaintext_data)} bytes with KMS envelope "
        f"(DEK: {len(encrypted_dek)} bytes, ciphertext: {len(ciphertext)} bytes)",
        flush=True,
    )

    return envelope


def decrypt_with_kms(envelope: Dict[str, bytes], kms_key_id: str) -> bytes:
    """
    Decrypt data using KMS envelope encryption.

    Process:
    1. Decrypt DEK using AWS KMS
    2. Decrypt data with DEK using AES-256-GCM
    3. Return plaintext data

    Args:
        envelope: Envelope dict from encrypt_with_kms containing:
            - encrypted_dek: KMS-encrypted data encryption key
            - iv: AES-GCM initialization vector
            - ciphertext: Encrypted data
            - auth_tag: AES-GCM authentication tag
        kms_key_id: KMS key ID or ARN (for logging/validation only)

    Returns:
        Decrypted plaintext data

    Raises:
        ValueError: If envelope is missing required fields
        ClientError: If KMS decryption fails
        Exception: If AES-GCM decryption fails (invalid auth tag, corrupted data)
    """
    # Validate envelope structure
    required_fields = {"encrypted_dek", "iv", "ciphertext", "auth_tag"}
    if not all(field in envelope for field in required_fields):
        missing = required_fields - set(envelope.keys())
        raise ValueError(f"Invalid KMS envelope: missing fields {missing}")

    # Step 1: Decrypt DEK with KMS
    kms_client = boto3.client("kms")
    try:
        response = kms_client.decrypt(
            CiphertextBlob=envelope["encrypted_dek"],
            # Note: KeyId is optional for decrypt (embedded in ciphertext blob)
            # but we can provide it for validation
        )
        dek = response["Plaintext"]
    except ClientError as e:
        print(f"[Enclave] KMS decryption failed: {e}", flush=True)
        raise

    # Step 2: Decrypt data with DEK using AES-256-GCM
    try:
        plaintext = decrypt_aes_gcm(
            key=dek,
            iv=envelope["iv"],
            ciphertext=envelope["ciphertext"],
            auth_tag=envelope["auth_tag"],
        )
    except Exception as e:
        print(f"[Enclave] AES-GCM decryption failed: {e}", flush=True)
        raise

    print(f"[Enclave] Decrypted {len(plaintext)} bytes from KMS envelope", flush=True)

    return plaintext
