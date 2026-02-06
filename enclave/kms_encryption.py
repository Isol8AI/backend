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

Uses botocore directly (not boto3) because the enclave has no IMDS for automatic
credential resolution. Credentials are injected from the parent instance via vsock.
Matches the pattern used by bedrock_client.py.
"""

import base64
import json
import os
from typing import Dict, Optional

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from crypto_primitives import encrypt_aes_gcm, decrypt_aes_gcm
from vsock_http_client import VsockHttpClient


# Module-level credentials (injected by bedrock_server.py)
_credentials: Optional[Credentials] = None
_region: str = os.environ.get("AWS_REGION", "us-east-1")
_http_client: Optional[VsockHttpClient] = None


def set_kms_credentials(
    access_key_id: str,
    secret_access_key: str,
    session_token: str,
    region: Optional[str] = None,
):
    """Set AWS credentials for KMS operations. Called by bedrock_server.py."""
    global _credentials, _region, _http_client
    _credentials = Credentials(
        access_key=access_key_id,
        secret_key=secret_access_key,
        token=session_token,
    )
    if region:
        _region = region
    if _http_client is None:
        _http_client = VsockHttpClient()


def _kms_request(action: str, payload: dict) -> dict:
    """Make a signed KMS API request via vsock proxy."""
    if _credentials is None:
        raise ValueError("KMS credentials not set. Call set_kms_credentials() first.")
    if _http_client is None:
        raise ValueError("HTTP client not initialized.")

    host = f"kms.{_region}.amazonaws.com"
    url = f"https://{host}/"
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": f"TrentService.{action}",
        "Host": host,
    }

    # Sign request with SigV4
    request = AWSRequest(method="POST", url=url, data=body, headers=headers)
    SigV4Auth(_credentials, "kms", _region).add_auth(request)
    signed_headers = dict(request.headers)

    # Make request via vsock proxy
    response = _http_client.request("POST", url, signed_headers, body)

    if response.status != 200:
        error_body = response.body.decode("utf-8", errors="replace")
        raise Exception(f"KMS {action} failed (HTTP {response.status}): {error_body}")

    return json.loads(response.body)


def encrypt_with_kms(plaintext_data: bytes, kms_key_id: str) -> Dict[str, bytes]:
    """
    Encrypt data using KMS envelope encryption.

    Process:
    1. Generate random 32-byte data encryption key (DEK)
    2. Encrypt plaintext data with DEK using AES-256-GCM
    3. Encrypt DEK with AWS KMS
    4. Return envelope containing encrypted DEK and encrypted data
    """
    if not kms_key_id:
        raise ValueError("KMS key ID is required for background mode encryption")

    # Step 1: Generate random 32-byte DEK
    dek = os.urandom(32)

    # Step 2: Encrypt data with DEK using AES-256-GCM
    iv, ciphertext, auth_tag = encrypt_aes_gcm(dek, plaintext_data)

    # Step 3: Encrypt DEK with KMS
    response = _kms_request("Encrypt", {
        "KeyId": kms_key_id,
        "Plaintext": base64.b64encode(dek).decode("ascii"),
    })
    encrypted_dek = base64.b64decode(response["CiphertextBlob"])

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
    """
    # Validate envelope structure
    required_fields = {"encrypted_dek", "iv", "ciphertext", "auth_tag"}
    if not all(field in envelope for field in required_fields):
        missing = required_fields - set(envelope.keys())
        raise ValueError(f"Invalid KMS envelope: missing fields {missing}")

    # Step 1: Decrypt DEK with KMS
    response = _kms_request("Decrypt", {
        "CiphertextBlob": base64.b64encode(envelope["encrypted_dek"]).decode("ascii"),
    })
    dek = base64.b64decode(response["Plaintext"])

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
