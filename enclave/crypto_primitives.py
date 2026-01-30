#!/usr/bin/env python3
"""
M3: Cryptographic Primitives for Nitro Enclave
==============================================

These primitives MUST produce identical outputs to the TypeScript frontend
implementation in frontend/src/lib/crypto/primitives.ts.

Algorithms:
- X25519 for ECDH key exchange
- HKDF-SHA512 for key derivation
- AES-256-GCM for authenticated encryption

Context strings (must match frontend):
- "client-to-enclave-transport": Messages from client to enclave
- "enclave-to-client-transport": Responses from enclave to client
- "user-message-storage": User messages stored in database
- "assistant-message-storage": Assistant messages stored in database
- "org-key-distribution": Org private key encrypted to member
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# =============================================================================
# Types
# =============================================================================


@dataclass
class KeyPair:
    """X25519 keypair container."""

    private_key: bytes  # 32 bytes
    public_key: bytes  # 32 bytes


@dataclass
class EncryptedPayload:
    """Standard encrypted payload structure for storage/transmission."""

    ephemeral_public_key: bytes  # 32 bytes
    iv: bytes  # 16 bytes
    ciphertext: bytes  # variable length
    auth_tag: bytes  # 16 bytes
    hkdf_salt: bytes  # 32 bytes

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with hex encoding."""
        return {
            "ephemeral_public_key": self.ephemeral_public_key.hex(),
            "iv": self.iv.hex(),
            "ciphertext": self.ciphertext.hex(),
            "auth_tag": self.auth_tag.hex(),
            "hkdf_salt": self.hkdf_salt.hex(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EncryptedPayload":
        """Create from dict with hex-encoded values."""
        return cls(
            ephemeral_public_key=bytes.fromhex(data["ephemeral_public_key"]),
            iv=bytes.fromhex(data["iv"]),
            ciphertext=bytes.fromhex(data["ciphertext"]),
            auth_tag=bytes.fromhex(data["auth_tag"]),
            hkdf_salt=bytes.fromhex(data["hkdf_salt"]),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "EncryptedPayload":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Utility Functions
# =============================================================================


def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to hex string."""
    return data.hex()


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string to bytes."""
    return bytes.fromhex(hex_str)


def secure_compare(a: bytes, b: bytes) -> bool:
    """
    Compare two byte arrays in constant time.
    Prevents timing attacks when comparing secrets.
    """
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0


# =============================================================================
# Key Generation
# =============================================================================


def generate_random_bytes(length: int = 32) -> bytes:
    """Generate cryptographically secure random bytes."""
    return os.urandom(length)


def generate_x25519_keypair() -> KeyPair:
    """
    Generate a new X25519 keypair for key exchange.

    Returns:
        KeyPair with 32-byte private and public keys
    """
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()

    return KeyPair(
        private_key=private_key.private_bytes_raw(),
        public_key=public_key.public_bytes_raw(),
    )


def get_public_key_from_private(private_key_bytes: bytes) -> bytes:
    """Derive public key from private key bytes."""
    private_key = X25519PrivateKey.from_private_bytes(private_key_bytes)
    return private_key.public_key().public_bytes_raw()


# =============================================================================
# Key Derivation
# =============================================================================


def derive_key_from_ecdh(
    private_key: bytes,
    public_key: bytes,
    context: str,
    salt: Optional[bytes] = None,
) -> Tuple[bytes, bytes]:
    """
    Derive a symmetric key from X25519 ECDH shared secret using HKDF-SHA512.

    This function:
    1. Computes the X25519 shared secret
    2. Generates a random salt if not provided
    3. Derives a 32-byte key using HKDF-SHA512

    Args:
        private_key: Our X25519 private key (32 bytes)
        public_key: Their X25519 public key (32 bytes)
        context: Context string for domain separation
        salt: Optional HKDF salt. If None, generates random 32-byte salt.

    Returns:
        Tuple of (derived_key, salt) - both 32 bytes

    Raises:
        ValueError: If keys are wrong length
    """
    if len(private_key) != 32:
        raise ValueError("Private key must be 32 bytes")
    if len(public_key) != 32:
        raise ValueError("Public key must be 32 bytes")

    # Generate random salt if not provided
    actual_salt = salt if salt is not None else generate_random_bytes(32)
    if len(actual_salt) != 32:
        raise ValueError("Salt must be 32 bytes")

    # Compute X25519 shared secret
    priv_key_obj = X25519PrivateKey.from_private_bytes(private_key)
    pub_key_obj = X25519PublicKey.from_public_bytes(public_key)
    shared_secret = priv_key_obj.exchange(pub_key_obj)

    # Derive key using HKDF-SHA512
    hkdf = HKDF(
        algorithm=hashes.SHA512(),
        length=32,
        salt=actual_salt,
        info=context.encode("utf-8"),
    )
    derived_key = hkdf.derive(shared_secret)

    return derived_key, actual_salt


# =============================================================================
# Symmetric Encryption (AES-256-GCM)
# =============================================================================


def encrypt_aes_gcm(
    key: bytes,
    plaintext: bytes,
    associated_data: Optional[bytes] = None,
) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypt data using AES-256-GCM.

    Args:
        key: 32-byte encryption key
        plaintext: Data to encrypt
        associated_data: Optional additional authenticated data (AAD)

    Returns:
        Tuple of (iv, ciphertext, auth_tag)
        - iv: 16 bytes
        - ciphertext: same length as plaintext
        - auth_tag: 16 bytes

    Raises:
        ValueError: If key is not 32 bytes
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")

    iv = generate_random_bytes(16)
    aesgcm = AESGCM(key)

    # AESGCM returns ciphertext + tag concatenated
    ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, associated_data)

    # Split: ciphertext is all but last 16 bytes, tag is last 16 bytes
    ciphertext = ciphertext_with_tag[:-16]
    auth_tag = ciphertext_with_tag[-16:]

    return iv, ciphertext, auth_tag


def decrypt_aes_gcm(
    key: bytes,
    iv: bytes,
    ciphertext: bytes,
    auth_tag: bytes,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """
    Decrypt data using AES-256-GCM.

    Args:
        key: 32-byte decryption key
        iv: 16-byte initialization vector
        ciphertext: Encrypted data
        auth_tag: 16-byte authentication tag
        associated_data: Optional AAD (must match encryption)

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If key/iv/tag are wrong length
        InvalidTag: If authentication fails
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")
    if len(auth_tag) != 16:
        raise ValueError("Auth tag must be 16 bytes")

    aesgcm = AESGCM(key)

    # Reconstruct ciphertext + tag
    ciphertext_with_tag = ciphertext + auth_tag

    return aesgcm.decrypt(iv, ciphertext_with_tag, associated_data)


# =============================================================================
# High-Level Encryption (Ephemeral ECDH Pattern)
# =============================================================================


def encrypt_to_public_key(
    recipient_public_key: bytes,
    plaintext: bytes,
    context: str,
) -> EncryptedPayload:
    """
    Encrypt data to a recipient's public key using ephemeral ECDH.

    This implements the ephemeral ECDH pattern:
    1. Generate ephemeral X25519 keypair
    2. Compute shared secret with recipient's public key
    3. Derive symmetric key via HKDF with random salt
    4. Encrypt with AES-256-GCM
    5. Discard ephemeral private key

    Args:
        recipient_public_key: Recipient's X25519 public key (32 bytes)
        plaintext: Data to encrypt
        context: Context string for domain separation

    Returns:
        EncryptedPayload containing all data needed for decryption
    """
    if len(recipient_public_key) != 32:
        raise ValueError("Recipient public key must be 32 bytes")

    # 1. Generate ephemeral keypair
    ephemeral = generate_x25519_keypair()

    # 2-3. ECDH + HKDF with random salt
    derived_key, salt = derive_key_from_ecdh(
        ephemeral.private_key,
        recipient_public_key,
        context,
    )

    # 4. Encrypt
    iv, ciphertext, auth_tag = encrypt_aes_gcm(derived_key, plaintext)

    # 5. Ephemeral private key is discarded (goes out of scope)

    return EncryptedPayload(
        ephemeral_public_key=ephemeral.public_key,
        iv=iv,
        ciphertext=ciphertext,
        auth_tag=auth_tag,
        hkdf_salt=salt,
    )


def decrypt_with_private_key(
    private_key: bytes,
    payload: EncryptedPayload,
    context: str,
) -> bytes:
    """
    Decrypt data encrypted with encrypt_to_public_key.

    This performs the inverse of the ephemeral ECDH pattern:
    1. Compute shared secret using our private key and sender's ephemeral public key
    2. Derive the same symmetric key via HKDF (using stored salt)
    3. Decrypt with AES-256-GCM

    Args:
        private_key: Our X25519 private key (32 bytes)
        payload: EncryptedPayload from encrypt_to_public_key
        context: Context string (MUST match encryption context)

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If private key is wrong length
        InvalidTag: If context doesn't match or decryption fails
    """
    if len(private_key) != 32:
        raise ValueError("Private key must be 32 bytes")

    # Derive the same symmetric key
    derived_key, _ = derive_key_from_ecdh(
        private_key,
        payload.ephemeral_public_key,
        context,
        payload.hkdf_salt,  # Use stored salt
    )

    # Decrypt
    return decrypt_aes_gcm(
        derived_key,
        payload.iv,
        payload.ciphertext,
        payload.auth_tag,
    )


# =============================================================================
# Test Vector Verification
# =============================================================================


def verify_ecdh_vector(
    private_key_hex: str,
    public_key_hex: str,
    context: str,
    salt_hex: str,
    expected_key_hex: str,
) -> bool:
    """Verify an ECDH test vector."""
    derived_key, _ = derive_key_from_ecdh(
        hex_to_bytes(private_key_hex),
        hex_to_bytes(public_key_hex),
        context,
        hex_to_bytes(salt_hex),
    )
    return bytes_to_hex(derived_key) == expected_key_hex


def verify_aes_gcm_vector(
    key_hex: str,
    iv_hex: str,
    plaintext_hex: str,
    ciphertext_hex: str,
    auth_tag_hex: str,
    aad_hex: Optional[str] = None,
) -> bool:
    """Verify an AES-GCM test vector by decrypting."""
    aad = hex_to_bytes(aad_hex) if aad_hex else None
    plaintext = decrypt_aes_gcm(
        hex_to_bytes(key_hex),
        hex_to_bytes(iv_hex),
        hex_to_bytes(ciphertext_hex),
        hex_to_bytes(auth_tag_hex),
        aad,
    )
    return bytes_to_hex(plaintext) == plaintext_hex


if __name__ == "__main__":
    # Quick self-test
    print("Testing crypto primitives...")

    # Test keypair generation
    kp = generate_x25519_keypair()
    print(f"Generated keypair: pub={kp.public_key.hex()[:16]}...")

    # Test encrypt/decrypt round-trip
    recipient = generate_x25519_keypair()
    message = b"Hello from enclave!"
    context = "test-context"

    encrypted = encrypt_to_public_key(recipient.public_key, message, context)
    decrypted = decrypt_with_private_key(recipient.private_key, encrypted, context)

    assert decrypted == message, "Round-trip failed!"
    print("Encrypt/decrypt round-trip: PASS")

    print("All self-tests passed!")
