"""Cryptographic primitives package for zero-trust encryption."""

from .primitives import (
    KeyPair,
    EncryptedPayload,
    generate_x25519_keypair,
    derive_key_from_ecdh,
    encrypt_aes_gcm,
    decrypt_aes_gcm,
    encrypt_to_public_key,
    decrypt_with_private_key,
    generate_salt,
    generate_recovery_code,
    secure_compare,
)

__all__ = [
    "KeyPair",
    "EncryptedPayload",
    "generate_x25519_keypair",
    "derive_key_from_ecdh",
    "encrypt_aes_gcm",
    "decrypt_aes_gcm",
    "encrypt_to_public_key",
    "decrypt_with_private_key",
    "generate_salt",
    "generate_recovery_code",
    "secure_compare",
]
