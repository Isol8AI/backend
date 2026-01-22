"""
Pydantic schemas for encryption-related API endpoints.

Security Note:
- CreateUserKeysRequest contains encrypted data only
- Server never sees plaintext private keys
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from core.crypto import EncryptedPayload as CryptoEncryptedPayload


def validate_hex_string(value: str, expected_length: Optional[int] = None) -> str:
    """Validate that a string is valid hex and optionally check length."""
    try:
        bytes.fromhex(value)
    except ValueError:
        raise ValueError("Must be a valid hex string")

    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"Must be {expected_length} hex characters")

    return value.lower()


class EncryptedPayload(BaseModel):
    """
    Standard encrypted payload structure.
    All fields are hex-encoded strings.

    Used for both storage and transmission of encrypted data.
    """

    ephemeral_public_key: str = Field(
        ..., min_length=64, max_length=64, description="Sender's ephemeral X25519 public key (32 bytes hex)"
    )
    iv: str = Field(..., min_length=32, max_length=32, description="AES-GCM initialization vector (16 bytes hex)")
    ciphertext: str = Field(..., min_length=1, description="Encrypted content (variable length hex)")
    auth_tag: str = Field(..., min_length=32, max_length=32, description="AES-GCM authentication tag (16 bytes hex)")
    hkdf_salt: str = Field(..., min_length=64, max_length=64, description="HKDF derivation salt (32 bytes hex)")

    @field_validator("ephemeral_public_key")
    @classmethod
    def validate_ephemeral_public_key(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    @field_validator("iv")
    @classmethod
    def validate_iv(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("ciphertext")
    @classmethod
    def validate_ciphertext(cls, v: str) -> str:
        return validate_hex_string(v)

    @field_validator("auth_tag")
    @classmethod
    def validate_auth_tag(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("hkdf_salt")
    @classmethod
    def validate_hkdf_salt(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    def to_crypto_payload(self) -> "CryptoEncryptedPayload":
        """
        Convert hex-encoded API payload to bytes-based crypto payload.

        Use this when passing encrypted data to crypto operations.
        """
        from core.crypto import EncryptedPayload as CryptoEncryptedPayload

        return CryptoEncryptedPayload(
            ephemeral_public_key=bytes.fromhex(self.ephemeral_public_key),
            iv=bytes.fromhex(self.iv),
            ciphertext=bytes.fromhex(self.ciphertext),
            auth_tag=bytes.fromhex(self.auth_tag),
            hkdf_salt=bytes.fromhex(self.hkdf_salt),
        )

    @classmethod
    def from_crypto_payload(cls, crypto_payload: "CryptoEncryptedPayload") -> "EncryptedPayload":
        """
        Convert bytes-based crypto payload to hex-encoded API payload.

        Use this when returning encrypted data from crypto operations to the API.
        """
        return cls(
            ephemeral_public_key=crypto_payload.ephemeral_public_key.hex(),
            iv=crypto_payload.iv.hex(),
            ciphertext=crypto_payload.ciphertext.hex(),
            auth_tag=crypto_payload.auth_tag.hex(),
            hkdf_salt=crypto_payload.hkdf_salt.hex(),
        )


class CreateUserKeysRequest(BaseModel):
    """
    Request to store user encryption keys.

    All private key data is already encrypted client-side.
    Server stores but cannot decrypt.
    """

    public_key: str = Field(..., min_length=64, max_length=64, description="X25519 public key (32 bytes hex)")

    # Passcode-encrypted private key
    encrypted_private_key: str = Field(..., min_length=1, description="AES-GCM encrypted private key (hex)")
    iv: str = Field(..., min_length=32, max_length=32, description="AES-GCM IV (16 bytes hex)")
    tag: str = Field(..., min_length=32, max_length=32, description="AES-GCM auth tag (16 bytes hex)")
    salt: str = Field(..., min_length=64, max_length=64, description="Argon2id salt (32 bytes hex)")

    # Recovery-encrypted private key
    recovery_encrypted_private_key: str = Field(..., min_length=1, description="Recovery-encrypted private key (hex)")
    recovery_iv: str = Field(..., min_length=32, max_length=32, description="Recovery AES-GCM IV (16 bytes hex)")
    recovery_tag: str = Field(..., min_length=32, max_length=32, description="Recovery AES-GCM auth tag (16 bytes hex)")
    recovery_salt: str = Field(..., min_length=64, max_length=64, description="Recovery Argon2id salt (32 bytes hex)")

    @field_validator("public_key")
    @classmethod
    def validate_public_key(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    @field_validator("iv")
    @classmethod
    def validate_iv(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("tag")
    @classmethod
    def validate_tag(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("salt")
    @classmethod
    def validate_salt(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    @field_validator("encrypted_private_key")
    @classmethod
    def validate_encrypted_private_key(cls, v: str) -> str:
        return validate_hex_string(v)

    @field_validator("recovery_iv")
    @classmethod
    def validate_recovery_iv(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("recovery_tag")
    @classmethod
    def validate_recovery_tag(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("recovery_salt")
    @classmethod
    def validate_recovery_salt(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    @field_validator("recovery_encrypted_private_key")
    @classmethod
    def validate_recovery_encrypted_private_key(cls, v: str) -> str:
        return validate_hex_string(v)


class UserKeysResponse(BaseModel):
    """Response with user's encrypted keys for client-side decryption."""

    public_key: str
    encrypted_private_key: str
    iv: str
    tag: str
    salt: str

    # Recovery keys only returned on specific request
    recovery_encrypted_private_key: Optional[str] = None
    recovery_iv: Optional[str] = None
    recovery_tag: Optional[str] = None
    recovery_salt: Optional[str] = None

    model_config = {"from_attributes": True}


class EncryptionStatusResponse(BaseModel):
    """User's encryption status."""

    has_encryption_keys: bool
    public_key: Optional[str] = None
    encryption_created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class EncryptedMessageResponse(BaseModel):
    """Message in API response - always encrypted in zero-trust model."""

    id: str
    session_id: str
    role: str
    model_used: Optional[str] = None
    created_at: datetime
    encrypted_content: EncryptedPayload

    model_config = {"from_attributes": True}


class SendEncryptedMessageRequest(BaseModel):
    """
    Request to send an encrypted message.

    The message content is encrypted TO the enclave's public key.
    The enclave will decrypt, process with LLM, and re-encrypt for storage.
    """

    session_id: Optional[str] = Field(None, description="Session ID. If None, creates new session.")
    model: str = Field(..., description="Model ID to use for response")
    encrypted_message: EncryptedPayload = Field(..., description="Message encrypted to enclave's public key")
    encrypted_history: Optional[list[EncryptedPayload]] = Field(
        None, description="Previous messages re-encrypted to enclave for context"
    )
    encrypted_memories: Optional[list[EncryptedPayload]] = Field(
        None, description="Relevant memories re-encrypted to enclave for context injection"
    )
    facts_context: Optional[str] = Field(
        None, description="Client-side formatted facts context string (already decrypted by client)"
    )
    client_transport_public_key: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Client's ephemeral X25519 public key for response encryption (32 bytes hex)",
    )

    @field_validator("client_transport_public_key")
    @classmethod
    def validate_client_transport_public_key(cls, v: str) -> str:
        return validate_hex_string(v, 64)


class EncryptedChatResponse(BaseModel):
    """Response from encrypted chat endpoint."""

    session_id: str
    message_id: str
    encrypted_response: EncryptedPayload = Field(..., description="Assistant response encrypted to user's public key")
    model_used: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
