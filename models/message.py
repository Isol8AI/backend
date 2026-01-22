"""
Encrypted message model.

Security Note:
- ALL messages are encrypted - there is NO plaintext storage
- Server cannot decrypt messages (needs user's or org's private key)
- ephemeral_public_key allows recipient to derive decryption key

Zero-Trust Principle:
  The server NEVER stores readable message content. Period.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import Base


class MessageRole(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(Base):
    """
    Encrypted message in a chat session.

    ALL messages are encrypted using the ephemeral ECDH pattern.
    There is NO plaintext content field - this is a zero-trust platform.

    The encrypted content is an EncryptedPayload:
    - ephemeral_public_key: Sender's ephemeral key for ECDH
    - iv: AES-GCM initialization vector
    - ciphertext: Encrypted message content
    - auth_tag: AES-GCM authentication tag
    - hkdf_salt: Salt used in HKDF key derivation
    """

    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    # Role
    role = Column(String, nullable=False)

    # =========================================================================
    # Encrypted Content (ALWAYS encrypted - no plaintext option)
    #
    # EncryptedPayload structure stored as hex strings:
    # - ephemeral_public_key: 32 bytes (64 hex chars)
    # - iv: 16 bytes (32 hex chars)
    # - ciphertext: variable length
    # - auth_tag: 16 bytes (32 hex chars)
    # - hkdf_salt: 32 bytes (64 hex chars)
    # =========================================================================

    ephemeral_public_key = Column(String(64), nullable=False)  # Required for decryption
    iv = Column(String(32), nullable=False)
    ciphertext = Column(Text, nullable=False)  # Variable length, hex encoded
    auth_tag = Column(String(32), nullable=False)
    hkdf_salt = Column(String(64), nullable=False)

    # =========================================================================
    # Metadata (not encrypted - needed for queries/billing)
    # =========================================================================

    # Model used (for assistant messages)
    model_used = Column(String, nullable=True)

    # Token counts (for billing/limits) - these come from enclave
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # =========================================================================
    # Relationships
    # =========================================================================

    session = relationship("Session", back_populates="messages")

    # =========================================================================
    # Indexes
    # =========================================================================

    __table_args__ = (Index("ix_messages_session_created", "session_id", "created_at"),)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def encrypted_payload(self) -> dict:
        """
        Get encrypted payload as dict for API response.

        Client will use this to decrypt the message content.
        """
        return {
            "ephemeral_public_key": self.ephemeral_public_key,
            "iv": self.iv,
            "ciphertext": self.ciphertext,
            "auth_tag": self.auth_tag,
            "hkdf_salt": self.hkdf_salt,
        }

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create_encrypted(
        cls,
        id: str,
        session_id: str,
        role: MessageRole,
        ephemeral_public_key: str,
        iv: str,
        ciphertext: str,
        auth_tag: str,
        hkdf_salt: str,
        model_used: str = None,
        input_tokens: int = None,
        output_tokens: int = None,
    ) -> "Message":
        """
        Create an encrypted message with validation.

        Args:
            id: Message UUID
            session_id: Parent session UUID
            role: USER, ASSISTANT, or SYSTEM
            ephemeral_public_key: 64 hex chars (32 bytes)
            iv: 32 hex chars (16 bytes)
            ciphertext: Variable length hex string
            auth_tag: 32 hex chars (16 bytes)
            hkdf_salt: 64 hex chars (32 bytes)
            model_used: Model ID (for assistant messages)
            input_tokens: Token count for billing
            output_tokens: Token count for billing

        Raises:
            ValueError: If any field has invalid length or format
        """
        # Validate lengths
        if len(ephemeral_public_key) != 64:
            raise ValueError("ephemeral_public_key must be 64 hex characters")
        if len(iv) != 32:
            raise ValueError("iv must be 32 hex characters")
        if len(auth_tag) != 32:
            raise ValueError("auth_tag must be 32 hex characters")
        if len(hkdf_salt) != 64:
            raise ValueError("hkdf_salt must be 64 hex characters")
        if not ciphertext:
            raise ValueError("ciphertext cannot be empty")

        # Validate hex strings
        for name, value in [
            ("ephemeral_public_key", ephemeral_public_key),
            ("iv", iv),
            ("auth_tag", auth_tag),
            ("hkdf_salt", hkdf_salt),
            ("ciphertext", ciphertext),
        ]:
            try:
                bytes.fromhex(value)
            except ValueError:
                raise ValueError(f"{name} must be a valid hex string")

        # Convert role to string if enum
        role_str = role.value if isinstance(role, MessageRole) else role

        return cls(
            id=id,
            session_id=session_id,
            role=role_str,
            ephemeral_public_key=ephemeral_public_key.lower(),
            iv=iv.lower(),
            ciphertext=ciphertext.lower(),
            auth_tag=auth_tag.lower(),
            hkdf_salt=hkdf_salt.lower(),
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # =========================================================================
    # API Response Methods
    # =========================================================================

    def to_api_response(self) -> dict:
        """
        Convert to API response format.

        Always returns encrypted_content - there is no plaintext option.
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "encrypted_content": self.encrypted_payload,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
