"""
User model with encryption key storage.

Security Note:
- encrypted_private_key can ONLY be decrypted with user's passcode
- recovery_encrypted_private_key can ONLY be decrypted with recovery code
- Server cannot access private key without user-provided secret
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Boolean, DateTime, Index
from sqlalchemy.orm import relationship

from .base import Base


class User(Base):
    """
    User model synced from Clerk with encryption key storage.

    Encryption Fields:
        public_key: X25519 public key (not secret, used for encryption TO user)
        encrypted_private_key: Private key encrypted with passcode-derived key
        key_salt: Salt for Argon2id passcode derivation
        recovery_*: Backup encryption using recovery code
    """

    __tablename__ = "users"

    # Primary key (from Clerk)
    id = Column(String, primary_key=True)  # Clerk User ID (user_xxx)

    # =========================================================================
    # Encryption Key Fields
    # =========================================================================

    # Public key - NOT secret, stored as hex string (64 chars = 32 bytes)
    public_key = Column(String(64), nullable=True, index=True)

    # Encrypted private key (passcode-protected)
    # Stored as hex strings for simplicity
    encrypted_private_key = Column(String, nullable=True)  # Variable length ciphertext
    encrypted_private_key_iv = Column(String(32), nullable=True)  # 16 bytes = 32 hex
    encrypted_private_key_tag = Column(String(32), nullable=True)  # 16 bytes = 32 hex
    key_salt = Column(String(64), nullable=True)  # 32 bytes = 64 hex (for Argon2id)

    # Recovery-encrypted private key (recovery code protected)
    recovery_encrypted_private_key = Column(String, nullable=True)
    recovery_iv = Column(String(32), nullable=True)
    recovery_tag = Column(String(32), nullable=True)
    recovery_salt = Column(String(64), nullable=True)

    # Encryption metadata
    has_encryption_keys = Column(Boolean, default=False, nullable=False)
    encryption_created_at = Column(DateTime, nullable=True)

    # =========================================================================
    # Relationships
    # =========================================================================

    memberships = relationship("OrganizationMembership", back_populates="user", cascade="all, delete-orphan")

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

    # =========================================================================
    # Indexes
    # =========================================================================

    __table_args__ = (Index("ix_users_has_encryption", "has_encryption_keys"),)

    # =========================================================================
    # Helper Properties
    # =========================================================================

    @property
    def can_receive_encrypted_messages(self) -> bool:
        """User can receive encrypted messages if they have a public key."""
        return self.public_key is not None

    @property
    def encryption_key_info(self) -> dict:
        """
        Return non-sensitive encryption info.

        NEVER include encrypted_private_key or recovery keys in API responses
        unless specifically requested by authenticated user.
        """
        return {
            "has_encryption_keys": bool(self.has_encryption_keys),
            "public_key": self.public_key,
            "encryption_created_at": (self.encryption_created_at.isoformat() if self.encryption_created_at else None),
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def set_encryption_keys(
        self,
        public_key: str,
        encrypted_private_key: str,
        iv: str,
        tag: str,
        salt: str,
        recovery_encrypted_private_key: str,
        recovery_iv: str,
        recovery_tag: str,
        recovery_salt: str,
    ) -> None:
        """
        Set all encryption key fields atomically.

        Args:
            public_key: Hex-encoded X25519 public key (64 chars)
            encrypted_private_key: Hex-encoded encrypted private key
            iv: Hex-encoded AES-GCM IV (32 chars)
            tag: Hex-encoded AES-GCM auth tag (32 chars)
            salt: Hex-encoded Argon2id salt (64 chars)
            recovery_*: Same but for recovery code encryption

        Raises:
            ValueError: If any field has invalid length
        """
        # Validate lengths
        if len(public_key) != 64:
            raise ValueError("public_key must be 64 hex characters (32 bytes)")
        if len(iv) != 32:
            raise ValueError("iv must be 32 hex characters (16 bytes)")
        if len(tag) != 32:
            raise ValueError("tag must be 32 hex characters (16 bytes)")
        if len(salt) != 64:
            raise ValueError("salt must be 64 hex characters (32 bytes)")
        if len(recovery_iv) != 32:
            raise ValueError("recovery_iv must be 32 hex characters (16 bytes)")
        if len(recovery_tag) != 32:
            raise ValueError("recovery_tag must be 32 hex characters (16 bytes)")
        if len(recovery_salt) != 64:
            raise ValueError("recovery_salt must be 64 hex characters (32 bytes)")

        # Validate hex strings
        for name, value in [
            ("public_key", public_key),
            ("iv", iv),
            ("tag", tag),
            ("salt", salt),
            ("recovery_iv", recovery_iv),
            ("recovery_tag", recovery_tag),
            ("recovery_salt", recovery_salt),
        ]:
            try:
                bytes.fromhex(value)
            except ValueError:
                raise ValueError(f"{name} must be a valid hex string")

        self.public_key = public_key.lower()
        self.encrypted_private_key = encrypted_private_key.lower()
        self.encrypted_private_key_iv = iv.lower()
        self.encrypted_private_key_tag = tag.lower()
        self.key_salt = salt.lower()

        self.recovery_encrypted_private_key = recovery_encrypted_private_key.lower()
        self.recovery_iv = recovery_iv.lower()
        self.recovery_tag = recovery_tag.lower()
        self.recovery_salt = recovery_salt.lower()

        self.has_encryption_keys = True
        self.encryption_created_at = datetime.utcnow()

    def clear_encryption_keys(self) -> None:
        """
        Clear all encryption keys.

        Use with caution - this makes all encrypted messages unrecoverable!
        """
        self.public_key = None
        self.encrypted_private_key = None
        self.encrypted_private_key_iv = None
        self.encrypted_private_key_tag = None
        self.key_salt = None

        self.recovery_encrypted_private_key = None
        self.recovery_iv = None
        self.recovery_tag = None
        self.recovery_salt = None

        self.has_encryption_keys = False
        self.encryption_created_at = None

    def get_encrypted_keys_for_unlock(self) -> Optional[dict]:
        """
        Get encrypted key data needed for client-side unlock.

        Returns None if user has no encryption keys.
        Only call this for the authenticated user themselves.
        """
        if not self.has_encryption_keys:
            return None

        return {
            "public_key": self.public_key,
            "encrypted_private_key": self.encrypted_private_key,
            "iv": self.encrypted_private_key_iv,
            "tag": self.encrypted_private_key_tag,
            "salt": self.key_salt,
        }

    def get_recovery_keys(self) -> Optional[dict]:
        """
        Get recovery-encrypted key data.

        Returns None if user has no encryption keys.
        Only call this for the authenticated user themselves.
        """
        if not self.has_encryption_keys:
            return None

        return {
            "recovery_encrypted_private_key": self.recovery_encrypted_private_key,
            "recovery_iv": self.recovery_iv,
            "recovery_tag": self.recovery_tag,
            "recovery_salt": self.recovery_salt,
        }
