"""
Organization model with encryption key storage for multi-tenant support.

Security Note:
- admin_encrypted_private_key can ONLY be decrypted with org passcode (known to admins)
- Members receive org private key encrypted TO their personal public key
- Server cannot access org private key without admin-provided passcode
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, String, Boolean, Index
from sqlalchemy.orm import relationship

from .base import Base


class Organization(Base):
    """Organization model representing a team/company workspace.

    Organizations are synced from Clerk and provide multi-tenant isolation.

    Encryption Fields:
        org_public_key: X25519 public key (not secret, used for encrypting TO org)
        admin_encrypted_private_key: Org private key encrypted with org passcode
        admin_key_salt: Salt for Argon2id org passcode derivation
    """

    __tablename__ = "organizations"

    id = Column(String, primary_key=True)  # Clerk org_id
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=True)

    # =========================================================================
    # Encryption Key Fields
    # =========================================================================

    # Org public key - NOT secret, stored as hex string (64 chars = 32 bytes)
    org_public_key = Column(String(64), nullable=True, index=True)

    # Admin-encrypted org private key (org passcode protected)
    # Only admins who know the org passcode can decrypt this
    admin_encrypted_private_key = Column(String, nullable=True)  # Variable length ciphertext
    admin_encrypted_private_key_iv = Column(String(32), nullable=True)  # 16 bytes = 32 hex
    admin_encrypted_private_key_tag = Column(String(32), nullable=True)  # 16 bytes = 32 hex
    admin_key_salt = Column(String(64), nullable=True)  # 32 bytes = 64 hex (for Argon2id)

    # Encryption metadata
    has_encryption_keys = Column(Boolean, default=False, nullable=False)
    encryption_created_at = Column(DateTime, nullable=True)
    encryption_created_by = Column(String, nullable=True)  # User ID of admin who created keys

    # =========================================================================
    # Future Features
    # =========================================================================

    # Future: model endpoint configuration
    custom_model_endpoint = Column(String, nullable=True)  # For self-hosted models
    fine_tuned_model_id = Column(String, nullable=True)  # For fine-tuned models

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # =========================================================================
    # Relationships
    # =========================================================================

    memberships = relationship("OrganizationMembership", back_populates="organization", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="organization", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="organization", cascade="all, delete-orphan")

    # =========================================================================
    # Indexes
    # =========================================================================

    __table_args__ = (Index("ix_organizations_has_encryption", "has_encryption_keys"),)

    # =========================================================================
    # Helper Properties
    # =========================================================================

    @property
    def can_receive_encrypted_messages(self) -> bool:
        """Organization can receive encrypted messages if it has a public key."""
        return self.org_public_key is not None

    @property
    def encryption_key_info(self) -> dict:
        """
        Return non-sensitive encryption info.

        NEVER include admin_encrypted_private_key in general API responses.
        """
        return {
            "has_encryption_keys": bool(self.has_encryption_keys),
            "org_public_key": self.org_public_key,
            "encryption_created_at": (self.encryption_created_at.isoformat() if self.encryption_created_at else None),
            "encryption_created_by": self.encryption_created_by,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def set_encryption_keys(
        self,
        org_public_key: str,
        admin_encrypted_private_key: str,
        iv: str,
        tag: str,
        salt: str,
        created_by: str,
    ) -> None:
        """
        Set organization encryption key fields atomically.

        Args:
            org_public_key: Hex-encoded X25519 public key (64 chars)
            admin_encrypted_private_key: Hex-encoded encrypted org private key
            iv: Hex-encoded AES-GCM IV (32 chars)
            tag: Hex-encoded AES-GCM auth tag (32 chars)
            salt: Hex-encoded Argon2id salt (64 chars)
            created_by: User ID of admin creating the keys

        Raises:
            ValueError: If any field has invalid length or format
        """
        # Validate lengths
        if len(org_public_key) != 64:
            raise ValueError("org_public_key must be 64 hex characters (32 bytes)")
        if len(iv) != 32:
            raise ValueError("iv must be 32 hex characters (16 bytes)")
        if len(tag) != 32:
            raise ValueError("tag must be 32 hex characters (16 bytes)")
        if len(salt) != 64:
            raise ValueError("salt must be 64 hex characters (32 bytes)")

        # Validate hex strings
        for name, value in [
            ("org_public_key", org_public_key),
            ("iv", iv),
            ("tag", tag),
            ("salt", salt),
        ]:
            try:
                bytes.fromhex(value)
            except ValueError:
                raise ValueError(f"{name} must be a valid hex string")

        self.org_public_key = org_public_key.lower()
        self.admin_encrypted_private_key = admin_encrypted_private_key.lower()
        self.admin_encrypted_private_key_iv = iv.lower()
        self.admin_encrypted_private_key_tag = tag.lower()
        self.admin_key_salt = salt.lower()

        self.has_encryption_keys = True
        self.encryption_created_at = datetime.utcnow()
        self.encryption_created_by = created_by

    def clear_encryption_keys(self) -> None:
        """
        Clear all organization encryption keys.

        Use with EXTREME caution - this makes all encrypted org messages unrecoverable!
        All member key distributions will also need to be cleared.
        """
        self.org_public_key = None
        self.admin_encrypted_private_key = None
        self.admin_encrypted_private_key_iv = None
        self.admin_encrypted_private_key_tag = None
        self.admin_key_salt = None

        self.has_encryption_keys = False
        self.encryption_created_at = None
        self.encryption_created_by = None

    def get_admin_encrypted_keys(self) -> Optional[dict]:
        """
        Get admin-encrypted org key data for org passcode unlock.

        Returns None if org has no encryption keys.
        Only call this for authenticated org admins.
        """
        if not self.has_encryption_keys:
            return None

        return {
            "org_public_key": self.org_public_key,
            "admin_encrypted_private_key": self.admin_encrypted_private_key,
            "iv": self.admin_encrypted_private_key_iv,
            "tag": self.admin_encrypted_private_key_tag,
            "salt": self.admin_key_salt,
        }
