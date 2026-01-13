"""
Organization membership model with encrypted org key distribution.

Security Note:
- Each member receives the org private key encrypted TO their personal public key
- Only the member can decrypt their copy (using their personal private key)
- Server cannot decrypt any member's copy of the org key
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, String, Boolean, Index, Enum, UniqueConstraint
from sqlalchemy.orm import relationship

from .base import Base


class MemberRole(str, PyEnum):
    """Member role in organization.

    Values match Clerk JWT claim format for consistency.
    """
    ADMIN = "org:admin"
    MEMBER = "org:member"


class OrganizationMembership(Base):
    """
    Membership record linking user to organization.

    For encrypted organizations, each member has their own copy of the org
    private key, encrypted to their personal public key. This allows:
    - Each member to decrypt org messages using their own passcode
    - Revocation by simply deleting the membership (and encrypted key)
    - No need to share the org passcode with every member

    The encrypted_org_key fields form an EncryptedPayload:
    - ephemeral_public_key: From admin's ephemeral key during distribution
    - iv, ciphertext, auth_tag, hkdf_salt: Standard AES-GCM encryption
    """

    __tablename__ = "organization_memberships"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)

    # Role in organization (admin or member)
    role = Column(Enum(MemberRole), default=MemberRole.MEMBER, nullable=False)

    # =========================================================================
    # Encrypted Org Key (distributed to this member)
    #
    # This is the org private key encrypted TO this user's public key.
    # Only this user can decrypt it using their personal private key.
    # =========================================================================

    # Ephemeral public key for ECDH (64 hex chars = 32 bytes)
    encrypted_org_key_ephemeral = Column(String(64), nullable=True)
    # AES-GCM IV (32 hex chars = 16 bytes)
    encrypted_org_key_iv = Column(String(32), nullable=True)
    # Encrypted org private key (variable length)
    encrypted_org_key_ciphertext = Column(String, nullable=True)
    # AES-GCM auth tag (32 hex chars = 16 bytes)
    encrypted_org_key_tag = Column(String(32), nullable=True)
    # HKDF salt (64 hex chars = 32 bytes)
    encrypted_org_key_hkdf_salt = Column(String(64), nullable=True)

    # Key distribution status
    has_org_key = Column(Boolean, default=False, nullable=False)
    key_distributed_at = Column(DateTime, nullable=True)
    key_distributed_by = Column(String, nullable=True)  # Admin user ID who distributed

    # Timestamps (Clerk webhook provides joined_at)
    joined_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # =========================================================================
    # Relationships
    # =========================================================================

    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="memberships")

    # =========================================================================
    # Constraints and Indexes
    # =========================================================================

    __table_args__ = (
        UniqueConstraint("user_id", "org_id", name="uq_membership_user_org"),
        Index("ix_memberships_org_id", "org_id"),
        Index("ix_memberships_user_id", "user_id"),
        Index("ix_memberships_has_org_key", "has_org_key"),
        Index("ix_memberships_role", "role"),
    )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_admin(self) -> bool:
        """Check if member is an admin."""
        return self.role == MemberRole.ADMIN

    @property
    def is_member(self) -> bool:
        """Check if member has basic member role."""
        return self.role == MemberRole.MEMBER

    @property
    def needs_key_distribution(self) -> bool:
        """
        Check if member needs org key distribution.

        True if:
        - Member doesn't have org key yet
        - Member has personal encryption keys (can receive encrypted data)
        """
        return not self.has_org_key and self.user and self.user.has_encryption_keys

    @property
    def encrypted_org_key_payload(self) -> Optional[dict]:
        """
        Get encrypted org key as payload dict.

        Returns None if key hasn't been distributed.
        """
        if not self.has_org_key:
            return None

        return {
            "ephemeral_public_key": self.encrypted_org_key_ephemeral,
            "iv": self.encrypted_org_key_iv,
            "ciphertext": self.encrypted_org_key_ciphertext,
            "auth_tag": self.encrypted_org_key_tag,
            "hkdf_salt": self.encrypted_org_key_hkdf_salt,
        }

    # =========================================================================
    # Methods
    # =========================================================================

    def set_encrypted_org_key(
        self,
        ephemeral_public_key: str,
        iv: str,
        ciphertext: str,
        auth_tag: str,
        hkdf_salt: str,
        distributed_by_user_id: str,
    ) -> None:
        """
        Store the encrypted org key for this member.

        Args:
            ephemeral_public_key: 64 hex chars (32 bytes)
            iv: 32 hex chars (16 bytes)
            ciphertext: Variable length hex
            auth_tag: 32 hex chars (16 bytes)
            hkdf_salt: 64 hex chars (32 bytes)
            distributed_by_user_id: Admin who distributed the key

        Raises:
            ValueError: If any field has invalid length or format
        """
        # Validate lengths
        if len(ephemeral_public_key) != 64:
            raise ValueError("ephemeral_public_key must be 64 hex characters (32 bytes)")
        if len(iv) != 32:
            raise ValueError("iv must be 32 hex characters (16 bytes)")
        if len(auth_tag) != 32:
            raise ValueError("auth_tag must be 32 hex characters (16 bytes)")
        if len(hkdf_salt) != 64:
            raise ValueError("hkdf_salt must be 64 hex characters (32 bytes)")
        if not ciphertext:
            raise ValueError("ciphertext cannot be empty")

        # Validate hex strings
        for name, value in [
            ("ephemeral_public_key", ephemeral_public_key),
            ("iv", iv),
            ("auth_tag", auth_tag),
            ("hkdf_salt", hkdf_salt),
        ]:
            try:
                bytes.fromhex(value)
            except ValueError:
                raise ValueError(f"{name} must be a valid hex string")

        self.encrypted_org_key_ephemeral = ephemeral_public_key.lower()
        self.encrypted_org_key_iv = iv.lower()
        self.encrypted_org_key_ciphertext = ciphertext.lower()
        self.encrypted_org_key_tag = auth_tag.lower()
        self.encrypted_org_key_hkdf_salt = hkdf_salt.lower()

        self.has_org_key = True
        self.key_distributed_at = datetime.utcnow()
        self.key_distributed_by = distributed_by_user_id

    def clear_encrypted_org_key(self) -> None:
        """
        Clear the encrypted org key from this membership.

        Used when revoking a member's access or rotating keys.
        """
        self.encrypted_org_key_ephemeral = None
        self.encrypted_org_key_iv = None
        self.encrypted_org_key_ciphertext = None
        self.encrypted_org_key_tag = None
        self.encrypted_org_key_hkdf_salt = None

        self.has_org_key = False
        self.key_distributed_at = None
        self.key_distributed_by = None

    def to_api_response(self, include_encrypted_key: bool = False) -> dict:
        """
        Convert to API response.

        Args:
            include_encrypted_key: If True, include the encrypted org key
                                   (only for the member themselves)
        """
        response = {
            "id": self.id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "role": self.role.value,
            "has_org_key": self.has_org_key,
            "key_distributed_at": (
                self.key_distributed_at.isoformat() if self.key_distributed_at else None
            ),
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

        if include_encrypted_key and self.has_org_key:
            response["encrypted_org_key"] = self.encrypted_org_key_payload

        return response
