"""
Pydantic schemas for organization encryption endpoints.

Security Note:
- CreateOrgKeysRequest contains encrypted data only
- Server never sees plaintext org private keys
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator

from .encryption import EncryptedPayload, validate_hex_string


class CreateOrgKeysRequest(BaseModel):
    """
    Request to create organization encryption keys.

    Admin creates org keypair client-side, encrypts private key with org passcode,
    and also encrypts it to their own public key for immediate use.
    """

    org_public_key: str = Field(..., min_length=64, max_length=64, description="Org X25519 public key (32 bytes hex)")

    # Admin-encrypted private key (org passcode protected)
    admin_encrypted_private_key: str = Field(
        ..., min_length=1, description="Org private key encrypted with org passcode (hex)"
    )
    admin_iv: str = Field(..., min_length=32, max_length=32, description="AES-GCM IV (16 bytes hex)")
    admin_tag: str = Field(..., min_length=32, max_length=32, description="AES-GCM auth tag (16 bytes hex)")
    admin_salt: str = Field(..., min_length=64, max_length=64, description="Argon2id salt (32 bytes hex)")

    # Admin's copy of org key encrypted to their personal public key
    # (so admin can use org encryption immediately)
    admin_member_encrypted_key: EncryptedPayload = Field(
        ..., description="Org private key encrypted to admin's personal public key"
    )

    @field_validator("org_public_key")
    @classmethod
    def validate_org_public_key(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    @field_validator("admin_iv")
    @classmethod
    def validate_admin_iv(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("admin_tag")
    @classmethod
    def validate_admin_tag(cls, v: str) -> str:
        return validate_hex_string(v, 32)

    @field_validator("admin_salt")
    @classmethod
    def validate_admin_salt(cls, v: str) -> str:
        return validate_hex_string(v, 64)

    @field_validator("admin_encrypted_private_key")
    @classmethod
    def validate_admin_encrypted_private_key(cls, v: str) -> str:
        return validate_hex_string(v)


class OrgEncryptionStatusResponse(BaseModel):
    """Organization encryption status."""

    has_encryption_keys: bool
    org_public_key: Optional[str] = None
    encryption_created_at: Optional[datetime] = None
    encryption_created_by: Optional[str] = None

    model_config = {"from_attributes": True}


class OrgAdminEncryptedKeysResponse(BaseModel):
    """Admin-encrypted org keys for recovery (admin-only endpoint)."""

    org_public_key: str
    admin_encrypted_private_key: str
    iv: str
    tag: str
    salt: str

    model_config = {"from_attributes": True}


class DistributeOrgKeyRequest(BaseModel):
    """Request to distribute org key to a member."""

    membership_id: str = Field(..., description="Membership ID of the member to receive the key")
    encrypted_org_key: EncryptedPayload = Field(..., description="Org private key encrypted to member's public key")


class BatchDistributeOrgKeyRequest(BaseModel):
    """Request to distribute org key to multiple members at once."""

    distributions: List[DistributeOrgKeyRequest] = Field(..., min_length=1, description="List of key distributions")


class PendingDistributionResponse(BaseModel):
    """Member who is ready for org key distribution (has personal keys)."""

    membership_id: str
    user_id: str
    user_public_key: str
    role: str
    joined_at: datetime

    model_config = {"from_attributes": True}


class NeedsPersonalSetupResponse(BaseModel):
    """Member who needs to set up personal encryption first."""

    membership_id: str
    user_id: str
    role: str
    joined_at: datetime

    model_config = {"from_attributes": True}


class PendingDistributionsResponse(BaseModel):
    """List of members needing key distribution, split by readiness."""

    org_id: str
    ready_for_distribution: List[PendingDistributionResponse]
    needs_personal_setup: List[NeedsPersonalSetupResponse]
    ready_count: int
    needs_setup_count: int


class MembershipResponse(BaseModel):
    """Member's view of their org membership."""

    id: str
    org_id: str
    org_name: Optional[str] = None
    role: str
    has_org_key: bool
    key_distributed_at: Optional[datetime] = None
    joined_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MembershipWithKeyResponse(MembershipResponse):
    """Member's membership including their encrypted org key."""

    encrypted_org_key: Optional[EncryptedPayload] = None

    model_config = {"from_attributes": True}


class OrgMemberResponse(BaseModel):
    """View of an org member (for admin listing)."""

    membership_id: str
    user_id: str
    role: str
    has_org_key: bool
    has_personal_keys: bool
    key_distributed_at: Optional[datetime] = None
    joined_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class OrgMembersListResponse(BaseModel):
    """List of org members (admin view)."""

    org_id: str
    members: List[OrgMemberResponse]
    total_count: int


class RevokeOrgKeyRequest(BaseModel):
    """Request to revoke a member's org key."""

    membership_id: str = Field(..., description="Membership ID to revoke key from")
    reason: Optional[str] = Field(None, max_length=500, description="Optional reason for revocation (logged for audit)")


class ChangeMemberRoleRequest(BaseModel):
    """Request to change a member's role."""

    membership_id: str = Field(..., description="Membership ID to change role")
    new_role: str = Field(..., pattern="^(admin|member)$", description="New role (admin or member)")


class AuditLogResponse(BaseModel):
    """Audit log entry."""

    id: str
    event_type: str
    actor_user_id: Optional[str] = None
    target_user_id: Optional[str] = None
    org_id: Optional[str] = None
    event_data: Optional[dict] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogsListResponse(BaseModel):
    """Paginated list of audit logs."""

    logs: List[AuditLogResponse]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class BulkDistributionResultResponse(BaseModel):
    """Result for a single distribution in bulk operation."""

    membership_id: str
    user_id: str
    success: bool
    error: Optional[str] = None

    model_config = {"from_attributes": True}


class BulkDistributionResponse(BaseModel):
    """Response for bulk key distribution."""

    org_id: str
    results: List[BulkDistributionResultResponse]
    success_count: int
    failure_count: int
