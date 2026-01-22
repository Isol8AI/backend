"""Pydantic schemas for API request/response validation."""

from .encryption import (
    EncryptedPayload,
    CreateUserKeysRequest,
    UserKeysResponse,
    EncryptionStatusResponse,
    EncryptedMessageResponse,
    SendEncryptedMessageRequest,
    EncryptedChatResponse,
)
from .organization_encryption import (
    CreateOrgKeysRequest,
    OrgEncryptionStatusResponse,
    OrgAdminEncryptedKeysResponse,
    DistributeOrgKeyRequest,
    BatchDistributeOrgKeyRequest,
    PendingDistributionResponse,
    PendingDistributionsResponse,
    MembershipResponse,
    MembershipWithKeyResponse,
    OrgMemberResponse,
    OrgMembersListResponse,
    RevokeOrgKeyRequest,
    ChangeMemberRoleRequest,
    AuditLogResponse,
    AuditLogsListResponse,
)

__all__ = [
    # User encryption
    "EncryptedPayload",
    "CreateUserKeysRequest",
    "UserKeysResponse",
    "EncryptionStatusResponse",
    "EncryptedMessageResponse",
    "SendEncryptedMessageRequest",
    "EncryptedChatResponse",
    # Organization encryption
    "CreateOrgKeysRequest",
    "OrgEncryptionStatusResponse",
    "OrgAdminEncryptedKeysResponse",
    "DistributeOrgKeyRequest",
    "BatchDistributeOrgKeyRequest",
    "PendingDistributionResponse",
    "PendingDistributionsResponse",
    "MembershipResponse",
    "MembershipWithKeyResponse",
    "OrgMemberResponse",
    "OrgMembersListResponse",
    "RevokeOrgKeyRequest",
    "ChangeMemberRoleRequest",
    "AuditLogResponse",
    "AuditLogsListResponse",
]
