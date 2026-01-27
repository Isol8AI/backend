"""
Core services for Isol8.

Services encapsulate business logic for encryption key management,
organization operations, and other domain-specific operations.
"""

from .user_key_service import (
    UserKeyService,
    UserKeyServiceError,
    KeysAlreadyExistError,
    KeysNotFoundError,
)
from .org_key_service import (
    OrgKeyService,
    OrgKeyServiceError,
    OrgKeysAlreadyExistError,
    OrgKeysNotFoundError,
    MembershipNotFoundError,
    NotAdminError,
)

# Note: MemoryService removed during migration to mem0.
# Will be re-implemented in Plan 2 with enclave-based mem0.

__all__ = [
    # User Key Service
    "UserKeyService",
    "UserKeyServiceError",
    "KeysAlreadyExistError",
    "KeysNotFoundError",
    # Org Key Service
    "OrgKeyService",
    "OrgKeyServiceError",
    "OrgKeysAlreadyExistError",
    "OrgKeysNotFoundError",
    "MembershipNotFoundError",
    "NotAdminError",
]
