"""Organizations router for managing Clerk organizations and encryption."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from core.auth import AuthContext, get_current_user, require_org_admin, require_org_context
from core.database import get_db, get_session_factory
from core.services.org_key_service import (
    OrgKeyService,
    OrgKeyServiceError,
    OrgKeysAlreadyExistError,
    OrgKeysNotFoundError,
    MembershipNotFoundError,
    MemberNotReadyError,
    NotAdminError,
)
from models.context_store import ContextStore
from models.organization import Organization
from models.organization_membership import MemberRole, OrganizationMembership
from schemas.encryption import EncryptedPayloadSchema
from schemas.organization_encryption import (
    CreateOrgKeysRequest,
    OrgEncryptionStatusResponse,
    DistributeOrgKeyRequest,
    BatchDistributeOrgKeyRequest,
    PendingDistributionResponse,
    NeedsPersonalSetupResponse,
    PendingDistributionsResponse,
    MembershipWithKeyResponse,
    BulkDistributionResponse,
    BulkDistributionResultResponse,
    CreateOrgKeysResponse,
    DistributeOrgKeyResponse,
    AdminRecoveryKeyResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/organizations", tags=["organizations"])


class SyncOrgRequest(BaseModel):
    """Request body for syncing organization from Clerk."""

    org_id: str  # Clerk organization ID from frontend
    name: str
    slug: str | None = None


class SyncOrgResponse(BaseModel):
    """Response for organization sync."""

    status: str  # "created", "updated"
    org_id: str


class CurrentOrgResponse(BaseModel):
    """Response for current organization context."""

    org_id: str | None
    org_name: str | None = None
    org_slug: str | None = None
    org_role: str | None = None
    is_personal_context: bool
    is_org_admin: bool = False


class OrgListItem(BaseModel):
    """Organization item in list response."""

    id: str
    name: str
    slug: str | None
    role: str


class ListOrgsResponse(BaseModel):
    """Response for listing user's organizations."""

    organizations: list[OrgListItem]


class OrgContextRequest(BaseModel):
    """Request body for updating organization context."""

    context_data: dict


class OrgContextResponse(BaseModel):
    """Response for organization context."""

    context_data: dict | None


@router.post(
    "/sync",
    response_model=SyncOrgResponse,
    summary="Sync organization from Clerk",
    description="Creates or updates the organization record and membership. Validates org membership via JWT claim or database record.",
    operation_id="sync_organization",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Not a member of this organization"},
    },
)
async def sync_organization(
    request: SyncOrgRequest,
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> SyncOrgResponse:
    org_id = request.org_id

    async with session_factory() as session:
        # Security validation: verify user is a member of this org
        if auth.org_id:
            # JWT has org claim - must match exactly
            if auth.org_id != org_id:
                raise HTTPException(status_code=403, detail="Cannot sync org you're not a member of")
        else:
            # No JWT org claim - check database membership (from webhook)
            # This handles first-time access when JWT hasn't refreshed yet
            result = await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == auth.user_id, OrganizationMembership.org_id == org_id
                )
            )
            existing_membership = result.scalar_one_or_none()
            if not existing_membership:
                raise HTTPException(status_code=403, detail="Not a member of this organization")

        # Check if organization exists
        result = await session.execute(select(Organization).where(Organization.id == org_id))
        org = result.scalar_one_or_none()

        if org is None:
            # Create new organization
            org = Organization(id=org_id, name=request.name, slug=request.slug)
            session.add(org)
            sync_status = "created"
        else:
            # Update existing organization
            org.name = request.name
            if request.slug:
                org.slug = request.slug
            sync_status = "updated"

        # Handle membership - use Clerk role directly (enum values match Clerk format)
        # Default to MEMBER if no role info from JWT (e.g., syncing from personal context)
        member_role = MemberRole(auth.org_role) if auth.org_role else MemberRole.MEMBER

        result = await session.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == auth.user_id, OrganizationMembership.org_id == org_id
            )
        )
        membership = result.scalar_one_or_none()

        if membership is None:
            # Create membership
            membership = OrganizationMembership(
                id=f"mem_{auth.user_id}_{org_id}", user_id=auth.user_id, org_id=org_id, role=member_role
            )
            session.add(membership)
        else:
            # Update role if changed and we have role info from JWT
            if auth.org_role and membership.role != member_role:
                membership.role = member_role

        await session.commit()

        return SyncOrgResponse(status=sync_status, org_id=org_id)


@router.get(
    "/current",
    response_model=CurrentOrgResponse,
    summary="Get current organization context",
    description="Get current organization context. Returns None for org fields when in personal mode.",
    operation_id="get_current_org",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def get_current_org(
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> CurrentOrgResponse:
    if auth.is_personal_context:
        return CurrentOrgResponse(org_id=None, is_personal_context=True, is_org_admin=False)

    # Fetch organization details
    async with session_factory() as session:
        result = await session.execute(select(Organization).where(Organization.id == auth.org_id))
        org = result.scalar_one_or_none()

        return CurrentOrgResponse(
            org_id=auth.org_id,
            org_name=org.name if org else None,
            org_slug=auth.org_slug,
            org_role=auth.org_role,
            is_personal_context=False,
            is_org_admin=auth.is_org_admin,
        )


@router.get(
    "/",
    response_model=ListOrgsResponse,
    summary="List user organizations",
    description="List all organizations the user is a member of.",
    operation_id="list_organizations",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def list_organizations(
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> ListOrgsResponse:
    async with session_factory() as session:
        result = await session.execute(
            select(OrganizationMembership, Organization)
            .join(Organization, OrganizationMembership.org_id == Organization.id)
            .where(OrganizationMembership.user_id == auth.user_id)
        )
        rows = result.all()

        organizations = [
            OrgListItem(id=org.id, name=org.name, slug=org.slug, role=membership.role) for membership, org in rows
        ]

        return ListOrgsResponse(organizations=organizations)


@router.get(
    "/context",
    response_model=OrgContextResponse,
    summary="Get organization context",
    description="Get shared organization context data used for RAG or shared settings.",
    operation_id="get_org_context",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Organization context required"},
    },
)
async def get_org_context(
    auth: AuthContext = Depends(require_org_context),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> OrgContextResponse:
    async with session_factory() as session:
        result = await session.execute(
            select(ContextStore).where(ContextStore.owner_type == "org", ContextStore.owner_id == auth.org_id)
        )
        context_store = result.scalar_one_or_none()

        return OrgContextResponse(context_data=context_store.context_data if context_store else None)


@router.put(
    "/context",
    response_model=OrgContextResponse,
    summary="Update organization context",
    description="Update shared organization context. Only organization admins can update the shared context.",
    operation_id="update_org_context",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
    },
)
async def update_org_context(
    request: OrgContextRequest,
    auth: AuthContext = Depends(require_org_admin),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> OrgContextResponse:
    async with session_factory() as session:
        result = await session.execute(
            select(ContextStore).where(ContextStore.owner_type == "org", ContextStore.owner_id == auth.org_id)
        )
        context_store = result.scalar_one_or_none()

        if context_store is None:
            # Create new context store
            context_store = ContextStore(
                id=f"ctx_org_{auth.org_id}", owner_type="org", owner_id=auth.org_id, context_data=request.context_data
            )
            session.add(context_store)
        else:
            # Update existing context
            context_store.context_data = request.context_data

        await session.commit()
        await session.refresh(context_store)

        return OrgContextResponse(context_data=context_store.context_data)


# =============================================================================
# Organization Encryption Endpoints
# =============================================================================


def _handle_org_key_service_error(e: OrgKeyServiceError):
    """Convert service errors to HTTP exceptions."""
    if isinstance(e, OrgKeysAlreadyExistError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    elif isinstance(e, (OrgKeysNotFoundError, MembershipNotFoundError)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    elif isinstance(e, NotAdminError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    elif isinstance(e, MemberNotReadyError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{org_id}/encryption-status",
    response_model=OrgEncryptionStatusResponse,
    summary="Get org encryption status",
    description="Get organization's encryption status. Any authenticated user can check status, but only members see full details.",
    operation_id="get_org_encryption_status",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Organization not found"},
    },
)
async def get_encryption_status(
    org_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        status_data = await service.get_org_encryption_status(org_id)
        return OrgEncryptionStatusResponse(**status_data)
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.post(
    "/{org_id}/keys",
    response_model=CreateOrgKeysResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create org encryption keys",
    description="Create organization encryption keys (admin only). The admin creates the org keypair client-side and sends the encrypted blobs.",
    operation_id="create_org_keys",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
        409: {"description": "Organization already has encryption keys"},
    },
)
async def create_org_keys(
    org_id: str,
    request: CreateOrgKeysRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        await service.create_org_keys(
            org_id=org_id,
            admin_user_id=auth.user_id,
            org_public_key=request.org_public_key,
            admin_encrypted_private_key=request.admin_encrypted_private_key,
            admin_iv=request.admin_iv,
            admin_tag=request.admin_tag,
            admin_salt=request.admin_salt,
            admin_member_key_ephemeral=request.admin_member_encrypted_key.ephemeral_public_key,
            admin_member_key_iv=request.admin_member_encrypted_key.iv,
            admin_member_key_ciphertext=request.admin_member_encrypted_key.ciphertext,
            admin_member_key_tag=request.admin_member_encrypted_key.auth_tag,
            admin_member_key_hkdf_salt=request.admin_member_encrypted_key.hkdf_salt,
        )
        return {"status": "created", "org_public_key": request.org_public_key}
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.get(
    "/{org_id}/pending-distributions",
    response_model=PendingDistributionsResponse,
    summary="Get pending key distributions",
    description="Get members needing org key distribution (admin only). Returns members ready for distribution and those needing personal key setup.",
    operation_id="get_pending_distributions",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
        404: {"description": "Organization not found"},
    },
)
async def get_pending_distributions(
    org_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        result = await service.get_pending_distributions(org_id, auth.user_id)
        return PendingDistributionsResponse(
            org_id=org_id,
            ready_for_distribution=[PendingDistributionResponse(**p) for p in result["ready_for_distribution"]],
            needs_personal_setup=[NeedsPersonalSetupResponse(**p) for p in result["needs_personal_setup"]],
            ready_count=result["ready_count"],
            needs_setup_count=result["needs_setup_count"],
        )
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.post(
    "/{org_id}/distribute-key",
    response_model=DistributeOrgKeyResponse,
    summary="Distribute org key to member",
    description="Distribute org key to a member (admin only). The admin re-encrypts the org key to the member's public key.",
    operation_id="distribute_org_key",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        400: {"description": "Member not ready for key distribution"},
        403: {"description": "Admin access required"},
        404: {"description": "Membership not found"},
    },
)
async def distribute_org_key(
    org_id: str,
    request: DistributeOrgKeyRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        membership = await service.distribute_org_key(
            org_id=org_id,
            admin_user_id=auth.user_id,
            membership_id=request.membership_id,
            ephemeral_public_key=request.encrypted_org_key.ephemeral_public_key,
            iv=request.encrypted_org_key.iv,
            ciphertext=request.encrypted_org_key.ciphertext,
            auth_tag=request.encrypted_org_key.auth_tag,
            hkdf_salt=request.encrypted_org_key.hkdf_salt,
        )
        return {"status": "distributed", "membership_id": membership.id}
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.post(
    "/{org_id}/distribute-keys-bulk",
    response_model=BulkDistributionResponse,
    summary="Bulk distribute org keys",
    description="Distribute org key to multiple members at once (admin only). Supports partial failure with per-member status.",
    operation_id="distribute_org_keys_bulk",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
    },
)
async def distribute_org_keys_bulk(
    org_id: str,
    request: BatchDistributeOrgKeyRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        # Convert request to service format
        distributions = [
            {
                "membership_id": d.membership_id,
                "ephemeral_public_key": d.encrypted_org_key.ephemeral_public_key,
                "iv": d.encrypted_org_key.iv,
                "ciphertext": d.encrypted_org_key.ciphertext,
                "auth_tag": d.encrypted_org_key.auth_tag,
                "hkdf_salt": d.encrypted_org_key.hkdf_salt,
            }
            for d in request.distributions
        ]

        results = await service.bulk_distribute_org_keys(
            org_id=org_id,
            admin_user_id=auth.user_id,
            distributions=distributions,
        )

        return BulkDistributionResponse(
            org_id=org_id,
            results=[
                BulkDistributionResultResponse(
                    membership_id=r.membership_id,
                    user_id=r.user_id,
                    success=r.success,
                    error=r.error,
                )
                for r in results
            ],
            success_count=sum(1 for r in results if r.success),
            failure_count=sum(1 for r in results if not r.success),
        )
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.get(
    "/{org_id}/membership",
    response_model=MembershipWithKeyResponse,
    summary="Get my membership",
    description="Get current user's membership with encrypted org key for client-side decryption.",
    operation_id="get_my_membership",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Not a member of this organization"},
    },
)
async def get_my_membership(
    org_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        membership = await service.get_membership(auth.user_id, org_id)
        if not membership:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not a member of this organization")

        encrypted_key = None
        if membership.has_org_key:
            payload = membership.encrypted_org_key_payload
            encrypted_key = EncryptedPayloadSchema(**payload)

        return MembershipWithKeyResponse(
            id=membership.id,
            org_id=org_id,
            role=membership.role.value,
            has_org_key=membership.has_org_key,
            encrypted_org_key=encrypted_key,
            key_distributed_at=membership.key_distributed_at,
            joined_at=membership.joined_at,
            created_at=membership.created_at,
        )
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.post(
    "/{org_id}/revoke-key/{member_user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke member org key",
    description="Revoke a member's org key (admin only). The member will no longer be able to decrypt org messages.",
    operation_id="revoke_member_key",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
        404: {"description": "Membership not found"},
    },
)
async def revoke_member_key(
    org_id: str,
    member_user_id: str,
    reason: str = None,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        await service.revoke_member_org_key(
            org_id=org_id,
            admin_user_id=auth.user_id,
            member_user_id=member_user_id,
            reason=reason,
        )
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.get(
    "/{org_id}/admin-recovery-key",
    response_model=AdminRecoveryKeyResponse,
    summary="Get admin recovery key",
    description="Get admin-encrypted org key for recovery (admin only). Used when admin needs to recover org key using org passcode.",
    operation_id="get_admin_recovery_key",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
        404: {"description": "Organization keys not found"},
    },
)
async def get_admin_recovery_key(
    org_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)
    try:
        return await service.get_admin_recovery_key(auth.user_id, org_id)
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)


@router.get(
    "/{org_id}/members",
    summary="List org members",
    description="List all organization members with key distribution status (admin only).",
    operation_id="list_org_members",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        403: {"description": "Admin access required"},
    },
)
async def list_org_members(
    org_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = OrgKeyService(db)

    # Verify user is admin
    try:
        await service.verify_admin(auth.user_id, org_id)
    except NotAdminError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    except OrgKeyServiceError as e:
        _handle_org_key_service_error(e)

    # Load memberships with user data
    result = await db.execute(
        select(OrganizationMembership)
        .where(OrganizationMembership.org_id == org_id)
        .options(selectinload(OrganizationMembership.user))
    )
    memberships = result.scalars().all()

    return {
        "org_id": org_id,
        "members": [
            {
                "membership_id": m.id,
                "user_id": m.user_id,
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "has_personal_keys": m.user.has_encryption_keys if m.user else False,
                "has_org_key": m.has_org_key,
                "key_distributed_at": m.key_distributed_at.isoformat() if m.key_distributed_at else None,
                "joined_at": m.joined_at.isoformat() if m.joined_at else None,
            }
            for m in memberships
        ],
        "total_count": len(memberships),
    }
