"""Organizations router for managing Clerk organizations."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.auth import AuthContext, get_current_user, require_org_admin, require_org_context
from core.database import get_session_factory
from models.context_store import ContextStore
from models.organization import Organization
from models.organization_membership import OrganizationMembership

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


@router.post("/sync", response_model=SyncOrgResponse)
async def sync_organization(
    request: SyncOrgRequest,
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> SyncOrgResponse:
    """Sync organization from Clerk to database.

    Creates the organization if it doesn't exist, updates it if it does.
    Also creates or updates the user's membership.

    Note: Uses org_id from request body rather than JWT claims because
    the frontend may call this before the JWT is refreshed with org claims.
    """
    org_id = request.org_id

    async with session_factory() as session:
        # Check if organization exists
        result = await session.execute(
            select(Organization).where(Organization.id == org_id)
        )
        org = result.scalar_one_or_none()

        if org is None:
            # Create new organization
            org = Organization(
                id=org_id,
                name=request.name,
                slug=request.slug
            )
            session.add(org)
            status = "created"
        else:
            # Update existing organization
            org.name = request.name
            if request.slug:
                org.slug = request.slug
            status = "updated"

        # Handle membership - use role from JWT if available, otherwise default
        role = auth.org_role or "org:member"

        result = await session.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == auth.user_id,
                OrganizationMembership.org_id == org_id
            )
        )
        membership = result.scalar_one_or_none()

        if membership is None:
            # Create membership
            membership = OrganizationMembership(
                id=f"mem_{auth.user_id}_{org_id}",
                user_id=auth.user_id,
                org_id=org_id,
                role=role
            )
            session.add(membership)
        else:
            # Update role if changed and we have role info
            if auth.org_role and membership.role != auth.org_role:
                membership.role = auth.org_role

        await session.commit()

        return SyncOrgResponse(status=status, org_id=org_id)


@router.get("/current", response_model=CurrentOrgResponse)
async def get_current_org(
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> CurrentOrgResponse:
    """Get current organization context.

    Returns None for org fields when in personal mode.
    """
    if auth.is_personal_context:
        return CurrentOrgResponse(
            org_id=None,
            is_personal_context=True,
            is_org_admin=False
        )

    # Fetch organization details
    async with session_factory() as session:
        result = await session.execute(
            select(Organization).where(Organization.id == auth.org_id)
        )
        org = result.scalar_one_or_none()

        return CurrentOrgResponse(
            org_id=auth.org_id,
            org_name=org.name if org else None,
            org_slug=auth.org_slug,
            org_role=auth.org_role,
            is_personal_context=False,
            is_org_admin=auth.is_org_admin
        )


@router.get("/", response_model=ListOrgsResponse)
async def list_organizations(
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> ListOrgsResponse:
    """List all organizations the user is a member of."""
    async with session_factory() as session:
        result = await session.execute(
            select(OrganizationMembership, Organization)
            .join(Organization, OrganizationMembership.org_id == Organization.id)
            .where(OrganizationMembership.user_id == auth.user_id)
        )
        rows = result.all()

        organizations = [
            OrgListItem(
                id=org.id,
                name=org.name,
                slug=org.slug,
                role=membership.role
            )
            for membership, org in rows
        ]

        return ListOrgsResponse(organizations=organizations)


@router.get("/context", response_model=OrgContextResponse)
async def get_org_context(
    auth: AuthContext = Depends(require_org_context),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> OrgContextResponse:
    """Get shared organization context.

    Returns the context data stored for the organization,
    which can be used for RAG or shared settings.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(ContextStore).where(
                ContextStore.owner_type == "org",
                ContextStore.owner_id == auth.org_id
            )
        )
        context_store = result.scalar_one_or_none()

        return OrgContextResponse(
            context_data=context_store.context_data if context_store else None
        )


@router.put("/context", response_model=OrgContextResponse)
async def update_org_context(
    request: OrgContextRequest,
    auth: AuthContext = Depends(require_org_admin),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> OrgContextResponse:
    """Update shared organization context.

    Only organization admins can update the shared context.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(ContextStore).where(
                ContextStore.owner_type == "org",
                ContextStore.owner_id == auth.org_id
            )
        )
        context_store = result.scalar_one_or_none()

        if context_store is None:
            # Create new context store
            context_store = ContextStore(
                id=f"ctx_org_{auth.org_id}",
                owner_type="org",
                owner_id=auth.org_id,
                context_data=request.context_data
            )
            session.add(context_store)
        else:
            # Update existing context
            context_store.context_data = request.context_data

        await session.commit()
        await session.refresh(context_store)

        return OrgContextResponse(context_data=context_store.context_data)
