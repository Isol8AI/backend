"""Context router for unified context access.

Provides a unified interface for getting/setting context based on
the current auth context (personal vs organization).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.auth import AuthContext, get_current_user
from core.database import get_session_factory
from models.context_store import ContextStore

router = APIRouter(prefix="/context", tags=["context"])


class ContextRequest(BaseModel):
    """Request body for updating context."""

    context_data: dict


class ContextResponse(BaseModel):
    """Response for context operations."""

    owner_type: str  # "user" or "org"
    context_data: dict | None


def _resolve_owner(auth: AuthContext) -> tuple[str, str]:
    """Resolve owner type and ID from auth context."""
    if auth.is_org_context:
        return "org", auth.org_id
    return "user", auth.user_id


@router.get("/", response_model=ContextResponse)
async def get_context(
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> ContextResponse:
    """Get context for the current auth context.

    Returns personal context in personal mode, org context in org mode.
    """
    owner_type, owner_id = _resolve_owner(auth)

    async with session_factory() as session:
        result = await session.execute(
            select(ContextStore).where(
                ContextStore.owner_type == owner_type,
                ContextStore.owner_id == owner_id,
            )
        )
        context_store = result.scalar_one_or_none()

        return ContextResponse(
            owner_type=owner_type,
            context_data=context_store.context_data if context_store else None,
        )


@router.put("/", response_model=ContextResponse)
async def update_context(
    request: ContextRequest,
    auth: AuthContext = Depends(get_current_user),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> ContextResponse:
    """Update context for the current auth context.

    In personal mode: Updates personal context.
    In org mode: Requires admin role to update org context.
    """
    # Org context requires admin privileges
    if auth.is_org_context and not auth.is_org_admin:
        raise HTTPException(
            status_code=403,
            detail="Only organization admins can update org context",
        )

    owner_type, owner_id = _resolve_owner(auth)

    async with session_factory() as session:
        result = await session.execute(
            select(ContextStore).where(
                ContextStore.owner_type == owner_type,
                ContextStore.owner_id == owner_id,
            )
        )
        context_store = result.scalar_one_or_none()

        if context_store is None:
            context_store = ContextStore(
                id=f"ctx_{owner_type}_{owner_id}",
                owner_type=owner_type,
                owner_id=owner_id,
                context_data=request.context_data,
            )
            session.add(context_store)
        else:
            context_store.context_data = request.context_data

        await session.commit()
        await session.refresh(context_store)

        return ContextResponse(
            owner_type=owner_type,
            context_data=context_store.context_data,
        )
