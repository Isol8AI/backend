"""
Memory API endpoints - stub during migration to mem0.

Note: Memory operations temporarily disabled during migration.
Plan 2 will implement real endpoints backed by mem0 in the enclave.

The settings page uses these endpoints to:
- List user memories
- Delete individual memories
- Delete all memories
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from core.auth import get_current_user, AuthContext

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class MemoryItem(BaseModel):
    """A memory item in search results or list."""

    id: str
    content: str  # Encrypted ciphertext
    primary_sector: str
    tags: List[str]
    metadata: dict
    score: Optional[float] = None
    salience: float
    created_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    is_org_memory: bool = False


class ListMemoriesResponse(BaseModel):
    """Response for listing memories."""

    memories: List[MemoryItem]
    total: int


class DeleteMemoriesRequest(BaseModel):
    """Request to delete all memories."""

    context: str = Field(default="personal", description="'personal' or 'org'")
    org_id: Optional[str] = Field(default=None, description="Org ID if deleting org memories")


# =============================================================================
# Endpoints - Stub implementations during migration
# =============================================================================


@router.get("", response_model=ListMemoriesResponse)
async def list_memories(
    org_id: Optional[str] = Query(default=None, description="Org ID for org context"),
    include_personal: bool = Query(default=False, description="Include personal in org context"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_current_user),
):
    """
    List all memories for the current user/org (for settings UI).

    Note: Temporarily disabled during migration to mem0.
    """
    logger.info(f"[memories] List memories called (disabled during migration) user_id={auth.user_id}")
    return ListMemoriesResponse(memories=[], total=0)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    org_id: Optional[str] = Query(default=None, description="Org ID if memory is in org context"),
    auth: AuthContext = Depends(get_current_user),
):
    """
    Delete a specific memory.

    Note: Temporarily disabled during migration to mem0.
    """
    logger.info(f"[memories] Delete memory called (disabled during migration) memory_id={memory_id}")
    # Return success (no-op during migration)
    return None


@router.delete("", status_code=status.HTTP_200_OK)
async def delete_all_memories(
    request: DeleteMemoriesRequest,
    auth: AuthContext = Depends(get_current_user),
):
    """
    Delete all memories for a context.

    Note: Temporarily disabled during migration to mem0.
    """
    logger.info(f"[memories] Delete all memories called (disabled during migration) context={request.context}")
    return {"deleted": 0, "context": request.context}
