"""
Memory API endpoints for encrypted memory storage and retrieval.

Security Note:
- All memory content is encrypted - server never sees plaintext
- Embeddings are pre-computed (from plaintext) by the enclave or client
- Server stores and searches encrypted blobs only
"""
import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from core.auth import get_current_user, AuthContext
from core.services.memory_service import (
    MemoryService,
    MemoryServiceError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Helper Functions for OpenMemory Data Parsing
# =============================================================================

def parse_metadata(r: dict) -> dict:
    """Parse metadata from OpenMemory's 'meta' column (JSON string)."""
    meta = r.get("meta") or r.get("metadata")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    elif isinstance(meta, dict):
        return meta
    return {}


def parse_tags(r: dict) -> list:
    """Parse tags from OpenMemory's 'tags' column (JSON string)."""
    tags = r.get("tags")
    if isinstance(tags, str):
        try:
            return json.loads(tags)
        except json.JSONDecodeError:
            return []
    elif isinstance(tags, list):
        return tags
    return []


def format_timestamp(ts) -> Optional[str]:
    """Convert Unix timestamp (milliseconds) to ISO 8601 string."""
    if ts is None:
        return None
    try:
        # OpenMemory stores timestamps in milliseconds
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(ts / 1000).isoformat() + "Z"
        return None
    except (ValueError, OSError):
        return None


# =============================================================================
# Request/Response Models
# =============================================================================

class StoreMemoryRequest(BaseModel):
    """Request to store an encrypted memory."""
    encrypted_content: str = Field(..., description="Encrypted memory content (ciphertext)")
    embedding: List[float] = Field(..., description="Pre-computed embedding vector")
    sector: str = Field(default="semantic", description="Memory sector")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")
    metadata: Optional[dict] = Field(default=None, description="Encryption metadata (iv, tag, key_id)")
    org_id: Optional[str] = Field(default=None, description="Org ID if storing to org context")


class StoreMemoryResponse(BaseModel):
    """Response after storing a memory."""
    id: Optional[str] = None
    primary_sector: Optional[str] = None
    salience: Optional[float] = None
    duplicate: bool = False


class SearchMemoriesRequest(BaseModel):
    """Request to search memories by embedding."""
    query_text: str = Field(..., description="Original query text (for token matching)")
    embedding: List[float] = Field(..., description="Pre-computed query embedding")
    limit: int = Field(default=10, ge=1, le=50)
    sector: Optional[str] = Field(default=None, description="Optional sector filter")
    org_id: Optional[str] = Field(default=None, description="Org ID for org context search")
    include_personal: bool = Field(default=True, description="Include personal memories in org search")


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


class SearchMemoriesResponse(BaseModel):
    """Response for memory search."""
    memories: List[MemoryItem]
    total: int


class ListMemoriesResponse(BaseModel):
    """Response for listing memories."""
    memories: List[MemoryItem]
    total: int


class DeleteMemoriesRequest(BaseModel):
    """Request to delete all memories."""
    context: str = Field(default="personal", description="'personal' or 'org'")
    org_id: Optional[str] = Field(default=None, description="Org ID if deleting org memories")


# =============================================================================
# Endpoints
# =============================================================================

async def get_memory_service() -> MemoryService:
    """Dependency to get memory service instance."""
    return MemoryService()


@router.post("/store", response_model=StoreMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(
    request: StoreMemoryRequest,
    auth: AuthContext = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Store an encrypted memory with pre-computed embedding.

    The content should be encrypted client-side or in the enclave.
    The embedding should be generated from plaintext before encryption.
    """
    try:
        result = await service.store_memory(
            encrypted_content=request.encrypted_content,
            embedding=request.embedding,
            user_id=auth.user_id,
            org_id=request.org_id,
            sector=request.sector,
            tags=request.tags,
            metadata=request.metadata,
        )

        if result is None:
            # Duplicate memory detected, return success but flag it
            return StoreMemoryResponse(duplicate=True)

        return StoreMemoryResponse(
            id=result.get("id", ""),
            primary_sector=result.get("primary_sector", request.sector),
            salience=result.get("salience", 0.5),
        )

    except MemoryServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/search", response_model=SearchMemoriesResponse)
async def search_memories(
    request: SearchMemoriesRequest,
    auth: AuthContext = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Search memories by embedding similarity.

    The query embedding should be generated client-side (e.g., via Transformers.js).
    Returns encrypted memory content that the client must decrypt.
    """
    try:
        results = await service.search_memories(
            query_text=request.query_text,
            query_embedding=request.embedding,
            user_id=auth.user_id,
            org_id=request.org_id,
            limit=request.limit,
            sector=request.sector,
            include_personal_in_org=request.include_personal,
        )

        memories = [
            MemoryItem(
                id=str(r.get("id", "")),  # Convert UUID to string
                content=r.get("content", ""),
                primary_sector=r.get("primary_sector", "semantic"),
                tags=parse_tags(r),
                metadata=parse_metadata(r),
                score=r.get("score"),
                salience=r.get("salience", 0.5),
                created_at=format_timestamp(r.get("created_at")),
                last_accessed_at=format_timestamp(r.get("last_seen_at")),
                is_org_memory=r.get("is_org_memory", False),
            )
            for r in results
        ]

        return SearchMemoriesResponse(memories=memories, total=len(memories))

    except MemoryServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=ListMemoriesResponse)
async def list_memories(
    org_id: Optional[str] = Query(default=None, description="Org ID for org context"),
    include_personal: bool = Query(default=False, description="Include personal in org context"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    """
    List all memories for the current user/org (for settings UI).

    Returns encrypted memory content that the client must decrypt.
    """
    try:
        results = await service.list_memories(
            user_id=auth.user_id,
            org_id=org_id,
            limit=limit,
            offset=offset,
            include_personal_in_org=include_personal,
        )

        memories = [
            MemoryItem(
                id=str(r.get("id", "")),  # Convert UUID to string
                content=r.get("content", ""),
                primary_sector=r.get("primary_sector", "semantic"),
                tags=parse_tags(r),
                metadata=parse_metadata(r),
                salience=r.get("salience", 0.5),
                created_at=format_timestamp(r.get("created_at")),
                last_accessed_at=format_timestamp(r.get("last_seen_at")),
                is_org_memory=r.get("is_org_memory", False),
            )
            for r in results
        ]

        return ListMemoriesResponse(memories=memories, total=len(memories))

    except MemoryServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{memory_id}", response_model=MemoryItem)
async def get_memory(
    memory_id: str,
    auth: AuthContext = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    """Get a specific memory by ID."""
    result = await service.get_memory(memory_id)

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    # Verify user has access
    # Memory user_id has prefixes (user_XXX, org_XXX) from MemoryService.get_memory_user_id
    memory_user_id = result.get("user_id", "")
    # Build allowed IDs using the same prefix pattern
    allowed_ids = [MemoryService.get_memory_user_id(auth.user_id)]
    # Note: For org access, we'd need to verify org membership
    # For now, allow if it matches user or any org the user might belong to

    if memory_user_id not in allowed_ids:
        # Check if it's an org memory - would need org membership check
        if not memory_user_id.startswith("org_"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return MemoryItem(
        id=str(result.get("id", "")),  # Convert UUID to string
        content=result.get("content", ""),
        primary_sector=result.get("primary_sector", "semantic"),
        tags=result.get("tags", []) if isinstance(result.get("tags"), list) else [],
        metadata=result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {},
        salience=result.get("salience", 0.5),
        created_at=str(result.get("created_at")) if result.get("created_at") else None,
        last_accessed_at=str(result.get("last_accessed_at")) if result.get("last_accessed_at") else None,
        is_org_memory=memory_user_id.startswith("org_"),
    )


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    org_id: Optional[str] = Query(default=None, description="Org ID if memory is in org context"),
    auth: AuthContext = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    """Delete a specific memory."""
    success = await service.delete_memory(
        memory_id=memory_id,
        user_id=auth.user_id,
        org_id=org_id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or access denied"
        )


@router.delete("", status_code=status.HTTP_200_OK)
async def delete_all_memories(
    request: DeleteMemoriesRequest,
    auth: AuthContext = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Delete all memories for a context.

    WARNING: This is destructive and cannot be undone!
    """
    try:
        count = await service.delete_all_memories(
            user_id=auth.user_id,
            org_id=request.org_id,
            context=request.context,
        )
        return {"deleted": count, "context": request.context}

    except MemoryServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
