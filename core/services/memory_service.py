"""
Memory Service - wraps OpenMemory SDK for encrypted memory storage.

Security Note:
- This service NEVER sees plaintext memory content
- Content is encrypted client-side or in the enclave before reaching here
- Embeddings are pre-computed from plaintext in the enclave
- The server stores encrypted blobs and embeddings only
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Add the memory package to Python path
MEMORY_PATH = Path(__file__).parent.parent.parent.parent / "memory" / "packages" / "openmemory-py" / "src"
if str(MEMORY_PATH) not in sys.path:
    sys.path.insert(0, str(MEMORY_PATH))

# Configure OpenMemory database path before importing
# OpenMemory SDK uses OM_DB_URL environment variable
OPENMEMORY_DB_PATH = Path(__file__).parent.parent.parent / "data" / "openmemory.db"
OPENMEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OM_DB_URL", f"sqlite:///{OPENMEMORY_DB_PATH}")


class MemoryServiceError(Exception):
    """Base exception for memory service errors."""
    pass


class MemoryNotFoundError(MemoryServiceError):
    """Memory not found."""
    pass


class MemoryService:
    """
    Service for managing encrypted memories via OpenMemory.

    All content is encrypted - this service only sees ciphertext.
    Embeddings are pre-computed from plaintext elsewhere (enclave or client).
    """

    _initialized = False
    _memory = None

    @classmethod
    def _ensure_initialized(cls):
        """Initialize OpenMemory connection once."""
        if cls._initialized:
            return

        try:
            from openmemory import Memory
            cls._memory = Memory(user=None)
            cls._initialized = True
            logger.info(f"OpenMemory initialized with database at {OPENMEMORY_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenMemory: {e}")
            raise MemoryServiceError(f"Failed to initialize memory service: {e}")

    def __init__(self):
        """Initialize the memory service."""
        self._ensure_initialized()

    @staticmethod
    def get_memory_user_id(user_id: str, org_id: Optional[str] = None) -> str:
        """
        Get the memory user_id based on context.

        Clerk IDs already have prefixes:
        - User: user_XXXXX (from JWT 'sub' claim)
        - Org: org_XXXXX (from JWT 'o.id' claim)

        Personal context: Use user_id as-is (already prefixed)
        Org context: Use org_id as-is (already prefixed)
        """
        if org_id:
            # Clerk org_id already has 'org_' prefix
            return org_id
        # Clerk user_id already has 'user_' prefix
        return user_id

    # Similarity threshold for deduplication (0.0 to 1.0)
    # Memories with similarity >= this threshold are considered duplicates
    DEDUP_SIMILARITY_THRESHOLD = 0.92

    async def store_memory(
        self,
        encrypted_content: str,
        embedding: List[float],
        user_id: str,
        org_id: Optional[str] = None,
        sector: str = "semantic",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Store an encrypted memory with pre-computed embedding.

        Performs deduplication by checking for existing similar memories.
        If a memory with high similarity exists, returns None without storing.

        Args:
            encrypted_content: Encrypted ciphertext (we never see plaintext)
            embedding: Pre-computed embedding vector from plaintext (generated in enclave)
            user_id: Clerk user ID
            org_id: Optional Clerk org ID (if in org context, writes to org store)
            sector: Memory sector (semantic, episodic, procedural, emotional, reflective)
            tags: Optional tags
            metadata: Encryption metadata (iv, tag, key_id, etc.)

        Returns:
            Dict with id, primary_sector, sectors, salience, or None if duplicate
        """
        memory_user_id = self.get_memory_user_id(user_id, org_id)

        # Check for duplicate memories using embedding similarity
        try:
            existing = await self._search_single(
                query_text="",  # Not used for vector search
                query_embedding=embedding,
                memory_user_id=memory_user_id,
                limit=1,
                sector=sector,
            )
            if existing and existing[0].get("score", 0) >= self.DEDUP_SIMILARITY_THRESHOLD:
                logger.info(
                    f"Skipping duplicate memory for {memory_user_id} "
                    f"(similarity: {existing[0].get('score', 0):.3f})"
                )
                return None
        except Exception as e:
            # If dedup check fails, proceed with storing (non-fatal)
            logger.warning(f"Dedup check failed (proceeding anyway): {e}")

        # Ensure metadata includes encryption info
        full_metadata = metadata or {}
        full_metadata["encrypted"] = True
        full_metadata["context"] = "org" if org_id else "personal"
        if org_id:
            full_metadata["org_id"] = org_id
        full_metadata["original_user_id"] = user_id

        try:
            result = await self._memory.add_with_embedding(
                content=encrypted_content,
                embedding=embedding,
                user_id=memory_user_id,
                sector=sector,
                tags=tags,
                metadata=full_metadata,
            )

            logger.info(f"Stored memory {result.get('id')} for {memory_user_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise MemoryServiceError(f"Failed to store memory: {e}")

    async def search_memories(
        self,
        query_text: str,
        query_embedding: List[float],
        user_id: str,
        org_id: Optional[str] = None,
        limit: int = 10,
        sector: Optional[str] = None,
        include_personal_in_org: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by embedding similarity.

        Args:
            query_text: Original query text (for token matching)
            query_embedding: Pre-computed query embedding (from client via Transformers.js)
            user_id: Clerk user ID
            org_id: Optional Clerk org ID
            limit: Max results to return
            sector: Optional sector filter
            include_personal_in_org: If True and in org context, search both org AND personal

        Returns:
            List of memory dicts (content is still encrypted)
        """
        results = []

        # Clerk IDs already have prefixes (user_XXX, org_XXX)
        if org_id and include_personal_in_org:
            # Org context: search both org and personal memories
            # org_id already has 'org_' prefix, user_id already has 'user_' prefix

            # Search org memories
            org_results = await self._search_single(
                query_text, query_embedding, org_id, limit, sector
            )

            # Search personal memories
            personal_results = await self._search_single(
                query_text, query_embedding, user_id, limit, sector
            )

            # Merge and sort by score
            results = org_results + personal_results
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            results = results[:limit]

        else:
            # Single context search - use the appropriate ID directly
            memory_user_id = self.get_memory_user_id(user_id, org_id)
            results = await self._search_single(
                query_text, query_embedding, memory_user_id, limit, sector
            )

        return results

    async def _search_single(
        self,
        query_text: str,
        query_embedding: List[float],
        memory_user_id: str,
        limit: int,
        sector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search a single memory store."""
        try:
            results = await self._memory.search_with_embedding(
                query_text=query_text,
                query_embedding=query_embedding,
                user_id=memory_user_id,
                limit=limit,
                sector=sector or "semantic",
            )

            # Add source context to results
            for r in results:
                r["memory_user_id"] = memory_user_id
                r["is_org_memory"] = memory_user_id.startswith("org_")

            return results

        except Exception as e:
            logger.error(f"Search failed for {memory_user_id}: {e}")
            return []

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        try:
            result = await self._memory.get(memory_id)
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    async def delete_memory(self, memory_id: str, user_id: str, org_id: Optional[str] = None) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete
            user_id: Clerk user ID (for authorization)
            org_id: Optional Clerk org ID

        Returns:
            True if deleted, False if not found or unauthorized
        """
        # First verify the memory belongs to this user/org
        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        memory_user_id = memory.get("user_id", "")
        # Clerk IDs already have prefixes (user_XXX, org_XXX)
        allowed_ids = [user_id]
        if org_id:
            allowed_ids.append(org_id)

        if memory_user_id not in allowed_ids:
            logger.warning(f"Unauthorized delete attempt: {user_id} tried to delete memory owned by {memory_user_id}")
            return False

        try:
            await self._memory.delete(memory_id)
            logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def list_memories(
        self,
        user_id: str,
        org_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_personal_in_org: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all memories for a user/org (for settings UI).

        Args:
            user_id: Clerk user ID
            org_id: Optional Clerk org ID
            limit: Max results
            offset: Pagination offset
            include_personal_in_org: If True, include personal memories when in org context

        Returns:
            List of memory dicts
        """
        results = []

        # Clerk IDs already have prefixes (user_XXX, org_XXX)
        if org_id:
            # Get org memories (org_id already has 'org_' prefix)
            org_memories = self._memory.history(
                user_id=org_id,
                limit=limit,
                offset=offset,
            )
            for m in org_memories:
                m["is_org_memory"] = True
            results.extend(org_memories)

            if include_personal_in_org:
                # Also get personal memories (user_id already has 'user_' prefix)
                personal_memories = self._memory.history(
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                )
                for m in personal_memories:
                    m["is_org_memory"] = False
                results.extend(personal_memories)
        else:
            # Personal context only (user_id already has 'user_' prefix)
            results = self._memory.history(
                user_id=user_id,
                limit=limit,
                offset=offset,
            )
            for m in results:
                m["is_org_memory"] = False

        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", 0), reverse=True)

        return results[:limit]

    async def delete_all_memories(
        self,
        user_id: str,
        org_id: Optional[str] = None,
        context: str = "personal",
    ) -> int:
        """
        Delete all memories for a user or org.

        Args:
            user_id: Clerk user ID
            org_id: Optional Clerk org ID
            context: "personal" or "org" - which memories to delete

        Returns:
            Number of memories deleted (approximate)
        """
        # Clerk IDs already have prefixes (user_XXX, org_XXX)
        if context == "org" and org_id:
            memory_user_id = org_id
        else:
            memory_user_id = user_id

        try:
            # Get count before deletion
            memories = self._memory.history(user_id=memory_user_id, limit=10000)
            count = len(memories)

            await self._memory.delete_all(user_id=memory_user_id)
            logger.info(f"Deleted {count} memories for {memory_user_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete all memories for {memory_user_id}: {e}")
            raise MemoryServiceError(f"Failed to delete memories: {e}")
