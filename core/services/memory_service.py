"""
Memory Service - manages encrypted memory storage via OpenMemory SDK.

Security Note:
- This service NEVER sees plaintext memory content
- Content is encrypted client-side or in the enclave before reaching here
- Embeddings are pre-computed from plaintext in the enclave
- The server stores encrypted blobs and embeddings only
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded decay module reference
_on_query_hit = None


def _get_on_query_hit():
    """Lazy-load the on_query_hit function from OpenMemory decay module."""
    global _on_query_hit
    if _on_query_hit is None:
        try:
            import sys
            from pathlib import Path
            memory_path = Path(__file__).parent.parent.parent.parent / "memory" / "packages" / "openmemory-py" / "src"
            if str(memory_path) not in sys.path:
                sys.path.insert(0, str(memory_path))
            from openmemory.memory.decay import on_query_hit
            _on_query_hit = on_query_hit
        except ImportError as e:
            logger.warning(f"Memory decay module not available: {e}")
            _on_query_hit = None
    return _on_query_hit


class MemoryServiceError(Exception):
    """Base exception for memory service errors."""
    pass


class MemoryNotFoundError(MemoryServiceError):
    """Memory not found."""
    pass


class MemoryService:
    """
    Service for managing encrypted memories via OpenMemory SDK.

    All content is encrypted - this service only sees ciphertext.
    Embeddings are pre-computed from plaintext elsewhere (enclave or client).
    """

    # Class-level OpenMemory instance (singleton pattern)
    _memory = None
    _initialized = False

    def __init__(self):
        """Initialize the MemoryService."""
        if not MemoryService._initialized:
            self._initialize_openmemory()

    @classmethod
    def _initialize_openmemory(cls):
        """Initialize the OpenMemory SDK instance."""
        if cls._initialized:
            return

        try:
            # Import OpenMemory from the memory package
            import sys
            from pathlib import Path

            # Add memory package to path if not already there
            # Path: backend/core/services/memory_service.py -> freebird/memory/packages/openmemory-py/src
            memory_path = Path(__file__).parent.parent.parent.parent / "memory" / "packages" / "openmemory-py" / "src"
            if str(memory_path) not in sys.path:
                sys.path.insert(0, str(memory_path))

            from openmemory import Memory
            cls._memory = Memory()
            cls._initialized = True
            logger.info("OpenMemory SDK initialized successfully")
        except ImportError as e:
            logger.warning(f"OpenMemory SDK not available: {e}")
            cls._memory = None
            cls._initialized = True  # Mark as initialized to prevent retry

    @staticmethod
    def get_memory_user_id(user_id: str, org_id: Optional[str] = None) -> str:
        """
        Get the memory user_id based on context.

        Personal context: Returns the Clerk user_id as-is (already prefixed with user_)
        Org context: Returns org_{org_id}

        This creates a namespace for memory isolation.
        Note: Clerk user IDs already have 'user_' prefix (e.g., user_37uHET...)
        and org IDs have 'org_' prefix (e.g., org_2abc...), so we use them directly.
        """
        if org_id:
            # Clerk org IDs already have 'org_' prefix, use as-is
            if org_id.startswith("org_"):
                return org_id
            return f"org_{org_id}"
        # Clerk user IDs already have 'user_' prefix, use as-is
        if user_id.startswith("user_"):
            return user_id
        return f"user_{user_id}"

    async def store_memory(
        self,
        encrypted_content: str,
        embedding: List[float],
        user_id: str,
        org_id: Optional[str] = None,
        sector: str = "semantic",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        salience: Optional[float] = None,
        summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store an encrypted memory with pre-computed embedding.

        Args:
            encrypted_content: Encrypted ciphertext (we never see plaintext)
            embedding: Pre-computed embedding vector from plaintext (generated in enclave)
            user_id: Clerk user ID
            org_id: Optional Clerk org ID (if in org context, writes to org store)
            sector: Memory sector (semantic, episodic, procedural, emotional, reflective)
            tags: Optional tags
            metadata: Encryption metadata (iv, tag, key_id, etc.)
            salience: Optional importance score (0.0-1.0), computed in enclave from plaintext
            summary: Optional encrypted summary text (generated in enclave from plaintext)

        Returns:
            Dict with id, primary_sector, salience
        """
        if self._memory is None:
            raise MemoryServiceError("OpenMemory SDK not initialized")

        memory_user_id = self.get_memory_user_id(user_id, org_id)

        # Ensure metadata includes encryption info
        full_metadata = metadata.copy() if metadata else {}
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
                salience=salience,
                summary=summary,
            )

            logger.info(f"Stored memory {result.get('id')} for {memory_user_id} (salience: {salience})")
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
            query_text: Original query text (used for token matching)
            query_embedding: Pre-computed query embedding (from client via Transformers.js)
            user_id: Clerk user ID
            org_id: Optional Clerk org ID
            limit: Max results to return
            sector: Optional sector filter
            include_personal_in_org: If True and in org context, search both org AND personal

        Returns:
            List of memory dicts (content is still encrypted)
        """
        if self._memory is None:
            raise MemoryServiceError("OpenMemory SDK not initialized")

        results = []

        if org_id and include_personal_in_org:
            # Org context: search both org and personal memories
            org_user_id = self.get_memory_user_id(user_id, org_id)
            personal_user_id = self.get_memory_user_id(user_id)

            org_results = await self._memory.search_with_embedding(
                query_text=query_text,
                query_embedding=query_embedding,
                user_id=org_user_id,
                limit=limit,
                sector=sector,
            )

            personal_results = await self._memory.search_with_embedding(
                query_text=query_text,
                query_embedding=query_embedding,
                user_id=personal_user_id,
                limit=limit,
                sector=sector,
            )

            # Add source context to results
            for r in org_results:
                r["memory_user_id"] = org_user_id
                r["is_org_memory"] = True
            for r in personal_results:
                r["memory_user_id"] = personal_user_id
                r["is_org_memory"] = False

            # Merge and sort by score
            results = org_results + personal_results
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            results = results[:limit]

        else:
            # Single context search
            memory_user_id = self.get_memory_user_id(user_id, org_id)
            results = await self._memory.search_with_embedding(
                query_text=query_text,
                query_embedding=query_embedding,
                user_id=memory_user_id,
                limit=limit,
                sector=sector,
            )

            # Add source context
            for r in results:
                r["memory_user_id"] = memory_user_id
                r["is_org_memory"] = memory_user_id.startswith("org_")

        # Boost salience of accessed memories (fire-and-forget, don't block response)
        if results:
            asyncio.create_task(self._boost_accessed_memories(results))

        return results

    async def _boost_accessed_memories(self, results: List[Dict[str, Any]]):
        """Boost salience of memories that were accessed during search."""
        on_query_hit = _get_on_query_hit()
        if on_query_hit is None:
            return

        for r in results:
            try:
                mem_id = str(r.get("id", ""))
                sector = r.get("primary_sector", "semantic")
                if mem_id:
                    await on_query_hit(mem_id, sector)
            except Exception as e:
                # Don't let decay errors affect search results
                logger.debug(f"Failed to boost memory {r.get('id')}: {e}")

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        if self._memory is None:
            raise MemoryServiceError("OpenMemory SDK not initialized")

        try:
            result = await self._memory.get(memory_id)
            if result:
                return dict(result) if hasattr(result, 'keys') else result
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
        if self._memory is None:
            raise MemoryServiceError("OpenMemory SDK not initialized")

        # First verify the memory belongs to this user/org
        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        memory_user_id = memory.get("user_id", "")
        allowed_ids = [self.get_memory_user_id(user_id)]
        if org_id:
            allowed_ids.append(self.get_memory_user_id(user_id, org_id))

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
        if self._memory is None:
            raise MemoryServiceError("OpenMemory SDK not initialized")

        results = []

        try:
            if org_id:
                # Get org memories
                org_user_id = self.get_memory_user_id(user_id, org_id)
                org_results = await self._memory.history(
                    user_id=org_user_id,
                    limit=limit,
                    offset=offset,
                )
                for r in org_results:
                    r["is_org_memory"] = True
                results.extend(org_results)

                if include_personal_in_org:
                    # Also get personal memories
                    personal_user_id = self.get_memory_user_id(user_id)
                    personal_results = await self._memory.history(
                        user_id=personal_user_id,
                        limit=limit,
                        offset=offset,
                    )
                    for r in personal_results:
                        r["is_org_memory"] = False
                    results.extend(personal_results)
            else:
                # Personal context only
                memory_user_id = self.get_memory_user_id(user_id)
                results = await self._memory.history(
                    user_id=memory_user_id,
                    limit=limit,
                    offset=offset,
                )
                for r in results:
                    r["is_org_memory"] = False

            # Sort by created_at descending
            results.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return []

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
            Number of memories deleted
        """
        if self._memory is None:
            raise MemoryServiceError("OpenMemory SDK not initialized")

        if context == "org" and org_id:
            memory_user_id = self.get_memory_user_id(user_id, org_id)
        else:
            memory_user_id = self.get_memory_user_id(user_id)

        try:
            # Get count before deletion
            existing = await self._memory.history(user_id=memory_user_id, limit=1000, offset=0)
            count = len(existing)

            # Delete all
            await self._memory.delete_all(user_id=memory_user_id)

            logger.info(f"Deleted {count} memories for {memory_user_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete all memories for {memory_user_id}: {e}")
            raise MemoryServiceError(f"Failed to delete memories: {e}")
