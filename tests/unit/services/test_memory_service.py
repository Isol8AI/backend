"""Tests for MemoryService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.services.memory_service import (
    MemoryService,
    MemoryServiceError,
    MemoryNotFoundError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_openmemory():
    """Create a mock OpenMemory Memory instance."""
    mock = MagicMock()
    mock.add_with_embedding = AsyncMock()
    mock.search_with_embedding = AsyncMock()
    mock.get = AsyncMock()
    mock.delete = AsyncMock()
    mock.delete_all = AsyncMock()
    mock.history = AsyncMock()
    return mock


@pytest.fixture
def sample_embedding():
    """Sample 384-dimensional embedding vector."""
    return [0.1] * 384


@pytest.fixture
def sample_memory_result():
    """Sample memory result from OpenMemory."""
    return {
        "id": "mem_123",
        "content": "<encrypted_ciphertext>",
        "primary_sector": "semantic",
        "tags": ["test", "sample"],
        "metadata": {"iv": "abc123", "tag": "def456", "key_id": "key_789"},
        "salience": 0.75,
        "created_at": 1704067200000,
        "last_seen_at": 1704067200000,
    }


@pytest.fixture
def sample_search_results():
    """Sample search results from OpenMemory."""
    return [
        {
            "id": "mem_1",
            "content": "<encrypted_1>",
            "primary_sector": "semantic",
            "tags": [],
            "metadata": {},
            "score": 0.95,
            "salience": 0.8,
        },
        {
            "id": "mem_2",
            "content": "<encrypted_2>",
            "primary_sector": "episodic",
            "tags": ["event"],
            "metadata": {},
            "score": 0.85,
            "salience": 0.6,
        },
    ]


# =============================================================================
# Test GetMemoryUserId
# =============================================================================

class TestGetMemoryUserId:
    """Tests for get_memory_user_id static method."""

    def test_personal_context_returns_user_prefix(self):
        """Personal context adds user_ prefix if not present."""
        result = MemoryService.get_memory_user_id("clerk_abc123")
        assert result == "user_clerk_abc123"

    def test_personal_context_preserves_existing_prefix(self):
        """Personal context preserves existing user_ prefix (Clerk format)."""
        result = MemoryService.get_memory_user_id("user_37uHETjuCsViqoAnQmG1tyzVWns")
        assert result == "user_37uHETjuCsViqoAnQmG1tyzVWns"

    def test_org_context_returns_org_prefix(self):
        """Org context adds org_ prefix if not present."""
        result = MemoryService.get_memory_user_id("clerk_abc123", org_id="clerk_org_456")
        assert result == "org_clerk_org_456"

    def test_org_context_preserves_existing_prefix(self):
        """Org context preserves existing org_ prefix (Clerk format)."""
        result = MemoryService.get_memory_user_id("user_123", org_id="org_2abc123def")
        assert result == "org_2abc123def"

    def test_org_id_takes_precedence(self):
        """When org_id is provided, it takes precedence over user_id."""
        result = MemoryService.get_memory_user_id("user_123", org_id="org_456")
        assert result == "org_456"
        assert "user_123" not in result


# =============================================================================
# Test StoreMemory
# =============================================================================

class TestStoreMemory:
    """Tests for store_memory method."""

    @pytest.mark.asyncio
    async def test_stores_memory_in_personal_context(self, mock_openmemory, sample_embedding):
        """Successfully stores memory for personal context."""
        mock_openmemory.add_with_embedding.return_value = {
            "id": "mem_new",
            "primary_sector": "semantic",
            "salience": 0.5,
        }

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.store_memory(
                    encrypted_content="<ciphertext>",
                    embedding=sample_embedding,
                    user_id="user_123",
                    sector="semantic",
                    tags=["test"],
                    metadata={"iv": "abc", "tag": "def"},
                )

        assert result["id"] == "mem_new"
        mock_openmemory.add_with_embedding.assert_called_once()
        call_kwargs = mock_openmemory.add_with_embedding.call_args.kwargs
        # user_123 already has user_ prefix (Clerk format), so no doubling
        assert call_kwargs["user_id"] == "user_123"
        assert call_kwargs["content"] == "<ciphertext>"
        assert call_kwargs["embedding"] == sample_embedding

    @pytest.mark.asyncio
    async def test_stores_memory_in_org_context(self, mock_openmemory, sample_embedding):
        """Successfully stores memory for org context."""
        mock_openmemory.add_with_embedding.return_value = {
            "id": "mem_org",
            "primary_sector": "semantic",
            "salience": 0.5,
        }

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.store_memory(
                    encrypted_content="<org_ciphertext>",
                    embedding=sample_embedding,
                    user_id="user_123",
                    org_id="org_456",
                    sector="episodic",
                )

        call_kwargs = mock_openmemory.add_with_embedding.call_args.kwargs
        # org_456 already has org_ prefix (Clerk format), so no doubling
        assert call_kwargs["user_id"] == "org_456"
        assert call_kwargs["metadata"]["context"] == "org"
        assert call_kwargs["metadata"]["org_id"] == "org_456"

    @pytest.mark.asyncio
    async def test_includes_encryption_metadata(self, mock_openmemory, sample_embedding):
        """Includes encryption metadata in stored memory."""
        mock_openmemory.add_with_embedding.return_value = {"id": "mem_1", "primary_sector": "semantic", "salience": 0.5}

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                await service.store_memory(
                    encrypted_content="<ciphertext>",
                    embedding=sample_embedding,
                    user_id="user_123",
                    metadata={"iv": "abc", "tag": "def", "key_id": "key_1"},
                )

        call_kwargs = mock_openmemory.add_with_embedding.call_args.kwargs
        assert call_kwargs["metadata"]["encrypted"] is True
        assert call_kwargs["metadata"]["iv"] == "abc"
        assert call_kwargs["metadata"]["tag"] == "def"
        assert call_kwargs["metadata"]["key_id"] == "key_1"

    @pytest.mark.asyncio
    async def test_raises_error_on_openmemory_failure(self, mock_openmemory, sample_embedding):
        """Raises MemoryServiceError when OpenMemory fails."""
        mock_openmemory.add_with_embedding.side_effect = Exception("OpenMemory error")

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                with pytest.raises(MemoryServiceError, match="Failed to store memory"):
                    await service.store_memory(
                        encrypted_content="<ciphertext>",
                        embedding=sample_embedding,
                        user_id="user_123",
                    )


# =============================================================================
# Test SearchMemories
# =============================================================================

class TestSearchMemories:
    """Tests for search_memories method."""

    @pytest.mark.asyncio
    async def test_searches_personal_context(self, mock_openmemory, sample_embedding, sample_search_results):
        """Searches personal memories only in personal context."""
        mock_openmemory.search_with_embedding.return_value = sample_search_results

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.search_memories(
                    query_text="test query",
                    query_embedding=sample_embedding,
                    user_id="user_123",
                    limit=10,
                )

        assert len(results) == 2
        mock_openmemory.search_with_embedding.assert_called_once()
        call_kwargs = mock_openmemory.search_with_embedding.call_args.kwargs
        assert call_kwargs["user_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_searches_org_and_personal_in_org_context(self, mock_openmemory, sample_embedding):
        """Searches both org and personal memories in org context when include_personal=True."""
        org_results = [{"id": "org_mem", "content": "<org>", "score": 0.9, "primary_sector": "semantic", "tags": [], "metadata": {}, "salience": 0.5}]
        personal_results = [{"id": "personal_mem", "content": "<personal>", "score": 0.8, "primary_sector": "semantic", "tags": [], "metadata": {}, "salience": 0.5}]

        mock_openmemory.search_with_embedding.side_effect = [org_results, personal_results]

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.search_memories(
                    query_text="test query",
                    query_embedding=sample_embedding,
                    user_id="user_123",
                    org_id="org_456",
                    include_personal_in_org=True,
                )

        assert len(results) == 2
        # Should have made two search calls
        assert mock_openmemory.search_with_embedding.call_count == 2

        # Check that org was searched first
        first_call = mock_openmemory.search_with_embedding.call_args_list[0].kwargs
        assert first_call["user_id"] == "org_456"

        # Then personal
        second_call = mock_openmemory.search_with_embedding.call_args_list[1].kwargs
        assert second_call["user_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_searches_org_only_when_include_personal_false(self, mock_openmemory, sample_embedding, sample_search_results):
        """Searches only org memories when include_personal=False."""
        mock_openmemory.search_with_embedding.return_value = sample_search_results

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.search_memories(
                    query_text="test query",
                    query_embedding=sample_embedding,
                    user_id="user_123",
                    org_id="org_456",
                    include_personal_in_org=False,
                )

        # Only one search call for org
        assert mock_openmemory.search_with_embedding.call_count == 1
        call_kwargs = mock_openmemory.search_with_embedding.call_args.kwargs
        assert call_kwargs["user_id"] == "org_456"

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, mock_openmemory, sample_embedding):
        """Results are sorted by score in descending order."""
        org_results = [{"id": "low_score", "content": "<>", "score": 0.5, "primary_sector": "semantic", "tags": [], "metadata": {}, "salience": 0.5}]
        personal_results = [{"id": "high_score", "content": "<>", "score": 0.9, "primary_sector": "semantic", "tags": [], "metadata": {}, "salience": 0.5}]

        mock_openmemory.search_with_embedding.side_effect = [org_results, personal_results]

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.search_memories(
                    query_text="test",
                    query_embedding=sample_embedding,
                    user_id="user_123",
                    org_id="org_456",
                )

        # High score should be first
        assert results[0]["id"] == "high_score"
        assert results[1]["id"] == "low_score"

    @pytest.mark.asyncio
    async def test_adds_source_context_to_results(self, mock_openmemory, sample_embedding, sample_search_results):
        """Adds memory_user_id and is_org_memory to results."""
        mock_openmemory.search_with_embedding.return_value = sample_search_results

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.search_memories(
                    query_text="test",
                    query_embedding=sample_embedding,
                    user_id="user_123",
                )

        for r in results:
            assert "memory_user_id" in r
            assert "is_org_memory" in r


# =============================================================================
# Test GetMemory
# =============================================================================

class TestGetMemory:
    """Tests for get_memory method."""

    @pytest.mark.asyncio
    async def test_returns_memory_dict(self, mock_openmemory, sample_memory_result):
        """Returns memory as dict."""
        # Mock the Row-like object that OpenMemory returns
        mock_row = MagicMock()
        mock_row.__iter__ = MagicMock(return_value=iter(sample_memory_result.items()))
        mock_row.keys = MagicMock(return_value=sample_memory_result.keys())
        for key, value in sample_memory_result.items():
            mock_row.__getitem__ = MagicMock(side_effect=lambda k: sample_memory_result[k])

        mock_openmemory.get.return_value = sample_memory_result

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.get_memory("mem_123")

        assert result["id"] == "mem_123"
        assert result["content"] == "<encrypted_ciphertext>"

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent(self, mock_openmemory):
        """Returns None for nonexistent memory."""
        mock_openmemory.get.return_value = None

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.get_memory("nonexistent")

        assert result is None


# =============================================================================
# Test DeleteMemory
# =============================================================================

class TestDeleteMemory:
    """Tests for delete_memory method."""

    @pytest.mark.asyncio
    async def test_deletes_owned_personal_memory(self, mock_openmemory, sample_memory_result):
        """Deletes memory owned by user in personal context."""
        sample_memory_result["user_id"] = "user_123"
        mock_openmemory.get.return_value = sample_memory_result
        mock_openmemory.delete.return_value = None

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.delete_memory(
                    memory_id="mem_123",
                    user_id="user_123",
                )

        assert result is True
        mock_openmemory.delete.assert_called_once_with("mem_123")

    @pytest.mark.asyncio
    async def test_deletes_org_memory_when_in_org_context(self, mock_openmemory, sample_memory_result):
        """Deletes org memory when user is in org context."""
        sample_memory_result["user_id"] = "org_456"
        mock_openmemory.get.return_value = sample_memory_result

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.delete_memory(
                    memory_id="mem_123",
                    user_id="user_123",
                    org_id="org_456",
                )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_unauthorized(self, mock_openmemory, sample_memory_result):
        """Returns False when user doesn't own the memory."""
        sample_memory_result["user_id"] = "user_other_user"
        mock_openmemory.get.return_value = sample_memory_result

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.delete_memory(
                    memory_id="mem_123",
                    user_id="user_123",
                )

        assert result is False
        mock_openmemory.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_for_nonexistent(self, mock_openmemory):
        """Returns False for nonexistent memory."""
        mock_openmemory.get.return_value = None

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                result = await service.delete_memory(
                    memory_id="nonexistent",
                    user_id="user_123",
                )

        assert result is False


# =============================================================================
# Test ListMemories
# =============================================================================

class TestListMemories:
    """Tests for list_memories method."""

    @pytest.mark.asyncio
    async def test_lists_personal_memories(self, mock_openmemory):
        """Lists personal memories in personal context."""
        mock_openmemory.history.return_value = [
            {"id": "mem_1", "created_at": 1000},
            {"id": "mem_2", "created_at": 900},
        ]

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.list_memories(
                    user_id="user_123",
                    limit=10,
                )

        assert len(results) == 2
        mock_openmemory.history.assert_called_once_with(
            user_id="user_123",
            limit=10,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_lists_org_memories_in_org_context(self, mock_openmemory):
        """Lists org memories when in org context."""
        mock_openmemory.history.return_value = [
            {"id": "org_mem", "created_at": 1000},
        ]

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.list_memories(
                    user_id="user_123",
                    org_id="org_456",
                    limit=10,
                )

        # Should call history with org user_id
        mock_openmemory.history.assert_called_with(
            user_id="org_456",
            limit=10,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_adds_is_org_memory_flag(self, mock_openmemory):
        """Adds is_org_memory flag to results."""
        mock_openmemory.history.return_value = [
            {"id": "mem_1", "created_at": 1000},
        ]

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                results = await service.list_memories(
                    user_id="user_123",
                )

        assert results[0]["is_org_memory"] is False


# =============================================================================
# Test DeleteAllMemories
# =============================================================================

class TestDeleteAllMemories:
    """Tests for delete_all_memories method."""

    @pytest.mark.asyncio
    async def test_deletes_personal_memories(self, mock_openmemory):
        """Deletes all personal memories."""
        mock_openmemory.history.return_value = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        mock_openmemory.delete_all.return_value = None

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                count = await service.delete_all_memories(
                    user_id="user_123",
                    context="personal",
                )

        assert count == 3
        mock_openmemory.delete_all.assert_called_once_with(user_id="user_123")

    @pytest.mark.asyncio
    async def test_deletes_org_memories(self, mock_openmemory):
        """Deletes all org memories when context=org."""
        mock_openmemory.history.return_value = [{"id": "1"}, {"id": "2"}]
        mock_openmemory.delete_all.return_value = None

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                count = await service.delete_all_memories(
                    user_id="user_123",
                    org_id="org_456",
                    context="org",
                )

        assert count == 2
        mock_openmemory.delete_all.assert_called_once_with(user_id="org_456")

    @pytest.mark.asyncio
    async def test_raises_error_on_failure(self, mock_openmemory):
        """Raises MemoryServiceError on failure."""
        mock_openmemory.history.return_value = []
        mock_openmemory.delete_all.side_effect = Exception("Delete failed")

        with patch.object(MemoryService, '_memory', mock_openmemory):
            with patch.object(MemoryService, '_initialized', True):
                service = MemoryService()
                with pytest.raises(MemoryServiceError, match="Failed to delete memories"):
                    await service.delete_all_memories(
                        user_id="user_123",
                        context="personal",
                    )
