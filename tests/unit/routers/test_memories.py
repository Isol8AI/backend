"""Tests for memories router endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_memory_service():
    """Create a mock MemoryService."""
    service = MagicMock()
    service.store_memory = AsyncMock()
    service.search_memories = AsyncMock()
    service.get_memory = AsyncMock()
    service.delete_memory = AsyncMock()
    service.list_memories = AsyncMock()
    service.delete_all_memories = AsyncMock()
    return service


@pytest.fixture
def sample_embedding():
    """Sample 384-dimensional embedding vector."""
    return [0.1] * 384


@pytest.fixture
def sample_stored_memory():
    """Sample response from store_memory."""
    return {
        "id": "mem_123",
        "primary_sector": "semantic",
        "salience": 0.5,
    }


@pytest.fixture
def sample_memory_list():
    """Sample list of memories."""
    return [
        {
            "id": "mem_1",
            "content": "<encrypted_1>",
            "primary_sector": "semantic",
            "tags": ["test"],
            "metadata": {"iv": "abc"},
            "salience": 0.8,
            "score": 0.95,
            "created_at": "1704067200000",
            "last_accessed_at": "1704067200000",
            "is_org_memory": False,
        },
        {
            "id": "mem_2",
            "content": "<encrypted_2>",
            "primary_sector": "episodic",
            "tags": [],
            "metadata": {},
            "salience": 0.6,
            "score": 0.85,
            "created_at": "1704067100000",
            "last_accessed_at": "1704067100000",
            "is_org_memory": True,
        },
    ]


# =============================================================================
# Test Store Memory Endpoint
# =============================================================================

class TestStoreMemoryEndpoint:
    """Tests for POST /memories/store endpoint."""

    @pytest.mark.asyncio
    async def test_stores_memory_successfully(self, async_client, mock_memory_service, sample_embedding, sample_stored_memory):
        """Successfully stores a memory."""
        mock_memory_service.store_memory.return_value = sample_stored_memory

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.post(
                "/api/v1/memories/store",
                json={
                    "encrypted_content": "<ciphertext>",
                    "embedding": sample_embedding,
                    "sector": "semantic",
                    "tags": ["test"],
                    "metadata": {"iv": "abc", "tag": "def"},
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "mem_123"
        assert data["primary_sector"] == "semantic"
        assert data["salience"] == 0.5

    @pytest.mark.asyncio
    async def test_stores_memory_with_org_context(self, async_client, mock_memory_service, sample_embedding, sample_stored_memory):
        """Stores memory in org context."""
        mock_memory_service.store_memory.return_value = sample_stored_memory

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.post(
                "/api/v1/memories/store",
                json={
                    "encrypted_content": "<ciphertext>",
                    "embedding": sample_embedding,
                    "org_id": "org_456",
                },
            )

        assert response.status_code == 201
        mock_memory_service.store_memory.assert_called_once()
        call_kwargs = mock_memory_service.store_memory.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"

    @pytest.mark.asyncio
    async def test_returns_500_on_service_error(self, async_client, mock_memory_service, sample_embedding):
        """Returns 500 when service raises error."""
        from core.services.memory_service import MemoryServiceError
        mock_memory_service.store_memory.side_effect = MemoryServiceError("Storage failed")

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.post(
                "/api/v1/memories/store",
                json={
                    "encrypted_content": "<ciphertext>",
                    "embedding": sample_embedding,
                },
            )

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_validates_required_fields(self, async_client, mock_memory_service):
        """Validates required fields in request."""
        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            # Missing embedding
            response = await async_client.post(
                "/api/v1/memories/store",
                json={
                    "encrypted_content": "<ciphertext>",
                },
            )

        assert response.status_code == 422


# =============================================================================
# Test Search Memories Endpoint
# =============================================================================

class TestSearchMemoriesEndpoint:
    """Tests for POST /memories/search endpoint."""

    @pytest.mark.asyncio
    async def test_searches_memories_successfully(self, async_client, mock_memory_service, sample_embedding, sample_memory_list):
        """Successfully searches memories."""
        mock_memory_service.search_memories.return_value = sample_memory_list

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.post(
                "/api/v1/memories/search",
                json={
                    "query_text": "test query",
                    "embedding": sample_embedding,
                    "limit": 10,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2
        assert data["memories"][0]["id"] == "mem_1"

    @pytest.mark.asyncio
    async def test_searches_with_org_context(self, async_client, mock_memory_service, sample_embedding, sample_memory_list):
        """Searches with org context."""
        mock_memory_service.search_memories.return_value = sample_memory_list

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.post(
                "/api/v1/memories/search",
                json={
                    "query_text": "test query",
                    "embedding": sample_embedding,
                    "org_id": "org_456",
                    "include_personal": True,
                },
            )

        assert response.status_code == 200
        call_kwargs = mock_memory_service.search_memories.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["include_personal_in_org"] is True

    @pytest.mark.asyncio
    async def test_respects_limit_bounds(self, async_client, mock_memory_service, sample_embedding):
        """Validates limit parameter bounds."""
        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            # Limit too high
            response = await async_client.post(
                "/api/v1/memories/search",
                json={
                    "query_text": "test",
                    "embedding": sample_embedding,
                    "limit": 100,  # Max is 50
                },
            )

        assert response.status_code == 422


# =============================================================================
# Test List Memories Endpoint
# =============================================================================

class TestListMemoriesEndpoint:
    """Tests for GET /memories endpoint."""

    @pytest.mark.asyncio
    async def test_lists_memories_successfully(self, async_client, mock_memory_service, sample_memory_list):
        """Successfully lists memories."""
        mock_memory_service.list_memories.return_value = sample_memory_list

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.get("/api/v1/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2

    @pytest.mark.asyncio
    async def test_lists_with_org_context(self, async_client, mock_memory_service, sample_memory_list):
        """Lists memories with org context."""
        mock_memory_service.list_memories.return_value = sample_memory_list

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.get(
                "/api/v1/memories",
                params={"org_id": "org_456", "include_personal": "true"},
            )

        assert response.status_code == 200
        call_kwargs = mock_memory_service.list_memories.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["include_personal_in_org"] is True

    @pytest.mark.asyncio
    async def test_supports_pagination(self, async_client, mock_memory_service):
        """Supports limit and offset parameters."""
        mock_memory_service.list_memories.return_value = []

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.get(
                "/api/v1/memories",
                params={"limit": 20, "offset": 10},
            )

        assert response.status_code == 200
        call_kwargs = mock_memory_service.list_memories.call_args.kwargs
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 10


# =============================================================================
# Test Get Memory Endpoint
# =============================================================================

class TestGetMemoryEndpoint:
    """Tests for GET /memories/{memory_id} endpoint."""

    @pytest.mark.asyncio
    async def test_gets_memory_successfully(self, async_client, mock_memory_service):
        """Successfully gets a memory."""
        mock_memory_service.get_memory.return_value = {
            "id": "mem_123",
            "content": "<encrypted>",
            "primary_sector": "semantic",
            "tags": [],
            "metadata": {},
            "salience": 0.5,
            "user_id": "user_user_test_123",
            "created_at": 1704067200000,
            "last_accessed_at": 1704067200000,
        }

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.get("/api/v1/memories/mem_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mem_123"
        assert data["content"] == "<encrypted>"

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent(self, async_client, mock_memory_service):
        """Returns 404 for nonexistent memory."""
        mock_memory_service.get_memory.return_value = None

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.get("/api/v1/memories/nonexistent")

        assert response.status_code == 404


# =============================================================================
# Test Delete Memory Endpoint
# =============================================================================

class TestDeleteMemoryEndpoint:
    """Tests for DELETE /memories/{memory_id} endpoint."""

    @pytest.mark.asyncio
    async def test_deletes_memory_successfully(self, async_client, mock_memory_service):
        """Successfully deletes a memory."""
        mock_memory_service.delete_memory.return_value = True

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.delete("/api/v1/memories/mem_123")

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_deletes_with_org_context(self, async_client, mock_memory_service):
        """Deletes memory with org context."""
        mock_memory_service.delete_memory.return_value = True

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.delete(
                "/api/v1/memories/mem_123",
                params={"org_id": "org_456"},
            )

        assert response.status_code == 204
        call_kwargs = mock_memory_service.delete_memory.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found_or_unauthorized(self, async_client, mock_memory_service):
        """Returns 404 when memory not found or user unauthorized."""
        mock_memory_service.delete_memory.return_value = False

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.delete("/api/v1/memories/mem_123")

        assert response.status_code == 404


# =============================================================================
# Test Delete All Memories Endpoint
# =============================================================================

class TestDeleteAllMemoriesEndpoint:
    """Tests for DELETE /memories endpoint."""

    @pytest.mark.asyncio
    async def test_deletes_all_personal_memories(self, async_client, mock_memory_service):
        """Deletes all personal memories."""
        mock_memory_service.delete_all_memories.return_value = 5

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.request(
                "DELETE",
                "/api/v1/memories",
                json={"context": "personal"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == 5
        assert data["context"] == "personal"

    @pytest.mark.asyncio
    async def test_deletes_all_org_memories(self, async_client, mock_memory_service):
        """Deletes all org memories."""
        mock_memory_service.delete_all_memories.return_value = 3

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.request(
                "DELETE",
                "/api/v1/memories",
                json={"context": "org", "org_id": "org_456"},
            )

        assert response.status_code == 200
        call_kwargs = mock_memory_service.delete_all_memories.call_args.kwargs
        assert call_kwargs["context"] == "org"
        assert call_kwargs["org_id"] == "org_456"

    @pytest.mark.asyncio
    async def test_returns_500_on_service_error(self, async_client, mock_memory_service):
        """Returns 500 when service raises error."""
        from core.services.memory_service import MemoryServiceError
        mock_memory_service.delete_all_memories.side_effect = MemoryServiceError("Delete failed")

        with patch("routers.memories.MemoryService", return_value=mock_memory_service):
            response = await async_client.request(
                "DELETE",
                "/api/v1/memories",
                json={"context": "personal"},
            )

        assert response.status_code == 500


# =============================================================================
# Test Authentication
# =============================================================================

class TestMemoriesAuthentication:
    """Tests for authentication on memories endpoints."""

    @pytest.mark.asyncio
    async def test_store_requires_auth(self, unauthenticated_async_client):
        """Store endpoint requires authentication."""
        response = await unauthenticated_async_client.post(
            "/api/v1/memories/store",
            json={
                "encrypted_content": "<ciphertext>",
                "embedding": [0.1] * 384,
            },
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_search_requires_auth(self, unauthenticated_async_client):
        """Search endpoint requires authentication."""
        response = await unauthenticated_async_client.post(
            "/api/v1/memories/search",
            json={
                "query_text": "test",
                "embedding": [0.1] * 384,
            },
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_list_requires_auth(self, unauthenticated_async_client):
        """List endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/memories")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_requires_auth(self, unauthenticated_async_client):
        """Get endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/memories/mem_123")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_requires_auth(self, unauthenticated_async_client):
        """Delete endpoint requires authentication."""
        response = await unauthenticated_async_client.delete("/api/v1/memories/mem_123")
        assert response.status_code == 403
