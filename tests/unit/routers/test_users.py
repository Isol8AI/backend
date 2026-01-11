"""Unit tests for users router."""
import pytest
from sqlalchemy import select

from models.user import User


class TestSyncUser:
    """Tests for POST /api/v1/users/sync endpoint."""

    @pytest.mark.asyncio
    async def test_sync_creates_new_user(self, async_client, db_session):
        """Sync creates new user when not exists."""
        response = await async_client.post("/api/v1/users/sync")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["user_id"] == "user_test_123"

    @pytest.mark.asyncio
    async def test_sync_returns_exists_for_existing_user(self, async_client, db_session, test_user):
        """Sync returns 'exists' for existing user."""
        response = await async_client.post("/api/v1/users/sync")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "exists"
        assert data["user_id"] == "user_test_123"

    @pytest.mark.asyncio
    async def test_sync_requires_authentication(self, unauthenticated_async_client):
        """Sync requires authentication."""
        response = await unauthenticated_async_client.post("/api/v1/users/sync")

        # Should fail without auth (403 or 401)
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_sync_user_persisted_to_database(self, async_client, db_session):
        """Sync actually persists user to database."""
        response = await async_client.post("/api/v1/users/sync")
        assert response.status_code == 200

        # Verify in database
        result = await db_session.execute(select(User).where(User.id == "user_test_123"))
        user = result.scalar_one_or_none()
        assert user is not None
        assert user.id == "user_test_123"

    @pytest.mark.asyncio
    async def test_sync_returns_user_id_from_token(self, async_client):
        """Sync returns user_id extracted from JWT token."""
        response = await async_client.post("/api/v1/users/sync")

        assert response.status_code == 200
        data = response.json()
        # user_id should match the 'sub' claim from mock_user_payload fixture
        assert "user_id" in data
