"""Unit tests for context router.

The context router provides a unified interface for getting/setting context
based on the current auth context (personal vs organization).
"""
import pytest
from sqlalchemy import select

from models.context_store import ContextStore


class TestGetContext:
    """Tests for GET /api/v1/context endpoint."""

    @pytest.mark.asyncio
    async def test_get_personal_context_empty_initially(self, async_client, db_session, test_user):
        """Personal context is empty initially."""
        response = await async_client.get("/api/v1/context/")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_type"] == "user"
        assert data["context_data"] is None or data["context_data"] == {}

    @pytest.mark.asyncio
    async def test_get_personal_context_after_update(self, async_client, db_session, test_user):
        """Can retrieve personal context after setting it."""
        # First set some context
        context_data = {"preferences": {"theme": "dark"}}
        await async_client.put(
            "/api/v1/context/",
            json={"context_data": context_data}
        )

        # Then retrieve it
        response = await async_client.get("/api/v1/context/")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_type"] == "user"
        assert data["context_data"] == context_data

    @pytest.mark.asyncio
    async def test_get_org_context_in_org_mode(
        self, async_client_org, db_session, test_user, test_organization, test_membership
    ):
        """Returns org context when in organization mode."""
        response = await async_client_org.get("/api/v1/context/")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_type"] == "org"

    @pytest.mark.asyncio
    async def test_get_context_requires_authentication(self, unauthenticated_async_client):
        """Get context requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/context/")
        assert response.status_code in [401, 403]


class TestUpdateContext:
    """Tests for PUT /api/v1/context endpoint."""

    @pytest.mark.asyncio
    async def test_update_personal_context(self, async_client, db_session, test_user):
        """Can update personal context."""
        context_data = {"preferences": {"theme": "dark"}, "notes": "test note"}

        response = await async_client.put(
            "/api/v1/context/",
            json={"context_data": context_data}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == context_data

        # Verify persisted to database
        result = await db_session.execute(
            select(ContextStore).where(
                ContextStore.owner_type == "user",
                ContextStore.owner_id == test_user.id
            )
        )
        store = result.scalar_one_or_none()
        assert store is not None
        assert store.context_data == context_data

    @pytest.mark.asyncio
    async def test_update_personal_context_overwrites(self, async_client, db_session, test_user):
        """Updating personal context overwrites previous values."""
        # First update
        await async_client.put(
            "/api/v1/context/",
            json={"context_data": {"old": "data"}}
        )

        # Second update
        new_data = {"new": "data"}
        response = await async_client.put(
            "/api/v1/context/",
            json={"context_data": new_data}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == new_data
        assert "old" not in data["context_data"]

    @pytest.mark.asyncio
    async def test_update_org_context_denied_for_member(
        self, async_client_org, db_session, test_user, test_organization, test_membership
    ):
        """Non-admin org member cannot update org context."""
        response = await async_client_org.put(
            "/api/v1/context/",
            json={"context_data": {"test": "data"}}
        )

        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_update_org_context_allowed_for_admin(
        self, async_client_org_admin, db_session, test_user, test_organization, test_admin_membership
    ):
        """Org admin can update org context."""
        context_data = {"shared_settings": {"api_key": "encrypted_key"}}

        response = await async_client_org_admin.put(
            "/api/v1/context/",
            json={"context_data": context_data}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == context_data

    @pytest.mark.asyncio
    async def test_update_context_requires_authentication(self, unauthenticated_async_client):
        """Update context requires authentication."""
        response = await unauthenticated_async_client.put(
            "/api/v1/context/",
            json={"context_data": {"test": "data"}}
        )
        assert response.status_code in [401, 403]


class TestContextIsolation:
    """Tests for context isolation between users and orgs."""

    @pytest.mark.asyncio
    async def test_personal_context_isolated_between_users(
        self, async_client, db_session, test_user, other_user
    ):
        """Each user has their own personal context."""
        # Set context for test_user
        my_context = {"my": "data"}
        await async_client.put(
            "/api/v1/context/",
            json={"context_data": my_context}
        )

        # Create context for other_user directly in database
        other_context = ContextStore(
            id="ctx_other",
            owner_type="user",
            owner_id=other_user.id,
            context_data={"other": "user_data"}
        )
        db_session.add(other_context)
        await db_session.flush()

        # Verify test_user only sees their context
        response = await async_client.get("/api/v1/context/")
        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == my_context
        assert "other" not in data["context_data"]

    @pytest.mark.asyncio
    async def test_personal_and_org_context_separate(
        self, async_client, db_session, test_user, test_organization
    ):
        """Personal and org context are stored separately."""
        # Set personal context
        personal_context = {"personal": "data"}
        await async_client.put(
            "/api/v1/context/",
            json={"context_data": personal_context}
        )

        # Create org context directly in database
        org_context_store = ContextStore(
            id="ctx_org_test",
            owner_type="org",
            owner_id=test_organization.id,
            context_data={"org": "data"}
        )
        db_session.add(org_context_store)
        await db_session.flush()

        # Verify personal context unchanged (not overwritten by org context)
        response = await async_client.get("/api/v1/context/")
        assert response.json()["owner_type"] == "user"
        assert response.json()["context_data"] == personal_context

        # Verify both exist in database
        result = await db_session.execute(select(ContextStore))
        stores = result.scalars().all()
        assert len(stores) == 2
        owner_types = [s.owner_type for s in stores]
        assert "user" in owner_types
        assert "org" in owner_types
