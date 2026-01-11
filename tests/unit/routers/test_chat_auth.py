"""Unit tests for chat authorization with organization scoping.

These tests verify that:
- Personal sessions are only visible to owner
- Org sessions are only visible to owner (not other org members)
- Sessions are scoped to the current context (personal vs org)
"""
import pytest

from models.session import Session


class TestPersonalSessionAuthorization:
    """Tests for personal session (no org) authorization."""

    @pytest.mark.asyncio
    async def test_personal_sessions_only_visible_to_owner(
        self, async_client, db_session, test_user, test_session
    ):
        """Personal sessions are visible to their owner."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        assert sessions[0]["id"] == test_session.id

    @pytest.mark.asyncio
    async def test_personal_sessions_not_visible_to_other_users(
        self, async_client, db_session, test_user, other_user_session
    ):
        """Other users' sessions are not visible."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        # test_user should not see other_user's session
        session_ids = [s["id"] for s in sessions]
        assert other_user_session.id not in session_ids

    @pytest.mark.asyncio
    async def test_personal_sessions_not_visible_in_org_context(
        self, async_client_org, db_session, test_user, test_session, test_membership
    ):
        """Personal sessions are not visible when in org context."""
        response = await async_client_org.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        # Personal session (org_id=None) should not appear in org context
        session_ids = [s["id"] for s in sessions]
        assert test_session.id not in session_ids


class TestOrgSessionAuthorization:
    """Tests for organization session authorization."""

    @pytest.mark.asyncio
    async def test_org_sessions_only_visible_to_owner(
        self, async_client_org, db_session, test_user, test_organization, test_org_session, test_membership
    ):
        """Org sessions are visible to their owner in org context."""
        response = await async_client_org.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        session_ids = [s["id"] for s in sessions]
        assert test_org_session.id in session_ids

    @pytest.mark.asyncio
    async def test_org_sessions_not_visible_to_other_org_members(
        self, async_client_org, db_session, test_user, other_user, test_organization,
        other_user_org_session, test_membership
    ):
        """Other org members' sessions are not visible (chats are private)."""
        response = await async_client_org.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        # test_user should not see other_user's org session
        session_ids = [s["id"] for s in sessions]
        assert other_user_org_session.id not in session_ids

    @pytest.mark.asyncio
    async def test_org_sessions_not_visible_in_personal_context(
        self, async_client, db_session, test_user, test_organization, test_org_session
    ):
        """Org sessions are not visible when in personal context."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        # Org session should not appear in personal context
        session_ids = [s["id"] for s in sessions]
        assert test_org_session.id not in session_ids

    @pytest.mark.asyncio
    async def test_cannot_access_sessions_from_different_org(
        self, async_client_org, db_session, test_user, other_organization, test_membership
    ):
        """Sessions from a different org are not visible."""
        # Create a session in a different org
        different_org_session = Session(
            id="session_different_org",
            user_id=test_user.id,
            org_id=other_organization.id,
            name="Different Org Session"
        )
        db_session.add(different_org_session)
        await db_session.flush()

        response = await async_client_org.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        session_ids = [s["id"] for s in sessions]
        # Session from org_other_456 should not appear in org_test_123 context
        assert "session_different_org" not in session_ids


class TestCreateSessionWithContext:
    """Tests for session creation with org context."""

    @pytest.mark.asyncio
    async def test_create_session_in_personal_context(
        self, async_client, db_session, test_user
    ):
        """New session in personal context has no org_id."""
        # This test will be implemented when we update the chat router
        # to use org_id from auth context
        pass

    @pytest.mark.asyncio
    async def test_create_session_in_org_context(
        self, async_client_org, db_session, test_user, test_organization, test_membership
    ):
        """New session in org context has org_id set."""
        # This test will be implemented when we update the chat router
        pass


class TestSessionMessageAuthorization:
    """Tests for session message access authorization."""

    @pytest.mark.asyncio
    async def test_can_access_own_session_messages(
        self, async_client, db_session, test_user, test_session, test_message
    ):
        """User can access messages in their own session."""
        response = await async_client.get(f"/api/v1/chat/sessions/{test_session.id}/messages")

        assert response.status_code == 200
        messages = response.json()
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_cannot_access_other_users_session_messages(
        self, async_client, db_session, test_user, other_user_session
    ):
        """User cannot access messages in another user's session."""
        response = await async_client.get(f"/api/v1/chat/sessions/{other_user_session.id}/messages")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cannot_access_personal_session_in_org_context(
        self, async_client_org, db_session, test_user, test_session, test_membership
    ):
        """Cannot access personal session when in org context."""
        response = await async_client_org.get(f"/api/v1/chat/sessions/{test_session.id}/messages")

        # Should return 404 because session is personal but context is org
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_can_access_org_session_in_org_context(
        self, async_client_org, db_session, test_user, test_organization, test_org_session, test_membership
    ):
        """Can access org session when in same org context."""
        response = await async_client_org.get(f"/api/v1/chat/sessions/{test_org_session.id}/messages")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cannot_access_org_session_in_personal_context(
        self, async_client, db_session, test_user, test_org_session
    ):
        """Cannot access org session when in personal context."""
        response = await async_client.get(f"/api/v1/chat/sessions/{test_org_session.id}/messages")

        assert response.status_code == 404
