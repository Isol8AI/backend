"""Unit tests for chat router."""
import pytest
import json
from unittest.mock import patch, AsyncMock

from models.session import Session
from models.message import Message
from core.config import AVAILABLE_MODELS


class TestGetAvailableModels:
    """Tests for GET /api/v1/chat/models endpoint.

    Note: This endpoint is intentionally public (no authentication required)
    to allow the frontend to display available models before user login.
    Model names are not considered sensitive information.
    """

    @pytest.mark.asyncio
    async def test_returns_available_models(self, async_client):
        """Returns list of available models."""
        response = await async_client.get("/api/v1/chat/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == len(AVAILABLE_MODELS)

    @pytest.mark.asyncio
    async def test_models_have_id_and_name(self, async_client):
        """Each model has id and name fields."""
        response = await async_client.get("/api/v1/chat/models")
        data = response.json()

        for model in data:
            assert "id" in model
            assert "name" in model

    @pytest.mark.asyncio
    async def test_models_endpoint_is_public(self, unauthenticated_async_client):
        """Models endpoint is intentionally public - no auth required.

        This allows the frontend to show available models on the login page
        or before user authentication completes.
        """
        response = await unauthenticated_async_client.get("/api/v1/chat/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestGetSessions:
    """Tests for GET /api/v1/chat/sessions endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_new_user(self, async_client, test_user):
        """Returns empty list when user has no sessions."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_returns_user_sessions(self, async_client, test_session):
        """Returns sessions belonging to the user."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == test_session.id
        assert data[0]["name"] == test_session.name

    @pytest.mark.asyncio
    async def test_does_not_return_other_users_sessions(self, async_client, test_user, other_user_session):
        """Does not return sessions belonging to other users."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        # Should not include other_user_session
        session_ids = [s["id"] for s in data]
        assert other_user_session.id not in session_ids

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Sessions endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/sessions")

        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_sessions_ordered_by_created_at_desc(self, async_client, db_session, test_user):
        """Sessions are ordered by created_at descending (newest first)."""
        import uuid
        from datetime import datetime, timedelta

        # Create sessions with different timestamps
        old_session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            name="Old Session"
        )
        old_session.created_at = datetime.utcnow() - timedelta(days=1)
        db_session.add(old_session)

        new_session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            name="New Session"
        )
        new_session.created_at = datetime.utcnow()
        db_session.add(new_session)

        await db_session.flush()

        response = await async_client.get("/api/v1/chat/sessions")
        data = response.json()

        assert len(data) >= 2
        # First session should be newer
        assert data[0]["name"] == "New Session"


class TestGetSessionMessages:
    """Tests for GET /api/v1/chat/sessions/{session_id}/messages endpoint."""

    @pytest.mark.asyncio
    async def test_returns_session_messages(self, async_client, test_session, test_message):
        """Returns messages for a session."""
        response = await async_client.get(f"/api/v1/chat/sessions/{test_session.id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == test_message.content

    @pytest.mark.asyncio
    async def test_returns_messages_in_chronological_order(self, async_client, test_session, test_conversation):
        """Messages are returned in chronological order."""
        response = await async_client.get(f"/api/v1/chat/sessions/{test_session.id}/messages")

        assert response.status_code == 200
        data = response.json()

        # Verify order: user, assistant, user
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "Hello!"
        assert data[1]["role"] == "assistant"
        assert data[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_session(self, async_client, test_user):
        """Returns 404 for non-existent session."""
        response = await async_client.get("/api/v1/chat/sessions/nonexistent-id/messages")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_returns_404_for_other_users_session(self, async_client, test_user, other_user_session):
        """Returns 404 when trying to access another user's session."""
        response = await async_client.get(f"/api/v1/chat/sessions/{other_user_session.id}/messages")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/sessions/some-id/messages")

        assert response.status_code in [401, 403]


class TestChatStream:
    """Tests for POST /api/v1/chat/stream endpoint."""

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM streaming response."""
        async def _mock_stream(*args, **kwargs):
            yield "Hello"
            yield " world"
            yield "!"
        return _mock_stream

    @pytest.mark.asyncio
    async def test_returns_404_if_user_not_synced(self, async_client):
        """Returns 404 if user not synced to database."""
        response = await async_client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"}
        )

        assert response.status_code == 404
        assert "sync" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_creates_new_session_if_none_provided(self, async_client, test_user, mock_llm_response):
        """Creates new session when session_id not provided."""
        with patch("routers.chat.llm_service.generate_response_stream", mock_llm_response):
            response = await async_client.post(
                "/api/v1/chat/stream",
                json={"message": "Hello, create a new session"}
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_stream_returns_sse_format(self, async_client, test_user, mock_llm_response):
        """Response is in Server-Sent Events format."""
        with patch("routers.chat.llm_service.generate_response_stream", mock_llm_response):
            response = await async_client.post(
                "/api/v1/chat/stream",
                json={"message": "Test"}
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    @pytest.mark.asyncio
    async def test_stream_sends_session_event_first(self, async_client, test_user, mock_llm_response):
        """First event contains session_id."""
        with patch("routers.chat.llm_service.generate_response_stream", mock_llm_response):
            response = await async_client.post(
                "/api/v1/chat/stream",
                json={"message": "Test"}
            )

            content = response.text
            lines = [l for l in content.split("\n") if l.startswith("data:")]

            assert len(lines) > 0
            first_event = json.loads(lines[0].replace("data: ", ""))
            assert first_event["type"] == "session"
            assert "session_id" in first_event

    @pytest.mark.asyncio
    async def test_stream_sends_content_events(self, async_client, test_user, mock_llm_response):
        """Content chunks are sent as content events."""
        with patch("routers.chat.llm_service.generate_response_stream", mock_llm_response):
            response = await async_client.post(
                "/api/v1/chat/stream",
                json={"message": "Test"}
            )

            content = response.text
            lines = [l for l in content.split("\n") if l.startswith("data:")]

            content_events = []
            for line in lines:
                event = json.loads(line.replace("data: ", ""))
                if event.get("type") == "content":
                    content_events.append(event["content"])

            # Should have Hello, " world", "!"
            assert len(content_events) == 3
            assert "".join(content_events) == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_sends_done_event_last(self, async_client, test_user, mock_llm_response):
        """Last event is 'done' type."""
        with patch("routers.chat.llm_service.generate_response_stream", mock_llm_response):
            response = await async_client.post(
                "/api/v1/chat/stream",
                json={"message": "Test"}
            )

            content = response.text
            lines = [l for l in content.split("\n") if l.startswith("data:")]

            last_event = json.loads(lines[-1].replace("data: ", ""))
            assert last_event["type"] == "done"

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"}
        )

        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_requires_message_field(self, async_client, test_user):
        """Request must include message field."""
        response = await async_client.post(
            "/api/v1/chat/stream",
            json={}
        )

        assert response.status_code == 422  # Validation error
