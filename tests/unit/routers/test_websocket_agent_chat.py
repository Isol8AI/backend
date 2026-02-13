"""
Unit tests for WebSocket agent_chat message type routing and validation.

Tests the agent_chat message type added to POST /ws/message:
- Routing: agent_chat messages are accepted and background task is queued
- Validation: invalid messages send errors via Management API
- Schema: AgentChatWSRequest Pydantic model validation

Uses the same pattern as test_websocket_chat.py:
- httpx AsyncClient with ASGITransport for HTTP endpoint testing
- Mocked ConnectionService and ManagementApiClient to isolate route logic
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routers.websocket_chat import router


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the websocket router."""
    app = FastAPI()
    app.include_router(router, prefix="/ws")
    return app


@pytest.fixture
def mock_connection_service():
    """Mock ConnectionService for testing."""
    with patch("routers.websocket_chat.get_connection_service") as mock_getter:
        mock_service = MagicMock()
        mock_getter.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_management_api():
    """Mock ManagementApiClient for testing."""
    with patch("routers.websocket_chat.get_management_api_client") as mock_getter:
        mock_client = MagicMock()
        mock_client.send_message = MagicMock(return_value=True)
        mock_getter.return_value = mock_client
        yield mock_client


@pytest.fixture
def valid_agent_chat_message():
    """Create a valid agent_chat message payload."""
    return {
        "type": "agent_chat",
        "agent_name": "my-agent",
        "encrypted_message": {
            "ephemeral_public_key": "a" * 64,
            "iv": "b" * 32,
            "ciphertext": "c" * 64,
            "auth_tag": "d" * 32,
            "hkdf_salt": "e" * 64,
        },
        "client_transport_public_key": "f" * 64,
        "user_public_key": "f" * 64,
    }


@pytest.fixture
def connected_user(mock_connection_service):
    """Set up mock connection service to return a connected user."""
    mock_connection_service.get_connection.return_value = {
        "user_id": "test-user-456",
        "org_id": None,
    }
    return "test-user-456"


class TestAgentChatMessageRouting:
    """Tests that the ws_message endpoint correctly routes agent_chat messages."""

    @pytest.mark.asyncio
    async def test_agent_chat_message_accepted(
        self, test_app, mock_connection_service, mock_management_api, valid_agent_chat_message, connected_user
    ):
        """Send valid agent_chat message, verify 200 response.

        A valid agent_chat message should be accepted and return 200.
        The background task is queued but not awaited in the HTTP handler.
        We mock the background processor to prevent it from running (it needs
        a real database); this test only verifies routing and validation.
        """
        with patch("routers.websocket_chat._process_agent_chat_background") as mock_bg:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    "/ws/message",
                    headers={"x-connection-id": "test-conn-123"},
                    json=valid_agent_chat_message,
                )

        assert response.status_code == 200
        # No error should be sent via Management API for a valid message
        mock_management_api.send_message.assert_not_called()
        # Background task should have been queued (called by BackgroundTasks)
        mock_bg.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_chat_invalid_format_sends_error(
        self, test_app, mock_connection_service, mock_management_api, connected_user
    ):
        """Send agent_chat with missing fields, verify error sent via Management API."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/ws/message",
                headers={"x-connection-id": "test-conn-123"},
                json={
                    "type": "agent_chat",
                    # All required fields missing
                },
            )

        # HTTP always returns 200 for accepted messages
        assert response.status_code == 200

        # Error should be sent via Management API
        mock_management_api.send_message.assert_called_once()
        call_args = mock_management_api.send_message.call_args
        assert call_args[0][0] == "test-conn-123"
        assert call_args[0][1]["type"] == "error"
        assert "invalid message format" in call_args[0][1]["message"].lower()

    @pytest.mark.asyncio
    async def test_agent_chat_missing_agent_name_sends_error(
        self, test_app, mock_connection_service, mock_management_api, connected_user
    ):
        """Missing agent_name field should send error via Management API."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/ws/message",
                headers={"x-connection-id": "test-conn-123"},
                json={
                    "type": "agent_chat",
                    # agent_name missing
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        "iv": "b" * 32,
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64,
                    },
                    "client_transport_public_key": "f" * 64,
                    "user_public_key": "f" * 64,
                },
            )

        assert response.status_code == 200

        mock_management_api.send_message.assert_called_once()
        call_args = mock_management_api.send_message.call_args
        assert call_args[0][0] == "test-conn-123"
        assert call_args[0][1]["type"] == "error"

    @pytest.mark.asyncio
    async def test_agent_chat_missing_encrypted_message_sends_error(
        self, test_app, mock_connection_service, mock_management_api, connected_user
    ):
        """Missing encrypted_message field should send error via Management API."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/ws/message",
                headers={"x-connection-id": "test-conn-123"},
                json={
                    "type": "agent_chat",
                    "agent_name": "my-agent",
                    # encrypted_message missing
                    "client_transport_public_key": "f" * 64,
                    "user_public_key": "f" * 64,
                },
            )

        assert response.status_code == 200

        mock_management_api.send_message.assert_called_once()
        call_args = mock_management_api.send_message.call_args
        assert call_args[0][0] == "test-conn-123"
        assert call_args[0][1]["type"] == "error"

    @pytest.mark.asyncio
    async def test_agent_chat_missing_transport_key_sends_error(
        self, test_app, mock_connection_service, mock_management_api, connected_user
    ):
        """Missing client_transport_public_key field should send error via Management API."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/ws/message",
                headers={"x-connection-id": "test-conn-123"},
                json={
                    "type": "agent_chat",
                    "agent_name": "my-agent",
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        "iv": "b" * 32,
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64,
                    },
                    # client_transport_public_key missing
                    "user_public_key": "f" * 64,
                },
            )

        assert response.status_code == 200

        mock_management_api.send_message.assert_called_once()
        call_args = mock_management_api.send_message.call_args
        assert call_args[0][0] == "test-conn-123"
        assert call_args[0][1]["type"] == "error"


class TestAgentChatWSRequestValidation:
    """Tests for the AgentChatWSRequest Pydantic schema."""

    def test_valid_request(self):
        """All fields present and valid should create a valid request."""
        from schemas.agent import AgentChatWSRequest
        from schemas.encryption import EncryptedPayloadSchema

        request = AgentChatWSRequest(
            agent_name="my-agent",
            encrypted_message=EncryptedPayloadSchema(
                ephemeral_public_key="a" * 64,
                iv="b" * 32,
                ciphertext="c" * 64,
                auth_tag="d" * 32,
                hkdf_salt="e" * 64,
            ),
            client_transport_public_key="f" * 64,
            user_public_key="f" * 64,
        )
        assert request.agent_name == "my-agent"
        assert request.client_transport_public_key == "f" * 64
        assert request.user_public_key == "f" * 64

    def test_agent_name_too_long(self):
        """agent_name > 50 chars should be rejected."""
        from pydantic import ValidationError
        from schemas.agent import AgentChatWSRequest
        from schemas.encryption import EncryptedPayloadSchema

        with pytest.raises(ValidationError) as exc_info:
            AgentChatWSRequest(
                agent_name="a" * 51,
                encrypted_message=EncryptedPayloadSchema(
                    ephemeral_public_key="a" * 64,
                    iv="b" * 32,
                    ciphertext="c" * 64,
                    auth_tag="d" * 32,
                    hkdf_salt="e" * 64,
                ),
                client_transport_public_key="f" * 64,
                user_public_key="f" * 64,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("agent_name",) for e in errors)

    def test_agent_name_empty(self):
        """Empty agent_name should be rejected."""
        from pydantic import ValidationError
        from schemas.agent import AgentChatWSRequest
        from schemas.encryption import EncryptedPayloadSchema

        with pytest.raises(ValidationError) as exc_info:
            AgentChatWSRequest(
                agent_name="",
                encrypted_message=EncryptedPayloadSchema(
                    ephemeral_public_key="a" * 64,
                    iv="b" * 32,
                    ciphertext="c" * 64,
                    auth_tag="d" * 32,
                    hkdf_salt="e" * 64,
                ),
                client_transport_public_key="f" * 64,
                user_public_key="f" * 64,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("agent_name",) for e in errors)
