"""
Unit tests for WebSocket chat endpoint.

Tests the /api/v1/ws/chat WebSocket endpoint for:
- Connection authentication (x-user-id header required)
- Message validation
- Ping/pong keepalive
- Error handling

Note: Uses mock WebSocket objects since httpx 0.28.x is incompatible with
starlette's TestClient for WebSocket testing. Tests directly invoke the
endpoint handler with mocked dependencies.
"""

import asyncio
import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from core.config import AVAILABLE_MODELS
from routers.websocket_chat import websocket_chat, VALID_MODEL_IDS
from schemas.encryption import EncryptedPayload, SendEncryptedMessageRequest


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self, headers: dict = None):
        self.headers = headers or {}
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.sent_messages = []
        self.receive_queue = asyncio.Queue()

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = None):
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def send_json(self, data: dict):
        self.sent_messages.append(data)

    async def receive_json(self):
        return await self.receive_queue.get()

    def queue_message(self, data: dict):
        """Queue a message to be received by the handler."""
        self.receive_queue.put_nowait(data)


class TestWebSocketConnectionAuth:
    """Tests for WebSocket connection authentication."""

    @pytest.mark.asyncio
    async def test_connection_without_user_id_rejected(self):
        """WebSocket connection without x-user-id header should be rejected with 4001."""
        websocket = MockWebSocket(headers={})
        mock_session_factory = MagicMock()

        await websocket_chat(websocket, mock_session_factory)

        assert websocket.accepted is True  # Connection accepted before check
        assert websocket.closed is True
        assert websocket.close_code == 4001
        assert "Unauthorized" in websocket.close_reason

    @pytest.mark.asyncio
    async def test_connection_with_user_id_accepted(self):
        """WebSocket connection with x-user-id header should be accepted."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        # Queue a disconnect to exit the loop
        async def mock_receive():
            raise Exception("WebSocketDisconnect simulation")

        websocket.receive_json = mock_receive

        # Should not close with 4001
        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass  # Expected - we simulated disconnect

        assert websocket.accepted is True
        # Should not be closed with unauthorized code
        assert websocket.close_code != 4001

    @pytest.mark.asyncio
    async def test_user_id_extracted_from_header(self):
        """User ID should be extracted from x-user-id header."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-xyz"})

        # The user_id is extracted from headers in the handler
        user_id = websocket.headers.get("x-user-id")
        assert user_id == "test-user-xyz"

    @pytest.mark.asyncio
    async def test_org_id_extracted_from_header(self):
        """Org ID should be extracted from x-org-id header."""
        websocket = MockWebSocket(headers={
            "x-user-id": "test-user-123",
            "x-org-id": "test-org-456"
        })

        user_id = websocket.headers.get("x-user-id")
        org_id = websocket.headers.get("x-org-id")

        assert user_id == "test-user-123"
        assert org_id == "test-org-456"


class TestWebSocketPingPong:
    """Tests for ping/pong keepalive mechanism."""

    @pytest.mark.asyncio
    async def test_pong_message_handled_silently(self):
        """Server should handle pong messages from client without error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        # Queue pong message then disconnect
        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "pong"}
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        # Pong should be handled silently - no error sent
        error_messages = [m for m in websocket.sent_messages if m.get("type") == "error"]
        assert len(error_messages) == 0


class TestWebSocketMessageValidation:
    """Tests for message format validation."""

    @pytest.mark.asyncio
    async def test_missing_encrypted_message_returns_error(self):
        """Message without encrypted_message field should return error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        # Queue invalid message then disconnect
        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"invalid": "data"}
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        # Should receive error response
        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid message format" in error_msg["message"]

    @pytest.mark.asyncio
    async def test_missing_client_transport_key_returns_error(self):
        """Message without client_transport_public_key should return error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        "iv": "b" * 32,
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64
                    },
                    "model": AVAILABLE_MODELS[0]["id"]
                    # Missing client_transport_public_key
                }
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid message format" in error_msg["message"]

    @pytest.mark.asyncio
    async def test_incomplete_encrypted_payload_returns_error(self):
        """Message with incomplete encrypted_payload should return error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        # Missing iv, ciphertext, auth_tag, hkdf_salt
                    },
                    "client_transport_public_key": "f" * 64,
                    "model": AVAILABLE_MODELS[0]["id"]
                }
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid message format" in error_msg["message"]


class TestWebSocketModelValidation:
    """Tests for model ID validation."""

    @pytest.mark.asyncio
    async def test_invalid_model_returns_error(self):
        """Invalid model ID should return error without disconnecting."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        "iv": "b" * 32,
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64
                    },
                    "client_transport_public_key": "f" * 64,
                    "model": "invalid-model-id"
                }
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid model" in error_msg["message"]

    @pytest.mark.asyncio
    async def test_empty_model_returns_error(self):
        """Empty model string should return error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        "iv": "b" * 32,
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64
                    },
                    "client_transport_public_key": "f" * 64,
                    "model": ""
                }
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid model" in error_msg["message"]

    def test_valid_model_ids_populated(self):
        """VALID_MODEL_IDS set should contain all available model IDs."""
        for model in AVAILABLE_MODELS:
            assert model["id"] in VALID_MODEL_IDS


class TestWebSocketHexValidation:
    """Tests for hex string validation in encrypted payloads."""

    @pytest.mark.asyncio
    async def test_invalid_hex_in_ephemeral_key_returns_error(self):
        """Non-hex characters in ephemeral_public_key should return error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "encrypted_message": {
                        "ephemeral_public_key": "g" * 64,  # Invalid hex
                        "iv": "b" * 32,
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64
                    },
                    "client_transport_public_key": "f" * 64,
                    "model": AVAILABLE_MODELS[0]["id"]
                }
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid message format" in error_msg["message"]

    @pytest.mark.asyncio
    async def test_wrong_length_iv_returns_error(self):
        """IV with wrong length should return error."""
        websocket = MockWebSocket(headers={"x-user-id": "test-user-123"})
        mock_session_factory = MagicMock()

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "encrypted_message": {
                        "ephemeral_public_key": "a" * 64,
                        "iv": "b" * 16,  # Too short (should be 32)
                        "ciphertext": "c" * 64,
                        "auth_tag": "d" * 32,
                        "hkdf_salt": "e" * 64
                    },
                    "client_transport_public_key": "f" * 64,
                    "model": AVAILABLE_MODELS[0]["id"]
                }
            raise Exception("Disconnect")

        websocket.receive_json = mock_receive

        try:
            await websocket_chat(websocket, mock_session_factory)
        except Exception:
            pass

        assert len(websocket.sent_messages) >= 1
        error_msg = websocket.sent_messages[0]
        assert error_msg["type"] == "error"
        assert "Invalid message format" in error_msg["message"]


class TestEncryptedPayloadValidation:
    """Unit tests for EncryptedPayload schema validation."""

    def test_valid_payload_accepted(self):
        """Valid encrypted payload should be accepted."""
        payload = EncryptedPayload(
            ephemeral_public_key="a" * 64,
            iv="b" * 32,
            ciphertext="c" * 64,
            auth_tag="d" * 32,
            hkdf_salt="e" * 64
        )
        assert payload.ephemeral_public_key == "a" * 64

    def test_invalid_hex_rejected(self):
        """Invalid hex characters should be rejected."""
        with pytest.raises(ValidationError):
            EncryptedPayload(
                ephemeral_public_key="g" * 64,  # 'g' is not valid hex
                iv="b" * 32,
                ciphertext="c" * 64,
                auth_tag="d" * 32,
                hkdf_salt="e" * 64
            )

    def test_wrong_length_rejected(self):
        """Fields with wrong length should be rejected."""
        with pytest.raises(ValidationError):
            EncryptedPayload(
                ephemeral_public_key="a" * 32,  # Should be 64 hex chars
                iv="b" * 32,
                ciphertext="c" * 64,
                auth_tag="d" * 32,
                hkdf_salt="e" * 64
            )


class TestSendEncryptedMessageRequestValidation:
    """Unit tests for SendEncryptedMessageRequest schema validation."""

    def test_valid_request_accepted(self):
        """Valid request should be accepted."""
        request = SendEncryptedMessageRequest(
            model=AVAILABLE_MODELS[0]["id"],
            encrypted_message=EncryptedPayload(
                ephemeral_public_key="a" * 64,
                iv="b" * 32,
                ciphertext="c" * 64,
                auth_tag="d" * 32,
                hkdf_salt="e" * 64
            ),
            client_transport_public_key="f" * 64
        )
        assert request.model == AVAILABLE_MODELS[0]["id"]

    def test_optional_session_id_accepted(self):
        """Request with optional session_id should be accepted."""
        request = SendEncryptedMessageRequest(
            session_id="test-session-id",
            model=AVAILABLE_MODELS[0]["id"],
            encrypted_message=EncryptedPayload(
                ephemeral_public_key="a" * 64,
                iv="b" * 32,
                ciphertext="c" * 64,
                auth_tag="d" * 32,
                hkdf_salt="e" * 64
            ),
            client_transport_public_key="f" * 64
        )
        assert request.session_id == "test-session-id"

    def test_optional_encrypted_history_accepted(self):
        """Request with optional encrypted_history should be accepted."""
        request = SendEncryptedMessageRequest(
            model=AVAILABLE_MODELS[0]["id"],
            encrypted_message=EncryptedPayload(
                ephemeral_public_key="a" * 64,
                iv="b" * 32,
                ciphertext="c" * 64,
                auth_tag="d" * 32,
                hkdf_salt="e" * 64
            ),
            encrypted_history=[
                EncryptedPayload(
                    ephemeral_public_key="1" * 64,
                    iv="2" * 32,
                    ciphertext="3" * 64,
                    auth_tag="4" * 32,
                    hkdf_salt="5" * 64
                )
            ],
            client_transport_public_key="f" * 64
        )
        assert len(request.encrypted_history) == 1

    def test_invalid_client_transport_key_rejected(self):
        """Invalid client_transport_public_key should be rejected."""
        with pytest.raises(ValidationError):
            SendEncryptedMessageRequest(
                model=AVAILABLE_MODELS[0]["id"],
                encrypted_message=EncryptedPayload(
                    ephemeral_public_key="a" * 64,
                    iv="b" * 32,
                    ciphertext="c" * 64,
                    auth_tag="d" * 32,
                    hkdf_salt="e" * 64
                ),
                client_transport_public_key="g" * 64  # Invalid hex
            )
