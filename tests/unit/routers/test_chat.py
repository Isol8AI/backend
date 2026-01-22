"""Unit tests for encrypted chat router."""

import uuid
from datetime import datetime, timedelta

import pytest

from core.config import AVAILABLE_MODELS
from models.session import Session


class TestGetAvailableModels:
    """Tests for GET /api/v1/chat/models endpoint."""

    @pytest.mark.asyncio
    async def test_returns_available_models(self, async_client):
        """Returns list of available models with expected count."""
        response = await async_client.get("/api/v1/chat/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == len(AVAILABLE_MODELS)

    @pytest.mark.asyncio
    async def test_models_have_id_and_name(self, async_client):
        """Each model has id and name fields."""
        response = await async_client.get("/api/v1/chat/models")

        for model in response.json():
            assert "id" in model
            assert "name" in model

    @pytest.mark.asyncio
    async def test_models_endpoint_is_public(self, unauthenticated_async_client):
        """Models endpoint accessible without authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/models")

        assert response.status_code == 200
        assert len(response.json()) > 0


class TestGetSessions:
    """Tests for GET /api/v1/chat/sessions endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_new_user(self, async_client, test_user):
        """Returns empty list when user has no sessions."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_user_sessions(self, async_client, test_session):
        """Returns sessions belonging to the user."""
        response = await async_client.get("/api/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == test_session.id
        assert data["sessions"][0]["name"] == test_session.name
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_does_not_return_other_users_sessions(self, async_client, test_user, other_user_session):
        """Excludes sessions belonging to other users."""
        response = await async_client.get("/api/v1/chat/sessions")

        session_ids = [s["id"] for s in response.json()["sessions"]]
        assert other_user_session.id not in session_ids

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Sessions endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/sessions")
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_sessions_ordered_by_updated_at_desc(self, async_client, db_session, test_user):
        """Sessions are ordered by updated_at descending (newest first)."""
        old_session = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="Old Session")
        old_session.created_at = datetime.utcnow() - timedelta(days=1)
        old_session.updated_at = datetime.utcnow() - timedelta(days=1)
        db_session.add(old_session)

        new_session = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="New Session")
        new_session.created_at = datetime.utcnow()
        new_session.updated_at = datetime.utcnow()
        db_session.add(new_session)

        await db_session.flush()

        response = await async_client.get("/api/v1/chat/sessions")
        data = response.json()

        assert len(data["sessions"]) >= 2
        assert data["sessions"][0]["name"] == "New Session"


class TestGetSessionMessages:
    """Tests for GET /api/v1/chat/sessions/{session_id}/messages endpoint.

    Note: Messages are always encrypted in the zero-trust model.
    """

    @pytest.mark.asyncio
    async def test_returns_session_messages(self, async_client, test_session, test_message):
        """Returns encrypted messages for a session."""
        response = await async_client.get(f"/api/v1/chat/sessions/{test_session.id}/messages")

        assert response.status_code == 200
        data = response.json()
        # Returns a dict with session_id and messages
        assert data["session_id"] == test_session.id
        messages = data["messages"]
        assert len(messages) == 1
        # Messages have encrypted_content, not plaintext content
        assert "encrypted_content" in messages[0]
        assert "ephemeral_public_key" in messages[0]["encrypted_content"]

    @pytest.mark.asyncio
    async def test_returns_messages_in_chronological_order(self, async_client, test_session, test_conversation):
        """Messages are returned in chronological order."""
        response = await async_client.get(f"/api/v1/chat/sessions/{test_session.id}/messages")
        data = response.json()
        messages = data["messages"]

        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_session(self, async_client, test_user):
        """Returns 404 for non-existent session."""
        response = await async_client.get("/api/v1/chat/sessions/nonexistent-id/messages")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_returns_404_for_other_users_session(self, async_client, test_user, other_user_session):
        """Returns 404 when accessing another user's session."""
        response = await async_client.get(f"/api/v1/chat/sessions/{other_user_session.id}/messages")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/sessions/some-id/messages")
        assert response.status_code in [401, 403]


class TestEncryptedChatStream:
    """Tests for POST /api/v1/chat/stream encrypted endpoint.

    Note: The new encrypted chat API requires:
    1. User to have encryption keys set up
    2. Message to be encrypted to enclave's public key
    3. Response is encrypted back to user's public key
    """

    @pytest.mark.asyncio
    async def test_returns_400_if_user_has_no_encryption_keys(self, async_client, test_user):
        """Returns 400 if user hasn't set up encryption keys."""
        # User exists but has no encryption keys
        encrypted_message = {
            "ephemeral_public_key": "aa" * 32,
            "iv": "bb" * 16,
            "ciphertext": "cc" * 32,
            "auth_tag": "dd" * 16,
            "hkdf_salt": "ee" * 32,
        }
        response = await async_client.post(
            "/api/v1/chat/encrypted/stream",
            json={
                "model": AVAILABLE_MODELS[0]["id"],
                "encrypted_message": encrypted_message,
                "client_transport_public_key": "ff" * 32,  # 32 bytes as hex
            },
        )

        assert response.status_code == 400
        assert "encryption" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.post("/api/v1/chat/encrypted/stream", json={})
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_requires_model_field(self, async_client, test_user):
        """Request must include model field."""
        response = await async_client.post(
            "/api/v1/chat/encrypted/stream",
            json={
                "encrypted_message": {
                    "ephemeral_public_key": "aa" * 32,
                    "iv": "bb" * 16,
                    "ciphertext": "cc" * 32,
                    "auth_tag": "dd" * 16,
                    "hkdf_salt": "ee" * 32,
                }
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_requires_encrypted_message_field(self, async_client, test_user):
        """Request must include encrypted_message field."""
        response = await async_client.post(
            "/api/v1/chat/encrypted/stream",
            json={
                "model": AVAILABLE_MODELS[0]["id"],
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_validates_model_exists(self, async_client, db_session, test_user):
        """Returns 400 for invalid model ID."""
        # Add encryption keys to the test_user
        test_user.set_encryption_keys(
            public_key="aa" * 32,
            encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            recovery_encrypted_private_key="ff" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )
        await db_session.flush()

        response = await async_client.post(
            "/api/v1/chat/encrypted/stream",
            json={
                "model": "invalid-model-id",
                "encrypted_message": {
                    "ephemeral_public_key": "aa" * 32,
                    "iv": "bb" * 16,
                    "ciphertext": "cc" * 32,
                    "auth_tag": "dd" * 16,
                    "hkdf_salt": "ee" * 32,
                },
                "client_transport_public_key": "ff" * 32,  # 32 bytes as hex
            },
        )

        assert response.status_code == 400
        assert "model" in response.json()["detail"].lower()


class TestEnclaveInfo:
    """Tests for GET /api/v1/chat/enclave/info endpoint."""

    @pytest.mark.asyncio
    async def test_returns_enclave_public_key(self, async_client, test_user):
        """Returns enclave's public key for message encryption."""
        response = await async_client.get("/api/v1/chat/enclave/info")

        assert response.status_code == 200
        data = response.json()
        assert "enclave_public_key" in data
        assert len(data["enclave_public_key"]) == 64  # 32 bytes as hex

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/enclave/info")
        assert response.status_code in [401, 403]


class TestEncryptionStatus:
    """Tests for GET /api/v1/chat/encryption-status endpoint."""

    @pytest.mark.asyncio
    async def test_returns_false_if_no_encryption_keys(self, async_client, test_user):
        """Returns can_send_encrypted=False if user has no keys."""
        response = await async_client.get("/api/v1/chat/encryption-status")

        assert response.status_code == 200
        data = response.json()
        assert data["can_send_encrypted"] is False
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_returns_true_if_has_encryption_keys(self, async_client, db_session, test_user):
        """Returns can_send_encrypted=True if user has keys."""
        # Add encryption keys to the test_user (which async_client is authenticated as)
        test_user.set_encryption_keys(
            public_key="aa" * 32,
            encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            recovery_encrypted_private_key="ff" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )
        await db_session.flush()

        response = await async_client.get("/api/v1/chat/encryption-status")

        assert response.status_code == 200
        data = response.json()
        assert data["can_send_encrypted"] is True
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/chat/encryption-status")
        assert response.status_code in [401, 403]


class TestCreateSession:
    """Tests for POST /api/v1/chat/sessions endpoint."""

    @pytest.mark.asyncio
    async def test_creates_session(self, async_client, test_user):
        """Creates a new chat session."""
        response = await async_client.post("/api/v1/chat/sessions", json={"name": "Test Chat"})

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Chat"
        assert data["user_id"] == test_user.id
        assert data["org_id"] is None

    @pytest.mark.asyncio
    async def test_creates_session_with_default_name(self, async_client, test_user):
        """Creates session with default name if not provided."""
        response = await async_client.post("/api/v1/chat/sessions", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Chat"

    @pytest.mark.asyncio
    async def test_requires_authentication(self, unauthenticated_async_client):
        """Endpoint requires authentication."""
        response = await unauthenticated_async_client.post("/api/v1/chat/sessions", json={})
        assert response.status_code in [401, 403]
