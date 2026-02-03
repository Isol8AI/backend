"""Tests for AgentHandler enclave integration."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from core.crypto import EncryptedPayload, generate_x25519_keypair
from core.enclave.agent_handler import AgentHandler, AgentMessageRequest, AgentMessageResponse
from core.enclave.mock_enclave import AgentRunResponse


class TestAgentHandler:
    """Test AgentHandler enclave integration."""

    @pytest.fixture
    def mock_enclave(self):
        """Create mock enclave that implements run_agent."""
        enclave = MagicMock()
        keypair = generate_x25519_keypair()
        enclave._keypair = keypair
        enclave.get_info.return_value = MagicMock(enclave_public_key=keypair.public_key)
        # Mock run_agent to return a successful response
        enclave.run_agent = AsyncMock(
            return_value=AgentRunResponse(
                success=True,
                encrypted_response=EncryptedPayload(
                    ephemeral_public_key=b"x" * 32,
                    iv=b"y" * 16,
                    ciphertext=b"encrypted_response",
                    auth_tag=b"z" * 16,
                    hkdf_salt=b"s" * 32,
                ),
                encrypted_state=EncryptedPayload(
                    ephemeral_public_key=b"a" * 32,
                    iv=b"b" * 16,
                    ciphertext=b"encrypted_state",
                    auth_tag=b"c" * 16,
                    hkdf_salt=b"d" * 32,
                ),
            )
        )
        return enclave

    @pytest.fixture
    def handler(self, mock_enclave):
        """Create handler with mocked enclave."""
        return AgentHandler(enclave=mock_enclave)

    @pytest.fixture
    def user_keypair(self):
        """Generate user keypair for testing."""
        return generate_x25519_keypair()

    @pytest.fixture
    def sample_encrypted_message(self):
        """Create a sample encrypted message."""
        return EncryptedPayload(
            ephemeral_public_key=b"m" * 32,
            iv=b"n" * 16,
            ciphertext=b"Hello!",
            auth_tag=b"o" * 16,
            hkdf_salt=b"p" * 32,
        )

    @pytest.mark.asyncio
    async def test_process_message_new_user(self, handler, mock_enclave, sample_encrypted_message, user_keypair):
        """Test processing message for a new user (no existing state)."""
        request = AgentMessageRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,  # New user
            user_public_key=user_keypair.public_key,
            model="claude-3-5-sonnet",
        )

        response = await handler.process_message(request)

        assert response.success is True
        assert response.encrypted_response is not None
        assert response.encrypted_state is not None

        # Verify enclave.run_agent was called with correct parameters
        mock_enclave.run_agent.assert_called_once_with(
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            agent_name="luna",
            model="claude-3-5-sonnet",
        )

    @pytest.mark.asyncio
    async def test_process_message_existing_user(self, handler, mock_enclave, sample_encrypted_message, user_keypair):
        """Test processing message for user with existing state."""
        existing_state = EncryptedPayload(
            ephemeral_public_key=b"e" * 32,
            iv=b"f" * 16,
            ciphertext=b"existing_state",
            auth_tag=b"g" * 16,
            hkdf_salt=b"h" * 32,
        )

        request = AgentMessageRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=existing_state,
            user_public_key=user_keypair.public_key,
            model="claude-3-5-sonnet",
        )

        response = await handler.process_message(request)

        assert response.success is True

        # Verify existing state was passed to enclave
        mock_enclave.run_agent.assert_called_once_with(
            encrypted_message=sample_encrypted_message,
            encrypted_state=existing_state,
            user_public_key=user_keypair.public_key,
            agent_name="luna",
            model="claude-3-5-sonnet",
        )

    @pytest.mark.asyncio
    async def test_process_message_enclave_error(self, handler, mock_enclave, sample_encrypted_message, user_keypair):
        """Test handling enclave errors gracefully."""
        mock_enclave.run_agent = AsyncMock(
            return_value=AgentRunResponse(
                success=False,
                error="Model not available",
            )
        )

        request = AgentMessageRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            model="invalid-model",
        )

        response = await handler.process_message(request)

        assert response.success is False
        assert "Model not available" in response.error

    @pytest.mark.asyncio
    async def test_process_message_exception(self, handler, mock_enclave, sample_encrypted_message, user_keypair):
        """Test that exceptions are handled gracefully."""
        mock_enclave.run_agent = AsyncMock(side_effect=Exception("Unexpected error"))

        request = AgentMessageRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            model="claude-3-5-sonnet",
        )

        response = await handler.process_message(request)

        assert response.success is False
        assert "Unexpected error" in response.error

    @pytest.mark.asyncio
    async def test_process_message_no_enclave(self, sample_encrypted_message, user_keypair):
        """Test error when enclave is not configured."""
        handler = AgentHandler(enclave=None)

        request = AgentMessageRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            model="claude-3-5-sonnet",
        )

        response = await handler.process_message(request)

        assert response.success is False
        assert "not configured" in response.error


class TestAgentMessageRequest:
    """Test AgentMessageRequest dataclass."""

    def test_required_fields(self):
        """Test that required fields are enforced."""
        keypair = generate_x25519_keypair()
        encrypted_msg = EncryptedPayload(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"test",
            auth_tag=b"z" * 16,
            hkdf_salt=b"s" * 32,
        )

        request = AgentMessageRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=encrypted_msg,
            encrypted_state=None,
            user_public_key=keypair.public_key,
            model="claude-3-5-sonnet",
        )

        assert request.user_id == "user_123"
        assert request.agent_name == "luna"


class TestAgentMessageResponse:
    """Test AgentMessageResponse dataclass."""

    def test_success_response(self):
        """Test successful response."""
        encrypted = EncryptedPayload(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"response",
            auth_tag=b"z" * 16,
            hkdf_salt=b"s" * 32,
        )
        response = AgentMessageResponse(
            success=True,
            encrypted_response=encrypted,
            encrypted_state=encrypted,
        )
        assert response.success is True
        assert response.error == ""

    def test_error_response(self):
        """Test error response."""
        response = AgentMessageResponse(
            success=False,
            error="Something went wrong",
        )
        assert response.success is False
        assert response.encrypted_response is None
