"""Tests for AgentHandler enclave integration."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from core.crypto import EncryptedPayload, generate_x25519_keypair
from core.enclave.agent_handler import AgentHandler, AgentMessageRequest, AgentMessageResponse


class TestAgentHandler:
    """Test AgentHandler enclave integration."""

    @pytest.fixture
    def mock_runner(self):
        """Create mock AgentRunner."""
        runner = MagicMock()
        runner.get_user_tmpfs_path.return_value = Path("/tmp/openclaw/user_123")
        runner.run_agent.return_value = MagicMock(
            success=True,
            response="Hello! I'm Luna, your AI companion.",
        )
        runner.pack_directory.return_value = b"packed_tarball_data"
        return runner

    @pytest.fixture
    def mock_enclave(self):
        """Create mock enclave for crypto operations."""
        enclave = MagicMock()
        keypair = generate_x25519_keypair()
        enclave._keypair = keypair
        enclave.get_info.return_value = MagicMock(
            enclave_public_key=keypair.public_key
        )
        enclave.decrypt_transport_message.return_value = b"Hello!"
        enclave.encrypt_for_transport.return_value = EncryptedPayload(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"encrypted_response",
            auth_tag=b"z" * 16,
            hkdf_salt=b"s" * 32,
        )
        enclave.encrypt_for_storage.return_value = EncryptedPayload(
            ephemeral_public_key=b"a" * 32,
            iv=b"b" * 16,
            ciphertext=b"encrypted_for_storage",
            auth_tag=b"c" * 16,
            hkdf_salt=b"d" * 32,
        )
        return enclave

    @pytest.fixture
    def handler(self, mock_runner, mock_enclave):
        """Create handler with mocked dependencies."""
        return AgentHandler(runner=mock_runner, enclave=mock_enclave)

    @pytest.fixture
    def user_keypair(self):
        """Generate user keypair for testing."""
        return generate_x25519_keypair()

    @pytest.fixture
    def sample_encrypted_message(self, mock_enclave):
        """Create a sample encrypted message."""
        from core.crypto import encrypt_to_public_key
        return encrypt_to_public_key(
            mock_enclave._keypair.public_key,
            b"Hello!",
            "client-to-enclave-transport",
        )

    @pytest.mark.asyncio
    async def test_process_message_new_user(
        self, handler, mock_runner, sample_encrypted_message, user_keypair
    ):
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
        mock_runner.create_fresh_agent.assert_called_once()
        mock_runner.run_agent.assert_called_once()
        mock_runner.cleanup_directory.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_existing_user(
        self, handler, mock_runner, mock_enclave, sample_encrypted_message, user_keypair
    ):
        """Test processing message for user with existing state."""
        # Mock decrypting existing state to tarball bytes
        mock_enclave.decrypt_transport_message.side_effect = [
            b"existing_tarball_data",  # First call: decrypt state
            b"Hello!",  # Second call: decrypt message
        ]

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
        mock_runner.unpack_tarball.assert_called_once()
        mock_runner.create_fresh_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_cli_error(
        self, handler, mock_runner, sample_encrypted_message, user_keypair
    ):
        """Test handling CLI errors gracefully."""
        mock_runner.run_agent.return_value = MagicMock(
            success=False,
            error="Model not available",
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
        mock_runner.cleanup_directory.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(
        self, handler, mock_runner, sample_encrypted_message, user_keypair
    ):
        """Test that cleanup happens even on exceptions."""
        mock_runner.run_agent.side_effect = Exception("Unexpected error")

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
        mock_runner.cleanup_directory.assert_called_once()


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
