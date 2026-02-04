"""Tests for AgentHandler streaming and AgentStreamChunk/AgentStreamRequest dataclasses."""

import pytest
from unittest.mock import MagicMock

from core.crypto import EncryptedPayload, generate_x25519_keypair
from core.enclave.agent_handler import AgentHandler, AgentStreamRequest
from core.enclave.mock_enclave import AgentStreamChunk, MockEnclave


class TestAgentStreamChunk:
    """Test AgentStreamChunk dataclass."""

    def test_default_values(self):
        """Test that default values are all None/False/empty/0."""
        chunk = AgentStreamChunk()

        assert chunk.encrypted_content is None
        assert chunk.encrypted_state is None
        assert chunk.is_final is False
        assert chunk.error == ""
        assert chunk.input_tokens == 0
        assert chunk.output_tokens == 0

    def test_content_chunk(self):
        """Test a streaming content chunk (encrypted_content set, is_final=False)."""
        encrypted_content = EncryptedPayload(
            ephemeral_public_key=b"a" * 32,
            iv=b"b" * 16,
            ciphertext=b"streaming text chunk",
            auth_tag=b"c" * 16,
            hkdf_salt=b"d" * 32,
        )

        chunk = AgentStreamChunk(encrypted_content=encrypted_content)

        assert chunk.encrypted_content is encrypted_content
        assert chunk.encrypted_content.ciphertext == b"streaming text chunk"
        assert chunk.encrypted_state is None
        assert chunk.is_final is False
        assert chunk.error == ""

    def test_final_chunk(self):
        """Test the final chunk with encrypted_state and is_final=True."""
        encrypted_state = EncryptedPayload(
            ephemeral_public_key=b"e" * 32,
            iv=b"f" * 16,
            ciphertext=b"updated tarball state",
            auth_tag=b"g" * 16,
            hkdf_salt=b"h" * 32,
        )

        chunk = AgentStreamChunk(
            encrypted_state=encrypted_state,
            is_final=True,
            input_tokens=150,
            output_tokens=300,
        )

        assert chunk.encrypted_content is None
        assert chunk.encrypted_state is encrypted_state
        assert chunk.encrypted_state.ciphertext == b"updated tarball state"
        assert chunk.is_final is True
        assert chunk.input_tokens == 150
        assert chunk.output_tokens == 300
        assert chunk.error == ""

    def test_error_chunk(self):
        """Test an error chunk with error message set and is_final=True."""
        chunk = AgentStreamChunk(
            error="Model not available",
            is_final=True,
        )

        assert chunk.encrypted_content is None
        assert chunk.encrypted_state is None
        assert chunk.is_final is True
        assert chunk.error == "Model not available"
        assert chunk.input_tokens == 0
        assert chunk.output_tokens == 0


class TestAgentStreamRequest:
    """Test AgentStreamRequest dataclass."""

    def test_required_fields(self):
        """Test that all required fields are set correctly."""
        keypair = generate_x25519_keypair()
        encrypted_msg = EncryptedPayload(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"test message",
            auth_tag=b"z" * 16,
            hkdf_salt=b"s" * 32,
        )
        encrypted_state = EncryptedPayload(
            ephemeral_public_key=b"a" * 32,
            iv=b"b" * 16,
            ciphertext=b"existing state",
            auth_tag=b"c" * 16,
            hkdf_salt=b"d" * 32,
        )

        request = AgentStreamRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=encrypted_msg,
            encrypted_state=encrypted_state,
            client_public_key=keypair.public_key,
        )

        assert request.user_id == "user_123"
        assert request.agent_name == "luna"
        assert request.encrypted_message is encrypted_msg
        assert request.encrypted_state is encrypted_state
        assert request.client_public_key == keypair.public_key

    def test_with_none_encrypted_state(self):
        """Test request for a new agent with no existing state."""
        keypair = generate_x25519_keypair()
        encrypted_msg = EncryptedPayload(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"first message",
            auth_tag=b"z" * 16,
            hkdf_salt=b"s" * 32,
        )

        request = AgentStreamRequest(
            user_id="user_456",
            agent_name="nova",
            encrypted_message=encrypted_msg,
            encrypted_state=None,
            client_public_key=keypair.public_key,
        )

        assert request.user_id == "user_456"
        assert request.agent_name == "nova"
        assert request.encrypted_state is None

    def test_with_existing_encrypted_state(self):
        """Test request with existing agent state tarball."""
        keypair = generate_x25519_keypair()
        encrypted_msg = EncryptedPayload(
            ephemeral_public_key=b"m" * 32,
            iv=b"n" * 16,
            ciphertext=b"followup message",
            auth_tag=b"o" * 16,
            hkdf_salt=b"p" * 32,
        )
        encrypted_state = EncryptedPayload(
            ephemeral_public_key=b"e" * 32,
            iv=b"f" * 16,
            ciphertext=b"tarball with sessions and memory",
            auth_tag=b"g" * 16,
            hkdf_salt=b"h" * 32,
        )

        request = AgentStreamRequest(
            user_id="user_789",
            agent_name="atlas",
            encrypted_message=encrypted_msg,
            encrypted_state=encrypted_state,
            client_public_key=keypair.public_key,
        )

        assert request.encrypted_state is not None
        assert request.encrypted_state.ciphertext == b"tarball with sessions and memory"


class TestAgentHandlerStreaming:
    """Test AgentHandler.process_message_streaming."""

    @pytest.fixture
    def mock_enclave(self):
        """Create mock enclave that implements agent_chat_streaming."""
        enclave = MagicMock()
        keypair = generate_x25519_keypair()
        enclave._keypair = keypair
        enclave.get_info.return_value = MagicMock(enclave_public_key=keypair.public_key)
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

    @pytest.fixture
    def sample_stream_request(self, sample_encrypted_message, user_keypair):
        """Create a sample AgentStreamRequest."""
        return AgentStreamRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,
            client_public_key=user_keypair.public_key,
        )

    @pytest.mark.asyncio
    async def test_process_message_streaming_yields_chunks(
        self, handler, mock_enclave, sample_stream_request
    ):
        """Test that process_message_streaming yields chunks from enclave."""
        content_payload = EncryptedPayload(
            ephemeral_public_key=b"r" * 32,
            iv=b"s" * 16,
            ciphertext=b"Hello from agent",
            auth_tag=b"t" * 16,
            hkdf_salt=b"u" * 32,
        )
        state_payload = EncryptedPayload(
            ephemeral_public_key=b"v" * 32,
            iv=b"w" * 16,
            ciphertext=b"updated state tarball",
            auth_tag=b"x" * 16,
            hkdf_salt=b"y" * 32,
        )

        async def mock_stream(*args, **kwargs):
            yield AgentStreamChunk(encrypted_content=content_payload)
            yield AgentStreamChunk(
                encrypted_state=state_payload,
                is_final=True,
                input_tokens=100,
                output_tokens=200,
            )

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(sample_stream_request):
            chunks.append(chunk)

        assert len(chunks) == 2

        # First chunk: streaming content
        assert chunks[0].encrypted_content is content_payload
        assert chunks[0].encrypted_content.ciphertext == b"Hello from agent"
        assert chunks[0].is_final is False
        assert chunks[0].encrypted_state is None

        # Second chunk: final state
        assert chunks[1].encrypted_state is state_payload
        assert chunks[1].encrypted_state.ciphertext == b"updated state tarball"
        assert chunks[1].is_final is True
        assert chunks[1].input_tokens == 100
        assert chunks[1].output_tokens == 200
        assert chunks[1].encrypted_content is None

    @pytest.mark.asyncio
    async def test_process_message_streaming_delegates_to_enclave(
        self, handler, mock_enclave, sample_stream_request, sample_encrypted_message, user_keypair
    ):
        """Test that process_message_streaming passes correct arguments to enclave."""
        received_kwargs = {}

        async def mock_stream(**kwargs):
            received_kwargs.update(kwargs)
            yield AgentStreamChunk(is_final=True)

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(sample_stream_request):
            chunks.append(chunk)

        assert received_kwargs["encrypted_message"] is sample_encrypted_message
        assert received_kwargs["encrypted_state"] is None
        assert received_kwargs["client_public_key"] == user_keypair.public_key
        assert received_kwargs["agent_name"] == "luna"

    @pytest.mark.asyncio
    async def test_process_message_streaming_handles_enclave_error(
        self, handler, mock_enclave, sample_stream_request
    ):
        """Test that enclave errors are yielded as error chunks."""
        async def mock_stream(*args, **kwargs):
            yield AgentStreamChunk(encrypted_content=EncryptedPayload(
                ephemeral_public_key=b"a" * 32,
                iv=b"b" * 16,
                ciphertext=b"partial",
                auth_tag=b"c" * 16,
                hkdf_salt=b"d" * 32,
            ))
            yield AgentStreamChunk(error="Bedrock throttled", is_final=True)

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(sample_stream_request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].encrypted_content is not None
        assert chunks[1].error == "Bedrock throttled"
        assert chunks[1].is_final is True

    @pytest.mark.asyncio
    async def test_process_message_streaming_handles_exception(
        self, handler, mock_enclave, sample_stream_request
    ):
        """Test that exceptions during streaming are caught and yield an error chunk."""
        async def mock_stream(*args, **kwargs):
            raise RuntimeError("Connection lost")
            yield  # type: ignore - make this a generator

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(sample_stream_request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "Connection lost" in chunks[0].error

    @pytest.mark.asyncio
    async def test_process_message_streaming_handles_mid_stream_exception(
        self, handler, mock_enclave, sample_stream_request
    ):
        """Test that an exception mid-stream yields whatever was received plus an error chunk."""
        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            yield AgentStreamChunk(encrypted_content=EncryptedPayload(
                ephemeral_public_key=b"a" * 32,
                iv=b"b" * 16,
                ciphertext=b"first chunk",
                auth_tag=b"c" * 16,
                hkdf_salt=b"d" * 32,
            ))
            raise Exception("Unexpected failure mid-stream")

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(sample_stream_request):
            chunks.append(chunk)

        # Should have the first content chunk plus the error chunk
        assert len(chunks) == 2
        assert chunks[0].encrypted_content is not None
        assert chunks[0].encrypted_content.ciphertext == b"first chunk"
        assert chunks[1].is_final is True
        assert "Unexpected failure mid-stream" in chunks[1].error

    @pytest.mark.asyncio
    async def test_process_message_streaming_no_enclave(self, sample_encrypted_message, user_keypair):
        """Test that process_message_streaming yields error chunk when enclave is None."""
        handler = AgentHandler(enclave=None)

        request = AgentStreamRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=None,
            client_public_key=user_keypair.public_key,
        )

        chunks = []
        async for chunk in handler.process_message_streaming(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "not configured" in chunks[0].error

    @pytest.mark.asyncio
    async def test_process_message_streaming_with_existing_state(
        self, handler, mock_enclave, sample_encrypted_message, user_keypair
    ):
        """Test streaming with an existing agent state passed through."""
        existing_state = EncryptedPayload(
            ephemeral_public_key=b"e" * 32,
            iv=b"f" * 16,
            ciphertext=b"existing_tarball",
            auth_tag=b"g" * 16,
            hkdf_salt=b"h" * 32,
        )

        request = AgentStreamRequest(
            user_id="user_123",
            agent_name="luna",
            encrypted_message=sample_encrypted_message,
            encrypted_state=existing_state,
            client_public_key=user_keypair.public_key,
        )

        received_kwargs = {}

        async def mock_stream(**kwargs):
            received_kwargs.update(kwargs)
            yield AgentStreamChunk(is_final=True)

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(request):
            chunks.append(chunk)

        assert received_kwargs["encrypted_state"] is existing_state

    @pytest.mark.asyncio
    async def test_process_message_streaming_multiple_content_chunks(
        self, handler, mock_enclave, sample_stream_request
    ):
        """Test streaming with multiple content chunks before final."""
        payloads = []
        for i in range(5):
            payloads.append(EncryptedPayload(
                ephemeral_public_key=bytes([i]) * 32,
                iv=bytes([i + 10]) * 16,
                ciphertext=f"chunk {i}".encode(),
                auth_tag=bytes([i + 20]) * 16,
                hkdf_salt=bytes([i + 30]) * 32,
            ))

        final_state = EncryptedPayload(
            ephemeral_public_key=b"z" * 32,
            iv=b"z" * 16,
            ciphertext=b"final state",
            auth_tag=b"z" * 16,
            hkdf_salt=b"z" * 32,
        )

        async def mock_stream(*args, **kwargs):
            for payload in payloads:
                yield AgentStreamChunk(encrypted_content=payload)
            yield AgentStreamChunk(
                encrypted_state=final_state,
                is_final=True,
                input_tokens=500,
                output_tokens=1000,
            )

        mock_enclave.agent_chat_streaming = mock_stream

        chunks = []
        async for chunk in handler.process_message_streaming(sample_stream_request):
            chunks.append(chunk)

        assert len(chunks) == 6  # 5 content + 1 final

        # All intermediate chunks should have content and not be final
        for i in range(5):
            assert chunks[i].encrypted_content is not None
            assert chunks[i].encrypted_content.ciphertext == f"chunk {i}".encode()
            assert chunks[i].is_final is False

        # Final chunk
        assert chunks[5].encrypted_state is final_state
        assert chunks[5].is_final is True
        assert chunks[5].input_tokens == 500
        assert chunks[5].output_tokens == 1000


class TestMockEnclaveAgentChatStreaming:
    """Test MockEnclave.agent_chat_streaming raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_mock_enclave_agent_chat_streaming_raises(self):
        """Test that MockEnclave.agent_chat_streaming raises NotImplementedError."""
        enclave = MockEnclave.__new__(MockEnclave)
        # Manually set the required attributes without full __init__
        enclave._keypair = generate_x25519_keypair()

        encrypted_msg = EncryptedPayload(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"test",
            auth_tag=b"z" * 16,
            hkdf_salt=b"s" * 32,
        )

        with pytest.raises(NotImplementedError, match="agent_chat_streaming is only available in Nitro Enclave mode"):
            async for _ in enclave.agent_chat_streaming(
                encrypted_message=encrypted_msg,
                encrypted_state=None,
                client_public_key=b"k" * 32,
                agent_name="test_agent",
            ):
                pass  # Should not reach here
