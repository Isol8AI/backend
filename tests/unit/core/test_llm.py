"""Unit tests for LLM service."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from core.llm import LLMService, DEFAULT_MODEL


class TestLLMServiceBuildMessages:
    """Tests for _build_messages method."""

    def test_build_messages_empty_history(self):
        """Build messages with empty history."""
        service = LLMService()
        messages = service._build_messages([], "Hello!")

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_build_messages_with_history(self):
        """Build messages with conversation history."""
        service = LLMService()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"}
        ]
        messages = service._build_messages(history, "What's the weather?")

        assert len(messages) == 4  # system + 2 history + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "What's the weather?"

    def test_build_messages_system_prompt_content(self):
        """System prompt contains model attribution note."""
        service = LLMService()
        messages = service._build_messages([], "Test")

        system_content = messages[0]["content"]
        assert "helpful AI assistant" in system_content
        assert "model-name" in system_content.lower() or "metadata" in system_content.lower()

    def test_build_messages_preserves_history_order(self):
        """History messages are in correct order."""
        service = LLMService()
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
        ]
        messages = service._build_messages(history, "Fifth")

        # Skip system message
        content_order = [m["content"] for m in messages[1:]]
        assert content_order == ["First", "Second", "Third", "Fourth", "Fifth"]


class TestLLMServiceGenerateResponse:
    """Tests for generate_response_stream method."""

    @pytest.fixture
    def service_with_token(self):
        """LLM service with mock token."""
        with patch("core.llm.settings") as mock_settings:
            mock_settings.HUGGINGFACE_TOKEN = "hf_test_token"
            mock_settings.HF_API_URL = "https://router.huggingface.co/v1"
            service = LLMService()
            service.token = "hf_test_token"
            service.api_url = "https://router.huggingface.co/v1"
            yield service

    @pytest.mark.asyncio
    async def test_missing_token_yields_error(self):
        """Missing HUGGINGFACE_TOKEN yields error message."""
        with patch("core.llm.settings") as mock_settings:
            mock_settings.HUGGINGFACE_TOKEN = None
            mock_settings.HF_API_URL = "https://router.huggingface.co/v1"
            service = LLMService()
            service.token = None

            chunks = []
            async for chunk in service.generate_response_stream("Hello"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "HUGGINGFACE_TOKEN" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_yields_content_chunks(self, service_with_token):
        """Stream correctly yields content chunks from SSE response."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            # Use MagicMock for the client, configure async context manager methods
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # stream() returns an async context manager (not a coroutine)
            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context

            mock_client_class.return_value = mock_client

            chunks = []
            async for chunk in service_with_token.generate_response_stream("Hi"):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_handles_api_error(self, service_with_token):
        """API error response yields error message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context

            mock_client_class.return_value = mock_client

            chunks = []
            async for chunk in service_with_token.generate_response_stream("Hi"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "500" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_handles_timeout(self, service_with_token):
        """Timeout yields appropriate error message."""
        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.side_effect = httpx.ReadTimeout("Timeout")
            mock_client_class.return_value = mock_client

            chunks = []
            async for chunk in service_with_token.generate_response_stream("Hi"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "too long" in chunks[0].lower()

    @pytest.mark.asyncio
    async def test_stream_handles_generic_exception(self, service_with_token):
        """Generic exception yields error message."""
        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.side_effect = Exception("Network failure")
            mock_client_class.return_value = mock_client

            chunks = []
            async for chunk in service_with_token.generate_response_stream("Hi"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "Network failure" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_uses_default_model(self, service_with_token):
        """Default model is used when none specified."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context

            mock_client_class.return_value = mock_client

            async for _ in service_with_token.generate_response_stream("Hi"):
                pass

            # Verify the call was made with default model
            call_args = mock_client.stream.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert payload.get("model") == DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_stream_uses_specified_model(self, service_with_token):
        """Specified model is used instead of default."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context

            mock_client_class.return_value = mock_client

            custom_model = "meta-llama/Llama-3.3-70B-Instruct"
            async for _ in service_with_token.generate_response_stream("Hi", model=custom_model):
                pass

            call_args = mock_client.stream.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert payload.get("model") == custom_model

    @pytest.mark.asyncio
    async def test_stream_skips_malformed_json(self, service_with_token):
        """Malformed JSON lines are skipped without error."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {malformed json}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context

            mock_client_class.return_value = mock_client

            chunks = []
            async for chunk in service_with_token.generate_response_stream("Hi"):
                chunks.append(chunk)

            # Should get Hello and ! but skip malformed line
            assert chunks == ["Hello", "!"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, service_with_token):
        """Lines with empty content are skipped."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context

            mock_client_class.return_value = mock_client

            chunks = []
            async for chunk in service_with_token.generate_response_stream("Hi"):
                chunks.append(chunk)

            assert chunks == ["Hello", "!"]


class TestLLMServiceConfig:
    """Tests for LLM service configuration."""

    def test_default_model_constant(self):
        """Default model constant is set correctly."""
        assert DEFAULT_MODEL == "Qwen/Qwen2.5-72B-Instruct"

    def test_service_initializes_from_settings(self):
        """Service initializes with settings values."""
        with patch("core.llm.settings") as mock_settings:
            mock_settings.HUGGINGFACE_TOKEN = "test_token"
            mock_settings.HF_API_URL = "https://test.api.com"

            service = LLMService()

            assert service.token == "test_token"
            assert service.api_url == "https://test.api.com"
