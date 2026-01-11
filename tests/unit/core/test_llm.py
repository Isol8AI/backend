"""Unit tests for LLM service."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.llm import DEFAULT_MODEL, LLMService


def create_mock_stream_client(response: MagicMock = None, error: Exception = None) -> MagicMock:
    """Create a mock httpx.AsyncClient configured for streaming."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    if error:
        mock_client.stream.side_effect = error
    else:
        mock_stream_context = MagicMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream.return_value = mock_stream_context

    return mock_client


def create_sse_response(lines: list[str], status_code: int = 200) -> MagicMock:
    """Create a mock SSE response with given lines."""
    async def mock_aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.aiter_lines = mock_aiter_lines
    return mock_response


class TestLLMServiceBuildMessages:
    """Tests for _build_messages method."""

    def test_empty_history_includes_system_and_user_message(self):
        """Build messages with empty history includes system and user message."""
        service = LLMService()
        messages = service._build_messages([], "Hello!")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_with_history_includes_all_messages(self):
        """Build messages includes history between system and new user message."""
        service = LLMService()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        messages = service._build_messages(history, "What's the weather?")

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["content"] == "What's the weather?"

    def test_system_prompt_content(self):
        """System prompt contains expected content."""
        service = LLMService()
        messages = service._build_messages([], "Test")
        assert "helpful AI assistant" in messages[0]["content"]

    def test_preserves_history_order(self):
        """History messages maintain chronological order."""
        service = LLMService()
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
        ]
        messages = service._build_messages(history, "Fifth")

        content_order = [m["content"] for m in messages[1:]]
        assert content_order == ["First", "Second", "Third", "Fourth", "Fifth"]


class TestLLMServiceGenerateResponse:
    """Tests for generate_response_stream method."""

    @pytest.fixture
    def service_with_token(self):
        """LLM service with mock token configured."""
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

            chunks = [chunk async for chunk in service.generate_response_stream("Hello")]

            assert len(chunks) == 1
            assert "HUGGINGFACE_TOKEN" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_yields_content_chunks(self, service_with_token):
        """Stream correctly yields content chunks from SSE response."""
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]
        mock_response = create_sse_response(sse_lines)

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = create_mock_stream_client(mock_response)

            chunks = [chunk async for chunk in service_with_token.generate_response_stream("Hi")]

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_handles_api_error(self, service_with_token):
        """API error response yields error message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = create_mock_stream_client(mock_response)

            chunks = [chunk async for chunk in service_with_token.generate_response_stream("Hi")]

            assert len(chunks) == 1
            assert "500" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_handles_timeout(self, service_with_token):
        """Timeout yields appropriate error message."""
        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = create_mock_stream_client(error=httpx.ReadTimeout("Timeout"))

            chunks = [chunk async for chunk in service_with_token.generate_response_stream("Hi")]

            assert len(chunks) == 1
            assert "too long" in chunks[0].lower()

    @pytest.mark.asyncio
    async def test_stream_handles_generic_exception(self, service_with_token):
        """Generic exception yields error message."""
        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = create_mock_stream_client(error=Exception("Network failure"))

            chunks = [chunk async for chunk in service_with_token.generate_response_stream("Hi")]

            assert len(chunks) == 1
            assert "Network failure" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_uses_default_model(self, service_with_token):
        """Default model is used when none specified."""
        mock_response = create_sse_response(['data: [DONE]'])

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = create_mock_stream_client(mock_response)
            mock_client_class.return_value = mock_client

            async for _ in service_with_token.generate_response_stream("Hi"):
                pass

            call_args = mock_client.stream.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert payload.get("model") == DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_stream_uses_specified_model(self, service_with_token):
        """Specified model is used instead of default."""
        mock_response = create_sse_response(['data: [DONE]'])

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = create_mock_stream_client(mock_response)
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
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {malformed json}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]
        mock_response = create_sse_response(sse_lines)

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = create_mock_stream_client(mock_response)

            chunks = [chunk async for chunk in service_with_token.generate_response_stream("Hi")]

            assert chunks == ["Hello", "!"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, service_with_token):
        """Lines with empty content are skipped."""
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]
        mock_response = create_sse_response(sse_lines)

        with patch("core.llm.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = create_mock_stream_client(mock_response)

            chunks = [chunk async for chunk in service_with_token.generate_response_stream("Hi")]

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
