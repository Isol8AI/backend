"""Tests for memory extraction functionality in the enclave."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from core.enclave import MockEnclave, ExtractedMemory, EncryptionContext


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_enclave():
    """Create a MockEnclave with mocked dependencies."""
    with patch.object(MockEnclave, '_get_embedding_model') as mock_embed:
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [0.1] * 384
        )
        mock_embed.return_value = mock_model

        enclave = MockEnclave(
            inference_url="https://test.api.com/v1",
            inference_token="test-token",
        )
        yield enclave


@pytest.fixture
def sample_extraction_response():
    """Sample response from extraction LLM."""
    return [
        {
            "text": "User's favorite programming language is Python",
            "sector": "semantic",
            "tags": ["programming", "preferences"]
        },
        {
            "text": "User prefers dark mode in their IDE",
            "sector": "procedural",
            "tags": ["preferences"]
        }
    ]


@pytest.fixture
def sample_storage_public_key():
    """Sample 32-byte public key."""
    return bytes.fromhex("a" * 64)


# =============================================================================
# Test Embedding Generation
# =============================================================================

class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    def test_lazy_loads_embedding_model(self, mock_enclave):
        """Embedding model is loaded lazily on first use."""
        # Initially None
        assert mock_enclave._embedding_model is None

        # After calling _get_embedding_model
        model = mock_enclave._get_embedding_model()
        assert model is not None

    def test_generates_384_dim_embedding(self, mock_enclave):
        """Generates 384-dimensional embedding vector."""
        embedding = mock_enclave._generate_embedding("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_normalizes_embeddings(self, mock_enclave):
        """Embeddings are normalized."""
        model = mock_enclave._get_embedding_model()
        mock_enclave._generate_embedding("test")

        # Verify encode was called with normalize_embeddings=True
        model.encode.assert_called_with("test", normalize_embeddings=True)


# =============================================================================
# Test Extraction Prompt
# =============================================================================

class TestExtractionPrompt:
    """Tests for extraction prompt building."""

    def test_builds_prompt_with_conversation(self, mock_enclave):
        """Prompt includes user message and assistant response."""
        prompt = mock_enclave._build_extraction_prompt(
            "What is Python?",
            "Python is a programming language."
        )

        assert "User: What is Python?" in prompt
        assert "Assistant: Python is a programming language." in prompt

    def test_prompt_includes_sector_definitions(self, mock_enclave):
        """Prompt includes definitions for all sectors."""
        prompt = mock_enclave._build_extraction_prompt("test", "test")

        assert "semantic:" in prompt
        assert "episodic:" in prompt
        assert "procedural:" in prompt
        assert "emotional:" in prompt
        assert "reflective:" in prompt

    def test_prompt_requests_json_output(self, mock_enclave):
        """Prompt requests JSON array output."""
        prompt = mock_enclave._build_extraction_prompt("test", "test")

        assert "JSON array" in prompt
        assert '{"text":' in prompt


# =============================================================================
# Test Extraction LLM Call
# =============================================================================

class TestExtractionLLMCall:
    """Tests for extraction LLM API call."""

    @pytest.mark.asyncio
    async def test_calls_extraction_llm(self, mock_enclave, sample_extraction_response):
        """Calls extraction LLM with correct parameters."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": json.dumps(sample_extraction_response)}}]
            }
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await mock_enclave._call_extraction_llm("test prompt")

            assert result == sample_extraction_response

    @pytest.mark.asyncio
    async def test_handles_markdown_code_block(self, mock_enclave, sample_extraction_response):
        """Handles JSON wrapped in markdown code blocks."""
        json_content = f"```json\n{json.dumps(sample_extraction_response)}\n```"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": json_content}}]
            }
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await mock_enclave._call_extraction_llm("test prompt")

            assert result == sample_extraction_response

    @pytest.mark.asyncio
    async def test_returns_empty_on_api_error(self, mock_enclave):
        """Returns empty list on API error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await mock_enclave._call_extraction_llm("test prompt")

            assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self, mock_enclave):
        """Returns empty list on invalid JSON response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "not valid json"}}]
            }
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await mock_enclave._call_extraction_llm("test prompt")

            assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_array_response(self, mock_enclave):
        """Handles empty array response correctly."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "[]"}}]
            }
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await mock_enclave._call_extraction_llm("test prompt")

            assert result == []


# =============================================================================
# Test Memory Encryption
# =============================================================================

class TestMemoryEncryption:
    """Tests for memory content encryption."""

    def test_encrypts_with_memory_storage_context(self, mock_enclave, sample_storage_public_key):
        """Encrypts using MEMORY_STORAGE context."""
        with patch("core.enclave.mock_enclave.encrypt_to_public_key") as mock_encrypt:
            mock_encrypt.return_value = MagicMock()

            mock_enclave.encrypt_for_memory_storage(
                b"test content",
                sample_storage_public_key,
            )

            mock_encrypt.assert_called_once_with(
                sample_storage_public_key,
                b"test content",
                EncryptionContext.MEMORY_STORAGE.value,
            )


# =============================================================================
# Test Extract Memories (Integration)
# =============================================================================

class TestExtractMemories:
    """Tests for the full extract_memories method."""

    @pytest.mark.asyncio
    async def test_extracts_memories_successfully(
        self,
        mock_enclave,
        sample_extraction_response,
        sample_storage_public_key,
    ):
        """Successfully extracts and encrypts memories."""
        with patch.object(mock_enclave, '_call_extraction_llm', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = sample_extraction_response

            with patch.object(mock_enclave, 'encrypt_for_memory_storage') as mock_encrypt:
                mock_payload = MagicMock()
                mock_payload.iv = b'\x00' * 16
                mock_payload.auth_tag = b'\x00' * 16
                mock_payload.ephemeral_public_key = b'\x00' * 32
                mock_encrypt.return_value = mock_payload

                memories = await mock_enclave.extract_memories(
                    "What is Python?",
                    "Python is a programming language.",
                    sample_storage_public_key,
                )

                assert len(memories) == 2
                assert all(isinstance(m, ExtractedMemory) for m in memories)

    @pytest.mark.asyncio
    async def test_returns_empty_when_nothing_extracted(
        self,
        mock_enclave,
        sample_storage_public_key,
    ):
        """Returns empty list when no memories extracted."""
        with patch.object(mock_enclave, '_call_extraction_llm', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = []

            memories = await mock_enclave.extract_memories(
                "Hello",
                "Hi there!",
                sample_storage_public_key,
            )

            assert memories == []

    @pytest.mark.asyncio
    async def test_validates_sector(
        self,
        mock_enclave,
        sample_storage_public_key,
    ):
        """Invalid sectors default to semantic."""
        with patch.object(mock_enclave, '_call_extraction_llm', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                {"text": "some fact", "sector": "invalid_sector", "tags": []}
            ]

            with patch.object(mock_enclave, 'encrypt_for_memory_storage') as mock_encrypt:
                mock_payload = MagicMock()
                mock_payload.iv = b'\x00' * 16
                mock_payload.auth_tag = b'\x00' * 16
                mock_payload.ephemeral_public_key = b'\x00' * 32
                mock_encrypt.return_value = mock_payload

                memories = await mock_enclave.extract_memories(
                    "test",
                    "test",
                    sample_storage_public_key,
                )

                assert len(memories) == 1
                assert memories[0].sector == "semantic"

    @pytest.mark.asyncio
    async def test_skips_empty_text(
        self,
        mock_enclave,
        sample_storage_public_key,
    ):
        """Skips items with empty text."""
        with patch.object(mock_enclave, '_call_extraction_llm', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                {"text": "", "sector": "semantic", "tags": []},
                {"text": "valid fact", "sector": "semantic", "tags": []}
            ]

            with patch.object(mock_enclave, 'encrypt_for_memory_storage') as mock_encrypt:
                mock_payload = MagicMock()
                mock_payload.iv = b'\x00' * 16
                mock_payload.auth_tag = b'\x00' * 16
                mock_payload.ephemeral_public_key = b'\x00' * 32
                mock_encrypt.return_value = mock_payload

                memories = await mock_enclave.extract_memories(
                    "test",
                    "test",
                    sample_storage_public_key,
                )

                assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_includes_metadata(
        self,
        mock_enclave,
        sample_extraction_response,
        sample_storage_public_key,
    ):
        """Extracted memories include encryption metadata."""
        with patch.object(mock_enclave, '_call_extraction_llm', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = sample_extraction_response[:1]

            with patch.object(mock_enclave, 'encrypt_for_memory_storage') as mock_encrypt:
                mock_payload = MagicMock()
                mock_payload.iv = bytes.fromhex("a" * 32)
                mock_payload.auth_tag = bytes.fromhex("b" * 32)
                mock_payload.ephemeral_public_key = bytes.fromhex("c" * 64)
                mock_encrypt.return_value = mock_payload

                memories = await mock_enclave.extract_memories(
                    "test",
                    "test",
                    sample_storage_public_key,
                )

                assert "iv" in memories[0].metadata
                assert "auth_tag" in memories[0].metadata
                assert "ephemeral_public_key" in memories[0].metadata
