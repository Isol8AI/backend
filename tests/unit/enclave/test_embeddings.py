"""Tests for enclave embedding generation."""

import pytest
import numpy as np


class TestEnclaveEmbeddings:
    """Tests for EnclaveEmbeddings class."""

    @pytest.fixture
    def embeddings(self):
        """Create EnclaveEmbeddings instance."""
        from core.enclave.embeddings import EnclaveEmbeddings

        return EnclaveEmbeddings()

    def test_generate_embedding_returns_384_dimensions(self, embeddings):
        """Single text returns 384-dimensional embedding."""
        result = embeddings.generate_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_generate_embedding_normalized(self, embeddings):
        """Embedding is L2 normalized (unit vector)."""
        result = embeddings.generate_embedding("Test text")

        # L2 norm should be approximately 1.0
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_generate_embeddings_batch(self, embeddings):
        """Batch generation returns list of embeddings."""
        texts = ["First text", "Second text", "Third text"]
        results = embeddings.generate_embeddings(texts)

        assert len(results) == 3
        assert all(len(emb) == 384 for emb in results)

    def test_similar_texts_have_high_similarity(self, embeddings):
        """Semantically similar texts have cosine similarity > 0.7."""
        emb1 = embeddings.generate_embedding("I love programming")
        emb2 = embeddings.generate_embedding("I enjoy coding")

        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        assert similarity > 0.7

    def test_different_texts_have_lower_similarity(self, embeddings):
        """Semantically different texts have lower similarity."""
        emb1 = embeddings.generate_embedding("I love programming")
        emb2 = embeddings.generate_embedding("The weather is sunny today")

        similarity = np.dot(emb1, emb2)
        assert similarity < 0.5

    def test_empty_text_returns_embedding(self, embeddings):
        """Empty text still returns valid embedding."""
        result = embeddings.generate_embedding("")

        assert len(result) == 384

    def test_model_property(self, embeddings):
        """Model name is accessible."""
        assert embeddings.model_name == "all-MiniLM-L6-v2"
        assert embeddings.dimension == 384
