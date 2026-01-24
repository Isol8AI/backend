"""Embedding generation for enclave processing.

This module provides server-side embedding generation using sentence-transformers.
The enclave generates embeddings from plaintext, avoiding client-side model loading.

Security Model:
- Embeddings are generated in the enclave where plaintext is available
- Embeddings themselves are not encrypted (required for similarity search)
- Only the enclave has access to plaintext for embedding generation
"""

from typing import List
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    """Lazily load the sentence-transformers model (cached)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
    return model


class EnclaveEmbeddings:
    """Generates embeddings using sentence-transformers.

    Uses all-MiniLM-L6-v2 model for 384-dimensional embeddings.
    Compatible with the frontend's Transformers.js embeddings.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.MODEL_NAME

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.DIMENSION

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            384-dimensional embedding as list of floats (L2 normalized)
        """
        model = _get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of 384-dimensional embeddings
        """
        if not texts:
            return []

        model = _get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]
