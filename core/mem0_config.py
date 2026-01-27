"""
mem0 configuration for server-side memory management.

mem0 handles:
- LLM-based fact extraction from messages
- Embedding generation
- Vector storage via pgvector
- Memory search and retrieval
"""

import logging
from typing import Any
from urllib.parse import urlparse

from core.config import settings

logger = logging.getLogger(__name__)


def parse_database_url(url: str) -> dict[str, Any]:
    """
    Parse DATABASE_URL into components for mem0 pgvector config.

    Handles both asyncpg and psycopg2 connection strings.
    """
    # Remove async driver prefix for parsing
    clean_url = url.replace("+asyncpg", "").replace("+psycopg2", "")
    parsed = urlparse(clean_url)

    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "securechat",
    }


def get_mem0_config() -> dict[str, Any]:
    """
    Get mem0 configuration using pgvector as vector store.

    Configuration uses:
    - pgvector for vector storage (same PostgreSQL as main app)
    - HuggingFace Inference API for LLM (fact extraction)
    - HuggingFace Inference API for embeddings
    """
    db_config = parse_database_url(settings.DATABASE_URL)

    config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": db_config["dbname"],
                "collection_name": "mem0_memories",
                "embedding_model_dims": 384,  # all-MiniLM-L6-v2 dimensions
                "host": db_config["host"],
                "port": db_config["port"],
                "user": db_config["user"],
                "password": db_config["password"],
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                # Use HuggingFace Inference Providers (OpenAI-compatible)
                "model": settings.EXTRACTION_MODEL,
                "api_key": settings.HUGGINGFACE_TOKEN,
                "openai_base_url": settings.HF_API_URL,
                "temperature": 0.1,  # Lower temperature for fact extraction
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "api_key": settings.HUGGINGFACE_TOKEN,
            },
        },
        "version": "v1.1",  # mem0 config version
    }

    logger.info(f"mem0 config: vector_store=pgvector, llm={settings.EXTRACTION_MODEL}")
    return config
