import os
import re
import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from core.config import settings


def _get_schema_and_clean_url(url: str) -> tuple[str, str]:
    """Extract schema from URL options and return (schema, clean_url).

    asyncpg doesn't support the 'options' URL parameter that sets search_path.
    We need to extract it and use server_settings instead.
    """
    # Match options=-csearch_path%3D{schema} or options=-c+search_path={schema}
    # URL-encoded: %3D is =, %26 is &
    match = re.search(r"[?&]options=-c(?:\+|%20)?search_path(?:%3D|=)(\w+)", url, re.IGNORECASE)
    if match:
        schema = match.group(1)
        # Remove the options parameter from URL
        clean_url = re.sub(r"[?&]options=-c(?:\+|%20)?search_path(?:%3D|=)\w+", "", url)
        # Fix URL if we removed the first query param (? becomes nothing)
        clean_url = re.sub(r"\?&", "?", clean_url)
        clean_url = re.sub(r"\?$", "", clean_url)
        return schema, clean_url

    # Fall back to ENVIRONMENT variable
    env = os.getenv("ENVIRONMENT", "").lower()
    if env in ("dev", "staging", "prod"):
        return env, url

    return "public", url


# Parse schema from DATABASE_URL and get clean URL for asyncpg
_db_schema, _clean_db_url = _get_schema_and_clean_url(settings.DATABASE_URL)

# Supabase pooler (pgbouncer transaction mode) configuration
# - NullPool: Let Supabase handle connection pooling, not SQLAlchemy
# - statement_cache_size=0: Disable asyncpg's prepared statement cache
# - prepared_statement_name_func: Use unnamed statements to avoid conflicts
# - server_settings: Set search_path for schema isolation + public for pgvector
_search_path = f"{_db_schema},public"
engine = create_async_engine(
    _clean_db_url,
    echo=settings.DEBUG,
    poolclass=NullPool,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_name_func": lambda: "",
        "server_settings": {"search_path": _search_path},
    },
)

# Single session factory using modern async_sessionmaker (SQLAlchemy 2.0+)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

# asyncpg pool for memory operations (pgvector)
_memory_pool: asyncpg.Pool | None = None


def _get_asyncpg_url() -> str:
    """Convert SQLAlchemy URL to asyncpg format (clean, without options)."""
    # Use the already-cleaned URL and remove '+asyncpg' if present
    url = _clean_db_url
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
    return url


async def get_memory_pool() -> asyncpg.Pool:
    """Get or create the asyncpg connection pool for memory operations."""
    global _memory_pool
    if _memory_pool is None:
        _memory_pool = await asyncpg.create_pool(
            _get_asyncpg_url(),
            min_size=1,
            max_size=5,
            command_timeout=30,
            # Supabase pooler compatibility
            statement_cache_size=0,
            # Set search_path for schema isolation + public for pgvector
            server_settings={"search_path": _search_path},
        )
    return _memory_pool


async def close_memory_pool() -> None:
    """Close the memory pool on shutdown."""
    global _memory_pool
    if _memory_pool is not None:
        await _memory_pool.close()
        _memory_pool = None


async def get_db():
    """Dependency that yields a database session for request scope."""
    async with async_session_factory() as session:
        yield session


def get_session_factory():
    """Dependency that returns the session factory.

    This allows tests to override the session factory used in streaming endpoints.
    """
    return async_session_factory


async def check_db_health() -> bool:
    """Verify database connectivity."""
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
