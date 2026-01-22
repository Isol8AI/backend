import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from core.config import settings

# Supabase pooler (pgbouncer transaction mode) configuration
# - NullPool: Let Supabase handle connection pooling, not SQLAlchemy
# - statement_cache_size=0: Disable asyncpg's prepared statement cache
# - prepared_statement_name_func: Use unnamed statements to avoid conflicts
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    poolclass=NullPool,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_name_func": lambda: "",
    },
)

# Single session factory using modern async_sessionmaker (SQLAlchemy 2.0+)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

# asyncpg pool for memory operations (pgvector)
_memory_pool: asyncpg.Pool | None = None


def _get_asyncpg_url() -> str:
    """Convert SQLAlchemy URL to asyncpg format."""
    url = settings.DATABASE_URL
    # Remove '+asyncpg' if present (SQLAlchemy format)
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
