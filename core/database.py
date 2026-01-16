from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from core.config import settings

# Production-ready connection pool settings
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=5,           # Base connections to keep open
    max_overflow=10,       # Extra connections under load
    pool_timeout=30,       # Seconds to wait for connection
    pool_recycle=1800,     # Recycle connections after 30 min (important for Supabase)
    pool_pre_ping=True,    # Verify connections before use
)

# Single session factory using modern async_sessionmaker (SQLAlchemy 2.0+)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


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
