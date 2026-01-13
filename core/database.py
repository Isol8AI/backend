from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from core.config import settings

# Only echo SQL in debug mode to avoid log pollution in production
engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Factory for creating sessions outside of dependency injection
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


def get_session_factory():
    """Dependency that returns the session factory.

    This allows tests to override the session factory used in streaming endpoints.
    """
    return async_session_factory
