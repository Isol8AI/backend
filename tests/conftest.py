"""
Shared test fixtures for Freebird backend tests.

This module provides:
- PostgreSQL test database (matches production environment)
- Authentication mocking for Clerk JWT
- Test client setup for FastAPI
- Factory fixtures for creating test data

IMPORTANT: Tests use a real PostgreSQL database to match production behavior.
Run `docker-compose up -d` before running tests to start the database.
"""
import os
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import patch

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from models.base import Base
from models.user import User
from models.session import Session
from models.message import Message


# --------------------------------------------------------------------------
# Test Database Configuration
# --------------------------------------------------------------------------

# Use the same database as development - tests drop/recreate tables
# For CI, set TEST_DATABASE_URL to a separate test database
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/securechat"
)


# --------------------------------------------------------------------------
# Database Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Create a database session for each test.

    Creates tables at start, yields session, then cleans up.
    Each test gets a fresh engine and session to avoid event loop issues.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        # Use rollback instead of commit to handle tests that expect exceptions
        await session.rollback()

    # Clean up tables after test
    async with async_session() as cleanup_session:
        # Delete in order to respect foreign key constraints
        await cleanup_session.execute(Message.__table__.delete())
        await cleanup_session.execute(Session.__table__.delete())
        await cleanup_session.execute(User.__table__.delete())
        await cleanup_session.commit()

    await engine.dispose()


@pytest.fixture
def override_get_db(db_session):
    """Create a dependency override for get_db."""
    async def _get_db():
        yield db_session
    return _get_db


@pytest.fixture
def override_get_session_factory(db_session):
    """Create a dependency override for get_session_factory.

    Returns a factory that yields the test's db_session instead of creating
    a new session. This allows streaming endpoints to use the same session
    as the test, avoiding database conflicts.
    """
    def _get_session_factory():
        # Return an async_sessionmaker-like object that returns db_session
        class TestSessionFactory:
            def __call__(self):
                return self

            async def __aenter__(self):
                return db_session

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        return TestSessionFactory()
    return _get_session_factory


# --------------------------------------------------------------------------
# Authentication Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def mock_user_payload():
    """Default mock user JWT payload."""
    return {
        "sub": "user_test_123",
        "email": "test@example.com",
        "iss": "https://test.clerk.accounts.dev",
        "aud": "test-audience",
        "exp": 9999999999,
        "iat": 1234567890,
    }


@pytest.fixture
def mock_current_user(mock_user_payload):
    """Override get_current_user dependency with mock payload."""
    async def _mock_get_current_user():
        return mock_user_payload
    return _mock_get_current_user


@pytest.fixture
def mock_jwks():
    """Mock JWKS response for JWT verification tests."""
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-key-id",
                "use": "sig",
                "n": "test-modulus",
                "e": "AQAB"
            }
        ]
    }


# --------------------------------------------------------------------------
# FastAPI Test Client Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a fresh FastAPI app instance for testing."""
    from main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app, override_get_db, mock_current_user) -> Generator:
    """Create a synchronous test client with mocked dependencies."""
    from core.database import get_db
    from core.auth import get_current_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = mock_current_user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(app, override_get_db, override_get_session_factory, mock_current_user) -> AsyncGenerator:
    """Create an async test client with mocked dependencies."""
    from core.database import get_db, get_session_factory
    from core.auth import get_current_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_session_factory] = override_get_session_factory
    app.dependency_overrides[get_current_user] = mock_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client(app, override_get_db) -> Generator:
    """Create a test client WITHOUT authentication (for testing auth failures)."""
    from core.database import get_db

    app.dependency_overrides[get_db] = override_get_db
    # Note: We don't override get_current_user, so it will require real auth

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
async def unauthenticated_async_client(app, override_get_db) -> AsyncGenerator:
    """Create an async test client WITHOUT authentication (for testing auth failures)."""
    from core.database import get_db

    app.dependency_overrides[get_db] = override_get_db
    # Note: We don't override get_current_user, so it will require real auth

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Test Data Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
async def test_user(db_session) -> User:
    """Create a test user in the database."""
    user = User(id="user_test_123")
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def other_user(db_session) -> User:
    """Create another test user (for authorization tests)."""
    user = User(id="user_other_456")
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def test_session(db_session, test_user) -> Session:
    """Create a test chat session in the database."""
    import uuid
    session = Session(
        id=str(uuid.uuid4()),
        user_id=test_user.id,
        name="Test Session"
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.fixture
async def other_user_session(db_session, other_user) -> Session:
    """Create a session belonging to another user (for authorization tests)."""
    import uuid
    session = Session(
        id=str(uuid.uuid4()),
        user_id=other_user.id,
        name="Other User's Session"
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.fixture
async def test_message(db_session, test_session) -> Message:
    """Create a test message in the database."""
    import uuid
    message = Message(
        id=str(uuid.uuid4()),
        session_id=test_session.id,
        role="user",
        content="Hello, this is a test message"
    )
    db_session.add(message)
    await db_session.flush()
    return message


@pytest.fixture
async def test_conversation(db_session, test_session) -> list[Message]:
    """Create a test conversation with multiple messages."""
    import uuid
    messages = [
        Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            content="Hello!"
        ),
        Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="assistant",
            content="Hi there! How can I help you today?",
            model_used="Qwen/Qwen2.5-72B-Instruct"
        ),
        Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            content="What's the weather like?"
        ),
    ]
    for msg in messages:
        db_session.add(msg)
    await db_session.flush()
    return messages


# --------------------------------------------------------------------------
# LLM Service Mocks
# --------------------------------------------------------------------------

@pytest.fixture
def mock_llm_stream():
    """Mock LLM streaming response generator."""
    async def _mock_stream():
        chunks = ["Hello", " ", "world", "!"]
        for chunk in chunks:
            yield chunk
    return _mock_stream


@pytest.fixture
def mock_hf_sse_response():
    """Mock HuggingFace SSE response data."""
    return [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]


# --------------------------------------------------------------------------
# Environment Mocks
# --------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Mock settings with test values."""
    with patch("core.config.settings") as mock:
        mock.CLERK_ISSUER = "https://test.clerk.accounts.dev"
        mock.CLERK_AUDIENCE = None
        mock.HUGGINGFACE_TOKEN = "hf_test_token"
        mock.HF_API_URL = "https://router.huggingface.co/v1"
        mock.DATABASE_URL = TEST_DATABASE_URL
        mock.PROJECT_NAME = "Freebird Test"
        mock.API_V1_STR = "/api/v1"
        yield mock
