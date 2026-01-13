"""
Shared test fixtures for Freebird backend tests.

Tests use a real PostgreSQL database to match production behavior.
Run `docker-compose up -d` before running tests to start the database.
"""
import json
import os
import uuid
from typing import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.auth import AuthContext
from models.base import Base
from models.context_store import ContextStore
from models.message import Message
from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole
from models.session import Session
from models.user import User

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/securechat"
)


def parse_sse_events(response_text: str) -> list[dict]:
    """Parse SSE events from response text into list of event dicts."""
    lines = [line for line in response_text.split("\n") if line.startswith("data:")]
    return [json.loads(line.replace("data: ", "")) for line in lines]


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for each test with automatic cleanup."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, pool_pre_ping=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        yield session
        await session.rollback()

    async with session_factory() as cleanup_session:
        await cleanup_session.execute(Message.__table__.delete())
        await cleanup_session.execute(Session.__table__.delete())
        await cleanup_session.execute(ContextStore.__table__.delete())
        await cleanup_session.execute(OrganizationMembership.__table__.delete())
        await cleanup_session.execute(Organization.__table__.delete())
        await cleanup_session.execute(User.__table__.delete())
        await cleanup_session.commit()

    await engine.dispose()


@pytest.fixture
def override_get_db(db_session):
    """Dependency override for get_db that uses the test session."""
    async def _get_db():
        yield db_session
    return _get_db


class _TestSessionContext:
    """Async context manager wrapper for test db_session."""

    def __init__(self, session: AsyncSession):
        self._session = session

    def __call__(self) -> "_TestSessionContext":
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *_) -> None:
        pass


@pytest.fixture
def override_get_session_factory(db_session):
    """Dependency override for get_session_factory that uses the test session."""
    def _get_session_factory():
        return _TestSessionContext(db_session)
    return _get_session_factory


@pytest.fixture
def mock_user_payload() -> dict:
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
def mock_auth_context() -> AuthContext:
    """Default mock auth context for personal mode (no org)."""
    return AuthContext(user_id="user_test_123")


@pytest.fixture
def mock_org_auth_context() -> AuthContext:
    """Mock auth context for organization mode."""
    return AuthContext(
        user_id="user_test_123",
        org_id="org_test_123",
        org_role="org:member",
        org_slug="test-org",
        org_permissions=[]
    )


@pytest.fixture
def mock_org_admin_auth_context() -> AuthContext:
    """Mock auth context for organization admin mode."""
    return AuthContext(
        user_id="user_test_123",
        org_id="org_test_123",
        org_role="org:admin",
        org_slug="test-org",
        org_permissions=["org:read", "org:write"]
    )


@pytest.fixture
def mock_current_user(mock_auth_context):
    """Dependency override for get_current_user with mock AuthContext (personal mode)."""
    async def _mock_get_current_user():
        return mock_auth_context
    return _mock_get_current_user


@pytest.fixture
def mock_current_user_org(mock_org_auth_context):
    """Dependency override for get_current_user with org context (member)."""
    async def _mock_get_current_user():
        return mock_org_auth_context
    return _mock_get_current_user


@pytest.fixture
def mock_current_user_org_admin(mock_org_admin_auth_context):
    """Dependency override for get_current_user with org context (admin)."""
    async def _mock_get_current_user():
        return mock_org_admin_auth_context
    return _mock_get_current_user


@pytest.fixture
def mock_jwks() -> dict:
    """Mock JWKS response for JWT verification tests."""
    return {
        "keys": [{
            "kty": "RSA",
            "kid": "test-key-id",
            "use": "sig",
            "n": "test-modulus",
            "e": "AQAB",
        }]
    }


@pytest.fixture
def app():
    """Create a fresh FastAPI app instance for testing."""
    from main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app, override_get_db, mock_current_user) -> Generator:
    """Synchronous test client with mocked auth and database."""
    from core.auth import get_current_user
    from core.database import get_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = mock_current_user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


async def _create_async_client(
    app, override_get_db, override_get_session_factory, auth_override=None
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with specified dependency overrides."""
    from core.auth import get_current_user
    from core.database import get_db, get_session_factory

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_session_factory] = override_get_session_factory
    if auth_override:
        app.dependency_overrides[get_current_user] = auth_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(app, override_get_db, override_get_session_factory, mock_current_user) -> AsyncGenerator:
    """Async test client with mocked auth and database (personal mode)."""
    async for client in _create_async_client(app, override_get_db, override_get_session_factory, mock_current_user):
        yield client


@pytest.fixture
async def async_client_org(app, override_get_db, override_get_session_factory, mock_current_user_org) -> AsyncGenerator:
    """Async test client with org member context."""
    async for client in _create_async_client(app, override_get_db, override_get_session_factory, mock_current_user_org):
        yield client


@pytest.fixture
async def async_client_org_admin(app, override_get_db, override_get_session_factory, mock_current_user_org_admin) -> AsyncGenerator:
    """Async test client with org admin context."""
    async for client in _create_async_client(app, override_get_db, override_get_session_factory, mock_current_user_org_admin):
        yield client


@pytest.fixture
def unauthenticated_client(app, override_get_db) -> Generator:
    """Synchronous test client without auth mocking (for auth failure tests)."""
    from core.database import get_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
async def unauthenticated_async_client(app, override_get_db, override_get_session_factory) -> AsyncGenerator:
    """Async test client without auth mocking (for auth failure tests)."""
    async for client in _create_async_client(app, override_get_db, override_get_session_factory, auth_override=None):
        yield client


@pytest.fixture
async def test_user(db_session) -> User:
    """Create a test user matching the mock_user_payload subject."""
    user = User(id="user_test_123")
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def test_user_with_keys(db_session) -> User:
    """Create a test user with encryption keys set up."""
    user = User(id="user_test_encrypted_123")
    user.set_encryption_keys(
        public_key="aa" * 32,  # 64 hex chars
        encrypted_private_key="bb" * 48,  # Variable length
        iv="cc" * 16,  # 32 hex chars
        tag="dd" * 16,  # 32 hex chars
        salt="ee" * 32,  # 64 hex chars
        recovery_encrypted_private_key="ff" * 48,
        recovery_iv="11" * 16,
        recovery_tag="22" * 16,
        recovery_salt="33" * 32,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def other_user(db_session) -> User:
    """Create another user for authorization tests."""
    user = User(id="user_other_456")
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def test_organization(db_session) -> Organization:
    """Create a test organization."""
    org = Organization(id="org_test_123", name="Test Organization", slug="test-org")
    db_session.add(org)
    await db_session.flush()
    return org


@pytest.fixture
async def other_organization(db_session) -> Organization:
    """Create another organization for authorization tests."""
    org = Organization(id="org_other_456", name="Other Organization", slug="other-org")
    db_session.add(org)
    await db_session.flush()
    return org


@pytest.fixture
async def test_membership(db_session, test_user, test_organization) -> OrganizationMembership:
    """Create a membership for test user in test organization."""
    membership = OrganizationMembership(
        id=f"mem_{test_user.id}_{test_organization.id}",
        user_id=test_user.id,
        org_id=test_organization.id,
        role=MemberRole.MEMBER
    )
    db_session.add(membership)
    await db_session.flush()
    return membership


@pytest.fixture
async def test_admin_membership(db_session, test_user, test_organization) -> OrganizationMembership:
    """Create an admin membership for test user in test organization."""
    membership = OrganizationMembership(
        id=f"mem_admin_{test_user.id}_{test_organization.id}",
        user_id=test_user.id,
        org_id=test_organization.id,
        role=MemberRole.ADMIN
    )
    db_session.add(membership)
    await db_session.flush()
    return membership


@pytest.fixture
async def test_session(db_session, test_user) -> Session:
    """Create a chat session for the test user."""
    session = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="Test Session")
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.fixture
async def other_user_session(db_session, other_user) -> Session:
    """Create a session belonging to another user for authorization tests."""
    session = Session(id=str(uuid.uuid4()), user_id=other_user.id, name="Other User's Session")
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.fixture
async def test_org_session(db_session, test_user, test_organization) -> Session:
    """Create a session for test user within an organization."""
    session = Session(
        id=str(uuid.uuid4()),
        user_id=test_user.id,
        org_id=test_organization.id,
        name="Org Session"
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.fixture
async def other_user_org_session(db_session, other_user, test_organization) -> Session:
    """Create a session for another user within the same organization."""
    session = Session(
        id=str(uuid.uuid4()),
        user_id=other_user.id,
        org_id=test_organization.id,
        name="Other User's Org Session"
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.fixture
async def test_message(db_session, test_session) -> Message:
    """Create a single encrypted user message in the test session."""
    from tests.factories.message_factory import generate_encrypted_payload

    payload = generate_encrypted_payload("Hello, this is a test message")
    message = Message(
        id=str(uuid.uuid4()),
        session_id=test_session.id,
        role="user",
        ephemeral_public_key=payload["ephemeral_public_key"],
        iv=payload["iv"],
        ciphertext=payload["ciphertext"],
        auth_tag=payload["auth_tag"],
        hkdf_salt=payload["hkdf_salt"],
    )
    db_session.add(message)
    await db_session.flush()
    return message


@pytest.fixture
async def test_conversation(db_session, test_session) -> list[Message]:
    """Create a multi-message encrypted conversation in the test session."""
    from tests.factories.message_factory import generate_encrypted_payload

    payload1 = generate_encrypted_payload("Hello!")
    payload2 = generate_encrypted_payload("Hi there! How can I help you today?")
    payload3 = generate_encrypted_payload("What's the weather like?")

    messages = [
        Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            ephemeral_public_key=payload1["ephemeral_public_key"],
            iv=payload1["iv"],
            ciphertext=payload1["ciphertext"],
            auth_tag=payload1["auth_tag"],
            hkdf_salt=payload1["hkdf_salt"],
        ),
        Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="assistant",
            ephemeral_public_key=payload2["ephemeral_public_key"],
            iv=payload2["iv"],
            ciphertext=payload2["ciphertext"],
            auth_tag=payload2["auth_tag"],
            hkdf_salt=payload2["hkdf_salt"],
            model_used="Qwen/Qwen2.5-72B-Instruct",
        ),
        Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            ephemeral_public_key=payload3["ephemeral_public_key"],
            iv=payload3["iv"],
            ciphertext=payload3["ciphertext"],
            auth_tag=payload3["auth_tag"],
            hkdf_salt=payload3["hkdf_salt"],
        ),
    ]
    for msg in messages:
        db_session.add(msg)
    await db_session.flush()
    return messages


@pytest.fixture
def mock_llm_stream():
    """Mock LLM streaming response generator."""
    async def _mock_stream():
        for chunk in ["Hello", " ", "world", "!"]:
            yield chunk
    return _mock_stream


@pytest.fixture
def mock_hf_sse_response() -> list[bytes]:
    """Mock HuggingFace SSE response data."""
    return [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]


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
