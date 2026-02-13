"""
Contract test fixtures.

Uses httpx AsyncClient with ASGITransport and mocked auth/enclave dependencies.
No real database, enclave, or Clerk needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

from core.auth import AuthContext


@pytest.fixture
def mock_auth():
    """Mock auth context for contract tests."""
    return AuthContext(user_id="contract_test_user")


@pytest.fixture
async def contract_client(mock_auth):
    """
    Async test client for contract tests.

    Overrides auth and database dependencies so tests don't need
    real infrastructure.
    """
    from main import app
    from core.auth import get_current_user
    from core.database import get_db, get_session_factory

    async def mock_get_current_user():
        return mock_auth

    async def mock_get_db():
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_result.scalars.return_value.all.return_value = []
        mock_result.scalar_one_or_none.return_value = None
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        yield mock_session

    def mock_get_session_factory():
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_result.scalars.return_value.all.return_value = []
        mock_result.scalar_one_or_none.return_value = None
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        class _Ctx:
            def __call__(self):
                return self

            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, *_):
                pass

        return _Ctx()

    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[get_db] = mock_get_db
    app.dependency_overrides[get_session_factory] = mock_get_session_factory

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
async def openapi_spec(contract_client):
    """Fetch and return the OpenAPI spec as a dict."""
    response = await contract_client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    return response.json()
