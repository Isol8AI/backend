"""
Tests for health check endpoint.

The health endpoint must return:
- HTTP 200 with {"status": "healthy"} when all checks pass
- HTTP 503 with {"status": "unhealthy"} when any check fails

This is critical for ALB health checks - returning 200 on failure
would cause ALB to route traffic to unhealthy instances.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200_when_database_connected(self):
        """Health endpoint returns 200 when database is healthy."""
        from main import app

        # Mock a successful database query
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock())

        async def mock_get_db():
            yield mock_db

        app.dependency_overrides = {}

        from core.database import get_db
        app.dependency_overrides[get_db] = mock_get_db

        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["database"] == "connected"
        finally:
            app.dependency_overrides = {}

    @pytest.mark.asyncio
    async def test_health_returns_503_when_database_disconnected(self):
        """Health endpoint returns 503 when database check fails."""
        from main import app

        # Mock a failed database query
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("Connection refused"))

        async def mock_get_db():
            yield mock_db

        app.dependency_overrides = {}

        from core.database import get_db
        app.dependency_overrides[get_db] = mock_get_db

        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")

            # ALB expects 503 for unhealthy instances
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["database"] == "disconnected"
        finally:
            app.dependency_overrides = {}
