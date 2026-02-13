"""
Tests for OpenAPI configuration on the FastAPI app.

Validates that the generated OpenAPI spec includes:
- Correct title, version, and description
- Server list with all environments
- Tag descriptions for all 9 route groups
- BearerAuth (JWT) security scheme
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock


EXPECTED_TAGS = [
    "users",
    "chat",
    "organizations",
    "agents",
    "context",
    "webhooks",
    "websocket",
    "debug",
    "town",
]


@pytest.fixture
async def openapi_spec():
    """Fetch the OpenAPI spec from the running app (no auth required)."""
    from main import app

    # Reset cached schema so custom_openapi regenerates it
    app.openapi_schema = None

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock())

    async def mock_get_db():
        yield mock_db

    from core.database import get_db

    app.dependency_overrides[get_db] = mock_get_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/openapi.json")
        assert response.status_code == 200, f"Failed to fetch OpenAPI spec: {response.status_code}"
        return response.json()
    finally:
        app.dependency_overrides.clear()


class TestOpenAPIMetadata:
    """Tests for top-level OpenAPI metadata fields."""

    @pytest.mark.asyncio
    async def test_title(self, openapi_spec):
        """Spec title should be 'Isol8 Chat API'."""
        assert openapi_spec["info"]["title"] == "Isol8 Chat API"

    @pytest.mark.asyncio
    async def test_version(self, openapi_spec):
        """Spec version should be '1.0.0'."""
        assert openapi_spec["info"]["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_description(self, openapi_spec):
        """Spec should have a non-empty description mentioning zero-trust."""
        description = openapi_spec["info"]["description"]
        assert len(description) > 0
        assert "zero-trust" in description.lower() or "Zero-trust" in description


class TestOpenAPIServers:
    """Tests for the servers list in the OpenAPI spec."""

    @pytest.mark.asyncio
    async def test_servers_present(self, openapi_spec):
        """Spec should include a servers list."""
        assert "servers" in openapi_spec
        assert len(openapi_spec["servers"]) == 4

    @pytest.mark.asyncio
    async def test_localhost_server(self, openapi_spec):
        """Servers list should include localhost:8000 for local dev."""
        urls = [s["url"] for s in openapi_spec["servers"]]
        assert "http://localhost:8000" in urls

    @pytest.mark.asyncio
    async def test_all_environments(self, openapi_spec):
        """Servers list should cover dev, staging, and production."""
        urls = [s["url"] for s in openapi_spec["servers"]]
        assert "https://api-dev.isol8.co" in urls
        assert "https://api-staging.isol8.co" in urls
        assert "https://api.isol8.co" in urls


class TestOpenAPITags:
    """Tests for tag metadata in the OpenAPI spec."""

    @pytest.mark.asyncio
    async def test_all_tags_present(self, openapi_spec):
        """Spec should include tag entries for all 9 route groups."""
        tag_names = [t["name"] for t in openapi_spec["tags"]]
        for expected in EXPECTED_TAGS:
            assert expected in tag_names, f"Missing tag: {expected}"

    @pytest.mark.asyncio
    async def test_all_tags_have_descriptions(self, openapi_spec):
        """Every tag should have a non-empty description."""
        for tag in openapi_spec["tags"]:
            assert "description" in tag, f"Tag '{tag['name']}' missing description"
            assert len(tag["description"]) > 0, f"Tag '{tag['name']}' has empty description"


class TestOpenAPISecurity:
    """Tests for the BearerAuth security scheme."""

    @pytest.mark.asyncio
    async def test_bearer_auth_scheme_exists(self, openapi_spec):
        """Spec should define a BearerAuth security scheme."""
        schemes = openapi_spec.get("components", {}).get("securitySchemes", {})
        assert "BearerAuth" in schemes

    @pytest.mark.asyncio
    async def test_bearer_auth_type(self, openapi_spec):
        """BearerAuth scheme should be type 'http' with scheme 'bearer'."""
        scheme = openapi_spec["components"]["securitySchemes"]["BearerAuth"]
        assert scheme["type"] == "http"
        assert scheme["scheme"] == "bearer"

    @pytest.mark.asyncio
    async def test_bearer_auth_format(self, openapi_spec):
        """BearerAuth scheme should specify bearerFormat as JWT."""
        scheme = openapi_spec["components"]["securitySchemes"]["BearerAuth"]
        assert scheme["bearerFormat"] == "JWT"

    @pytest.mark.asyncio
    async def test_global_security_requirement(self, openapi_spec):
        """Spec should have a global security requirement for BearerAuth."""
        assert "security" in openapi_spec
        security_names = [list(s.keys())[0] for s in openapi_spec["security"]]
        assert "BearerAuth" in security_names
