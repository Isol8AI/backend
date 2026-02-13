"""Test that all remaining router endpoints are properly documented in OpenAPI."""

import pytest


@pytest.mark.asyncio
async def test_every_endpoint_has_summary(async_client):
    """Every endpoint in the API should have a summary."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    missing = []
    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                if "summary" not in details:
                    missing.append(f"{method.upper()} {path}")
    assert not missing, f"Endpoints missing summary: {missing}"


@pytest.mark.asyncio
async def test_every_endpoint_has_operation_id(async_client):
    """Every endpoint in the API should have an operationId."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    missing = []
    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                if "operationId" not in details:
                    missing.append(f"{method.upper()} {path}")
    assert not missing, f"Endpoints missing operationId: {missing}"


@pytest.mark.asyncio
async def test_context_endpoints_have_401_responses(async_client):
    """Context router endpoints should document 401 error response."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    context_paths = {k: v for k, v in spec["paths"].items() if "/context" in k}
    assert context_paths, "No context paths found"
    for path, methods in context_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "401" in details["responses"], f"{method.upper()} {path} missing 401 response"


@pytest.mark.asyncio
async def test_debug_encryption_report_has_response_model(async_client):
    """GET /debug/encryption/report should have a response model."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    path = spec["paths"]["/api/v1/debug/encryption/report"]["get"]
    assert "content" in path["responses"]["200"]


@pytest.mark.asyncio
async def test_webhook_endpoint_has_summary(async_client):
    """POST /webhooks/clerk should have a summary."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    path = spec["paths"]["/api/v1/webhooks/clerk"]["post"]
    assert "summary" in path


@pytest.mark.asyncio
async def test_health_and_root_endpoints_have_summaries(async_client):
    """Root, health, and protected endpoints should have summaries."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    for endpoint in ["/", "/health", "/protected"]:
        assert endpoint in spec["paths"], f"{endpoint} not found in spec"
        get_op = spec["paths"][endpoint]["get"]
        assert "summary" in get_op, f"GET {endpoint} missing summary"


@pytest.mark.asyncio
async def test_ws_endpoints_have_summaries(async_client):
    """WebSocket HTTP POST endpoints should have summaries."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    ws_paths = {k: v for k, v in spec["paths"].items() if "/ws/" in k}
    assert ws_paths, "No websocket paths found"
    for path, methods in ws_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "summary" in details, f"{method.upper()} {path} missing summary"
