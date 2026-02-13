"""Test that chat router endpoints are properly documented in OpenAPI."""

import pytest


@pytest.mark.asyncio
async def test_all_chat_endpoints_have_summary(async_client):
    """All chat endpoints should have a summary."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    chat_paths = {k: v for k, v in spec["paths"].items() if "/chat/" in k}
    for path, methods in chat_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "summary" in details, f"{method.upper()} {path} missing summary"


@pytest.mark.asyncio
async def test_chat_encryption_status_has_response_model(async_client):
    """GET /chat/encryption-status should have a documented response schema."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    path = spec["paths"]["/api/v1/chat/encryption-status"]["get"]
    assert "content" in path["responses"]["200"]


@pytest.mark.asyncio
async def test_all_chat_endpoints_have_error_responses(async_client):
    """All authenticated chat endpoints should document 401 error response."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    # /chat/models is unauthenticated, skip it
    unauthenticated = {"/api/v1/chat/models"}
    chat_paths = {k: v for k, v in spec["paths"].items() if "/chat/" in k and k not in unauthenticated}
    for path, methods in chat_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "401" in details["responses"], f"{method.upper()} {path} missing 401 response"


@pytest.mark.asyncio
async def test_chat_endpoints_have_operation_ids(async_client):
    """All chat endpoints should have unique operation IDs."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    chat_paths = {k: v for k, v in spec["paths"].items() if "/chat/" in k}
    operation_ids = []
    for path, methods in chat_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "operationId" in details, f"{method.upper()} {path} missing operationId"
                operation_ids.append(details["operationId"])
    # All operation IDs should be unique
    assert len(operation_ids) == len(set(operation_ids)), "Duplicate operation IDs found"
