"""Test that organizations router endpoints are properly documented in OpenAPI."""

import pytest


@pytest.mark.asyncio
async def test_all_org_endpoints_have_summary(async_client):
    """All org endpoints should have a summary."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    org_paths = {k: v for k, v in spec["paths"].items() if "/organizations" in k}
    for path, methods in org_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "summary" in details, f"{method.upper()} {path} missing summary"


@pytest.mark.asyncio
async def test_create_org_keys_has_response_model(async_client):
    """POST /organizations/{org_id}/keys should have a response model."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    path = spec["paths"]["/api/v1/organizations/{org_id}/keys"]["post"]
    assert "content" in path["responses"]["201"]


@pytest.mark.asyncio
async def test_distribute_key_has_response_model(async_client):
    """POST /organizations/{org_id}/distribute-key should have a response model."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    path = spec["paths"]["/api/v1/organizations/{org_id}/distribute-key"]["post"]
    assert "content" in path["responses"]["200"]


@pytest.mark.asyncio
async def test_all_org_endpoints_have_error_responses(async_client):
    """All org endpoints should document 401 error response."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    org_paths = {k: v for k, v in spec["paths"].items() if "/organizations" in k}
    for path, methods in org_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "401" in details["responses"], f"{method.upper()} {path} missing 401 response"
