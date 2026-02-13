"""Test that agents router endpoints are properly documented in OpenAPI."""

import pytest


@pytest.mark.asyncio
async def test_all_agent_endpoints_have_summary(async_client):
    """All agent endpoints should have a summary."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    agent_paths = {k: v for k, v in spec["paths"].items() if "/agents" in k}
    assert agent_paths, "No agent paths found in OpenAPI spec"
    for path, methods in agent_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "summary" in details, f"{method.upper()} {path} missing summary"


@pytest.mark.asyncio
async def test_agent_endpoints_have_operation_ids(async_client):
    """All agent endpoints should have unique operation IDs."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    agent_paths = {k: v for k, v in spec["paths"].items() if "/agents" in k}
    operation_ids = set()
    for path, methods in agent_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                op_id = details.get("operationId")
                assert op_id, f"{method.upper()} {path} missing operationId"
                assert op_id not in operation_ids, f"Duplicate operationId: {op_id}"
                operation_ids.add(op_id)


@pytest.mark.asyncio
async def test_all_agent_endpoints_have_error_responses(async_client):
    """All agent endpoints should document 401 error response."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    agent_paths = {k: v for k, v in spec["paths"].items() if "/agents" in k}
    for path, methods in agent_paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                assert "401" in details["responses"], f"{method.upper()} {path} missing 401 response"


@pytest.mark.asyncio
async def test_create_agent_has_201_response(async_client):
    """POST /agents should return 201 Created."""
    response = await async_client.get("/api/v1/openapi.json")
    spec = response.json()
    path = spec["paths"]["/api/v1/agents"]["post"]
    assert "201" in path["responses"]
