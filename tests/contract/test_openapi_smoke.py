"""Smoke test that OpenAPI spec is valid and complete."""

import pytest


@pytest.mark.asyncio
async def test_openapi_spec_is_valid(openapi_spec):
    """OpenAPI spec should be valid JSON with required fields."""
    assert "openapi" in openapi_spec
    assert openapi_spec["openapi"].startswith("3.")
    assert "info" in openapi_spec
    assert "paths" in openapi_spec


@pytest.mark.asyncio
async def test_all_paths_have_operations(openapi_spec):
    """Every path should have at least one HTTP method."""
    for path, methods in openapi_spec["paths"].items():
        ops = [m for m in methods if m in ("get", "post", "put", "delete", "patch")]
        assert len(ops) > 0, f"Path {path} has no operations"


@pytest.mark.asyncio
async def test_no_empty_descriptions(openapi_spec):
    """No endpoint should have an empty description."""
    for path, methods in openapi_spec["paths"].items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete"):
                desc = details.get("description", "")
                if desc is not None:
                    assert desc != "", f"{method.upper()} {path} has empty description"
