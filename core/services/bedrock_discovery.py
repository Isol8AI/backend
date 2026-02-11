"""
Discover available Bedrock foundation models via ListFoundationModels API.

Replaces the hardcoded AVAILABLE_MODELS list in config.py with dynamic discovery.
Uses the same filtering logic as the OpenClaw TypeScript bedrock-discovery:
  - Active models only
  - Streaming supported
  - Text output modality

Results are cached for 1 hour to avoid repeated API calls.
"""

import logging
import time
from typing import TypedDict

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_cache: dict[str, list["DiscoveredModel"]] = {}
_cache_expires_at: float = 0
_CACHE_TTL_SECONDS = 3600  # 1 hour
_has_logged_error = False


class DiscoveredModel(TypedDict):
    id: str
    name: str


def _is_active(summary: dict) -> bool:
    status = summary.get("modelLifecycle", {}).get("status", "")
    return status.upper() == "ACTIVE"


def _has_text_output(summary: dict) -> bool:
    modalities = summary.get("outputModalities", [])
    return any(m.upper() == "TEXT" for m in modalities)


def _supports_streaming(summary: dict) -> bool:
    return summary.get("responseStreamingSupported", False) is True


def _should_include(summary: dict) -> bool:
    model_id = summary.get("modelId", "").strip()
    if not model_id:
        return False
    if not _supports_streaming(summary):
        return False
    if not _has_text_output(summary):
        return False
    if not _is_active(summary):
        return False
    return True


def discover_models(region: str = "us-east-1") -> list[DiscoveredModel]:
    """
    Call ListFoundationModels and return filtered, sorted model list.
    Results are cached for 1 hour. Returns empty list on error.
    """
    global _cache_expires_at, _has_logged_error

    now = time.time()
    cache_key = region
    if cache_key in _cache and _cache_expires_at > now:
        return _cache[cache_key]

    try:
        client = boto3.client("bedrock", region_name=region)
        response = client.list_foundation_models()
        summaries = response.get("modelSummaries", [])

        models: list[DiscoveredModel] = []
        for summary in summaries:
            if not _should_include(summary):
                continue
            model_id = summary["modelId"].strip()
            model_name = summary.get("modelName", "").strip() or model_id
            models.append({"id": model_id, "name": model_name})

        models.sort(key=lambda m: m["name"])

        _cache[cache_key] = models
        _cache_expires_at = now + _CACHE_TTL_SECONDS
        return models

    except (ClientError, Exception) as e:
        if not _has_logged_error:
            _has_logged_error = True
            logger.warning(f"Failed to discover Bedrock models: {e}")
        return _cache.get(cache_key, [])
