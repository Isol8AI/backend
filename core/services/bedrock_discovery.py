"""
Discover available Bedrock models via ListFoundationModels + ListInferenceProfiles.

Some Bedrock models (e.g., Claude 3.5 Sonnet, DeepSeek R1) require inference
profile IDs (us.anthropic.claude-...) for invocation and don't support
on-demand throughput with base model IDs (anthropic.claude-...).

This module:
  1. Calls ListFoundationModels for model metadata (name, capabilities)
  2. Calls ListInferenceProfiles to map base IDs → inference profile IDs
  3. Returns models with the correct invocation ID (inference profile if
     available, otherwise base model ID)

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


def _build_inference_profile_map(client) -> dict[str, str]:
    """
    Call ListInferenceProfiles and build a mapping from base model ID
    to inference profile ID.

    Example mapping:
      "anthropic.claude-3-5-sonnet-20241022-v2:0"
        -> "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    """
    mapping: dict[str, str] = {}
    try:
        paginator = client.get_paginator("list_inference_profiles")
        for page in paginator.paginate():
            for profile in page.get("inferenceProfileSummaries", []):
                profile_id = (profile.get("inferenceProfileId") or "").strip()
                if not profile_id:
                    continue
                for model_ref in profile.get("models", []):
                    model_arn = model_ref.get("modelArn") or ""
                    # ARN: arn:aws:bedrock:region::foundation-model/base-model-id
                    if "foundation-model/" in model_arn:
                        base_id = model_arn.split("foundation-model/")[-1].strip()
                        if base_id:
                            mapping[base_id] = profile_id
    except (ClientError, Exception) as e:
        logger.debug(f"ListInferenceProfiles unavailable (non-fatal): {e}")
    return mapping


def discover_models(region: str = "us-east-1") -> list[DiscoveredModel]:
    """
    Call ListFoundationModels + ListInferenceProfiles, return filtered,
    sorted model list with correct invocation IDs.

    Results are cached for 1 hour. Returns empty list on error.
    """
    global _cache_expires_at, _has_logged_error

    now = time.time()
    cache_key = region
    if cache_key in _cache and _cache_expires_at > now:
        return _cache[cache_key]

    try:
        client = boto3.client("bedrock", region_name=region)

        # Get model metadata
        response = client.list_foundation_models()
        summaries = response.get("modelSummaries", [])

        # Get base-ID → inference-profile-ID mapping
        profile_map = _build_inference_profile_map(client)

        models: list[DiscoveredModel] = []
        for summary in summaries:
            if not _should_include(summary):
                continue
            base_id = summary["modelId"].strip()
            model_name = summary.get("modelName", "").strip() or base_id
            # Use inference profile ID for invocation if available
            invocation_id = profile_map.get(base_id, base_id)
            models.append({"id": invocation_id, "name": model_name})

        models.sort(key=lambda m: m["name"])

        _cache[cache_key] = models
        _cache_expires_at = now + _CACHE_TTL_SECONDS
        _has_logged_error = False
        return models

    except (ClientError, Exception) as e:
        if not _has_logged_error:
            _has_logged_error = True
            logger.warning(f"Failed to discover Bedrock models: {e}")
        return _cache.get(cache_key, [])
