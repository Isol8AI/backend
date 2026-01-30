#!/usr/bin/env python3
"""
M4: AWS Bedrock Client for Nitro Enclave
=========================================

This module provides AWS Bedrock API access from inside a Nitro Enclave
using the Converse API - a unified interface that works with ALL Bedrock
models (Claude, Llama, Nova, Titan, Mistral, etc.).

Key security properties:
1. AWS credentials are retrieved by parent from IAM role (IMDS)
2. Credentials are passed to enclave via vsock
3. Plaintext prompts/responses only exist in enclave memory
4. TLS termination happens inside the enclave

Credential flow for M4:
1. Parent retrieves temporary credentials from EC2 IMDS (IAM role)
2. Parent sends credentials to enclave via SET_CREDENTIALS command
3. Enclave uses credentials with SigV4 for Bedrock API calls

For M6, we'll use KMS attestation to securely retrieve credentials
directly inside the enclave.
"""

import json
import hashlib
import hmac
import datetime
from typing import Dict, Optional, List, Any, Generator
from dataclasses import dataclass

from vsock_http_client import VsockHttpClient


@dataclass
class ConverseTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class BedrockResponse:
    """Response from Bedrock Converse API."""

    content: str
    model_id: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    raw_response: dict


class SigV4Signer:
    """AWS Signature Version 4 signer."""

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        service: str,
        session_token: Optional[str] = None,
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.service = service
        self.session_token = session_token

    def _sign(self, key: bytes, msg: str) -> bytes:
        """HMAC-SHA256 signing."""
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _get_signature_key(self, date_stamp: str) -> bytes:
        """Derive signing key."""
        k_date = self._sign(f"AWS4{self.secret_access_key}".encode("utf-8"), date_stamp)
        k_region = self._sign(k_date, self.region)
        k_service = self._sign(k_region, self.service)
        k_signing = self._sign(k_service, "aws4_request")
        return k_signing

    def sign_request(
        self,
        method: str,
        host: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, str]:
        """Sign an AWS API request. Returns headers dict with Authorization added."""
        t = datetime.datetime.utcnow()
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = t.strftime("%Y%m%d")

        signed_headers_dict = dict(headers)
        signed_headers_dict["host"] = host
        signed_headers_dict["x-amz-date"] = amz_date

        if self.session_token:
            signed_headers_dict["x-amz-security-token"] = self.session_token

        payload_hash = hashlib.sha256(body).hexdigest()
        signed_headers_dict["x-amz-content-sha256"] = payload_hash

        sorted_headers = sorted(signed_headers_dict.keys())
        canonical_headers = ""
        for key in sorted_headers:
            canonical_headers += f"{key.lower()}:{signed_headers_dict[key].strip()}\n"
        signed_headers = ";".join(key.lower() for key in sorted_headers)

        canonical_request = "\n".join(
            [
                method,
                path,
                "",  # empty query string
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )

        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/aws4_request"
        string_to_sign = "\n".join(
            [
                algorithm,
                amz_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )

        signing_key = self._get_signature_key(date_stamp)
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        authorization_header = (
            f"{algorithm} "
            f"Credential={self.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        result_headers = dict(signed_headers_dict)
        result_headers["Authorization"] = authorization_header
        return result_headers


class BedrockClient:
    """
    AWS Bedrock client for Nitro Enclave using the Converse API.

    The Converse API provides a unified interface for ALL Bedrock models:
    - Anthropic Claude
    - Meta Llama
    - Amazon Nova/Titan
    - Mistral
    - And more

    No model-specific request/response parsing needed.
    """

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.host = f"bedrock-runtime.{region}.amazonaws.com"
        self.http_client = VsockHttpClient()

        # Credentials (set by parent via SET_CREDENTIALS command)
        self._access_key_id: Optional[str] = None
        self._secret_access_key: Optional[str] = None
        self._session_token: Optional[str] = None
        self._credentials_expiration: Optional[str] = None

    def set_credentials(
        self,
        access_key_id: str,
        secret_access_key: str,
        session_token: str,
        expiration: Optional[str] = None,
    ):
        """Set AWS credentials from IAM role (via parent IMDS)."""
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._credentials_expiration = expiration

    def has_credentials(self) -> bool:
        """Check if credentials are set."""
        return bool(self._access_key_id and self._secret_access_key and self._session_token)

    def _create_signer(self) -> SigV4Signer:
        """Create SigV4 signer with current credentials."""
        if not self.has_credentials():
            raise ValueError("AWS credentials not set.")
        return SigV4Signer(
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            region=self.region,
            service="bedrock",
            session_token=self._session_token,
        )

    def converse(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, str]]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> BedrockResponse:
        """
        Call Bedrock Converse API (non-streaming).

        This unified API works with ALL Bedrock models.

        Args:
            model_id: Model identifier (e.g., "us.anthropic.claude-3-5-haiku-20241022-v1:0")
            messages: List of messages in Converse format:
                [{"role": "user", "content": [{"text": "Hello"}]}]
            system: Optional system prompts: [{"text": "You are helpful."}]
            inference_config: Optional config: {"maxTokens": 4096, "temperature": 0.7}

        Returns:
            BedrockResponse with unified response format
        """
        # Build request body
        body = {"modelId": model_id, "messages": messages}
        if system:
            body["system"] = system
        if inference_config:
            body["inferenceConfig"] = inference_config

        body_bytes = json.dumps(body).encode("utf-8")

        # Converse API endpoint
        path = "/model/{}/converse".format(model_id.replace("/", "%2F"))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Sign request
        signer = self._create_signer()
        signed_headers = signer.sign_request("POST", self.host, path, headers, body_bytes)

        # Make request
        url = f"https://{self.host}{path}"
        response = self.http_client.request("POST", url, signed_headers, body_bytes)

        if response.status != 200:
            error_msg = f"Bedrock Converse error: {response.status}"
            try:
                error_body = response.json()
                error_msg += f" - {error_body.get('message', response.body.decode('utf-8'))}"
            except Exception:
                error_msg += f" - {response.body.decode('utf-8')}"
            raise Exception(error_msg)

        return self._parse_converse_response(model_id, response.json())

    def _parse_converse_response(self, model_id: str, data: dict) -> BedrockResponse:
        """Parse Converse API response (unified format for all models)."""
        # Extract content from output message
        output_message = data.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        content = ""
        for block in content_blocks:
            if "text" in block:
                content += block["text"]

        # Extract usage
        usage = data.get("usage", {})

        return BedrockResponse(
            content=content,
            model_id=model_id,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            stop_reason=data.get("stopReason", "end_turn"),
            raw_response=data,
        )

    def converse_stream(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, str]]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Call Bedrock Converse API with streaming.

        Yields events as they arrive:
        - {"contentBlockDelta": {"delta": {"text": "chunk"}}}
        - {"messageStop": {"stopReason": "end_turn"}}
        - {"metadata": {"usage": {"inputTokens": N, "outputTokens": N}}}

        Args:
            model_id: Model identifier
            messages: Conversation messages
            system: System prompts
            inference_config: Inference configuration

        Yields:
            Event dictionaries from the stream
        """
        # Build request body
        body = {"modelId": model_id, "messages": messages}
        if system:
            body["system"] = system
        if inference_config:
            body["inferenceConfig"] = inference_config

        body_bytes = json.dumps(body).encode("utf-8")

        # Streaming endpoint
        path = "/model/{}/converse-stream".format(model_id.replace("/", "%2F"))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/vnd.amazon.eventstream",
        }

        # Sign request
        signer = self._create_signer()
        signed_headers = signer.sign_request("POST", self.host, path, headers, body_bytes)

        # Make streaming request
        url = f"https://{self.host}{path}"

        # For streaming, we need a different approach - connect and read events
        # This is a simplified implementation for M4
        # Full implementation would parse AWS event stream format
        response = self.http_client.request("POST", url, signed_headers, body_bytes)

        if response.status != 200:
            error_msg = f"Bedrock stream error: {response.status}"
            try:
                error_body = response.json()
                error_msg += f" - {error_body.get('message', response.body.decode('utf-8'))}"
            except Exception:
                error_msg += f" - {response.body.decode('utf-8')}"
            raise Exception(error_msg)

        # Parse event stream response
        # AWS event stream format is binary, but for M4 testing we'll use non-streaming
        # and simulate events from the full response
        full_response = self._parse_converse_response(model_id, response.json())

        # Simulate streaming events for compatibility
        yield {"contentBlockStart": {"contentBlockIndex": 0}}
        yield {"contentBlockDelta": {"delta": {"text": full_response.content}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": full_response.stop_reason}}
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": full_response.input_tokens,
                    "outputTokens": full_response.output_tokens,
                }
            }
        }


def build_converse_messages(
    history: List[ConverseTurn],
    current_message: str,
) -> List[Dict[str, Any]]:
    """
    Build messages array for Converse API.

    Args:
        history: Previous conversation turns
        current_message: Current user message

    Returns:
        Messages in Converse API format
    """
    messages = []
    for turn in history:
        messages.append(
            {
                "role": turn.role,
                "content": [{"text": turn.content}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"text": current_message}],
        }
    )
    return messages


if __name__ == "__main__":
    print("Bedrock Client (Converse API) initialized.")
    print("Supports ALL Bedrock models through unified Converse API:")
    print("  - Anthropic Claude")
    print("  - Meta Llama")
    print("  - Amazon Nova/Titan")
    print("  - Mistral")
    print("  - And more")
