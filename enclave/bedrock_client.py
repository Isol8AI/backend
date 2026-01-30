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
3. Enclave uses botocore for SigV4 signing, VsockHttpClient for HTTP

For M6, we'll use KMS attestation to securely retrieve credentials
directly inside the enclave.
"""

import json
from typing import Dict, Optional, List, Any, Generator
from dataclasses import dataclass

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

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

    def _get_credentials(self) -> Credentials:
        """Get botocore Credentials object."""
        if not self.has_credentials():
            raise ValueError("AWS credentials not set.")
        return Credentials(
            access_key=self._access_key_id,
            secret_key=self._secret_access_key,
            token=self._session_token,
        )

    def _sign_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, str]:
        """Sign request using botocore SigV4Auth. Returns signed headers."""
        credentials = self._get_credentials()
        request = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(credentials, "bedrock", self.region).add_auth(request)
        return dict(request.headers)

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
        path = f"/model/{model_id}/converse"
        url = f"https://{self.host}{path}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Sign request using botocore (handles all SigV4 complexity)
        signed_headers = self._sign_request("POST", url, headers, body_bytes)

        # Make request via vsock
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

    def _parse_event_stream_message(self, data: bytes) -> dict:
        """
        Parse single AWS event stream message into event dict.

        Format:
        - Bytes 0-3: total length (big-endian)
        - Bytes 4-7: headers length (big-endian)
        - Bytes 8-11: prelude CRC
        - Bytes 12 to 12+headers_length: headers
        - Remaining (minus 4 byte CRC): payload
        """
        if len(data) < 16:
            return {}

        headers_length = int.from_bytes(data[4:8], "big")
        headers_end = 12 + headers_length

        # Extract payload (skip 4-byte message CRC at end)
        payload = data[headers_end:-4]

        if payload:
            try:
                return json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
        return {}

    def converse_stream(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, str]]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Call Bedrock Converse API with real streaming.

        Yields event dictionaries:
        - {"type": "content", "text": "chunk"}
        - {"type": "stop", "reason": "end_turn"}
        - {"type": "metadata", "usage": {"inputTokens": N, "outputTokens": N}}
        """
        # Build request body
        body = {"modelId": model_id, "messages": messages}
        if system:
            body["system"] = system
        if inference_config:
            body["inferenceConfig"] = inference_config

        body_bytes = json.dumps(body).encode("utf-8")

        # Streaming endpoint
        path = f"/model/{model_id}/converse-stream"
        url = f"https://{self.host}{path}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/vnd.amazon.eventstream",
        }

        # Sign request
        signed_headers = self._sign_request("POST", url, headers, body_bytes)

        # Stream response
        try:
            for raw_message in self.http_client.request_stream("POST", url, signed_headers, body_bytes):
                event = self._parse_event_stream_message(raw_message)

                if not event:
                    continue

                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield {"type": "content", "text": text}

                elif "messageStop" in event:
                    reason = event["messageStop"].get("stopReason", "end_turn")
                    yield {"type": "stop", "reason": reason}

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    yield {"type": "metadata", "usage": usage}

        except Exception as e:
            yield {"type": "error", "message": str(e)}


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
