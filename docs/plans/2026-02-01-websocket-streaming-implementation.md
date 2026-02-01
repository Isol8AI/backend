
# WebSocket Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SSE streaming with WebSocket via API Gateway WebSocket API to eliminate the 30-second timeout buffering issue.

**Architecture:** Add a new API Gateway WebSocket API alongside the existing HTTP API. WebSocket connections are authenticated via Lambda authorizer that validates Clerk JWT on `$connect`. The backend FastAPI server handles WebSocket connections and reuses the existing `ChatService.process_encrypted_message_stream()` for actual message processing.

**Tech Stack:** FastAPI WebSocket, AWS API Gateway WebSocket API, AWS Lambda (Python 3.11), Terraform, TypeScript/React

---

## Task 1: Backend WebSocket Router

**Files:**
- Create: `backend/routers/websocket_chat.py`
- Modify: `backend/main.py:56-63`

**Step 1: Create the WebSocket router file**

Create `backend/routers/websocket_chat.py`:

```python
"""
WebSocket chat endpoint for real-time streaming.

Replaces SSE streaming to avoid API Gateway HTTP API 30s timeout/buffering.
Uses the same ChatService and encryption flow as the SSE endpoint.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session_factory
from core.services.chat_service import ChatService
from core.crypto import EncryptedPayload
from schemas.encryption import EncryptedPayload as EncryptedPayloadSchema

logger = logging.getLogger(__name__)
router = APIRouter()

# Ping interval to keep connection alive (ALB has 300s idle timeout)
PING_INTERVAL_SECONDS = 30


@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
):
    """
    WebSocket endpoint for encrypted chat streaming.

    Authentication:
    - In production: Lambda authorizer validates JWT on $connect
    - User ID passed via x-user-id header from API Gateway
    - For local dev: token query param validated here

    Message flow:
    1. Client connects with token
    2. Client sends encrypted message JSON
    3. Server streams encrypted response chunks
    4. Server sends 'done' when complete
    """
    await websocket.accept()

    # Get user ID from API Gateway context (set by Lambda authorizer)
    # API Gateway passes authorizer context via headers
    user_id = websocket.headers.get("x-amzn-request-context-userid") or websocket.headers.get("x-user-id")
    org_id = websocket.headers.get("x-amzn-request-context-orgid") or websocket.headers.get("x-org-id")

    # For local development without API Gateway
    if not user_id and token:
        # Validate token locally (simplified for dev)
        from core.auth import validate_clerk_token
        try:
            auth_context = await validate_clerk_token(token)
            user_id = auth_context.user_id
            org_id = auth_context.org_id
        except Exception as e:
            logger.warning(f"WebSocket auth failed: {e}")
            await websocket.close(code=4001, reason="Unauthorized")
            return

    if not user_id:
        logger.warning("WebSocket connection without user_id")
        await websocket.close(code=4001, reason="Unauthorized")
        return

    logger.info(f"WebSocket connected: user_id={user_id}, org_id={org_id}")

    # Keepalive ping task
    async def send_pings():
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    ping_task = asyncio.create_task(send_pings())

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Handle pong response
            if data.get("type") == "pong":
                continue

            # Process chat message
            await _handle_chat_message(websocket, data, user_id, org_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "Internal error"})
        except Exception:
            pass
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass


async def _handle_chat_message(
    websocket: WebSocket,
    data: dict,
    user_id: str,
    org_id: Optional[str],
):
    """Handle a single chat message request."""
    try:
        # Parse request data
        encrypted_message_data = data.get("encrypted_message")
        encrypted_history_data = data.get("encrypted_history", [])
        client_transport_public_key = data.get("client_transport_public_key")
        model = data.get("model", "us.amazon.nova-micro-v1:0")
        session_id = data.get("session_id")
        facts_context = data.get("facts_context")
        request_org_id = data.get("org_id") or org_id

        if not encrypted_message_data or not client_transport_public_key:
            await websocket.send_json({
                "type": "error",
                "message": "Missing encrypted_message or client_transport_public_key"
            })
            return

        # Convert to crypto payloads
        encrypted_msg = EncryptedPayloadSchema(**encrypted_message_data).to_crypto_payload()
        encrypted_history = [
            EncryptedPayloadSchema(**h).to_crypto_payload()
            for h in encrypted_history_data
        ]

        async with async_session_factory() as db:
            service = ChatService(db)

            # Verify user can send encrypted messages
            can_send, error_msg = await service.verify_can_send_encrypted(
                user_id=user_id,
                org_id=request_org_id,
            )
            if not can_send:
                await websocket.send_json({"type": "error", "message": error_msg})
                return

            # Get or create session
            if session_id:
                session = await service.get_session(
                    session_id=session_id,
                    user_id=user_id,
                    org_id=request_org_id,
                )
                if not session:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not found or access denied"
                    })
                    return
            else:
                session = await service.create_session(
                    user_id=user_id,
                    name="New Chat",
                    org_id=request_org_id,
                )
                session_id = session.id

            # Send session ID
            await websocket.send_json({
                "type": "session",
                "session_id": session_id
            })

            # Stream response
            chunk_count = 0
            async for chunk in service.process_encrypted_message_stream(
                session_id=session_id,
                user_id=user_id,
                org_id=request_org_id,
                encrypted_message=encrypted_msg,
                encrypted_history=encrypted_history,
                facts_context=facts_context,
                model=model,
                client_transport_public_key=client_transport_public_key,
            ):
                if chunk.error:
                    await websocket.send_json({
                        "type": "error",
                        "message": chunk.error
                    })
                    return

                if chunk.encrypted_content:
                    chunk_count += 1
                    api_payload = EncryptedPayloadSchema.from_crypto_payload(chunk.encrypted_content)
                    await websocket.send_json({
                        "type": "encrypted_chunk",
                        "encrypted_content": api_payload.model_dump()
                    })

                if chunk.encrypted_thinking:
                    api_payload = EncryptedPayloadSchema.from_crypto_payload(chunk.encrypted_thinking)
                    await websocket.send_json({
                        "type": "thinking",
                        "encrypted_content": api_payload.model_dump()
                    })

                if chunk.is_final and chunk.stored_user_message and chunk.stored_assistant_message:
                    await websocket.send_json({
                        "type": "stored",
                        "model_used": chunk.model_used,
                        "input_tokens": chunk.input_tokens,
                        "output_tokens": chunk.output_tokens,
                    })

            # Send done
            await websocket.send_json({"type": "done"})
            logger.debug(f"WebSocket stream complete: session={session_id}, chunks={chunk_count}")

    except json.JSONDecodeError as e:
        await websocket.send_json({"type": "error", "message": f"Invalid JSON: {e}"})
    except Exception as e:
        logger.exception(f"Error handling chat message: {e}")
        await websocket.send_json({"type": "error", "message": "Internal error"})
```

**Step 2: Register the WebSocket router in main.py**

Modify `backend/main.py` - add import and include router:

After line 17 (existing router imports), add:
```python
from routers import users, chat, organizations, context, webhooks, debug_encryption, websocket_chat
```

After line 63 (debug routes), add:
```python
# WebSocket routes (no prefix - path includes /ws)
app.include_router(websocket_chat.router, tags=["websocket"])
```

**Step 3: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff format routers/websocket_chat.py && ruff check routers/websocket_chat.py --fix`

**Step 4: Test locally with wscat**

Run backend: `cd /Users/prasiddhaparthsarthy/Desktop/freebird && ./start_dev.sh`

In another terminal: `wscat -c "ws://localhost:8000/ws/chat?token=<your-clerk-token>"`

Expected: Connection established, can send/receive JSON messages

**Step 5: Commit**

```bash
git add routers/websocket_chat.py main.py
git commit -m "feat: add WebSocket endpoint for chat streaming

Adds /ws/chat WebSocket endpoint that reuses ChatService for
encrypted message processing. Supports:
- Authentication via token query param (local) or API Gateway headers
- Keepalive ping/pong every 30s
- Same encryption flow as SSE endpoint

Part of WebSocket streaming implementation to fix API Gateway
30s timeout issue."
```

---

## Task 2: Backend WebSocket Unit Tests

**Files:**
- Create: `backend/tests/unit/routers/test_websocket_chat.py`

**Step 1: Create test file**

Create `backend/tests/unit/routers/test_websocket_chat.py`:

```python
"""Unit tests for WebSocket chat endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


class TestWebSocketConnection:
    """Tests for WebSocket connection handling."""

    @pytest.mark.asyncio
    async def test_connection_without_auth_closes_with_4001(self, app):
        """WebSocket without authentication should close with 4001."""
        client = TestClient(app)

        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat") as websocket:
                # Should not reach here
                pass

    @pytest.mark.asyncio
    async def test_connection_with_invalid_token_closes(self, app):
        """WebSocket with invalid token should close."""
        client = TestClient(app)

        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat?token=invalid") as websocket:
                pass


class TestWebSocketPingPong:
    """Tests for keepalive ping/pong."""

    @pytest.mark.asyncio
    async def test_pong_response_is_handled(self, app, mock_auth_token):
        """Client pong responses should be handled without error."""
        client = TestClient(app)

        with patch("routers.websocket_chat.validate_clerk_token") as mock_validate:
            mock_validate.return_value = MagicMock(user_id="user_123", org_id=None)

            with client.websocket_connect(f"/ws/chat?token={mock_auth_token}") as websocket:
                # Send pong (simulating response to ping)
                websocket.send_json({"type": "pong"})
                # Connection should stay open - send another message
                websocket.send_json({"type": "pong"})


class TestWebSocketMessageHandling:
    """Tests for chat message handling."""

    @pytest.mark.asyncio
    async def test_missing_encrypted_message_returns_error(self, app, mock_auth_token):
        """Request without encrypted_message should return error."""
        client = TestClient(app)

        with patch("routers.websocket_chat.validate_clerk_token") as mock_validate:
            mock_validate.return_value = MagicMock(user_id="user_123", org_id=None)

            with client.websocket_connect(f"/ws/chat?token={mock_auth_token}") as websocket:
                websocket.send_json({
                    "client_transport_public_key": "abc123",
                    "model": "test-model"
                    # Missing encrypted_message
                })

                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "encrypted_message" in response["message"]

    @pytest.mark.asyncio
    async def test_missing_transport_key_returns_error(self, app, mock_auth_token):
        """Request without client_transport_public_key should return error."""
        client = TestClient(app)

        with patch("routers.websocket_chat.validate_clerk_token") as mock_validate:
            mock_validate.return_value = MagicMock(user_id="user_123", org_id=None)

            with client.websocket_connect(f"/ws/chat?token={mock_auth_token}") as websocket:
                websocket.send_json({
                    "encrypted_message": {"ephemeral_public_key": "...", "iv": "...", "ciphertext": "...", "tag": "..."},
                    "model": "test-model"
                    # Missing client_transport_public_key
                })

                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "client_transport_public_key" in response["message"]
```

**Step 2: Run tests**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && pytest tests/unit/routers/test_websocket_chat.py -v`

Expected: All tests pass (some may be skipped if fixtures need adjustment)

**Step 3: Commit**

```bash
git add tests/unit/routers/test_websocket_chat.py
git commit -m "test: add WebSocket chat endpoint unit tests

Tests connection handling, ping/pong keepalive, and message
validation for the WebSocket chat endpoint."
```

---

## Task 3: Lambda Authorizer for Clerk JWT

**Files:**
- Create: `terraform/lambda/websocket-authorizer/index.py`
- Create: `terraform/lambda/websocket-authorizer/requirements.txt`

**Step 1: Create Lambda authorizer directory and code**

Create `terraform/lambda/websocket-authorizer/index.py`:

```python
"""
Lambda authorizer for WebSocket API Gateway.

Validates Clerk JWT tokens on $connect and passes user context
to the backend via the authorizer response context.
"""

import os
import logging
from typing import Any, Dict

import jwt
from jwt import PyJWKClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables (set by Terraform)
CLERK_JWKS_URL = os.environ.get("CLERK_JWKS_URL", "")
CLERK_ISSUER = os.environ.get("CLERK_ISSUER", "")

# Cache JWKS client for performance
_jwks_client = None


def get_jwks_client() -> PyJWKClient:
    """Get cached JWKS client."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(CLERK_JWKS_URL, cache_keys=True, lifespan=3600)
    return _jwks_client


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda authorizer handler for WebSocket $connect.

    Args:
        event: API Gateway WebSocket authorizer event
        context: Lambda context

    Returns:
        Policy document allowing/denying connection
    """
    logger.info(f"Authorizer event: {event}")

    # Extract token from query string
    query_params = event.get("queryStringParameters") or {}
    token = query_params.get("token")

    if not token:
        logger.warning("No token provided")
        return {"isAuthorized": False}

    if not CLERK_JWKS_URL or not CLERK_ISSUER:
        logger.error("CLERK_JWKS_URL or CLERK_ISSUER not configured")
        return {"isAuthorized": False}

    try:
        # Get signing key from JWKS
        jwks_client = get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Decode and validate token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={
                "verify_aud": False,  # Clerk doesn't always set audience
                "verify_exp": True,
            },
        )

        user_id = payload.get("sub")
        org_id = payload.get("org_id")

        if not user_id:
            logger.warning("Token missing 'sub' claim")
            return {"isAuthorized": False}

        logger.info(f"Authorized: user_id={user_id}, org_id={org_id}")

        # Return authorization with context
        # Context is passed to backend via request headers
        return {
            "isAuthorized": True,
            "context": {
                "userId": user_id,
                "orgId": org_id or "",
            },
        }

    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return {"isAuthorized": False}
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return {"isAuthorized": False}
    except Exception as e:
        logger.exception(f"Authorizer error: {e}")
        return {"isAuthorized": False}
```

**Step 2: Create requirements.txt**

Create `terraform/lambda/websocket-authorizer/requirements.txt`:

```
PyJWT>=2.8.0
cryptography>=41.0.0
```

**Step 3: Commit**

```bash
git add terraform/lambda/websocket-authorizer/
git commit -m "feat: add Lambda authorizer for WebSocket JWT validation

Lambda function that validates Clerk JWT tokens on WebSocket
$connect and passes user/org context to the backend."
```

---

## Task 4: Terraform WebSocket API Module

**Files:**
- Create: `terraform/modules/websocket-api/main.tf`
- Create: `terraform/modules/websocket-api/lambda.tf`
- Create: `terraform/modules/websocket-api/variables.tf`
- Create: `terraform/modules/websocket-api/outputs.tf`

**Step 1: Create main.tf**

Create `terraform/modules/websocket-api/main.tf`:

```hcl
# =============================================================================
# WebSocket API Gateway Module
# =============================================================================
# Creates a WebSocket API for real-time chat streaming.
# Uses Lambda authorizer to validate Clerk JWT on $connect.
#
# Architecture:
#   Browser → WebSocket API Gateway → VPC Link → ALB → EC2
# =============================================================================

# -----------------------------------------------------------------------------
# WebSocket API
# -----------------------------------------------------------------------------
resource "aws_apigatewayv2_api" "websocket" {
  name                       = "${var.project}-${var.environment}-websocket"
  protocol_type              = "WEBSOCKET"
  route_selection_expression = "$request.body.action"

  tags = {
    Name = "${var.project}-${var.environment}-websocket"
  }
}

# -----------------------------------------------------------------------------
# Lambda Authorizer
# -----------------------------------------------------------------------------
resource "aws_apigatewayv2_authorizer" "clerk_jwt" {
  api_id                     = aws_apigatewayv2_api.websocket.id
  authorizer_type            = "REQUEST"
  authorizer_uri             = aws_lambda_function.authorizer.invoke_arn
  identity_sources           = ["route.request.querystring.token"]
  name                       = "clerk-jwt-authorizer"
  authorizer_payload_format_version = "2.0"
  enable_simple_responses    = true
}

# -----------------------------------------------------------------------------
# VPC Link (reuse existing or create new)
# -----------------------------------------------------------------------------
resource "aws_apigatewayv2_vpc_link" "websocket" {
  name               = "${var.project}-${var.environment}-websocket-vpc-link"
  security_group_ids = [var.alb_security_group_id]
  subnet_ids         = var.subnet_ids

  tags = {
    Name = "${var.project}-${var.environment}-websocket-vpc-link"
  }
}

# -----------------------------------------------------------------------------
# Integration with ALB
# -----------------------------------------------------------------------------
resource "aws_apigatewayv2_integration" "alb" {
  api_id             = aws_apigatewayv2_api.websocket.id
  integration_type   = "HTTP_PROXY"
  integration_uri    = var.alb_listener_arn
  integration_method = "ANY"
  connection_type    = "VPC_LINK"
  connection_id      = aws_apigatewayv2_vpc_link.websocket.id

  # Pass authorizer context to backend via headers
  request_parameters = {
    "integration.request.header.x-user-id" = "context.authorizer.userId"
    "integration.request.header.x-org-id"  = "context.authorizer.orgId"
  }
}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

# $connect - with authorization
resource "aws_apigatewayv2_route" "connect" {
  api_id             = aws_apigatewayv2_api.websocket.id
  route_key          = "$connect"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.clerk_jwt.id
  target             = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

# $disconnect
resource "aws_apigatewayv2_route" "disconnect" {
  api_id    = aws_apigatewayv2_api.websocket.id
  route_key = "$disconnect"
  target    = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

# $default - for all messages
resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.websocket.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

# -----------------------------------------------------------------------------
# Stage
# -----------------------------------------------------------------------------
resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.websocket.id
  name        = var.environment
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = var.throttling_burst_limit
    throttling_rate_limit  = var.throttling_rate_limit
  }

  tags = {
    Name = "${var.project}-${var.environment}-websocket-stage"
  }
}

# -----------------------------------------------------------------------------
# Custom Domain
# -----------------------------------------------------------------------------
resource "aws_apigatewayv2_domain_name" "websocket" {
  count       = var.domain_name != "" ? 1 : 0
  domain_name = var.domain_name

  domain_name_configuration {
    certificate_arn = var.certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }

  tags = {
    Name = "${var.project}-${var.environment}-websocket-domain"
  }
}

resource "aws_apigatewayv2_api_mapping" "websocket" {
  count       = var.domain_name != "" ? 1 : 0
  api_id      = aws_apigatewayv2_api.websocket.id
  domain_name = aws_apigatewayv2_domain_name.websocket[0].id
  stage       = aws_apigatewayv2_stage.main.id
}
```

**Step 2: Create lambda.tf**

Create `terraform/modules/websocket-api/lambda.tf`:

```hcl
# =============================================================================
# Lambda Authorizer for WebSocket API
# =============================================================================

# -----------------------------------------------------------------------------
# Lambda Function
# -----------------------------------------------------------------------------
data "archive_file" "authorizer" {
  type        = "zip"
  source_dir  = "${path.module}/../../../lambda/websocket-authorizer"
  output_path = "${path.module}/authorizer.zip"
}

resource "aws_lambda_function" "authorizer" {
  filename         = data.archive_file.authorizer.output_path
  function_name    = "${var.project}-${var.environment}-websocket-authorizer"
  role             = aws_iam_role.authorizer.arn
  handler          = "index.handler"
  runtime          = "python3.11"
  timeout          = 10
  memory_size      = 256
  source_code_hash = data.archive_file.authorizer.output_base64sha256

  environment {
    variables = {
      CLERK_JWKS_URL = var.clerk_jwks_url
      CLERK_ISSUER   = var.clerk_issuer
    }
  }

  layers = [aws_lambda_layer_version.authorizer_deps.arn]

  tags = {
    Name = "${var.project}-${var.environment}-websocket-authorizer"
  }
}

# -----------------------------------------------------------------------------
# Lambda Layer for Dependencies
# -----------------------------------------------------------------------------
resource "null_resource" "authorizer_deps" {
  triggers = {
    requirements = filemd5("${path.module}/../../../lambda/websocket-authorizer/requirements.txt")
  }

  provisioner "local-exec" {
    command = <<-EOT
      cd ${path.module}/../../../lambda/websocket-authorizer
      pip install -r requirements.txt -t python/lib/python3.11/site-packages/ --upgrade
      zip -r ${path.module}/authorizer-deps.zip python
      rm -rf python
    EOT
  }
}

resource "aws_lambda_layer_version" "authorizer_deps" {
  depends_on          = [null_resource.authorizer_deps]
  filename            = "${path.module}/authorizer-deps.zip"
  layer_name          = "${var.project}-${var.environment}-authorizer-deps"
  compatible_runtimes = ["python3.11"]
}

# -----------------------------------------------------------------------------
# Lambda IAM Role
# -----------------------------------------------------------------------------
resource "aws_iam_role" "authorizer" {
  name = "${var.project}-${var.environment}-websocket-authorizer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project}-${var.environment}-websocket-authorizer-role"
  }
}

resource "aws_iam_role_policy_attachment" "authorizer_basic" {
  role       = aws_iam_role.authorizer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# -----------------------------------------------------------------------------
# Lambda Permission for API Gateway
# -----------------------------------------------------------------------------
resource "aws_lambda_permission" "authorizer" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.authorizer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.websocket.execution_arn}/*"
}
```

**Step 3: Create variables.tf**

Create `terraform/modules/websocket-api/variables.tf`:

```hcl
# =============================================================================
# WebSocket API Module - Variables
# =============================================================================

variable "project" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for VPC Link"
  type        = list(string)
}

variable "alb_listener_arn" {
  description = "ARN of the ALB listener to integrate with"
  type        = string
}

variable "alb_security_group_id" {
  description = "Security group ID of the ALB"
  type        = string
}

variable "clerk_jwks_url" {
  description = "Clerk JWKS URL for JWT validation"
  type        = string
}

variable "clerk_issuer" {
  description = "Clerk issuer URL for JWT validation"
  type        = string
}

variable "throttling_burst_limit" {
  description = "Throttling burst limit (connections)"
  type        = number
  default     = 500
}

variable "throttling_rate_limit" {
  description = "Throttling rate limit (connections per second)"
  type        = number
  default     = 100
}

variable "domain_name" {
  description = "Custom domain name (optional)"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for custom domain"
  type        = string
  default     = ""
}
```

**Step 4: Create outputs.tf**

Create `terraform/modules/websocket-api/outputs.tf`:

```hcl
# =============================================================================
# WebSocket API Module - Outputs
# =============================================================================

output "api_id" {
  description = "WebSocket API ID"
  value       = aws_apigatewayv2_api.websocket.id
}

output "api_endpoint" {
  description = "WebSocket API endpoint URL"
  value       = aws_apigatewayv2_api.websocket.api_endpoint
}

output "stage_name" {
  description = "Stage name"
  value       = aws_apigatewayv2_stage.main.name
}

output "custom_domain_name" {
  description = "Custom domain name for DNS"
  value       = var.domain_name != "" ? aws_apigatewayv2_domain_name.websocket[0].domain_name_configuration[0].target_domain_name : null
}

output "custom_domain_zone_id" {
  description = "Custom domain hosted zone ID for DNS"
  value       = var.domain_name != "" ? aws_apigatewayv2_domain_name.websocket[0].domain_name_configuration[0].hosted_zone_id : null
}
```

**Step 5: Commit**

```bash
git add terraform/modules/websocket-api/
git commit -m "feat: add Terraform module for WebSocket API Gateway

Creates WebSocket API with:
- Lambda authorizer for Clerk JWT validation
- VPC Link integration to ALB
- Custom domain support
- Rate limiting/throttling"
```

---

## Task 5: Wire WebSocket Module into Main Terraform

**Files:**
- Modify: `terraform/main.tf`
- Modify: `terraform/variables.tf`

**Step 1: Add WebSocket module to main.tf**

Add after the API Gateway module (around line 158) in `terraform/main.tf`:

```hcl
# -----------------------------------------------------------------------------
# WebSocket API Gateway Module (Real-time streaming)
# -----------------------------------------------------------------------------
module "websocket_api" {
  source = "./modules/websocket-api"

  project     = "isol8"
  environment = var.environment
  subnet_ids  = module.vpc.private_subnet_ids

  # ALB integration
  alb_listener_arn      = module.alb.http_listener_arn
  alb_security_group_id = module.alb.security_group_id

  # Clerk JWT validation
  clerk_jwks_url = "https://${var.clerk_issuer}/.well-known/jwks.json"
  clerk_issuer   = "https://${var.clerk_issuer}"

  # Custom domain
  domain_name     = var.websocket_domain_name
  certificate_arn = module.acm.certificate_arn

  # Rate limiting
  throttling_burst_limit = 500
  throttling_rate_limit  = 100
}

# -----------------------------------------------------------------------------
# Route53 DNS Record for WebSocket API
# -----------------------------------------------------------------------------
resource "aws_route53_record" "websocket" {
  count   = var.websocket_domain_name != "" ? 1 : 0
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.websocket_domain_name
  type    = "A"

  alias {
    name                   = module.websocket_api.custom_domain_name
    zone_id                = module.websocket_api.custom_domain_zone_id
    evaluate_target_health = false
  }
}
```

**Step 2: Add variable to variables.tf**

Add to `terraform/variables.tf`:

```hcl
variable "websocket_domain_name" {
  description = "Custom domain for WebSocket API (e.g., ws-dev.isol8.co)"
  type        = string
  default     = ""
}
```

**Step 3: Commit**

```bash
git add terraform/main.tf terraform/variables.tf
git commit -m "feat: wire WebSocket API module into main Terraform

Adds websocket_api module with Route53 DNS record for
custom domain support."
```

---

## Task 6: Frontend WebSocket Hook

**Files:**
- Create: `frontend/src/hooks/useChatWebSocket.ts`

**Step 1: Create the WebSocket hook**

Create `frontend/src/hooks/useChatWebSocket.ts`:

```typescript
/**
 * WebSocket-based chat hook with streaming support.
 *
 * Replaces SSE-based useChat to avoid API Gateway HTTP API buffering.
 * Uses the same encryption flow - only the transport protocol changes.
 *
 * Features:
 * - Persistent WebSocket connection
 * - Automatic reconnection with exponential backoff
 * - Ping/pong keepalive
 * - Same ChatMessage interface as useChat
 */

'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useEncryption } from './useEncryption';
import type {
  ChatMessage,
  UseChatOptions,
  UseChatReturn,
} from './useChat';
import type { SerializedEncryptedPayload } from '@/lib/crypto/message-crypto';

// =============================================================================
// Configuration
// =============================================================================

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000]; // Exponential backoff

// =============================================================================
// Types
// =============================================================================

interface WSMessage {
  type: 'session' | 'encrypted_chunk' | 'thinking' | 'stored' | 'done' | 'error' | 'ping';
  session_id?: string;
  encrypted_content?: SerializedEncryptedPayload;
  message?: string;
  model_used?: string;
  input_tokens?: number;
  output_tokens?: number;
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useChatWebSocket(options: UseChatOptions = {}): UseChatReturn {
  const { initialSessionId, orgId, onSessionChange } = options;
  const { getToken } = useAuth();
  const encryption = useEncryption();
  const isOrgContext = !!orgId;

  // State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId ?? null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const currentAssistantIdRef = useRef<string | null>(null);

  // =============================================================================
  // WebSocket Connection
  // =============================================================================

  const connect = useCallback(async (): Promise<WebSocket> => {
    // Return existing connection if open
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return wsRef.current;
    }

    // Close any existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const token = await getToken();
    if (!token) {
      throw new Error('Not authenticated');
    }

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${WS_URL}/ws/chat?token=${token}`);

      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        reconnectAttemptRef.current = 0;
        setError(null);
        wsRef.current = ws;
        resolve(ws);
      };

      ws.onclose = (event) => {
        console.log(`[WebSocket] Closed: code=${event.code}, reason=${event.reason}`);
        wsRef.current = null;

        // Don't reconnect on normal closure or auth failure
        if (event.code === 1000 || event.code === 4001) {
          if (event.code === 4001) {
            setError('Authentication failed. Please refresh the page.');
          }
          return;
        }

        // Attempt reconnect with backoff
        if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
          const delay = RECONNECT_DELAYS[reconnectAttemptRef.current];
          reconnectAttemptRef.current++;
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptRef.current})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect().catch(console.error);
          }, delay);
        } else {
          setError('Connection lost. Please refresh the page.');
        }
      };

      ws.onerror = (event) => {
        console.error('[WebSocket] Error:', event);
        reject(new Error('WebSocket connection failed'));
      };

      ws.onmessage = (event) => {
        try {
          const data: WSMessage = JSON.parse(event.data);
          handleMessage(data);
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };
    });
  }, [getToken]);

  // =============================================================================
  // Message Handling
  // =============================================================================

  const handleMessage = useCallback((data: WSMessage) => {
    switch (data.type) {
      case 'ping':
        // Respond to server ping
        wsRef.current?.send(JSON.stringify({ type: 'pong' }));
        break;

      case 'session':
        if (data.session_id) {
          setSessionId(data.session_id);
          onSessionChange?.(data.session_id);
          window.dispatchEvent(new CustomEvent('sessionUpdated'));
        }
        break;

      case 'encrypted_chunk':
        if (data.encrypted_content && currentAssistantIdRef.current) {
          const decrypted = encryption.decryptTransportResponse(data.encrypted_content);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === currentAssistantIdRef.current
                ? { ...msg, content: msg.content + decrypted }
                : msg
            )
          );
        }
        break;

      case 'thinking':
        if (data.encrypted_content && currentAssistantIdRef.current) {
          const decrypted = encryption.decryptTransportResponse(data.encrypted_content);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === currentAssistantIdRef.current
                ? { ...msg, thinking: (msg.thinking || '') + decrypted }
                : msg
            )
          );
        }
        break;

      case 'stored':
        console.log('[WebSocket] Message stored:', {
          model: data.model_used,
          inputTokens: data.input_tokens,
          outputTokens: data.output_tokens,
        });
        break;

      case 'done':
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === currentAssistantIdRef.current
              ? { ...msg, isStreaming: false }
              : msg
          )
        );
        currentAssistantIdRef.current = null;
        break;

      case 'error':
        setError(data.message || 'Unknown error');
        setIsStreaming(false);
        if (currentAssistantIdRef.current) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === currentAssistantIdRef.current
                ? { ...msg, content: `Error: ${data.message}`, isStreaming: false }
                : msg
            )
          );
        }
        currentAssistantIdRef.current = null;
        break;
    }
  }, [encryption, onSessionChange]);

  // =============================================================================
  // Send Message
  // =============================================================================

  const sendMessage = useCallback(
    async (content: string, model: string): Promise<void> => {
      // Validate encryption state
      if (isOrgContext) {
        if (!encryption.isOrgUnlocked) {
          throw new Error('Organization encryption keys not unlocked');
        }
      } else {
        if (!encryption.state.isUnlocked) {
          throw new Error('Encryption keys not unlocked');
        }
      }
      if (!encryption.state.enclavePublicKey) {
        throw new Error('Enclave public key not available');
      }

      setError(null);

      // Create placeholder messages
      const userMsgId = `user-${Date.now()}`;
      const assistantMsgId = `assistant-${Date.now()}`;
      currentAssistantIdRef.current = assistantMsgId;

      const userMessage: ChatMessage = {
        id: userMsgId,
        role: 'user',
        content,
      };

      const assistantMessage: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        model,
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);

      try {
        // Ensure connection
        const ws = await connect();

        // Generate transport keypair
        const transportKeypair = encryption.generateTransportKeypair();

        // Encrypt message
        const encryptedMessage = encryption.encryptMessage(content);

        // Prepare history
        const historyMessages = messages.filter((m) => m.encryptedPayload);
        const encryptedHistory =
          historyMessages.length > 0
            ? encryption.prepareHistoryForTransport(
                historyMessages.map((m) => ({
                  role: m.role,
                  encrypted_content: m.encryptedPayload!,
                })),
                isOrgContext
              )
            : [];

        // Send message
        ws.send(
          JSON.stringify({
            session_id: sessionId,
            encrypted_message: encryptedMessage,
            encrypted_history: encryptedHistory,
            client_transport_public_key: transportKeypair.publicKey,
            model,
            ...(orgId && { org_id: orgId }),
          })
        );
      } catch (err) {
        console.error('[WebSocket] Send error:', err);
        const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
        setError(errorMessage);
        setIsStreaming(false);

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: `Error: ${errorMessage}`, isStreaming: false }
              : msg
          )
        );
        currentAssistantIdRef.current = null;
      }
    },
    [connect, encryption, messages, sessionId, isOrgContext, orgId]
  );

  // =============================================================================
  // Load Session
  // =============================================================================

  const loadSession = useCallback(
    async (id: string): Promise<void> => {
      // Same implementation as useChat - fetch via HTTP
      if (isOrgContext) {
        if (!encryption.isOrgUnlocked) {
          throw new Error('Organization encryption keys not unlocked');
        }
      } else {
        if (!encryption.state.isUnlocked) {
          throw new Error('Encryption keys not unlocked');
        }
      }

      try {
        const token = await getToken();
        if (!token) {
          throw new Error('Not authenticated');
        }

        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
        const url = new URL(`${BACKEND_URL}/chat/sessions/${id}/messages`);
        if (orgId) {
          url.searchParams.set('org_id', orgId);
        }

        const res = await fetch(url.toString(), {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!res.ok) {
          throw new Error('Failed to load session messages');
        }

        const data = await res.json();

        if (data.messages?.[0]?.encrypted_content) {
          const encryptedMessages = data.messages.map(
            (msg: { role: 'user' | 'assistant'; encrypted_content: SerializedEncryptedPayload }) => ({
              role: msg.role,
              encrypted_content: msg.encrypted_content,
            })
          );

          const decryptedContents = encryption.decryptStoredMessages(encryptedMessages, isOrgContext);

          const loadedMessages: ChatMessage[] = data.messages.map(
            (
              msg: { id: string; role: 'user' | 'assistant'; encrypted_content: SerializedEncryptedPayload },
              index: number
            ) => ({
              id: msg.id,
              role: msg.role,
              content: decryptedContents[index],
              encryptedPayload: msg.encrypted_content,
            })
          );

          setMessages(loadedMessages);
        } else {
          // Fallback for unencrypted messages (legacy)
          const loadedMessages: ChatMessage[] = data.messages.map(
            (msg: { id: string; role: 'user' | 'assistant'; content: string }) => ({
              id: msg.id,
              role: msg.role,
              content: msg.content,
            })
          );
          setMessages(loadedMessages);
        }

        setSessionId(id);
        setError(null);
      } catch (err) {
        console.error('Failed to load session:', err);
        setError(err instanceof Error ? err.message : 'Failed to load session');
        throw err;
      }
    },
    [encryption, getToken, isOrgContext, orgId]
  );

  // =============================================================================
  // Clear / Abort
  // =============================================================================

  const clearSession = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setError(null);
  }, []);

  const abort = useCallback(() => {
    // Close WebSocket to abort streaming
    wsRef.current?.close(1000);
    setIsStreaming(false);
  }, []);

  // =============================================================================
  // Cleanup
  // =============================================================================

  useEffect(() => {
    return () => {
      reconnectTimeoutRef.current && clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close(1000);
    };
  }, []);

  return {
    messages,
    sessionId,
    isStreaming,
    error,
    sendMessage,
    loadSession,
    clearSession,
    abort,
  };
}
```

**Step 2: Add environment variable**

Add to `frontend/.env.local` (and `.env.development`, `.env.production`):

```
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

For deployed environments:
- Dev: `wss://ws-dev.isol8.co`
- Staging: `wss://ws-staging.isol8.co`
- Prod: `wss://ws.isol8.co`

**Step 3: Commit**

```bash
git add frontend/src/hooks/useChatWebSocket.ts
git commit -m "feat: add WebSocket chat hook for streaming

New useChatWebSocket hook that uses WebSocket instead of SSE:
- Same ChatMessage interface as useChat
- Automatic reconnection with exponential backoff
- Ping/pong keepalive
- Same encryption flow"
```

---

## Task 7: Switch ChatWindow to WebSocket

**Files:**
- Modify: `frontend/src/components/chat/ChatWindow.tsx`

**Step 1: Update import and hook usage**

In `frontend/src/components/chat/ChatWindow.tsx`, change the import from `useChat` to `useChatWebSocket`:

```typescript
// Change this:
import { useChat } from '@/hooks/useChat';

// To this:
import { useChatWebSocket as useChat } from '@/hooks/useChatWebSocket';
```

This is a drop-in replacement since both hooks implement the same `UseChatReturn` interface.

**Step 2: Commit**

```bash
git add frontend/src/components/chat/ChatWindow.tsx
git commit -m "feat: switch ChatWindow to WebSocket streaming

Use useChatWebSocket hook instead of SSE-based useChat.
Drop-in replacement with same interface."
```

---

## Task 8: Deploy and Test

**Step 1: Set environment variables for dev**

Update Terraform variables or tfvars for dev environment:

```hcl
websocket_domain_name = "ws-dev.isol8.co"
```

**Step 2: Deploy Terraform**

```bash
cd /Users/prasiddhaparthsarthy/Desktop/freebird/terraform
terraform init
terraform plan -var-file=environments/dev/terraform.tfvars
terraform apply -var-file=environments/dev/terraform.tfvars
```

**Step 3: Update Vercel environment variables**

Add to Vercel project settings:
- `NEXT_PUBLIC_WS_URL=wss://ws-dev.isol8.co`

**Step 4: Deploy frontend**

Push to main branch or manually deploy via Vercel.

**Step 5: Test streaming**

1. Open dev site
2. Send a chat message
3. Verify response streams progressively (not all at once)
4. Check browser Network tab for WebSocket connection

**Step 6: Final commit**

```bash
git add .
git commit -m "deploy: WebSocket streaming to dev environment

Complete WebSocket streaming implementation:
- Backend: WebSocket endpoint with keepalive
- Frontend: WebSocket hook with reconnection
- Terraform: WebSocket API Gateway with Lambda authorizer
- Verified streaming works end-to-end"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `routers/websocket_chat.py`, `main.py` | Backend WebSocket endpoint |
| 2 | `tests/unit/routers/test_websocket_chat.py` | Backend unit tests |
| 3 | `terraform/lambda/websocket-authorizer/` | Lambda authorizer code |
| 4 | `terraform/modules/websocket-api/` | Terraform module |
| 5 | `terraform/main.tf`, `variables.tf` | Wire module into main |
| 6 | `frontend/src/hooks/useChatWebSocket.ts` | Frontend WebSocket hook |
| 7 | `frontend/src/components/chat/ChatWindow.tsx` | Switch to WebSocket |
| 8 | Deploy | Test end-to-end |

**Total estimated commits:** 8
