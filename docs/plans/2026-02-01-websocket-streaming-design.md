# WebSocket Streaming Design

**Date:** 2026-02-01
**Status:** Proposed
**Problem:** API Gateway HTTP API buffers SSE responses (30s timeout), breaking LLM streaming

## Overview

Replace SSE streaming with WebSocket via API Gateway WebSocket API. This eliminates buffering while maintaining rate limiting, authentication, and API Gateway management features.

## Architecture

```
Streaming:     Browser → API Gateway WebSocket API → VPC Link → ALB → EC2
Non-streaming: Browser → API Gateway HTTP API → VPC Link → ALB → EC2
```

### Components

| Component | Purpose |
|-----------|---------|
| API Gateway WebSocket API | Real-time streaming, no buffering |
| API Gateway HTTP API | Existing REST endpoints (unchanged) |
| Lambda Authorizer | Validates Clerk JWT on `$connect` |
| ALB | Routes to EC2 (already supports WebSocket) |
| FastAPI WebSocket endpoint | New `/ws/chat` endpoint |

### Domains

- `api-{env}.isol8.co` → HTTP API (existing)
- `ws-{env}.isol8.co` → WebSocket API (new)

### WebSocket API Routes

- `$connect` → Lambda authorizer validates JWT
- `$disconnect` → Cleanup logging
- `$default` → Forward messages to ALB → FastAPI

## Encryption Compatibility

The encryption flow is transport-agnostic. WebSocket changes only the transport layer:

```
Client ◄──── WebSocket ────► Parent (FastAPI) ◄──── vsock ────► Enclave
       │                            │                              │
       └── Transport changes        └── No changes                 └── No changes
```

### Message Flow

1. Client connects: `wss://ws-dev.isol8.co?token=eyJ...`
2. Lambda authorizer validates Clerk JWT
3. Client sends encrypted message (same format as SSE):
   ```json
   {
     "encrypted_message": { "ephemeral_public_key": "...", "iv": "...", "ciphertext": "...", "tag": "..." },
     "encrypted_history": [...],
     "client_transport_public_key": "abc123...",
     "model": "claude-3",
     "session_id": "optional-existing-session"
   }
   ```
4. Server streams encrypted chunks:
   ```json
   { "type": "session", "session_id": "..." }
   { "type": "encrypted_chunk", "encrypted_content": {...} }
   { "type": "encrypted_chunk", "encrypted_content": {...} }
   { "type": "done", "stored_user_message": {...}, "stored_assistant_message": {...} }
   ```
5. Client decrypts each chunk with transport private key

## Lambda Authorizer

Validates Clerk JWT on WebSocket connection:

```python
import jwt
from jwt import PyJWKClient

CLERK_JWKS_URL = "https://your-clerk-domain/.well-known/jwks.json"
CLERK_ISSUER = "https://your-clerk-domain"

jwks_client = PyJWKClient(CLERK_JWKS_URL, cache_keys=True)

def handler(event, context):
    token = event.get("queryStringParameters", {}).get("token")

    if not token:
        return {"isAuthorized": False}

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
        )
        return {
            "isAuthorized": True,
            "context": {
                "userId": payload.get("sub"),
                "orgId": payload.get("org_id"),
            }
        }
    except Exception:
        return {"isAuthorized": False}
```

## Backend WebSocket Endpoint

New file: `routers/websocket_chat.py`

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.services.chat_service import ChatService
from core.crypto import EncryptedPayload
import asyncio

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    # User ID from Lambda authorizer context
    user_id = websocket.headers.get("x-user-id")
    org_id = websocket.headers.get("x-org-id")

    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    chat_service = ChatService()

    # Keepalive ping task
    async def send_pings():
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_json({"type": "ping"})
            except:
                break

    ping_task = asyncio.create_task(send_pings())

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "pong":
                continue

            # Parse encrypted payload
            encrypted_message = EncryptedPayload.from_dict(data["encrypted_message"])
            encrypted_history = [EncryptedPayload.from_dict(h) for h in data.get("encrypted_history", [])]
            client_transport_key = bytes.fromhex(data["client_transport_public_key"])
            model = data.get("model", "default")
            session_id = data.get("session_id")

            # Stream response using existing ChatService
            async for chunk in chat_service.process_encrypted_message_stream(
                user_id=user_id,
                org_id=org_id,
                session_id=session_id,
                encrypted_message=encrypted_message,
                encrypted_history=encrypted_history,
                client_transport_public_key=client_transport_key,
                model=model,
            ):
                await websocket.send_json(chunk.to_dict())

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
```

## Frontend WebSocket Hook

New file: `hooks/useChatWebSocket.ts`

```typescript
'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useEncryption } from './useEncryption';
import type { ChatMessage, UseChatOptions, UseChatReturn } from './useChat';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000];

export function useChatWebSocket(options: UseChatOptions = {}): UseChatReturn {
  const { initialSessionId, orgId, onSessionChange } = options;
  const { getToken } = useAuth();
  const encryption = useEncryption();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId ?? null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const transportKeypairRef = useRef<{ publicKey: string; privateKey: Uint8Array } | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const token = await getToken();
    if (!token) throw new Error('Not authenticated');

    const ws = new WebSocket(`${WS_URL}/ws/chat?token=${token}`);

    ws.onopen = () => {
      reconnectAttemptRef.current = 0;
      setError(null);
    };

    ws.onclose = (event) => {
      wsRef.current = null;

      if (event.code === 1000 || event.code === 4001) return;

      if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = RECONNECT_DELAYS[reconnectAttemptRef.current];
        reconnectAttemptRef.current++;
        reconnectTimeoutRef.current = setTimeout(() => connect(), delay);
      } else {
        setError('Connection lost. Please refresh the page.');
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
        return;
      }

      if (data.type === 'session') {
        setSessionId(data.session_id);
        onSessionChange?.(data.session_id);
      } else if (data.type === 'encrypted_chunk') {
        const decrypted = encryption.decryptTransportResponse(data.encrypted_content);
        setMessages(prev => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          if (lastMsg?.role === 'assistant') {
            lastMsg.content += decrypted;
          }
          return updated;
        });
      } else if (data.type === 'done') {
        setIsStreaming(false);
      } else if (data.type === 'error') {
        setError(data.message);
        setIsStreaming(false);
      }
    };

    wsRef.current = ws;
  }, [getToken, encryption, onSessionChange]);

  const sendMessage = useCallback(async (content: string, model: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      await connect();
    }

    transportKeypairRef.current = encryption.generateTransportKeypair();
    const encryptedMessage = encryption.encryptMessage(content);

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
    };

    const assistantMessage: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);
    setError(null);

    wsRef.current!.send(JSON.stringify({
      session_id: sessionId,
      encrypted_message: encryptedMessage,
      encrypted_history: [], // Prepare history same as SSE
      client_transport_public_key: transportKeypairRef.current.publicKey,
      model,
      ...(orgId && { org_id: orgId }),
    }));
  }, [connect, encryption, sessionId, orgId]);

  const clearSession = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setError(null);
  }, []);

  const abort = useCallback(() => {
    wsRef.current?.close(1000);
    setIsStreaming(false);
  }, []);

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
    loadSession: async () => {}, // TODO: implement
    clearSession,
    abort,
  };
}
```

## Terraform Infrastructure

### New module: `terraform/modules/websocket-api/main.tf`

```hcl
resource "aws_apigatewayv2_api" "websocket" {
  name                       = "${var.environment}-isol8-websocket"
  protocol_type              = "WEBSOCKET"
  route_selection_expression = "$request.body.action"
}

resource "aws_apigatewayv2_authorizer" "clerk_jwt" {
  api_id                     = aws_apigatewayv2_api.websocket.id
  authorizer_type            = "REQUEST"
  authorizer_uri             = aws_lambda_function.authorizer.invoke_arn
  identity_sources           = ["route.request.querystring.token"]
  name                       = "clerk-jwt-authorizer"
}

resource "aws_apigatewayv2_route" "connect" {
  api_id             = aws_apigatewayv2_api.websocket.id
  route_key          = "$connect"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.clerk_jwt.id
  target             = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

resource "aws_apigatewayv2_route" "disconnect" {
  api_id    = aws_apigatewayv2_api.websocket.id
  route_key = "$disconnect"
  target    = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.websocket.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

resource "aws_apigatewayv2_integration" "alb" {
  api_id             = aws_apigatewayv2_api.websocket.id
  integration_type   = "HTTP_PROXY"
  integration_uri    = var.alb_listener_arn
  integration_method = "ANY"
  connection_type    = "VPC_LINK"
  connection_id      = var.vpc_link_id
}

resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.websocket.id
  name        = var.environment
  auto_deploy = true
}

resource "aws_apigatewayv2_domain_name" "websocket" {
  domain_name = "ws-${var.environment}.isol8.co"

  domain_name_configuration {
    certificate_arn = var.certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }
}

resource "aws_apigatewayv2_api_mapping" "websocket" {
  api_id      = aws_apigatewayv2_api.websocket.id
  domain_name = aws_apigatewayv2_domain_name.websocket.id
  stage       = aws_apigatewayv2_stage.main.id
}
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Network drop | Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, 16s) |
| Token expired | Close with 4001, prompt re-auth |
| Server restart | Auto-reconnect |
| Idle timeout (ALB 300s) | Ping/pong every 30s keeps alive |
| Mid-stream disconnect | Frontend shows error, allows retry |
| Max reconnect attempts | Show "Connection lost. Please refresh." |

## Files to Create

| File | Purpose |
|------|---------|
| `backend/routers/websocket_chat.py` | WebSocket endpoint |
| `frontend/src/hooks/useChatWebSocket.ts` | Frontend hook |
| `terraform/modules/websocket-api/main.tf` | WebSocket API Gateway |
| `terraform/modules/websocket-api/lambda.tf` | Authorizer Lambda |
| `terraform/modules/websocket-api/variables.tf` | Module variables |
| `terraform/modules/websocket-api/outputs.tf` | Module outputs |
| `terraform/lambda/websocket-authorizer/index.py` | Authorizer code |

## Files to Modify

| File | Change |
|------|--------|
| `backend/main.py` | Register WebSocket router |
| `frontend/src/components/chat/ChatWindow.tsx` | Use new hook |
| `frontend/.env.*` | Add `NEXT_PUBLIC_WS_URL` |
| `terraform/environments/*/main.tf` | Add websocket-api module |

## Implementation Order

1. Lambda authorizer (test independently)
2. Backend WebSocket endpoint (test with wscat)
3. Terraform WebSocket API (deploy to dev)
4. Frontend hook (integrate and test)
5. E2E tests

## Environment Variables

```bash
# Frontend
NEXT_PUBLIC_WS_URL=wss://ws-dev.isol8.co

# Lambda
CLERK_JWKS_URL=https://your-clerk-domain/.well-known/jwks.json
CLERK_ISSUER=https://your-clerk-domain
```

## Testing Strategy

- Unit tests for Lambda authorizer (valid/expired/missing token)
- Unit tests for backend WebSocket endpoint
- Frontend hook tests with jest-websocket-mock
- E2E test: full encrypted chat flow over WebSocket
