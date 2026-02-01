"""
WebSocket endpoint for encrypted chat streaming.

Replaces SSE streaming to avoid API Gateway HTTP API buffering issues.
Uses same encryption flow as SSE endpoint - only transport layer changes.

Security Note:
- Server acts as BLIND RELAY - cannot read message content
- All messages encrypted to enclave (transport) or user/org (storage)
- WebSocket delivers encrypted chunks to client
"""

import asyncio
import logging
from typing import Optional

from clerk_backend_api import Clerk
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import AVAILABLE_MODELS, settings
from core.database import get_session_factory
from core.services.chat_service import ChatService
from schemas.encryption import EncryptedPayload, SendEncryptedMessageRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

# Build set of valid model IDs for fast validation
VALID_MODEL_IDS = {model["id"] for model in AVAILABLE_MODELS}


@router.websocket("/chat")
async def websocket_chat(
    websocket: WebSocket,
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """
    WebSocket endpoint for encrypted chat streaming.

    Authentication: Clerk JWT passed via query param (validated by Lambda authorizer
    in production, or via x-user-id header in development).

    Message format (client -> server):
    {
        "encrypted_message": {...},
        "encrypted_history": [...],
        "client_transport_public_key": "hex...",
        "model": "model-id",
        "session_id": "optional-existing-session",
        "org_id": "optional-org-id",
        "facts_context": "optional-plaintext-facts"
    }

    Response format (server -> client):
    {"type": "session", "session_id": "..."}
    {"type": "encrypted_chunk", "encrypted_content": {...}}
    {"type": "thinking", "encrypted_content": {...}}
    {"type": "stored", "model_used": "...", "input_tokens": N, "output_tokens": N}
    {"type": "done"}
    {"type": "error", "message": "..."}
    {"type": "ping"}  # Server sends, client responds with {"type": "pong"}
    """
    await websocket.accept()

    # Get user ID from headers (set by Lambda authorizer via API Gateway)
    # In dev, can be passed directly for testing
    user_id = websocket.headers.get("x-user-id")
    org_id = websocket.headers.get("x-org-id")

    if not user_id:
        logger.warning("WebSocket connection without user_id")
        await websocket.close(code=4001, reason="Unauthorized")
        return

    logger.info("WebSocket connected: user_id=%s, org_id=%s", user_id, org_id)

    # Keepalive ping task
    async def send_pings():
        """Send ping every 30s to keep connection alive (ALB has 300s idle timeout)."""
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    ping_task = asyncio.create_task(send_pings())

    try:
        while True:
            data = await websocket.receive_json()

            # Handle pong response (client keepalive acknowledgment)
            if data.get("type") == "pong":
                continue

            # Parse and validate request using Pydantic
            try:
                # Build request object for validation
                request = SendEncryptedMessageRequest(
                    session_id=data.get("session_id"),
                    model=data.get("model", ""),
                    encrypted_message=EncryptedPayload(**data["encrypted_message"]),
                    encrypted_history=[
                        EncryptedPayload(**h) for h in data.get("encrypted_history", [])
                    ] or None,
                    facts_context=data.get("facts_context"),
                    client_transport_public_key=data["client_transport_public_key"],
                )

                # Allow org_id from message to override header
                msg_org_id = data.get("org_id") or org_id

            except (KeyError, ValidationError) as e:
                logger.error("Invalid message format: %s", e)
                await websocket.send_json({"type": "error", "message": f"Invalid message format: {e}"})
                continue

            # Validate model
            if request.model not in VALID_MODEL_IDS:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid model. Available models: {list(VALID_MODEL_IDS)}"
                })
                continue

            # Process the message
            try:
                await _process_chat_message(
                    websocket=websocket,
                    session_factory=session_factory,
                    user_id=user_id,
                    org_id=msg_org_id,
                    request=request,
                )
            except Exception as e:
                logger.exception("Error processing message: %s", e)
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: user_id=%s", user_id)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass


async def _process_chat_message(
    websocket: WebSocket,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: str,
    org_id: Optional[str],
    request: SendEncryptedMessageRequest,
) -> None:
    """
    Process a single chat message and stream response via WebSocket.

    This follows the same flow as the SSE endpoint but uses WebSocket for transport.
    """
    logger.debug(
        "WebSocket chat request - user_id=%s, org_id=%s, model=%s, session_id=%s, history_count=%d, has_facts=%s",
        user_id,
        org_id or "personal",
        request.model,
        request.session_id or "new",
        len(request.encrypted_history) if request.encrypted_history else 0,
        bool(request.facts_context),
    )

    # Get or create session
    async with session_factory() as service_db:
        service = ChatService(service_db)

        # Verify user can send encrypted messages
        can_send, error_msg = await service.verify_can_send_encrypted(
            user_id=user_id,
            org_id=org_id,
        )
        if not can_send:
            await websocket.send_json({"type": "error", "message": error_msg})
            return

        session_id = request.session_id
        if session_id:
            session = await service.get_session(
                session_id=session_id,
                user_id=user_id,
                org_id=org_id,
            )
            if not session:
                await websocket.send_json({"type": "error", "message": "Session not found or access denied"})
                return
        else:
            # Create new session
            session = await service.create_session(
                user_id=user_id,
                name="New Chat",
                org_id=org_id,
            )
            session_id = session.id

    # Send session ID first
    await websocket.send_json({"type": "session", "session_id": session_id})

    # Convert hex-encoded API payloads to bytes-based crypto payloads
    encrypted_msg = request.encrypted_message.to_crypto_payload()

    encrypted_history = []
    if request.encrypted_history:
        for h in request.encrypted_history:
            encrypted_history.append(h.to_crypto_payload())

    # Fetch user/org metadata from Clerk for AWS credential resolution
    user_metadata = None
    org_metadata = None
    if settings.CLERK_SECRET_KEY:
        try:
            clerk = Clerk(bearer_auth=settings.CLERK_SECRET_KEY)
            user = clerk.users.get(user_id=user_id)
            user_metadata = user.private_metadata

            if org_id:
                org = clerk.organizations.get(organization_id=org_id)
                org_metadata = org.private_metadata
        except Exception as e:
            logger.warning("Could not fetch Clerk metadata: %s", e)
            # Continue without custom credentials - will use IAM role

    # Stream response
    try:
        chunk_count = 0

        async with session_factory() as stream_db:
            stream_service = ChatService(stream_db)

            async for chunk in stream_service.process_encrypted_message_stream(
                session_id=session_id,
                user_id=user_id,
                org_id=org_id,
                encrypted_message=encrypted_msg,
                encrypted_history=encrypted_history,
                facts_context=request.facts_context,
                model=request.model,
                client_transport_public_key=request.client_transport_public_key,
                user_metadata=user_metadata,
                org_metadata=org_metadata,
            ):
                if chunk.error:
                    logger.debug("Enclave error for session_id=%s: %s", session_id, chunk.error)
                    await websocket.send_json({"type": "error", "message": chunk.error})
                    return

                if chunk.encrypted_content:
                    chunk_count += 1
                    # Convert bytes-based crypto payload to hex-encoded API payload
                    api_payload = EncryptedPayload.from_crypto_payload(chunk.encrypted_content)
                    await websocket.send_json({
                        "type": "encrypted_chunk",
                        "encrypted_content": api_payload.model_dump()
                    })

                if chunk.encrypted_thinking:
                    # Send thinking chunk
                    api_payload = EncryptedPayload.from_crypto_payload(chunk.encrypted_thinking)
                    await websocket.send_json({
                        "type": "thinking",
                        "encrypted_content": api_payload.model_dump()
                    })

                if chunk.is_final and chunk.stored_user_message and chunk.stored_assistant_message:
                    # Send stored message info
                    logger.debug("Messages stored for session_id=%s", session_id)
                    await websocket.send_json({
                        "type": "stored",
                        "model_used": chunk.model_used,
                        "input_tokens": chunk.input_tokens,
                        "output_tokens": chunk.output_tokens,
                    })

        logger.debug("WebSocket stream complete for session_id=%s, chunks=%d", session_id, chunk_count)
        await websocket.send_json({"type": "done"})

    except ValueError as e:
        logger.error("Streaming error for session_id=%s: %s", session_id, e)
        await websocket.send_json({"type": "error", "message": str(e)})
    except Exception:
        logger.exception("Unexpected streaming error for session_id=%s", session_id)
        await websocket.send_json({"type": "error", "message": "Internal error during streaming"})
