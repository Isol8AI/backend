"""
HTTP routes for API Gateway WebSocket integration.

API Gateway WebSocket converts WebSocket frames into HTTP POST requests:
- $connect  -> POST /ws/connect
- $disconnect -> POST /ws/disconnect
- $default (messages) -> POST /ws/message

Responses are pushed via Management API, not returned in HTTP response body.
The HTTP response only indicates whether the request was accepted (200) or rejected (4xx).

Security Note:
- Server acts as BLIND RELAY - cannot read message content
- All messages encrypted to enclave (transport) or user/org (storage)
- Management API delivers encrypted chunks to client
"""

import logging
from typing import Any, Dict, Optional

from clerk_backend_api import Clerk
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Response
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import AVAILABLE_MODELS, settings
from core.database import get_session_factory as db_get_session_factory
from core.services.chat_service import ChatService
from core.services.connection_service import ConnectionService, ConnectionServiceError
from core.services.management_api_client import ManagementApiClient, ManagementApiClientError
from schemas.agent import AgentChatWSRequest
from schemas.encryption import EncryptedPayload, SendEncryptedMessageRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Build set of valid model IDs for fast validation
VALID_MODEL_IDS = {model["id"] for model in AVAILABLE_MODELS}

# Singleton instances (created lazily)
_connection_service: Optional[ConnectionService] = None
_management_api_client: Optional[ManagementApiClient] = None


def get_connection_service() -> ConnectionService:
    """Get or create ConnectionService singleton."""
    global _connection_service
    if _connection_service is None:
        _connection_service = ConnectionService()
    return _connection_service


def get_management_api_client() -> ManagementApiClient:
    """Get or create ManagementApiClient singleton."""
    global _management_api_client
    if _management_api_client is None:
        _management_api_client = ManagementApiClient()
    return _management_api_client


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get database session factory."""
    return db_get_session_factory()


@router.post("/connect", status_code=200)
async def ws_connect(
    x_connection_id: Optional[str] = Header(None, alias="x-connection-id"),
    x_user_id: Optional[str] = Header(None, alias="x-user-id"),
    x_org_id: Optional[str] = Header(None, alias="x-org-id"),
) -> Response:
    """
    Handle WebSocket $connect event from API Gateway.

    Called when a client establishes a WebSocket connection. The Lambda authorizer
    has already validated the JWT and extracted user identity into headers.

    Headers (set by Lambda authorizer):
    - x-connection-id: API Gateway connection ID (required)
    - x-user-id: Authenticated user ID from Clerk JWT (required)
    - x-org-id: Organization ID from Clerk JWT (optional)

    Returns:
        200: Connection stored successfully (empty body to avoid forwarding)
        400: Missing connection-id header
        401: Missing user-id header (unauthorized)

    Note: Returns empty body because API Gateway may forward HTTP response body
    as WebSocket message when route responses are configured.
    """
    if not x_connection_id:
        raise HTTPException(status_code=400, detail="Missing x-connection-id header")

    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing x-user-id header")

    logger.info(
        "WebSocket connect: connection_id=%s, user_id=%s, org_id=%s",
        x_connection_id,
        x_user_id,
        x_org_id or "personal",
    )

    connection_service = get_connection_service()
    connection_service.store_connection(
        connection_id=x_connection_id,
        user_id=x_user_id,
        org_id=x_org_id,
    )

    # Return empty body - actual communication happens via Management API
    return Response(status_code=200)


@router.post("/disconnect", status_code=200)
async def ws_disconnect(
    x_connection_id: Optional[str] = Header(None, alias="x-connection-id"),
) -> Response:
    """
    Handle WebSocket $disconnect event from API Gateway.

    Called when a client's WebSocket connection is closed. This is a best-effort
    cleanup - we always return 200 even if the connection wasn't found or deletion
    fails. This prevents API Gateway from retrying disconnect events.

    Headers:
    - x-connection-id: API Gateway connection ID (optional for best-effort)

    Returns:
        200: Always (best-effort cleanup, empty body)
    """
    if not x_connection_id:
        logger.debug("Disconnect without connection_id, ignoring")
        return Response(status_code=200)

    logger.info("WebSocket disconnect: connection_id=%s", x_connection_id)

    try:
        connection_service = get_connection_service()
        connection_service.delete_connection(x_connection_id)
    except ConnectionServiceError as e:
        # Log but don't fail - best effort cleanup
        logger.warning("Failed to delete connection %s: %s", x_connection_id, e)
    except Exception as e:
        logger.exception("Unexpected error deleting connection %s: %s", x_connection_id, e)

    # Return empty body - actual communication happens via Management API
    return Response(status_code=200)


@router.post("/message", status_code=200)
async def ws_message(
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
    x_connection_id: Optional[str] = Header(None, alias="x-connection-id"),
) -> Response:
    """
    Handle WebSocket $default (message) event from API Gateway.

    Processes incoming WebSocket messages from clients. The message body is
    JSON parsed and routed based on message type:
    - ping: Responds with pong (keepalive)
    - pong: Acknowledges silently (client keepalive response)
    - chat: Validates and queues for background processing

    Headers:
    - x-connection-id: API Gateway connection ID (required)

    Body: JSON message from client

    Returns:
        200: Message received and being processed (empty body)
        400: Missing connection-id header
        401: Unknown connection (not in DynamoDB)

    Note: All responses (pong, errors, chat) are sent via Management API,
    not HTTP response. Returns empty body to avoid forwarding to WebSocket.
    """
    if not x_connection_id:
        raise HTTPException(status_code=400, detail="Missing x-connection-id header")

    # Look up connection to get user context
    connection_service = get_connection_service()
    connection = connection_service.get_connection(x_connection_id)

    if not connection:
        raise HTTPException(status_code=401, detail="Unknown connection")

    user_id = connection["user_id"]
    conn_org_id = connection["org_id"]

    # Handle message types
    msg_type = body.get("type")

    if msg_type == "ping":
        # Respond with pong via Management API
        management_api = get_management_api_client()
        management_api.send_message(x_connection_id, {"type": "pong"})
        return Response(status_code=200)

    if msg_type == "pong":
        # Client acknowledged our ping - no action needed
        return Response(status_code=200)

    if msg_type == "agent_chat":
        try:
            _validate_and_process_agent_chat(
                connection_id=x_connection_id,
                user_id=user_id,
                body=body,
                background_tasks=background_tasks,
            )
        except ValueError as e:
            management_api = get_management_api_client()
            management_api.send_message(x_connection_id, {"type": "error", "message": str(e)})
        return Response(status_code=200)

    # Assume it's a chat message - validate and process
    try:
        _validate_and_process_chat(
            connection_id=x_connection_id,
            user_id=user_id,
            conn_org_id=conn_org_id,
            body=body,
            background_tasks=background_tasks,
        )
    except ValueError as e:
        # Validation failed - send error via Management API
        management_api = get_management_api_client()
        management_api.send_message(x_connection_id, {"type": "error", "message": str(e)})

    # Return empty body - actual response comes via Management API
    return Response(status_code=200)


def _validate_and_process_chat(
    connection_id: str,
    user_id: str,
    conn_org_id: Optional[str],
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> None:
    """
    Validate chat message and queue for background processing.

    Raises:
        ValueError: If message format is invalid or model is not supported
    """
    # Parse and validate request using Pydantic
    try:
        request = SendEncryptedMessageRequest(
            session_id=body.get("session_id"),
            model=body.get("model", ""),
            encrypted_message=EncryptedPayload(**body["encrypted_message"]),
            encrypted_history=[EncryptedPayload(**h) for h in body.get("encrypted_history", [])] or None,
            facts_context=body.get("facts_context"),
            client_transport_public_key=body["client_transport_public_key"],
        )
    except (KeyError, ValidationError) as e:
        logger.error("Invalid message format: %s", e)
        raise ValueError(f"Invalid message format: {e}")

    # Validate model
    if request.model not in VALID_MODEL_IDS:
        raise ValueError(f"Invalid model. Available models: {list(VALID_MODEL_IDS)}")

    # Allow org_id from message to override connection org
    msg_org_id = body.get("org_id") or conn_org_id

    # Queue chat processing as background task
    background_tasks.add_task(
        _process_chat_message_background,
        connection_id=connection_id,
        user_id=user_id,
        org_id=msg_org_id,
        request=request,
    )


async def _process_chat_message_background(
    connection_id: str,
    user_id: str,
    org_id: Optional[str],
    request: SendEncryptedMessageRequest,
) -> None:
    """
    Process chat message in background task.

    Streams response chunks via Management API.
    """
    logger.debug(
        "Processing chat - connection_id=%s, user_id=%s, org_id=%s, model=%s, session_id=%s",
        connection_id,
        user_id,
        org_id or "personal",
        request.model,
        request.session_id or "new",
    )

    management_api = get_management_api_client()
    session_factory = get_session_factory()

    try:
        # Get or create session
        async with session_factory() as service_db:
            service = ChatService(service_db)

            # Verify user can send encrypted messages
            can_send, error_msg = await service.verify_can_send_encrypted(
                user_id=user_id,
                org_id=org_id,
            )
            if not can_send:
                management_api.send_message(connection_id, {"type": "error", "message": error_msg})
                return

            session_id = request.session_id
            is_new_session = False
            if session_id:
                session = await service.get_session(
                    session_id=session_id,
                    user_id=user_id,
                    org_id=org_id,
                )
                if not session:
                    management_api.send_message(
                        connection_id,
                        {"type": "error", "message": "Session not found or access denied"},
                    )
                    return
            else:
                # Create new session (deferred - not committed yet)
                session = await service.create_session_deferred(
                    user_id=user_id,
                    name="New Chat",
                    org_id=org_id,
                )
                session_id = session.id
                is_new_session = True

        # Send session ID first
        management_api.send_message(connection_id, {"type": "session", "session_id": session_id})

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
                is_new_session=is_new_session,
            ):
                if chunk.error:
                    logger.debug("Enclave error for session_id=%s: %s", session_id, chunk.error)
                    management_api.send_message(
                        connection_id,
                        {"type": "error", "message": chunk.error},
                    )
                    return

                if chunk.encrypted_content:
                    chunk_count += 1
                    # Convert bytes-based crypto payload to hex-encoded API payload
                    api_payload = EncryptedPayload.from_crypto_payload(chunk.encrypted_content)
                    if not management_api.send_message(
                        connection_id,
                        {"type": "encrypted_chunk", "encrypted_content": api_payload.model_dump()},
                    ):
                        # Connection gone - stop streaming
                        logger.warning("Connection %s gone during streaming", connection_id)
                        return

                if chunk.encrypted_thinking:
                    # Send thinking chunk
                    api_payload = EncryptedPayload.from_crypto_payload(chunk.encrypted_thinking)
                    management_api.send_message(
                        connection_id,
                        {"type": "thinking", "encrypted_content": api_payload.model_dump()},
                    )

                if chunk.is_final and chunk.stored_user_message and chunk.stored_assistant_message:
                    # Send stored message info
                    logger.debug("Messages stored for session_id=%s", session_id)
                    management_api.send_message(
                        connection_id,
                        {
                            "type": "stored",
                            "model_used": chunk.model_used,
                            "input_tokens": chunk.input_tokens,
                            "output_tokens": chunk.output_tokens,
                        },
                    )

        logger.debug(
            "Stream complete for connection_id=%s, session_id=%s, chunks=%d",
            connection_id,
            session_id,
            chunk_count,
        )
        management_api.send_message(connection_id, {"type": "done"})

    except ManagementApiClientError as e:
        logger.error("Management API error for connection %s: %s", connection_id, e)
        # Can't send error to client - Management API failed
    except Exception as e:
        logger.exception("Unexpected error processing chat for connection %s: %s", connection_id, e)
        try:
            management_api.send_message(
                connection_id,
                {"type": "error", "message": "Internal error during processing"},
            )
        except Exception:
            pass  # Best effort


# =============================================================================
# Agent Chat (Streaming)
# =============================================================================


def _validate_and_process_agent_chat(
    connection_id: str,
    user_id: str,
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> None:
    """
    Validate agent chat message and queue for background processing.

    Raises:
        ValueError: If message format is invalid
    """
    try:
        encrypted_soul = None
        if body.get("encrypted_soul_content"):
            encrypted_soul = EncryptedPayload(**body["encrypted_soul_content"])

        encrypted_state_from_client = None
        if body.get("encrypted_state"):
            encrypted_state_from_client = EncryptedPayload(**body["encrypted_state"])

        request = AgentChatWSRequest(
            agent_name=body["agent_name"],
            encrypted_message=EncryptedPayload(**body["encrypted_message"]),
            client_transport_public_key=body["client_transport_public_key"],
            encrypted_soul_content=encrypted_soul,
            encrypted_state=encrypted_state_from_client,
        )
    except (KeyError, ValidationError) as e:
        logger.error("Invalid agent chat message format: %s", e)
        raise ValueError(f"Invalid message format: {e}")

    background_tasks.add_task(
        _process_agent_chat_background,
        connection_id=connection_id,
        user_id=user_id,
        agent_name=request.agent_name,
        encrypted_message=request.encrypted_message,
        client_transport_public_key=request.client_transport_public_key,
        encrypted_soul_content=request.encrypted_soul_content,
        encrypted_state_from_client=request.encrypted_state,
    )


async def _process_agent_chat_background(
    connection_id: str,
    user_id: str,
    agent_name: str,
    encrypted_message: EncryptedPayload,
    client_transport_public_key: str,
    encrypted_soul_content: Optional[EncryptedPayload] = None,
    encrypted_state_from_client: Optional[EncryptedPayload] = None,
) -> None:
    """
    Process agent chat message in background task with streaming.

    For zero_trust mode: client provides encrypted_state_from_client (re-encrypted to enclave)
    For background mode: server loads encrypted_dek from DB
    """
    import json as json_module

    from core.enclave import get_enclave, AgentHandler, AgentStreamRequest
    from core.services.agent_service import AgentService

    logger.debug(
        "Processing agent chat - connection_id=%s, user_id=%s, agent=%s",
        connection_id,
        user_id,
        agent_name,
    )

    management_api = get_management_api_client()
    session_factory = get_session_factory()

    try:
        # Look up agent to determine encryption mode
        async with session_factory() as db:
            service = AgentService(db)
            agent_state = await service.get_agent_state(user_id, agent_name)

        if not agent_state:
            management_api.send_message(
                connection_id,
                {"type": "error", "message": f"Agent '{agent_name}' not found"},
            )
            return

        encryption_mode = agent_state.encryption_mode

        # For zero_trust mode: client provides re-encrypted state in request
        # For background mode: server loads encrypted_dek from DB (not implemented)

        encrypted_state_for_enclave = None
        if encryption_mode == "zero_trust":
            # For zero_trust mode: client provides re-encrypted state in the request
            if encrypted_state_from_client:
                # Client already decrypted (with user key) and re-encrypted (to enclave transport key)
                encrypted_state_for_enclave = encrypted_state_from_client.to_crypto_payload()
            # else: no state (new agent or first message)
        elif encryption_mode == "background":
            # Background mode: would load encrypted_dek from DB and pass to enclave
            # Not yet implemented
            management_api.send_message(
                connection_id,
                {"type": "error", "message": "Background mode not yet implemented"},
            )
            return

        # Convert API encrypted_message to crypto payload
        encrypted_msg = encrypted_message.to_crypto_payload()
        client_pub_key = bytes.fromhex(client_transport_public_key)

        # Convert encrypted_soul_content if provided
        encrypted_soul = None
        if encrypted_soul_content:
            encrypted_soul = encrypted_soul_content.to_crypto_payload()

        # Create stream request
        enclave = get_enclave()
        handler = AgentHandler(enclave=enclave)
        request = AgentStreamRequest(
            user_id=user_id,
            agent_name=agent_name,
            encrypted_message=encrypted_msg,
            encrypted_state=encrypted_state_for_enclave,
            client_public_key=client_pub_key,
            encrypted_soul_content=encrypted_soul,
            encryption_mode=encryption_mode,
        )

        # Stream response chunks
        chunk_count = 0
        async for chunk in handler.process_message_streaming(request):
            if chunk.error:
                logger.debug("Agent enclave error for %s: %s", agent_name, chunk.error)
                management_api.send_message(
                    connection_id,
                    {"type": "error", "message": chunk.error},
                )
                return

            if chunk.encrypted_content:
                chunk_count += 1
                api_payload = EncryptedPayload.from_crypto_payload(chunk.encrypted_content)
                if not management_api.send_message(
                    connection_id,
                    {"type": "encrypted_chunk", "encrypted_content": api_payload.model_dump()},
                ):
                    logger.warning("Connection %s gone during agent streaming", connection_id)
                    return

            if chunk.is_final and chunk.encrypted_state:
                # Serialize updated state and store in DB
                state_json = json_module.dumps(chunk.encrypted_state.to_dict()).encode("utf-8")

                async with session_factory() as db:
                    service = AgentService(db)
                    await service.update_agent_state(user_id, agent_name, state_json)
                    await db.commit()

                logger.debug("Agent state updated for %s/%s", user_id, agent_name)

        logger.debug(
            "Agent stream complete for connection_id=%s, agent=%s, chunks=%d",
            connection_id,
            agent_name,
            chunk_count,
        )
        management_api.send_message(connection_id, {"type": "done"})

    except ManagementApiClientError as e:
        logger.error("Management API error for connection %s: %s", connection_id, e)
    except Exception as e:
        logger.exception("Unexpected error processing agent chat for connection %s: %s", connection_id, e)
        try:
            management_api.send_message(
                connection_id,
                {"type": "error", "message": "Internal error during processing"},
            )
        except Exception:
            pass
