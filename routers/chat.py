"""
Encrypted chat API endpoints.

Security Note:
- Server acts as BLIND RELAY - cannot read message content
- All messages encrypted to enclave (transport) or user/org (storage)
- SSE streaming delivers encrypted chunks to client
"""

import json
import logging
from typing import Optional

from clerk_backend_api import Clerk
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.auth import AuthContext, get_current_user
from core.config import get_available_models, settings
from core.database import get_db, get_session_factory
from core.services.chat_service import ChatService
from schemas.encryption import EncryptedPayloadSchema, SendEncryptedMessageRequest
from schemas.chat import (
    CreateSessionRequest,
    SessionResponse,
    SessionListResponse,
    SessionMessagesResponse,
    EnclaveInfoResponse,
    EnclaveHealthResponse,
    DeleteSessionsResponse,
    EncryptionCheckResponse,
)
from schemas.encryption import EncryptedMessageResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_valid_model_ids() -> set[str]:
    return {model["id"] for model in get_available_models()}


# =============================================================================
# Schema for API Responses
# =============================================================================


class SessionOut(BaseModel):
    """Session for API response."""

    id: str
    name: str
    org_id: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ModelOut(BaseModel):
    """Model info for API response."""

    id: str
    name: str


# =============================================================================
# Enclave Info
# =============================================================================


@router.get(
    "/enclave/info",
    response_model=EnclaveInfoResponse,
    summary="Get enclave info",
    description="Get enclave's public key for message encryption. Client MUST encrypt messages to this key.",
    operation_id="get_enclave_info",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        503: {"description": "Enclave not available"},
    },
)
async def get_enclave_info(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    info = service.get_enclave_info()
    return EnclaveInfoResponse(
        enclave_public_key=info["enclave_public_key"],
        attestation={"document": info["attestation_document"]} if info.get("attestation_document") else None,
    )


@router.get(
    "/enclave/health",
    summary="Check enclave health",
    description="Check enclave health and connectivity. Returns enclave status, mode (mock/nitro), and credential status.",
    operation_id="get_enclave_health",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        503: {"description": "Enclave not available"},
    },
)
async def get_enclave_health(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from core.enclave import get_enclave

    try:
        enclave = get_enclave()

        # For NitroEnclaveClient, use health_check method
        if hasattr(enclave, "health_check"):
            return enclave.health_check()

        # For MockEnclave, return basic info
        info = enclave.get_info()
        return {
            "status": "healthy",
            "mode": "mock",
            "public_key": info.enclave_public_key.hex()[:16] + "...",
        }

    except Exception as e:
        logger.error(f"Enclave health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enclave not available",
        )


# =============================================================================
# Models
# =============================================================================


@router.get(
    "/models",
    response_model=list[ModelOut],
    summary="List available models",
    description="Get list of available LLM models via Bedrock discovery.",
    operation_id="list_models",
)
async def list_models() -> list[ModelOut]:
    return get_available_models()


# =============================================================================
# Session Management
# =============================================================================


@router.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create chat session",
    description="Create a new chat session. Creates in current context (personal or org based on auth).",
    operation_id="create_session",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        400: {"description": "Invalid request data"},
    },
)
async def create_session(
    request: CreateSessionRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)

    # Use org_id from request or auth context
    org_id = request.org_id or auth.org_id

    try:
        session = await service.create_session(
            user_id=auth.user_id,
            name=request.name or "New Chat",
            org_id=org_id,
        )
        return SessionResponse.model_validate(session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List chat sessions",
    description="Get all chat sessions for the current user in current context with pagination. Sessions are scoped to personal or org context.",
    operation_id="list_sessions",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def get_sessions(
    limit: int = Query(default=50, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    sessions, total = await service.list_sessions(
        user_id=auth.user_id,
        org_id=auth.org_id,
        limit=limit,
        offset=offset,
    )

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                user_id=s.user_id,
                name=s.name,
                org_id=s.org_id,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
            for s in sessions
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
    summary="Get session messages",
    description="Get all messages for a session (encrypted). Returns encrypted messages that client must decrypt with their key.",
    operation_id="get_session_messages",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Session not found or access denied"},
    },
)
async def get_session_messages(
    session_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionMessagesResponse:
    service = ChatService(db)

    try:
        messages = await service.get_session_messages(
            session_id=session_id,
            user_id=auth.user_id,
            org_id=auth.org_id,
        )

        return SessionMessagesResponse(
            session_id=session_id,
            messages=[
                EncryptedMessageResponse(
                    id=m.id,
                    session_id=m.session_id,
                    role=m.role,
                    encrypted_content=EncryptedPayloadSchema(**m.encrypted_payload),
                    model_used=m.model_used,
                    created_at=m.created_at,
                )
                for m in messages
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chat session",
    description="Delete a session and all its messages (GDPR compliance). This is a permanent deletion - messages cannot be recovered.",
    operation_id="delete_session",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Session not found or access denied"},
    },
)
async def delete_session(
    session_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    deleted = await service.delete_session(
        session_id=session_id,
        user_id=auth.user_id,
        org_id=auth.org_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found or access denied")


@router.delete(
    "/sessions",
    response_model=DeleteSessionsResponse,
    summary="Delete all sessions",
    description="Delete all sessions for user in current context (GDPR compliance). This is a permanent deletion. Deletes only sessions in the current context (personal or org).",
    operation_id="delete_all_sessions",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def delete_all_sessions(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    count = await service.delete_all_sessions(
        user_id=auth.user_id,
        org_id=auth.org_id,
    )
    return DeleteSessionsResponse(deleted_count=count)


# =============================================================================
# Encrypted Message Streaming
# =============================================================================


@router.post(
    "/encrypted/stream",
    summary="Stream encrypted chat response",
    description=(
        "Send encrypted message and stream encrypted response via SSE. "
        "SSE event types: session (session_id), encrypted_chunk (encrypted content), "
        "thinking (encrypted thinking), stored (final stored messages), done (complete), error (error occurred)."
    ),
    operation_id="chat_stream_encrypted",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        400: {"description": "Invalid model or encryption not set up"},
        404: {"description": "Session not found or access denied"},
    },
)
async def chat_stream_encrypted(
    request: SendEncryptedMessageRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    logger.debug(
        "Encrypted chat request - user_id=%s, org_id=%s, model=%s, session_id=%s, history_count=%d, has_facts=%s",
        auth.user_id,
        auth.org_id or "personal",
        request.model,
        request.session_id or "new",
        len(request.encrypted_history) if request.encrypted_history else 0,
        bool(request.facts_context),
    )

    # Validate model
    if request.model not in _get_valid_model_ids():
        raise HTTPException(status_code=400, detail=f"Invalid model. Available models: {list(_get_valid_model_ids())}")

    async with session_factory() as service_db:
        service = ChatService(service_db)

        # Verify user can send encrypted messages
        can_send, error_msg = await service.verify_can_send_encrypted(
            user_id=auth.user_id,
            org_id=auth.org_id,
        )
        if not can_send:
            raise HTTPException(status_code=400, detail=error_msg)

        # Get or create session
        session_id = request.session_id
        is_new_session = False
        if session_id:
            session = await service.get_session(
                session_id=session_id,
                user_id=auth.user_id,
                org_id=auth.org_id,
            )
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or access denied")
        else:
            # Create new session (deferred - not committed yet)
            session = await service.create_session_deferred(
                user_id=auth.user_id,
                name="New Chat",
                org_id=auth.org_id,
            )
            session_id = session.id
            is_new_session = True

    # Convert hex-encoded API payloads to bytes-based crypto payloads
    encrypted_msg = request.encrypted_message.to_crypto()

    encrypted_history = []
    if request.encrypted_history:
        for h in request.encrypted_history:
            encrypted_history.append(h.to_crypto())

    # Fetch user/org metadata from Clerk for AWS credential resolution
    user_metadata = None
    org_metadata = None
    if settings.CLERK_SECRET_KEY:
        try:
            clerk = Clerk(bearer_auth=settings.CLERK_SECRET_KEY)
            user = clerk.users.get(user_id=auth.user_id)
            user_metadata = user.private_metadata

            if auth.org_id:
                org = clerk.organizations.get(organization_id=auth.org_id)
                org_metadata = org.private_metadata
        except Exception as e:
            logger.warning(f"Could not fetch Clerk metadata: {e}")
            # Continue without custom credentials - will use IAM role

    async def generate():
        """Generate SSE stream with encrypted content."""
        logger.debug("SSE stream started for session_id=%s", session_id)
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        try:
            chunk_count = 0

            async with session_factory() as stream_db:
                stream_service = ChatService(stream_db)

                async for chunk in stream_service.process_encrypted_message_stream(
                    session_id=session_id,
                    user_id=auth.user_id,
                    org_id=auth.org_id,
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
                        yield f"data: {json.dumps({'type': 'error', 'message': chunk.error})}\n\n"
                        return

                    if chunk.encrypted_content:
                        chunk_count += 1
                        # Convert bytes-based crypto payload to hex-encoded API payload
                        api_payload = EncryptedPayloadSchema.from_crypto(chunk.encrypted_content)
                        yield f"data: {json.dumps({'type': 'encrypted_chunk', 'encrypted_content': api_payload.model_dump()})}\n\n"

                    if chunk.encrypted_thinking:
                        # Send thinking chunk
                        api_payload = EncryptedPayloadSchema.from_crypto(chunk.encrypted_thinking)
                        yield f"data: {json.dumps({'type': 'thinking', 'encrypted_content': api_payload.model_dump()})}\n\n"

                    if chunk.is_final and chunk.stored_user_message and chunk.stored_assistant_message:
                        # Send stored message info
                        logger.debug("Messages stored for session_id=%s", session_id)
                        yield f"data: {json.dumps({'type': 'stored', 'model_used': chunk.model_used, 'input_tokens': chunk.input_tokens, 'output_tokens': chunk.output_tokens})}\n\n"

            logger.debug("SSE stream complete for session_id=%s, chunks=%d", session_id, chunk_count)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except ValueError as e:
            logger.error("Streaming error for session_id=%s: %s", session_id, e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception:
            logger.exception("Unexpected streaming error for session_id=%s", session_id)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal error during streaming'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# =============================================================================
# Encryption Status Check
# =============================================================================


@router.get(
    "/encryption-status",
    response_model=EncryptionCheckResponse,
    summary="Check encryption status",
    description="Check if user can send encrypted messages in current context. Returns status for both personal and org contexts.",
    operation_id="get_encryption_status",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def get_encryption_status(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)

    can_send, error = await service.verify_can_send_encrypted(
        user_id=auth.user_id,
        org_id=auth.org_id,
    )

    return {
        "can_send_encrypted": can_send,
        "error": error if not can_send else None,
        "context": "organization" if auth.org_id else "personal",
        "org_id": auth.org_id,
    }
