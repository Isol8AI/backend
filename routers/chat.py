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

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.auth import AuthContext, get_current_user
from core.config import AVAILABLE_MODELS
from core.database import get_db, get_session_factory
from core.services.chat_service import ChatService
from schemas.encryption import EncryptedPayload, SendEncryptedMessageRequest
from schemas.chat import (
    CreateSessionRequest,
    SessionResponse,
    SessionListResponse,
    SessionMessagesResponse,
    EnclaveInfoResponse,
    DeleteSessionsResponse,
)
from schemas.encryption import EncryptedMessageResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Build set of valid model IDs for fast validation
VALID_MODEL_IDS = {model["id"] for model in AVAILABLE_MODELS}


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

@router.get("/enclave/info", response_model=EnclaveInfoResponse)
async def get_enclave_info(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get enclave's public key for message encryption.

    Client MUST encrypt messages to this key before sending.
    The enclave is the only entity that can decrypt and process them.
    """
    service = ChatService(db)
    info = service.get_enclave_info()
    return EnclaveInfoResponse(
        enclave_public_key=info["enclave_public_key"],
        attestation={"document": info["attestation_document"]} if info.get("attestation_document") else None,
    )


# =============================================================================
# Models
# =============================================================================

@router.get("/models", response_model=list[ModelOut])
async def get_available_models() -> list[ModelOut]:
    """Get list of available LLM models."""
    return AVAILABLE_MODELS


# =============================================================================
# Session Management
# =============================================================================

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new chat session.

    Creates in current context (personal or org based on auth).
    """
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


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(
    limit: int = Query(default=50, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all chat sessions for the current user in current context with pagination.

    Sessions are scoped to the current context:
    - Personal mode: Only sessions with org_id=None
    - Org mode: Only sessions with matching org_id
    """
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


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionMessagesResponse:
    """
    Get all messages for a session (encrypted).

    Returns encrypted messages that client must decrypt with their key.
    """
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
                    encrypted_content=EncryptedPayload(**m.encrypted_payload),
                    model_used=m.model_used,
                    created_at=m.created_at,
                )
                for m in messages
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a session and all its messages (GDPR compliance).

    This is a permanent deletion - messages cannot be recovered.
    """
    service = ChatService(db)
    deleted = await service.delete_session(
        session_id=session_id,
        user_id=auth.user_id,
        org_id=auth.org_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found or access denied")


@router.delete("/sessions", response_model=DeleteSessionsResponse)
async def delete_all_sessions(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete all sessions for user in current context (GDPR compliance).

    This is a permanent deletion - messages cannot be recovered.
    Deletes only sessions in the current context (personal or org).
    """
    service = ChatService(db)
    count = await service.delete_all_sessions(
        user_id=auth.user_id,
        org_id=auth.org_id,
    )
    return DeleteSessionsResponse(deleted_count=count)


# =============================================================================
# Encrypted Message Streaming
# =============================================================================

@router.post("/encrypted/stream")
async def chat_stream_encrypted(
    request: SendEncryptedMessageRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """
    Send encrypted message and stream encrypted response.

    This is the main chat endpoint. The flow is:
    1. Client encrypts message TO enclave's public key
    2. Server relays encrypted message to enclave (cannot read it)
    3. Enclave decrypts, processes with LLM, re-encrypts for storage
    4. Encrypted response chunks streamed back via SSE
    5. Client decrypts response with their private key

    SSE event types:
    - session: {session_id} - Sent first with session ID
    - chunk: {encrypted_content} - Encrypted response chunk
    - stored: {user_message, assistant_message} - Final stored messages
    - done: {} - Streaming complete
    - error: {message} - Error occurred
    """
    logger.debug(
        "Encrypted chat request - user_id=%s, org_id=%s, model=%s, session_id=%s, history_count=%d, memory_count=%d, has_facts=%s",
        auth.user_id,
        auth.org_id or "personal",
        request.model,
        request.session_id or "new",
        len(request.encrypted_history) if request.encrypted_history else 0,
        len(request.encrypted_memories) if request.encrypted_memories else 0,
        bool(request.facts_context),
    )

    # Validate model
    if request.model not in VALID_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {list(VALID_MODEL_IDS)}"
        )

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
        if session_id:
            session = await service.get_session(
                session_id=session_id,
                user_id=auth.user_id,
                org_id=auth.org_id,
            )
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or access denied")
        else:
            # Create new session
            session = await service.create_session(
                user_id=auth.user_id,
                name="New Chat",
                org_id=auth.org_id,
            )
            session_id = session.id

    # Convert hex-encoded API payloads to bytes-based crypto payloads
    encrypted_msg = request.encrypted_message.to_crypto_payload()

    encrypted_history = []
    if request.encrypted_history:
        for h in request.encrypted_history:
            encrypted_history.append(h.to_crypto_payload())

    encrypted_memories = []
    if request.encrypted_memories:
        for m in request.encrypted_memories:
            encrypted_memories.append(m.to_crypto_payload())

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
                    encrypted_memories=encrypted_memories,
                    facts_context=request.facts_context,
                    model=request.model,
                    client_transport_public_key=request.client_transport_public_key,
                ):
                    if chunk.error:
                        logger.debug("Enclave error for session_id=%s: %s", session_id, chunk.error)
                        yield f"data: {json.dumps({'type': 'error', 'message': chunk.error})}\n\n"
                        return

                    if chunk.encrypted_content:
                        chunk_count += 1
                        # Convert bytes-based crypto payload to hex-encoded API payload
                        api_payload = EncryptedPayload.from_crypto_payload(chunk.encrypted_content)
                        yield f"data: {json.dumps({'type': 'encrypted_chunk', 'encrypted_content': api_payload.model_dump()})}\n\n"

                    if chunk.is_final and chunk.stored_user_message and chunk.stored_assistant_message:
                        # Send stored message info
                        logger.debug("Messages stored for session_id=%s", session_id)
                        yield f"data: {json.dumps({'type': 'stored', 'model_used': chunk.model_used, 'input_tokens': chunk.input_tokens, 'output_tokens': chunk.output_tokens})}\n\n"

                        # Send extracted facts (encrypted for client-side storage)
                        if chunk.extracted_facts:
                            facts_data = []
                            for fact in chunk.extracted_facts:
                                api_payload = EncryptedPayload.from_crypto_payload(fact.encrypted_payload)
                                facts_data.append({
                                    'fact_id': fact.fact_id,
                                    'encrypted_payload': api_payload.model_dump(),
                                })
                            logger.debug("Sending %d extracted facts for session_id=%s", len(facts_data), session_id)
                            yield f"data: {json.dumps({'type': 'extracted_facts', 'facts': facts_data})}\n\n"

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
        }
    )


# =============================================================================
# Encryption Status Check
# =============================================================================

@router.get("/encryption-status")
async def get_encryption_status(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Check if user can send encrypted messages in current context.

    Returns status for both personal and org contexts.
    """
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
