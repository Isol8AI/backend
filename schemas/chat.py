"""
Pydantic schemas for encrypted chat API endpoints.

Security Note:
- All message content is encrypted - server never sees plaintext
- Server acts as blind relay between client and enclave
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from schemas.encryption import EncryptedPayloadSchema, EncryptedMessageResponse


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""

    name: Optional[str] = Field("New Chat", description="Display name for the session")
    org_id: Optional[str] = Field(None, description="Organization ID for org sessions. None for personal sessions.")


class SessionResponse(BaseModel):
    """Chat session data."""

    id: str
    user_id: str
    org_id: Optional[str] = None
    name: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    """Paginated list of user's chat sessions."""

    sessions: list[SessionResponse]
    total: int
    limit: int
    offset: int


class SessionMessagesResponse(BaseModel):
    """Messages for a session - all encrypted."""

    session_id: str
    messages: list[EncryptedMessageResponse]


class EnclaveInfoResponse(BaseModel):
    """Enclave public key for message encryption."""

    enclave_public_key: str = Field(
        ..., description="Enclave's X25519 public key (32 bytes hex) - encrypt messages to this"
    )
    attestation: Optional[dict] = Field(None, description="Attestation document (production only)")


class StreamingChunkResponse(BaseModel):
    """A chunk of streaming encrypted response."""

    chunk_index: int
    encrypted_chunk: EncryptedPayloadSchema
    is_final: bool = False


class StreamCompleteResponse(BaseModel):
    """Final message after streaming completes."""

    session_id: str
    message_id: str
    stored_user_message: EncryptedMessageResponse
    stored_assistant_message: EncryptedMessageResponse
    model_used: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ModelsResponse(BaseModel):
    """Available LLM models."""

    models: list[dict]


class DeleteSessionsResponse(BaseModel):
    """Response for bulk session deletion."""

    deleted_count: int


class EnclaveHealthResponse(BaseModel):
    """Response from GET /chat/enclave/health."""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    enclave_type: str = Field(..., description="'nitro' or 'mock'")
    enclave_public_key: Optional[str] = Field(None, description="Enclave public key if available")


class EncryptionCheckResponse(BaseModel):
    """Response from GET /chat/encryption-status."""

    can_send_encrypted: bool = Field(..., description="Whether user can send encrypted messages")
    error: Optional[str] = Field(None, description="Error reason if cannot send")
    context: str = Field(..., description="'personal' or 'organization'")
    org_id: Optional[str] = Field(None, description="Current org context if any")
