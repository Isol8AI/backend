"""Pydantic schemas for agent API."""

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from schemas.encryption import EncryptedPayload


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""

    agent_name: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    soul_content: Optional[str] = Field(None, max_length=10000)
    model: str = Field(default="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    encryption_mode: Literal["zero_trust", "background"] = Field(
        default="zero_trust",
        description="Encryption mode: zero_trust (user key, default) or background (KMS, opt-in)",
    )


class AgentResponse(BaseModel):
    """Agent details response."""

    agent_name: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    tarball_size_bytes: Optional[int] = None
    encryption_mode: Literal["zero_trust", "background"] = Field(
        default="zero_trust",
        description="Encryption mode for this agent",
    )

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """List of agents response."""

    agents: List[AgentResponse]


class SendAgentMessageRequest(BaseModel):
    """Request to send a message to an agent."""

    encrypted_message: EncryptedPayload
    model: str = Field(default="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    # For zero_trust mode: client decrypts state, re-encrypts to enclave transport key
    encrypted_state: Optional[EncryptedPayload] = Field(
        default=None,
        description="Agent state encrypted to enclave transport key (zero_trust mode only)",
    )


class AgentMessageResponse(BaseModel):
    """Response from agent message."""

    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    error: Optional[str] = None


class AgentChatWSRequest(BaseModel):
    """WebSocket request for streaming agent chat."""

    agent_name: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    encrypted_message: EncryptedPayload
    client_transport_public_key: str
    # Optional: encrypted soul/personality content for first message (new agent)
    # Encrypted to enclave's public key so server cannot read it
    encrypted_soul_content: Optional[EncryptedPayload] = None
    # For zero_trust mode: client provides decrypted state re-encrypted to enclave
    encrypted_state: Optional[EncryptedPayload] = Field(
        default=None,
        description="Agent state encrypted to enclave transport key (zero_trust mode only)",
    )
