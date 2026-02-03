"""Pydantic schemas for agent API."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from schemas.encryption import EncryptedPayload


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""

    agent_name: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    soul_content: Optional[str] = Field(None, max_length=10000)
    model: str = Field(default="us.anthropic.claude-3-5-sonnet-20241022-v2:0")


class AgentResponse(BaseModel):
    """Agent details response."""

    agent_name: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    tarball_size_bytes: Optional[int] = None

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """List of agents response."""

    agents: List[AgentResponse]


class SendAgentMessageRequest(BaseModel):
    """Request to send a message to an agent."""

    encrypted_message: EncryptedPayload
    model: str = Field(default="us.anthropic.claude-3-5-sonnet-20241022-v2:0")


class AgentMessageResponse(BaseModel):
    """Response from agent message."""

    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    error: Optional[str] = None
