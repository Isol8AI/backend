"""
Agent API endpoints.

Handles agent CRUD operations and message processing.
All agent data is encrypted - the server cannot read it.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import AuthContext, get_current_user
from core.database import get_db
from core.enclave import get_enclave
from core.enclave.agent_handler import AgentHandler, AgentMessageRequest
from core.crypto import EncryptedPayload as CryptoEncryptedPayload
from core.services.agent_service import AgentService
from models.user import User
from schemas.agent import (
    CreateAgentRequest,
    AgentResponse,
    AgentListResponse,
    SendAgentMessageRequest,
    AgentMessageResponse,
)
from schemas.encryption import EncryptedPayload

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Agent CRUD Operations
# =============================================================================


@router.get("", response_model=AgentListResponse)
async def list_agents(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all agents for the current user.

    Returns basic metadata - actual agent content is encrypted.
    """
    service = AgentService(db)
    agents = await service.list_user_agents(user_id=auth.user_id)

    return AgentListResponse(
        agents=[
            AgentResponse(
                agent_name=a.agent_name,
                user_id=a.user_id,
                created_at=a.created_at,
                updated_at=a.updated_at,
                tarball_size_bytes=a.tarball_size_bytes,
            )
            for a in agents
        ]
    )


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: CreateAgentRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new agent.

    Stores metadata only. The actual agent state (SOUL.md, config, memory)
    is created inside the enclave on first message, preserving zero-trust:
    the server never sees the personality content.

    The client passes soul_content encrypted to the enclave in the first
    AGENT_CHAT_STREAM message.
    """
    service = AgentService(db)

    # Check if agent already exists
    existing = await service.get_agent_state(
        user_id=auth.user_id,
        agent_name=request.agent_name,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{request.agent_name}' already exists",
        )

    # Store metadata only — no tarball, no plaintext soul content
    # The enclave creates the fresh agent state on first message
    state = await service.create_agent_state(
        user_id=auth.user_id,
        agent_name=request.agent_name,
    )
    await db.commit()

    return AgentResponse(
        agent_name=state.agent_name,
        user_id=state.user_id,
        created_at=state.created_at,
        updated_at=state.updated_at,
        tarball_size_bytes=state.tarball_size_bytes,
    )


@router.get("/{agent_name}", response_model=AgentResponse)
async def get_agent(
    agent_name: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get agent details.

    Returns metadata only - actual content is encrypted.
    """
    service = AgentService(db)
    state = await service.get_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
    )

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    return AgentResponse(
        agent_name=state.agent_name,
        user_id=state.user_id,
        created_at=state.created_at,
        updated_at=state.updated_at,
        tarball_size_bytes=state.tarball_size_bytes,
    )


@router.delete("/{agent_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_name: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an agent and all its data.

    This is permanent - the agent's memory and history cannot be recovered.
    """
    service = AgentService(db)
    deleted = await service.delete_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    await db.commit()


# =============================================================================
# Agent Messaging
# =============================================================================


@router.post("/{agent_name}/message", response_model=AgentMessageResponse)
async def send_agent_message(
    agent_name: str,
    request: SendAgentMessageRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to an agent.

    If the agent doesn't exist, it's created automatically.
    The message and response are end-to-end encrypted.

    Flow:
    1. Client encrypts message to enclave's public key
    2. Server relays encrypted message to enclave (cannot read it)
    3. Enclave decrypts, runs OpenClaw, re-encrypts response
    4. Server stores encrypted state, returns encrypted response
    5. Client decrypts response with their private key
    """
    service = AgentService(db)
    enclave = get_enclave()

    # Get user's public key for response encryption
    result = await db.execute(select(User).where(User.id == auth.user_id))
    user = result.scalar_one_or_none()

    if not user or not user.public_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User encryption keys not set up",
        )

    user_public_key = bytes.fromhex(user.public_key)

    # Get existing state if any
    existing_state = await service.get_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
    )

    encrypted_state = None
    if existing_state:
        encrypted_state = _deserialize_encrypted_payload(existing_state.encrypted_tarball)

    # Convert API payload to crypto payload
    encrypted_message = request.encrypted_message.to_crypto_payload()

    # Process through enclave
    handler = AgentHandler(enclave=enclave)
    agent_request = AgentMessageRequest(
        user_id=auth.user_id,
        agent_name=agent_name,
        encrypted_message=encrypted_message,
        encrypted_state=encrypted_state,
        user_public_key=user_public_key,
        model=request.model,
    )

    response = await handler.process_message(agent_request)

    if not response.success:
        return AgentMessageResponse(
            success=False,
            error=response.error,
        )

    # Store updated state
    encrypted_state_bytes = _serialize_encrypted_payload(response.encrypted_state)

    if existing_state:
        await service.update_agent_state(
            user_id=auth.user_id,
            agent_name=agent_name,
            encrypted_tarball=encrypted_state_bytes,
        )
    else:
        await service.create_agent_state(
            user_id=auth.user_id,
            agent_name=agent_name,
            encrypted_tarball=encrypted_state_bytes,
        )

    await db.commit()

    # Convert crypto payload to API payload
    api_response = EncryptedPayload.from_crypto_payload(response.encrypted_response)

    return AgentMessageResponse(
        success=True,
        encrypted_response=api_response,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _serialize_encrypted_payload(payload: CryptoEncryptedPayload) -> bytes:
    """Serialize encrypted payload to bytes for storage."""
    return json.dumps(
        {
            "ephemeral_public_key": payload.ephemeral_public_key.hex(),
            "iv": payload.iv.hex(),
            "ciphertext": payload.ciphertext.hex(),
            "auth_tag": payload.auth_tag.hex(),
            "hkdf_salt": payload.hkdf_salt.hex() if payload.hkdf_salt else None,
        }
    ).encode()


def _deserialize_encrypted_payload(data: bytes) -> CryptoEncryptedPayload:
    """Deserialize encrypted payload from storage."""
    obj = json.loads(data.decode())
    return CryptoEncryptedPayload(
        ephemeral_public_key=bytes.fromhex(obj["ephemeral_public_key"]),
        iv=bytes.fromhex(obj["iv"]),
        ciphertext=bytes.fromhex(obj["ciphertext"]),
        auth_tag=bytes.fromhex(obj["auth_tag"]),
        hkdf_salt=bytes.fromhex(obj["hkdf_salt"]) if obj.get("hkdf_salt") else None,
    )
