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
from core.crypto import EncryptedPayload
from core.services.agent_service import AgentService
from models.user import User
from schemas.agent import (
    CreateAgentRequest,
    AgentResponse,
    AgentListResponse,
    SendAgentMessageRequest,
    AgentMessageResponse,
    UpdateAgentStateRequest,
    ExtractAgentFilesRequest,
    ExtractAgentFilesResponse,
    PackAgentFilesRequest,
)
from schemas.encryption import EncryptedPayloadSchema

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Agent CRUD Operations
# =============================================================================


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List agents",
    description="List all agents for the current user. Returns basic metadata - actual agent content is encrypted.",
    operation_id="list_agents",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def list_agents(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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
                encryption_mode=a.encryption_mode,
            )
            for a in agents
        ]
    )


@router.post(
    "",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create agent",
    description="Create a new agent. Stores metadata only - actual agent state is created inside the enclave on first message.",
    operation_id="create_agent",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        409: {"description": "Agent with this name already exists"},
    },
)
async def create_agent(
    request: CreateAgentRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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
        encryption_mode=request.encryption_mode,
    )
    await db.commit()

    return AgentResponse(
        agent_name=state.agent_name,
        user_id=state.user_id,
        created_at=state.created_at,
        updated_at=state.updated_at,
        tarball_size_bytes=state.tarball_size_bytes,
        encryption_mode=state.encryption_mode,
    )


@router.get(
    "/{agent_name}",
    response_model=AgentResponse,
    summary="Get agent details",
    description="Get agent metadata. Returns metadata only - actual content is encrypted.",
    operation_id="get_agent",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Agent not found"},
    },
)
async def get_agent(
    agent_name: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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
        encryption_mode=state.encryption_mode,
    )


@router.delete(
    "/{agent_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent",
    description="Delete an agent and all its data permanently. The agent's memory and history cannot be recovered.",
    operation_id="delete_agent",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Agent not found"},
    },
)
async def delete_agent(
    agent_name: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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


@router.get(
    "/{agent_name}/state",
    summary="Get agent state",
    description="Get encrypted agent state for zero_trust mode. Client decrypts with user's private key, re-encrypts to enclave transport key.",
    operation_id="get_agent_state",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def get_agent_state(
    agent_name: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = AgentService(db)
    state = await service.get_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
    )

    if not state or not state.encrypted_tarball:
        return {"encrypted_state": None, "encryption_mode": state.encryption_mode if state else "zero_trust"}

    if state.encryption_mode == "background":
        # Background mode: client doesn't need state (server loads it directly)
        return {"encrypted_state": None, "encryption_mode": "background"}

    # Zero trust: deserialize and return so client can decrypt/re-encrypt
    encrypted_payload = _deserialize_encrypted_payload(state.encrypted_tarball)
    api_payload = EncryptedPayloadSchema.from_crypto(encrypted_payload)

    return {
        "encrypted_state": api_payload,
        "encryption_mode": state.encryption_mode,
    }


@router.put(
    "/{agent_name}/state",
    summary="Update agent state",
    description="Upload a modified encrypted tarball. Works for zero_trust mode (client encrypts to user's public key).",
    operation_id="update_agent_state",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Agent not found"},
    },
)
async def update_agent_state(
    agent_name: str,
    request: UpdateAgentStateRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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

    encrypted_state_bytes = _serialize_encrypted_payload(request.encrypted_state.to_crypto())
    await service.update_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
        encrypted_tarball=encrypted_state_bytes,
    )
    await db.commit()

    return {"status": "ok"}


@router.post(
    "/{agent_name}/files/extract",
    response_model=ExtractAgentFilesResponse,
    summary="Extract agent files (background mode)",
    description="Extract files from a background-mode agent's KMS-encrypted tarball via enclave.",
    operation_id="extract_agent_files",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Agent not found or no state"},
        400: {"description": "Agent is not in background mode"},
    },
)
async def extract_agent_files(
    agent_name: str,
    request: ExtractAgentFilesRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = AgentService(db)
    state = await service.get_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
    )

    if not state or not state.encrypted_tarball:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found or has no state",
        )

    if state.encryption_mode != "background":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Extract endpoint is only for background-mode agents",
        )

    enclave = get_enclave()
    kms_envelope = json.loads(state.encrypted_tarball.decode())

    response = await enclave.extract_agent_files(
        kms_envelope=kms_envelope,
        user_public_key=request.ephemeral_public_key,
    )

    return ExtractAgentFilesResponse(
        encrypted_files=EncryptedPayloadSchema.from_crypto(response),
    )


@router.post(
    "/{agent_name}/files/pack",
    summary="Pack agent files (background mode)",
    description="Pack modified files into a new KMS-encrypted tarball via enclave.",
    operation_id="pack_agent_files",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        404: {"description": "Agent not found"},
        400: {"description": "Agent is not in background mode"},
    },
)
async def pack_agent_files(
    agent_name: str,
    request: PackAgentFilesRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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

    if state.encryption_mode != "background":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pack endpoint is only for background-mode agents",
        )

    enclave = get_enclave()
    files_for_enclave = [
        {
            "path": f.path,
            "encrypted_content": f.encrypted_content.to_crypto().to_dict(),
        }
        for f in request.files
    ]

    kms_envelope = await enclave.pack_agent_files(files=files_for_enclave)

    # Store updated KMS envelope
    kms_envelope_serialized = json.dumps(kms_envelope).encode()
    await service.update_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
        encrypted_tarball=kms_envelope_serialized,
    )
    await db.commit()

    return {"status": "ok"}


# =============================================================================
# Agent Messaging
# =============================================================================


@router.post(
    "/{agent_name}/message",
    response_model=AgentMessageResponse,
    summary="Send agent message",
    description="Send an encrypted message to an agent. Auto-creates agent if it doesn't exist. Supports zero_trust and background encryption modes.",
    operation_id="send_agent_message",
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
        400: {"description": "User encryption keys not set up"},
    },
)
async def send_agent_message(
    agent_name: str,
    request: SendAgentMessageRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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

    # Get or create agent state record
    existing_state = await service.get_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
    )

    # Determine encryption mode
    encryption_mode = "zero_trust"  # Default
    if existing_state:
        encryption_mode = existing_state.encryption_mode

    # For zero_trust mode: client provides re-encrypted state in request
    # For background mode: load KMS envelope from DB
    encrypted_state = None
    kms_envelope = None

    if encryption_mode == "zero_trust" and request.encrypted_state:
        # Client decrypted and re-encrypted state to enclave transport key
        encrypted_state = request.encrypted_state.to_crypto()
    elif encryption_mode == "background" and existing_state and existing_state.encrypted_tarball:
        # Background mode: load KMS envelope from DB
        kms_envelope = json.loads(existing_state.encrypted_tarball.decode())
        # Convert hex strings back to bytes for enclave
        kms_envelope = {
            "encrypted_dek": bytes.fromhex(kms_envelope["encrypted_dek"]),
            "iv": bytes.fromhex(kms_envelope["iv"]),
            "ciphertext": bytes.fromhex(kms_envelope["ciphertext"]),
            "auth_tag": bytes.fromhex(kms_envelope["auth_tag"]),
        }

    # Convert API payload to crypto payload
    encrypted_message = request.encrypted_message.to_crypto()

    # Process through enclave
    handler = AgentHandler(enclave=enclave)
    agent_request = AgentMessageRequest(
        user_id=auth.user_id,
        agent_name=agent_name,
        encrypted_message=encrypted_message,
        encrypted_state=encrypted_state,
        user_public_key=user_public_key,
        model=request.model,
        encryption_mode=encryption_mode,
        kms_envelope=kms_envelope,
    )

    response = await handler.process_message(agent_request)

    if not response.success:
        return AgentMessageResponse(
            success=False,
            error=response.error,
        )

    # Store updated state based on encryption mode
    if encryption_mode == "background":
        # Background mode: store KMS envelope (already has hex strings from enclave)
        kms_envelope_serialized = json.dumps(response.kms_envelope).encode()

        if existing_state:
            await service.update_agent_state(
                user_id=auth.user_id,
                agent_name=agent_name,
                encrypted_tarball=kms_envelope_serialized,
            )
        else:
            await service.create_agent_state(
                user_id=auth.user_id,
                agent_name=agent_name,
                encrypted_tarball=kms_envelope_serialized,
                encryption_mode="background",
            )
    else:
        # Zero trust mode: store encrypted state
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
                encryption_mode="zero_trust",
            )

    await db.commit()

    # Convert crypto payload to API payload
    api_response = EncryptedPayloadSchema.from_crypto(response.encrypted_response)

    return AgentMessageResponse(
        success=True,
        encrypted_response=api_response,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _serialize_encrypted_payload(payload: EncryptedPayload) -> bytes:
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


def _deserialize_encrypted_payload(data: bytes) -> EncryptedPayload:
    """Deserialize encrypted payload from storage."""
    obj = json.loads(data.decode())
    return EncryptedPayload(
        ephemeral_public_key=bytes.fromhex(obj["ephemeral_public_key"]),
        iv=bytes.fromhex(obj["iv"]),
        ciphertext=bytes.fromhex(obj["ciphertext"]),
        auth_tag=bytes.fromhex(obj["auth_tag"]),
        hkdf_salt=bytes.fromhex(obj["hkdf_salt"]) if obj.get("hkdf_salt") else None,
    )
