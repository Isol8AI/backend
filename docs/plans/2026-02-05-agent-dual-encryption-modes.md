


# Agent Dual Encryption Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the critical agent state encryption bug and implement two encryption modes: Zero Trust (default, user-key encryption) and Background Mode (opt-in, KMS encryption).

**Architecture:** Agent state is currently encrypted to the enclave's ephemeral key (bug - lost on reboot). We fix this by defaulting to user-key encryption (zero trust) with an opt-in KMS mode for background features. Zero trust mode requires the client to decrypt/re-encrypt state; background mode allows the enclave to decrypt autonomously.

**Tech Stack:** Python/FastAPI, SQLAlchemy, AES-256-GCM, X25519 ECDH, AWS KMS (for background mode), TypeScript/React (frontend)

---

## Task 1: Add encryption_mode column to AgentState model

**Files:**
- Modify: `backend/models/agent_state.py`
- Create: `backend/alembic/versions/2026_02_05_add_encryption_mode.py`
- Test: `backend/tests/unit/models/test_agent_state.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/models/test_agent_state.py
# Add to existing test file

def test_agent_state_has_encryption_mode_column():
    """Test that AgentState has encryption_mode with default zero_trust."""
    from models.agent_state import AgentState, EncryptionMode

    state = AgentState(
        user_id="user_123",
        agent_name="test_agent",
        encrypted_tarball=b"encrypted_data",
    )

    assert state.encryption_mode == EncryptionMode.ZERO_TRUST
    assert hasattr(EncryptionMode, "ZERO_TRUST")
    assert hasattr(EncryptionMode, "BACKGROUND")


def test_agent_state_encryption_mode_can_be_background():
    """Test that encryption_mode can be set to background."""
    from models.agent_state import AgentState, EncryptionMode

    state = AgentState(
        user_id="user_123",
        agent_name="test_agent",
        encrypted_tarball=b"encrypted_data",
        encryption_mode=EncryptionMode.BACKGROUND,
    )

    assert state.encryption_mode == EncryptionMode.BACKGROUND


def test_agent_state_has_encrypted_dek_column():
    """Test that AgentState has encrypted_dek for background mode."""
    from models.agent_state import AgentState, EncryptionMode

    state = AgentState(
        user_id="user_123",
        agent_name="test_agent",
        encrypted_tarball=b"encrypted_data",
        encryption_mode=EncryptionMode.BACKGROUND,
        encrypted_dek=b"kms_encrypted_key",
    )

    assert state.encrypted_dek == b"kms_encrypted_key"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/models/test_agent_state.py::test_agent_state_has_encryption_mode_column -v`
Expected: FAIL with "cannot import name 'EncryptionMode'"

**Step 3: Write minimal implementation**

```python
# backend/models/agent_state.py
# Add imports at top
import enum
from sqlalchemy import Enum as SQLEnum

# Add enum class before AgentState class
class EncryptionMode(str, enum.Enum):
    """Agent state encryption mode."""
    ZERO_TRUST = "zero_trust"    # Encrypted to user's key (default)
    BACKGROUND = "background"    # Encrypted with KMS (opt-in)


# Add columns to AgentState class (after existing columns)
class AgentState(Base):
    # ... existing columns ...

    # Encryption mode for this agent
    encryption_mode = Column(
        SQLEnum(EncryptionMode),
        default=EncryptionMode.ZERO_TRUST,
        nullable=False,
        server_default="zero_trust",
    )

    # KMS-encrypted data encryption key (only for background mode)
    encrypted_dek = Column(LargeBinary, nullable=True)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/models/test_agent_state.py -v -k "encryption_mode or encrypted_dek"`
Expected: PASS (3 tests)

**Step 5: Create migration**

```python
# backend/alembic/versions/2026_02_05_add_encryption_mode.py
"""Add encryption_mode and encrypted_dek to agent_states

Revision ID: 2026_02_05_001
Revises: <previous_revision>
Create Date: 2026-02-05
"""
from alembic import op
import sqlalchemy as sa

revision = '2026_02_05_001'
down_revision = None  # Update with actual previous revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'agent_states',
        sa.Column(
            'encryption_mode',
            sa.Enum('zero_trust', 'background', name='encryptionmode'),
            nullable=False,
            server_default='zero_trust',
        )
    )
    op.add_column(
        'agent_states',
        sa.Column('encrypted_dek', sa.LargeBinary(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('agent_states', 'encrypted_dek')
    op.drop_column('agent_states', 'encryption_mode')
    op.execute("DROP TYPE IF EXISTS encryptionmode")
```

**Step 6: Commit**

```bash
git add backend/models/agent_state.py backend/tests/unit/models/test_agent_state.py backend/alembic/versions/2026_02_05_add_encryption_mode.py
git commit -m "feat(agent): add encryption_mode and encrypted_dek columns

- Add EncryptionMode enum (zero_trust, background)
- Default to zero_trust for all agents
- Add encrypted_dek for KMS data key storage (background mode only)"
```

---

## Task 2: Update agent schemas for encryption mode

**Files:**
- Modify: `backend/schemas/agent.py`
- Test: `backend/tests/unit/schemas/test_agent_schemas.py` (create if needed)

**Step 1: Write the failing test**

```python
# backend/tests/unit/schemas/test_agent_schemas.py
import pytest
from pydantic import ValidationError


class TestAgentSchemas:
    def test_create_agent_request_has_encryption_mode(self):
        """Test CreateAgentRequest accepts encryption_mode."""
        from schemas.agent import CreateAgentRequest

        request = CreateAgentRequest(
            agent_name="test_agent",
            encryption_mode="zero_trust",
        )
        assert request.encryption_mode == "zero_trust"

    def test_create_agent_request_defaults_to_zero_trust(self):
        """Test encryption_mode defaults to zero_trust."""
        from schemas.agent import CreateAgentRequest

        request = CreateAgentRequest(agent_name="test_agent")
        assert request.encryption_mode == "zero_trust"

    def test_create_agent_request_rejects_invalid_mode(self):
        """Test invalid encryption_mode is rejected."""
        from schemas.agent import CreateAgentRequest

        with pytest.raises(ValidationError):
            CreateAgentRequest(
                agent_name="test_agent",
                encryption_mode="invalid_mode",
            )

    def test_agent_response_includes_encryption_mode(self):
        """Test AgentResponse includes encryption_mode."""
        from schemas.agent import AgentResponse

        response = AgentResponse(
            agent_name="test_agent",
            encryption_mode="background",
            created_at="2026-02-05T00:00:00Z",
            updated_at="2026-02-05T00:00:00Z",
        )
        assert response.encryption_mode == "background"

    def test_send_message_request_accepts_encrypted_state(self):
        """Test SendAgentMessageRequest accepts client-provided encrypted_state."""
        from schemas.agent import SendAgentMessageRequest
        from schemas.encryption import EncryptedPayload

        payload = EncryptedPayload(
            ephemeral_public_key="a" * 64,
            iv="b" * 32,
            ciphertext="c" * 64,
            auth_tag="d" * 32,
            hkdf_salt="e" * 64,
        )

        request = SendAgentMessageRequest(
            encrypted_message=payload,
            encrypted_state=payload,  # Client provides decrypted/re-encrypted state
        )
        assert request.encrypted_state is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/schemas/test_agent_schemas.py -v`
Expected: FAIL (file may not exist or schemas missing fields)

**Step 3: Write minimal implementation**

```python
# backend/schemas/agent.py
# Update/add these schema classes

from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from schemas.encryption import EncryptedPayload


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""
    agent_name: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    encryption_mode: Literal["zero_trust", "background"] = "zero_trust"
    soul_md: Optional[str] = None
    model: Optional[str] = None


class AgentResponse(BaseModel):
    """Response containing agent metadata."""
    agent_name: str
    encryption_mode: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SendAgentMessageRequest(BaseModel):
    """Request to send a message to an agent."""
    encrypted_message: EncryptedPayload
    # For zero_trust mode: client decrypts state with their key,
    # re-encrypts to enclave transport key, provides it here
    encrypted_state: Optional[EncryptedPayload] = None


class AgentMessageResponse(BaseModel):
    """Response from agent message."""
    encrypted_response: EncryptedPayload
    # For zero_trust mode: state encrypted to user's key
    # For background mode: state encrypted with KMS (stored server-side)
    encrypted_state: Optional[EncryptedPayload] = None


class UpdateAgentEncryptionRequest(BaseModel):
    """Request to change agent's encryption mode."""
    encryption_mode: Literal["zero_trust", "background"]
    # When switching TO zero_trust, client must provide state encrypted to their key
    # When switching TO background, client provides state for KMS encryption
    encrypted_state: Optional[EncryptedPayload] = None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/schemas/test_agent_schemas.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add backend/schemas/agent.py backend/tests/unit/schemas/test_agent_schemas.py
git commit -m "feat(agent): add encryption_mode to agent schemas

- CreateAgentRequest accepts encryption_mode (default: zero_trust)
- AgentResponse includes encryption_mode
- SendAgentMessageRequest accepts client-provided encrypted_state
- Add UpdateAgentEncryptionRequest for mode switching"
```

---

## Task 3: Fix mock_enclave to encrypt state to user's key (Zero Trust)

**Files:**
- Modify: `backend/core/enclave/mock_enclave.py:1142-1227`
- Test: `backend/tests/unit/enclave/test_mock_enclave.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/enclave/test_mock_enclave.py
# Add to existing test file in the agent tests section

class TestAgentStateEncryption:
    """Test that agent state is encrypted to user's key, not enclave's key."""

    @pytest.fixture
    def user_keypair(self):
        from core.crypto.primitives import generate_x25519_keypair
        return generate_x25519_keypair()

    @pytest.fixture
    def enclave(self):
        from core.enclave.mock_enclave import MockEnclave
        return MockEnclave()

    @pytest.mark.asyncio
    async def test_agent_state_encrypted_to_user_key_not_enclave_key(
        self, enclave, user_keypair
    ):
        """Critical test: agent state must be encrypted to USER's key."""
        from core.crypto.primitives import (
            encrypt_to_public_key,
            decrypt_with_private_key,
        )
        from core.enclave import EncryptionContext

        # Create a simple message
        message = "Hello agent"
        encrypted_message = encrypt_to_public_key(
            enclave.public_key,
            message.encode(),
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Run agent (no existing state)
        response = await enclave.run_agent(
            encrypted_message=encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            agent_name="test_agent",
            model="anthropic.claude-sonnet-4-20250514",
            encryption_mode="zero_trust",
        )

        assert response.success
        assert response.encrypted_state is not None

        # THE CRITICAL ASSERTION:
        # User should be able to decrypt the state with THEIR private key
        state_bytes = decrypt_with_private_key(
            user_keypair.private_key,  # User's key, NOT enclave's
            response.encrypted_state,
            EncryptionContext.AGENT_STATE_STORAGE.value,
        )

        # State should be a valid tarball
        assert len(state_bytes) > 0
        # Verify it's gzip by checking magic bytes
        assert state_bytes[:2] == b'\x1f\x8b'  # gzip magic number

    @pytest.mark.asyncio
    async def test_enclave_cannot_decrypt_zero_trust_state(
        self, enclave, user_keypair
    ):
        """Enclave should NOT be able to decrypt zero_trust state."""
        from core.crypto.primitives import encrypt_to_public_key, decrypt_with_private_key
        from core.enclave import EncryptionContext
        import pytest

        message = "Hello agent"
        encrypted_message = encrypt_to_public_key(
            enclave.public_key,
            message.encode(),
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        response = await enclave.run_agent(
            encrypted_message=encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            agent_name="test_agent",
            model="anthropic.claude-sonnet-4-20250514",
            encryption_mode="zero_trust",
        )

        # Enclave trying to decrypt should fail
        with pytest.raises(Exception):  # Decryption will fail
            decrypt_with_private_key(
                enclave._keypair.private_key,  # Enclave's key
                response.encrypted_state,
                EncryptionContext.AGENT_STATE_STORAGE.value,
            )
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/enclave/test_mock_enclave.py::TestAgentStateEncryption -v`
Expected: FAIL - currently encrypts to enclave's key, not user's key

**Step 3: Write minimal implementation**

```python
# backend/core/enclave/mock_enclave.py
# Modify the run_agent method (around line 1142)

async def run_agent(
    self,
    encrypted_message: EncryptedPayload,
    encrypted_state: Optional[EncryptedPayload],
    user_public_key: bytes,
    agent_name: str,
    model: str,
    encryption_mode: str = "zero_trust",  # Add this parameter
) -> AgentRunResponse:
    """Run an OpenClaw agent with an encrypted message.

    Args:
        encrypted_message: Message encrypted to enclave's transport key
        encrypted_state: For zero_trust: state encrypted to enclave transport key
                        For background: None (server passes KMS state separately)
        user_public_key: User's public key for encrypting response and state
        agent_name: Name of the agent to run
        model: LLM model to use
        encryption_mode: "zero_trust" (default) or "background"
    """
    tmpfs_path = None

    try:
        # Create tmpfs directory for agent state
        tmpfs_path = Path(tempfile.mkdtemp(prefix=f"agent_{agent_name}_"))
        logger.info(f"[MockEnclave] Created tmpfs: {tmpfs_path}")

        # Decrypt existing state if provided
        if encrypted_state:
            # In zero_trust mode, client decrypted with their key and
            # re-encrypted to enclave transport key
            state_bytes = decrypt_with_private_key(
                self._keypair.private_key,
                encrypted_state,
                EncryptionContext.CLIENT_TO_ENCLAVE.value,  # Transport context
            )
            self._unpack_tarball(state_bytes, tmpfs_path)
            logger.info(f"[MockEnclave] Extracted existing state ({len(state_bytes)} bytes)")
        else:
            self._create_fresh_agent(tmpfs_path, agent_name, model)
            logger.info("[MockEnclave] Created fresh agent directory")

        # Decrypt user message
        message_bytes = decrypt_with_private_key(
            self._keypair.private_key,
            encrypted_message,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )
        message = message_bytes.decode("utf-8")
        logger.info(f"[MockEnclave] Decrypted message: {message[:50]}...")

        # Run OpenClaw CLI
        result = self._run_openclaw(tmpfs_path, message, agent_name, model)

        if not result["success"]:
            return AgentRunResponse(
                success=False,
                error=result["error"],
            )

        logger.info(f"[MockEnclave] Agent response: {result['response'][:50]}...")

        # Pack updated state
        tarball_bytes = self._pack_directory(tmpfs_path)
        logger.info(f"[MockEnclave] Packed state: {len(tarball_bytes)} bytes")

        # FIXED: Encrypt state based on mode
        if encryption_mode == "zero_trust":
            # Encrypt to USER's public key (only they can decrypt)
            encrypted_state_out = encrypt_to_public_key(
                user_public_key,  # USER's key, not enclave's!
                tarball_bytes,
                EncryptionContext.AGENT_STATE_STORAGE.value,
            )
        else:
            # Background mode: would use KMS here (Task 5)
            # For now, still encrypt to user's key
            encrypted_state_out = encrypt_to_public_key(
                user_public_key,
                tarball_bytes,
                EncryptionContext.AGENT_STATE_STORAGE.value,
            )

        # Encrypt response for transport (to user's key)
        encrypted_response = encrypt_to_public_key(
            user_public_key,
            result["response"].encode("utf-8"),
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )

        return AgentRunResponse(
            success=True,
            encrypted_response=encrypted_response,
            encrypted_state=encrypted_state_out,
        )

    except Exception as e:
        logger.exception(f"[MockEnclave] run_agent error: {e}")
        return AgentRunResponse(
            success=False,
            error=str(e),
        )

    finally:
        # Always cleanup tmpfs
        if tmpfs_path and tmpfs_path.exists():
            shutil.rmtree(tmpfs_path, ignore_errors=True)
            logger.debug(f"[MockEnclave] Cleaned up tmpfs: {tmpfs_path}")
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/enclave/test_mock_enclave.py::TestAgentStateEncryption -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add backend/core/enclave/mock_enclave.py backend/tests/unit/enclave/test_mock_enclave.py
git commit -m "fix(enclave): encrypt agent state to user's key, not enclave's

BREAKING: Agent state is now encrypted to user's public key for zero_trust mode.
This fixes the critical bug where state was encrypted to enclave's ephemeral key
and lost on reboot.

- Add encryption_mode parameter to run_agent()
- zero_trust: encrypt to user_public_key (only user can decrypt)
- Change encrypted_state context to CLIENT_TO_ENCLAVE (transport)"
```

---

## Task 4: Update agent router to handle zero_trust flow

**Files:**
- Modify: `backend/routers/agents.py`
- Test: `backend/tests/unit/routers/test_agents.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/routers/test_agents.py
# Add to existing test file

class TestAgentZeroTrustFlow:
    """Test zero_trust encryption flow in agent router."""

    @pytest.mark.asyncio
    async def test_create_agent_with_encryption_mode(self, client, mock_db):
        """Test creating agent with explicit encryption_mode."""
        response = await client.post(
            "/api/v1/agents",
            json={
                "agent_name": "test_agent",
                "encryption_mode": "zero_trust",
            },
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["encryption_mode"] == "zero_trust"

    @pytest.mark.asyncio
    async def test_create_agent_defaults_to_zero_trust(self, client, mock_db):
        """Test agent defaults to zero_trust mode."""
        response = await client.post(
            "/api/v1/agents",
            json={"agent_name": "test_agent"},
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["encryption_mode"] == "zero_trust"

    @pytest.mark.asyncio
    async def test_message_response_includes_encrypted_state(self, client, mock_db, mock_enclave):
        """Test message response includes encrypted_state for client storage."""
        # First create agent
        await client.post(
            "/api/v1/agents",
            json={"agent_name": "test_agent"},
            headers={"Authorization": "Bearer test_token"},
        )

        # Send message
        response = await client.post(
            "/api/v1/agents/test_agent/message",
            json={
                "encrypted_message": {
                    "ephemeral_public_key": "a" * 64,
                    "iv": "b" * 32,
                    "ciphertext": "c" * 64,
                    "auth_tag": "d" * 32,
                    "hkdf_salt": "e" * 64,
                },
            },
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200
        data = response.json()
        # Zero trust mode: response includes encrypted_state for client
        assert "encrypted_state" in data
        assert data["encrypted_state"] is not None

    @pytest.mark.asyncio
    async def test_get_agent_state_endpoint(self, client, mock_db):
        """Test endpoint to fetch encrypted agent state."""
        # Create agent with some state
        await client.post(
            "/api/v1/agents",
            json={"agent_name": "test_agent"},
            headers={"Authorization": "Bearer test_token"},
        )

        # Fetch state
        response = await client.get(
            "/api/v1/agents/test_agent/state",
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "encrypted_state" in data
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/routers/test_agents.py::TestAgentZeroTrustFlow -v`
Expected: FAIL - endpoints don't exist or return wrong data

**Step 3: Write minimal implementation**

```python
# backend/routers/agents.py
# Add/modify these endpoints

from models.agent_state import EncryptionMode


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(
    request: CreateAgentRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new agent with specified encryption mode."""
    service = AgentService(db)

    # Check if agent already exists
    existing = await service.get_agent_state(auth.user_id, request.agent_name)
    if existing:
        raise HTTPException(status_code=409, detail="Agent already exists")

    # Map string to enum
    mode = EncryptionMode(request.encryption_mode)

    # Create agent state (initially empty - will be populated on first message)
    agent = await service.create_agent_state(
        user_id=auth.user_id,
        agent_name=request.agent_name,
        encrypted_tarball=b"",  # Empty until first message
        encryption_mode=mode,
    )

    return AgentResponse(
        agent_name=agent.agent_name,
        encryption_mode=agent.encryption_mode.value,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.get("/{agent_name}/state", response_model=AgentStateResponse)
async def get_agent_state(
    agent_name: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get encrypted agent state for client-side decryption (zero_trust mode)."""
    service = AgentService(db)

    agent = await service.get_agent_state(auth.user_id, agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent.encryption_mode != EncryptionMode.ZERO_TRUST:
        raise HTTPException(
            status_code=400,
            detail="State fetch only available for zero_trust mode"
        )

    if not agent.encrypted_tarball:
        return AgentStateResponse(encrypted_state=None)

    # Deserialize and return
    encrypted_state = _deserialize_encrypted_payload(agent.encrypted_tarball)
    return AgentStateResponse(
        encrypted_state=EncryptedPayloadSchema.from_crypto_payload(encrypted_state)
    )


@router.post("/{agent_name}/message", response_model=AgentMessageResponse)
async def send_agent_message(
    agent_name: str,
    request: SendAgentMessageRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send message to agent. For zero_trust, client provides encrypted_state."""
    service = AgentService(db)
    handler = get_agent_handler()

    # Get agent metadata
    agent = await service.get_agent_state(auth.user_id, agent_name)

    # Get user's public key for response encryption
    user_keys = await get_user_keys(db, auth.user_id)
    if not user_keys:
        raise HTTPException(status_code=400, detail="User encryption not set up")

    user_public_key = bytes.fromhex(user_keys.public_key)

    # Convert request to crypto types
    encrypted_message = request.encrypted_message.to_crypto_payload()

    # Handle state based on encryption mode
    if agent and agent.encryption_mode == EncryptionMode.ZERO_TRUST:
        # Client must provide state (they decrypted and re-encrypted it)
        if request.encrypted_state:
            encrypted_state = request.encrypted_state.to_crypto_payload()
        else:
            encrypted_state = None
    else:
        # Background mode or new agent: server handles state
        encrypted_state = None
        if agent and agent.encrypted_tarball:
            encrypted_state = _deserialize_encrypted_payload(agent.encrypted_tarball)

    # Process through enclave
    encryption_mode = agent.encryption_mode.value if agent else "zero_trust"

    response = await handler.run_agent(
        encrypted_message=encrypted_message,
        encrypted_state=encrypted_state,
        user_public_key=user_public_key,
        agent_name=agent_name,
        model="anthropic.claude-sonnet-4-20250514",
        encryption_mode=encryption_mode,
    )

    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)

    # Store updated state
    encrypted_state_bytes = _serialize_encrypted_payload(response.encrypted_state)

    if agent:
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
            encryption_mode=EncryptionMode.ZERO_TRUST,
        )

    # Return response with encrypted_state for zero_trust
    return AgentMessageResponse(
        encrypted_response=EncryptedPayloadSchema.from_crypto_payload(
            response.encrypted_response
        ),
        encrypted_state=EncryptedPayloadSchema.from_crypto_payload(
            response.encrypted_state
        ) if encryption_mode == "zero_trust" else None,
    )


# Add to schemas/agent.py
class AgentStateResponse(BaseModel):
    """Response containing encrypted agent state for client decryption."""
    encrypted_state: Optional[EncryptedPayload] = None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/routers/test_agents.py::TestAgentZeroTrustFlow -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add backend/routers/agents.py backend/schemas/agent.py backend/tests/unit/routers/test_agents.py
git commit -m "feat(agent): implement zero_trust flow in agent router

- Add GET /agents/{name}/state endpoint for fetching encrypted state
- Update POST /agents to accept encryption_mode
- Update POST /agents/{name}/message to return encrypted_state
- Client can now decrypt state with their key and re-encrypt for enclave"
```

---

## Task 5: Implement KMS encryption for background mode

**Files:**
- Modify: `backend/core/enclave/mock_enclave.py`
- Create: `backend/core/enclave/kms_encryption.py`
- Test: `backend/tests/unit/enclave/test_kms_encryption.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/enclave/test_kms_encryption.py
import pytest
from unittest.mock import Mock, patch


class TestKMSEncryption:
    """Test KMS envelope encryption for background mode."""

    def test_generate_data_key_returns_plaintext_and_encrypted(self):
        """Test KMS generates both plaintext and encrypted DEK."""
        from core.enclave.kms_encryption import KMSEncryption

        with patch('boto3.client') as mock_boto:
            mock_kms = Mock()
            mock_kms.generate_data_key.return_value = {
                'Plaintext': b'a' * 32,
                'CiphertextBlob': b'encrypted_key_data',
            }
            mock_boto.return_value = mock_kms

            kms = KMSEncryption(key_id="alias/test-key")
            plaintext_key, encrypted_key = kms.generate_data_key()

            assert len(plaintext_key) == 32
            assert encrypted_key == b'encrypted_key_data'

    def test_decrypt_data_key_returns_plaintext(self):
        """Test KMS decrypts encrypted DEK."""
        from core.enclave.kms_encryption import KMSEncryption

        with patch('boto3.client') as mock_boto:
            mock_kms = Mock()
            mock_kms.decrypt.return_value = {
                'Plaintext': b'a' * 32,
            }
            mock_boto.return_value = mock_kms

            kms = KMSEncryption(key_id="alias/test-key")
            plaintext_key = kms.decrypt_data_key(b'encrypted_key_data')

            assert len(plaintext_key) == 32

    def test_encrypt_with_dek(self):
        """Test AES-GCM encryption with DEK."""
        from core.enclave.kms_encryption import KMSEncryption

        kms = KMSEncryption(key_id="alias/test-key")

        dek = b'a' * 32  # 256-bit key
        plaintext = b"secret agent state data"

        iv, ciphertext, tag = kms.encrypt_with_dek(dek, plaintext)

        assert len(iv) == 12  # AES-GCM standard IV
        assert len(tag) == 16  # AES-GCM standard tag
        assert ciphertext != plaintext

    def test_decrypt_with_dek(self):
        """Test AES-GCM decryption with DEK."""
        from core.enclave.kms_encryption import KMSEncryption

        kms = KMSEncryption(key_id="alias/test-key")

        dek = b'a' * 32
        plaintext = b"secret agent state data"

        iv, ciphertext, tag = kms.encrypt_with_dek(dek, plaintext)
        recovered = kms.decrypt_with_dek(dek, iv, ciphertext, tag)

        assert recovered == plaintext

    def test_envelope_encrypt_and_decrypt(self):
        """Test full envelope encryption flow."""
        from core.enclave.kms_encryption import KMSEncryption

        with patch('boto3.client') as mock_boto:
            mock_kms = Mock()
            # Store the "encrypted" key for later decryption
            stored_key = b'a' * 32
            mock_kms.generate_data_key.return_value = {
                'Plaintext': stored_key,
                'CiphertextBlob': b'encrypted_key_blob',
            }
            mock_kms.decrypt.return_value = {
                'Plaintext': stored_key,
            }
            mock_boto.return_value = mock_kms

            kms = KMSEncryption(key_id="alias/test-key")

            # Encrypt
            plaintext = b"agent tarball data here"
            envelope = kms.envelope_encrypt(plaintext)

            assert 'encrypted_dek' in envelope
            assert 'iv' in envelope
            assert 'ciphertext' in envelope
            assert 'tag' in envelope

            # Decrypt
            recovered = kms.envelope_decrypt(envelope)
            assert recovered == plaintext
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/enclave/test_kms_encryption.py -v`
Expected: FAIL with "No module named 'core.enclave.kms_encryption'"

**Step 3: Write minimal implementation**

```python
# backend/core/enclave/kms_encryption.py
"""KMS envelope encryption for background mode agent state."""

import os
from typing import Tuple, Dict
import boto3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class KMSEncryption:
    """KMS-based envelope encryption for agent state.

    Uses AWS KMS to generate/decrypt data encryption keys (DEK),
    then uses AES-256-GCM for actual data encryption.

    In production Nitro Enclave, KMS calls include attestation document
    which KMS validates against PCR values in key policy.
    """

    def __init__(self, key_id: str, region: str = "us-east-1"):
        """Initialize KMS encryption.

        Args:
            key_id: KMS key ID or alias (e.g., "alias/isol8-dev-enclave")
            region: AWS region
        """
        self.key_id = key_id
        self.region = region
        self._client = None

    @property
    def client(self):
        """Lazy-load KMS client."""
        if self._client is None:
            self._client = boto3.client("kms", region_name=self.region)
        return self._client

    def generate_data_key(self) -> Tuple[bytes, bytes]:
        """Generate a data encryption key using KMS.

        Returns:
            Tuple of (plaintext_key, encrypted_key)
            - plaintext_key: 32-byte key for AES-256
            - encrypted_key: KMS-encrypted blob for storage
        """
        response = self.client.generate_data_key(
            KeyId=self.key_id,
            KeySpec="AES_256",
        )
        return response["Plaintext"], response["CiphertextBlob"]

    def decrypt_data_key(self, encrypted_key: bytes) -> bytes:
        """Decrypt a data encryption key using KMS.

        In production, this includes attestation document for PCR validation.

        Args:
            encrypted_key: KMS-encrypted key blob

        Returns:
            32-byte plaintext key
        """
        response = self.client.decrypt(
            KeyId=self.key_id,
            CiphertextBlob=encrypted_key,
            # In production Nitro Enclave:
            # Recipient={
            #     'KeyEncryptionAlgorithm': 'RSAES_OAEP_SHA_256',
            #     'AttestationDocument': attestation_doc,
            # }
        )
        return response["Plaintext"]

    def encrypt_with_dek(self, dek: bytes, plaintext: bytes) -> Tuple[bytes, bytes, bytes]:
        """Encrypt data with AES-256-GCM using the DEK.

        Args:
            dek: 32-byte data encryption key
            plaintext: Data to encrypt

        Returns:
            Tuple of (iv, ciphertext, tag)
        """
        iv = os.urandom(12)  # 96-bit IV for AES-GCM
        aesgcm = AESGCM(dek)
        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, None)
        # AES-GCM appends 16-byte tag to ciphertext
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]
        return iv, ciphertext, tag

    def decrypt_with_dek(self, dek: bytes, iv: bytes, ciphertext: bytes, tag: bytes) -> bytes:
        """Decrypt data with AES-256-GCM using the DEK.

        Args:
            dek: 32-byte data encryption key
            iv: 12-byte initialization vector
            ciphertext: Encrypted data
            tag: 16-byte authentication tag

        Returns:
            Decrypted plaintext
        """
        aesgcm = AESGCM(dek)
        return aesgcm.decrypt(iv, ciphertext + tag, None)

    def envelope_encrypt(self, plaintext: bytes) -> Dict[str, bytes]:
        """Envelope encrypt data using KMS.

        Args:
            plaintext: Data to encrypt

        Returns:
            Dict with encrypted_dek, iv, ciphertext, tag (all bytes)
        """
        plaintext_dek, encrypted_dek = self.generate_data_key()
        iv, ciphertext, tag = self.encrypt_with_dek(plaintext_dek, plaintext)

        # Clear plaintext key from memory
        del plaintext_dek

        return {
            "encrypted_dek": encrypted_dek,
            "iv": iv,
            "ciphertext": ciphertext,
            "tag": tag,
        }

    def envelope_decrypt(self, envelope: Dict[str, bytes]) -> bytes:
        """Envelope decrypt data using KMS.

        Args:
            envelope: Dict with encrypted_dek, iv, ciphertext, tag

        Returns:
            Decrypted plaintext
        """
        plaintext_dek = self.decrypt_data_key(envelope["encrypted_dek"])
        plaintext = self.decrypt_with_dek(
            plaintext_dek,
            envelope["iv"],
            envelope["ciphertext"],
            envelope["tag"],
        )

        # Clear plaintext key from memory
        del plaintext_dek

        return plaintext
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/enclave/test_kms_encryption.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add backend/core/enclave/kms_encryption.py backend/tests/unit/enclave/test_kms_encryption.py
git commit -m "feat(enclave): add KMS envelope encryption for background mode

- Add KMSEncryption class with generate/decrypt data key
- Implement AES-256-GCM encryption/decryption with DEK
- Add envelope_encrypt/envelope_decrypt convenience methods
- Placeholder for attestation document in production"
```

---

## Task 6: Integrate KMS into mock_enclave for background mode

**Files:**
- Modify: `backend/core/enclave/mock_enclave.py`
- Modify: `backend/core/enclave/__init__.py`
- Test: `backend/tests/unit/enclave/test_mock_enclave.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/enclave/test_mock_enclave.py
# Add to TestAgentStateEncryption class

@pytest.mark.asyncio
async def test_background_mode_uses_kms_encryption(self, enclave, user_keypair):
    """Test background mode uses KMS envelope encryption."""
    from core.crypto.primitives import encrypt_to_public_key
    from core.enclave import EncryptionContext
    from unittest.mock import patch, Mock

    message = "Hello agent"
    encrypted_message = encrypt_to_public_key(
        enclave.public_key,
        message.encode(),
        EncryptionContext.CLIENT_TO_ENCLAVE.value,
    )

    with patch('core.enclave.mock_enclave.KMSEncryption') as mock_kms_class:
        mock_kms = Mock()
        mock_kms.envelope_encrypt.return_value = {
            "encrypted_dek": b"encrypted_key",
            "iv": b"123456789012",
            "ciphertext": b"encrypted_tarball",
            "tag": b"1234567890123456",
        }
        mock_kms_class.return_value = mock_kms

        response = await enclave.run_agent(
            encrypted_message=encrypted_message,
            encrypted_state=None,
            user_public_key=user_keypair.public_key,
            agent_name="test_agent",
            model="anthropic.claude-sonnet-4-20250514",
            encryption_mode="background",  # Background mode
        )

        assert response.success
        # Background mode should have encrypted_dek
        assert response.encrypted_dek is not None
        # KMS envelope_encrypt should have been called
        mock_kms.envelope_encrypt.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/enclave/test_mock_enclave.py::TestAgentStateEncryption::test_background_mode_uses_kms_encryption -v`
Expected: FAIL - background mode not implemented

**Step 3: Write minimal implementation**

```python
# backend/core/enclave/mock_enclave.py
# Add import at top
from core.enclave.kms_encryption import KMSEncryption

# Update AgentRunResponse in __init__.py
# backend/core/enclave/__init__.py
@dataclass
class AgentRunResponse:
    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    encrypted_state: Optional[EncryptedPayload] = None
    encrypted_dek: Optional[bytes] = None  # For background mode
    kms_envelope: Optional[Dict[str, bytes]] = None  # Full KMS envelope
    error: Optional[str] = None

# Update run_agent in mock_enclave.py
class MockEnclave:
    def __init__(self):
        # ... existing init ...
        self._kms = None
        self._kms_key_id = os.environ.get(
            "KMS_KEY_ID",
            "alias/isol8-dev-enclave"
        )

    @property
    def kms(self) -> KMSEncryption:
        """Lazy-load KMS encryption."""
        if self._kms is None:
            self._kms = KMSEncryption(key_id=self._kms_key_id)
        return self._kms

    async def run_agent(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        user_public_key: bytes,
        agent_name: str,
        model: str,
        encryption_mode: str = "zero_trust",
        kms_envelope: Optional[Dict[str, bytes]] = None,  # For background mode
    ) -> AgentRunResponse:
        """Run an OpenClaw agent with an encrypted message."""
        tmpfs_path = None

        try:
            tmpfs_path = Path(tempfile.mkdtemp(prefix=f"agent_{agent_name}_"))
            logger.info(f"[MockEnclave] Created tmpfs: {tmpfs_path}")

            # Decrypt existing state based on mode
            if encryption_mode == "zero_trust" and encrypted_state:
                # Client provided state encrypted to enclave transport key
                state_bytes = decrypt_with_private_key(
                    self._keypair.private_key,
                    encrypted_state,
                    EncryptionContext.CLIENT_TO_ENCLAVE.value,
                )
                self._unpack_tarball(state_bytes, tmpfs_path)
                logger.info(f"[MockEnclave] Extracted zero_trust state")
            elif encryption_mode == "background" and kms_envelope:
                # Server provided KMS-encrypted state
                state_bytes = self.kms.envelope_decrypt(kms_envelope)
                self._unpack_tarball(state_bytes, tmpfs_path)
                logger.info(f"[MockEnclave] Extracted background state via KMS")
            else:
                self._create_fresh_agent(tmpfs_path, agent_name, model)
                logger.info("[MockEnclave] Created fresh agent directory")

            # Decrypt user message
            message_bytes = decrypt_with_private_key(
                self._keypair.private_key,
                encrypted_message,
                EncryptionContext.CLIENT_TO_ENCLAVE.value,
            )
            message = message_bytes.decode("utf-8")

            # Run OpenClaw
            result = self._run_openclaw(tmpfs_path, message, agent_name, model)

            if not result["success"]:
                return AgentRunResponse(success=False, error=result["error"])

            # Pack updated state
            tarball_bytes = self._pack_directory(tmpfs_path)

            # Encrypt state based on mode
            if encryption_mode == "zero_trust":
                # Encrypt to user's public key
                encrypted_state_out = encrypt_to_public_key(
                    user_public_key,
                    tarball_bytes,
                    EncryptionContext.AGENT_STATE_STORAGE.value,
                )
                encrypted_dek = None
                kms_envelope_out = None
            else:  # background
                # Use KMS envelope encryption
                kms_envelope_out = self.kms.envelope_encrypt(tarball_bytes)
                encrypted_state_out = None
                encrypted_dek = kms_envelope_out["encrypted_dek"]

            # Encrypt response for transport
            encrypted_response = encrypt_to_public_key(
                user_public_key,
                result["response"].encode("utf-8"),
                EncryptionContext.ENCLAVE_TO_CLIENT.value,
            )

            return AgentRunResponse(
                success=True,
                encrypted_response=encrypted_response,
                encrypted_state=encrypted_state_out,
                encrypted_dek=encrypted_dek,
                kms_envelope=kms_envelope_out,
            )

        except Exception as e:
            logger.exception(f"[MockEnclave] run_agent error: {e}")
            return AgentRunResponse(success=False, error=str(e))

        finally:
            if tmpfs_path and tmpfs_path.exists():
                shutil.rmtree(tmpfs_path, ignore_errors=True)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/enclave/test_mock_enclave.py::TestAgentStateEncryption -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add backend/core/enclave/mock_enclave.py backend/core/enclave/__init__.py backend/tests/unit/enclave/test_mock_enclave.py
git commit -m "feat(enclave): integrate KMS encryption for background mode

- Add kms property to MockEnclave for lazy KMS client
- Update run_agent to handle both zero_trust and background modes
- zero_trust: encrypt state to user's key
- background: use KMS envelope encryption
- Add kms_envelope to AgentRunResponse"
```

---

## Task 7: Update router to handle background mode with KMS

**Files:**
- Modify: `backend/routers/agents.py`
- Test: `backend/tests/unit/routers/test_agents.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/routers/test_agents.py
# Add to test file

class TestAgentBackgroundMode:
    """Test background mode KMS flow in agent router."""

    @pytest.mark.asyncio
    async def test_create_agent_background_mode(self, client, mock_db):
        """Test creating agent with background encryption mode."""
        response = await client.post(
            "/api/v1/agents",
            json={
                "agent_name": "test_agent",
                "encryption_mode": "background",
            },
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["encryption_mode"] == "background"

    @pytest.mark.asyncio
    async def test_background_mode_stores_kms_envelope(
        self, client, mock_db, mock_enclave
    ):
        """Test background mode stores KMS envelope in database."""
        from models.agent_state import AgentState, EncryptionMode

        # Create background mode agent
        await client.post(
            "/api/v1/agents",
            json={
                "agent_name": "test_agent",
                "encryption_mode": "background",
            },
            headers={"Authorization": "Bearer test_token"},
        )

        # Send message
        response = await client.post(
            "/api/v1/agents/test_agent/message",
            json={
                "encrypted_message": {
                    "ephemeral_public_key": "a" * 64,
                    "iv": "b" * 32,
                    "ciphertext": "c" * 64,
                    "auth_tag": "d" * 32,
                    "hkdf_salt": "e" * 64,
                },
            },
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200

        # Verify database has encrypted_dek
        agent = await mock_db.get(AgentState, ...)
        assert agent.encrypted_dek is not None
        assert agent.encryption_mode == EncryptionMode.BACKGROUND

    @pytest.mark.asyncio
    async def test_background_mode_no_state_in_response(self, client, mock_db, mock_enclave):
        """Test background mode doesn't return encrypted_state to client."""
        await client.post(
            "/api/v1/agents",
            json={
                "agent_name": "test_agent",
                "encryption_mode": "background",
            },
            headers={"Authorization": "Bearer test_token"},
        )

        response = await client.post(
            "/api/v1/agents/test_agent/message",
            json={
                "encrypted_message": {
                    "ephemeral_public_key": "a" * 64,
                    "iv": "b" * 32,
                    "ciphertext": "c" * 64,
                    "auth_tag": "d" * 32,
                    "hkdf_salt": "e" * 64,
                },
            },
            headers={"Authorization": "Bearer test_token"},
        )

        data = response.json()
        # Background mode: state stays server-side
        assert data.get("encrypted_state") is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/routers/test_agents.py::TestAgentBackgroundMode -v`
Expected: FAIL - KMS envelope not stored

**Step 3: Write minimal implementation**

```python
# backend/routers/agents.py
# Update send_agent_message to handle background mode

import json

def _serialize_kms_envelope(envelope: Dict[str, bytes]) -> bytes:
    """Serialize KMS envelope to bytes for storage."""
    return json.dumps({
        "encrypted_dek": envelope["encrypted_dek"].hex(),
        "iv": envelope["iv"].hex(),
        "ciphertext": envelope["ciphertext"].hex(),
        "tag": envelope["tag"].hex(),
    }).encode()


def _deserialize_kms_envelope(data: bytes) -> Dict[str, bytes]:
    """Deserialize KMS envelope from storage."""
    obj = json.loads(data.decode())
    return {
        "encrypted_dek": bytes.fromhex(obj["encrypted_dek"]),
        "iv": bytes.fromhex(obj["iv"]),
        "ciphertext": bytes.fromhex(obj["ciphertext"]),
        "tag": bytes.fromhex(obj["tag"]),
    }


@router.post("/{agent_name}/message", response_model=AgentMessageResponse)
async def send_agent_message(
    agent_name: str,
    request: SendAgentMessageRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send message to agent."""
    service = AgentService(db)
    handler = get_agent_handler()

    agent = await service.get_agent_state(auth.user_id, agent_name)

    user_keys = await get_user_keys(db, auth.user_id)
    if not user_keys:
        raise HTTPException(status_code=400, detail="User encryption not set up")

    user_public_key = bytes.fromhex(user_keys.public_key)
    encrypted_message = request.encrypted_message.to_crypto_payload()

    # Determine encryption mode
    encryption_mode = agent.encryption_mode.value if agent else "zero_trust"

    # Prepare state based on mode
    encrypted_state = None
    kms_envelope = None

    if encryption_mode == "zero_trust":
        if request.encrypted_state:
            encrypted_state = request.encrypted_state.to_crypto_payload()
    elif encryption_mode == "background":
        if agent and agent.encrypted_tarball:
            kms_envelope = _deserialize_kms_envelope(agent.encrypted_tarball)

    # Process through enclave
    response = await handler.run_agent(
        encrypted_message=encrypted_message,
        encrypted_state=encrypted_state,
        user_public_key=user_public_key,
        agent_name=agent_name,
        model="anthropic.claude-sonnet-4-20250514",
        encryption_mode=encryption_mode,
        kms_envelope=kms_envelope,
    )

    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)

    # Store updated state based on mode
    if encryption_mode == "zero_trust":
        encrypted_tarball = _serialize_encrypted_payload(response.encrypted_state)
        encrypted_dek = None
    else:  # background
        encrypted_tarball = _serialize_kms_envelope(response.kms_envelope)
        encrypted_dek = response.encrypted_dek

    if agent:
        await service.update_agent_state(
            user_id=auth.user_id,
            agent_name=agent_name,
            encrypted_tarball=encrypted_tarball,
            encrypted_dek=encrypted_dek,
        )
    else:
        await service.create_agent_state(
            user_id=auth.user_id,
            agent_name=agent_name,
            encrypted_tarball=encrypted_tarball,
            encryption_mode=EncryptionMode(encryption_mode),
            encrypted_dek=encrypted_dek,
        )

    # Return response
    return AgentMessageResponse(
        encrypted_response=EncryptedPayloadSchema.from_crypto_payload(
            response.encrypted_response
        ),
        # Only return state for zero_trust (client needs it)
        encrypted_state=EncryptedPayloadSchema.from_crypto_payload(
            response.encrypted_state
        ) if encryption_mode == "zero_trust" else None,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/routers/test_agents.py::TestAgentBackgroundMode -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add backend/routers/agents.py backend/tests/unit/routers/test_agents.py
git commit -m "feat(agent): implement background mode with KMS in router

- Add _serialize/_deserialize_kms_envelope helpers
- Update send_agent_message to handle both modes
- background: store KMS envelope, don't return state to client
- zero_trust: store user-encrypted state, return to client"
```

---

## Task 8: Add mode switching endpoint

**Files:**
- Modify: `backend/routers/agents.py`
- Test: `backend/tests/unit/routers/test_agents.py`

**Step 1: Write the failing test**

```python
# backend/tests/unit/routers/test_agents.py

class TestAgentModeSwitch:
    """Test switching agent encryption modes."""

    @pytest.mark.asyncio
    async def test_switch_zero_trust_to_background(self, client, mock_db):
        """Test switching from zero_trust to background mode."""
        # Create zero_trust agent
        await client.post(
            "/api/v1/agents",
            json={"agent_name": "test_agent", "encryption_mode": "zero_trust"},
            headers={"Authorization": "Bearer test_token"},
        )

        # Switch to background
        response = await client.patch(
            "/api/v1/agents/test_agent/encryption",
            json={
                "encryption_mode": "background",
                "encrypted_state": {  # Client provides current state
                    "ephemeral_public_key": "a" * 64,
                    "iv": "b" * 32,
                    "ciphertext": "c" * 64,
                    "auth_tag": "d" * 32,
                    "hkdf_salt": "e" * 64,
                },
            },
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["encryption_mode"] == "background"

    @pytest.mark.asyncio
    async def test_switch_background_to_zero_trust(self, client, mock_db, mock_enclave):
        """Test switching from background to zero_trust mode."""
        # Create background agent
        await client.post(
            "/api/v1/agents",
            json={"agent_name": "test_agent", "encryption_mode": "background"},
            headers={"Authorization": "Bearer test_token"},
        )

        # Send a message to create state
        await client.post(
            "/api/v1/agents/test_agent/message",
            json={
                "encrypted_message": {
                    "ephemeral_public_key": "a" * 64,
                    "iv": "b" * 32,
                    "ciphertext": "c" * 64,
                    "auth_tag": "d" * 32,
                    "hkdf_salt": "e" * 64,
                },
            },
            headers={"Authorization": "Bearer test_token"},
        )

        # Switch to zero_trust
        response = await client.patch(
            "/api/v1/agents/test_agent/encryption",
            json={"encryption_mode": "zero_trust"},
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["encryption_mode"] == "zero_trust"
        # Response should include state encrypted to user's key
        assert "encrypted_state" in data
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/routers/test_agents.py::TestAgentModeSwitch -v`
Expected: FAIL - endpoint doesn't exist

**Step 3: Write minimal implementation**

```python
# backend/routers/agents.py

@router.patch("/{agent_name}/encryption", response_model=AgentEncryptionResponse)
async def update_agent_encryption(
    agent_name: str,
    request: UpdateAgentEncryptionRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Switch agent encryption mode.

    When switching TO zero_trust:
    - If agent has KMS state, enclave decrypts and re-encrypts to user's key
    - Response includes encrypted_state for client

    When switching TO background:
    - Client must provide encrypted_state (their current state)
    - Enclave decrypts and re-encrypts with KMS
    """
    service = AgentService(db)
    handler = get_agent_handler()

    agent = await service.get_agent_state(auth.user_id, agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    new_mode = EncryptionMode(request.encryption_mode)
    current_mode = agent.encryption_mode

    if new_mode == current_mode:
        return AgentEncryptionResponse(
            agent_name=agent_name,
            encryption_mode=new_mode.value,
            encrypted_state=None,
        )

    user_keys = await get_user_keys(db, auth.user_id)
    user_public_key = bytes.fromhex(user_keys.public_key)

    # Get current state
    if current_mode == EncryptionMode.ZERO_TRUST:
        if not request.encrypted_state:
            raise HTTPException(
                status_code=400,
                detail="Must provide encrypted_state when switching from zero_trust"
            )
        current_state_payload = request.encrypted_state.to_crypto_payload()
        current_kms_envelope = None
    else:  # background
        current_state_payload = None
        if agent.encrypted_tarball:
            current_kms_envelope = _deserialize_kms_envelope(agent.encrypted_tarball)
        else:
            current_kms_envelope = None

    # Re-encrypt state through enclave
    response = await handler.reencrypt_state(
        encrypted_state=current_state_payload,
        kms_envelope=current_kms_envelope,
        user_public_key=user_public_key,
        from_mode=current_mode.value,
        to_mode=new_mode.value,
    )

    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)

    # Store with new mode
    if new_mode == EncryptionMode.ZERO_TRUST:
        encrypted_tarball = _serialize_encrypted_payload(response.encrypted_state)
        encrypted_dek = None
    else:
        encrypted_tarball = _serialize_kms_envelope(response.kms_envelope)
        encrypted_dek = response.encrypted_dek

    await service.update_agent_state(
        user_id=auth.user_id,
        agent_name=agent_name,
        encrypted_tarball=encrypted_tarball,
        encryption_mode=new_mode,
        encrypted_dek=encrypted_dek,
    )

    return AgentEncryptionResponse(
        agent_name=agent_name,
        encryption_mode=new_mode.value,
        encrypted_state=EncryptedPayloadSchema.from_crypto_payload(
            response.encrypted_state
        ) if new_mode == EncryptionMode.ZERO_TRUST else None,
    )


# Add to schemas/agent.py
class AgentEncryptionResponse(BaseModel):
    """Response after changing encryption mode."""
    agent_name: str
    encryption_mode: str
    # When switching TO zero_trust, includes state encrypted to user's key
    encrypted_state: Optional[EncryptedPayload] = None
```

**Step 4: Add reencrypt_state to mock_enclave**

```python
# backend/core/enclave/mock_enclave.py

async def reencrypt_state(
    self,
    encrypted_state: Optional[EncryptedPayload],
    kms_envelope: Optional[Dict[str, bytes]],
    user_public_key: bytes,
    from_mode: str,
    to_mode: str,
) -> AgentRunResponse:
    """Re-encrypt agent state from one mode to another."""
    try:
        # Decrypt current state
        if from_mode == "zero_trust" and encrypted_state:
            state_bytes = decrypt_with_private_key(
                self._keypair.private_key,
                encrypted_state,
                EncryptionContext.CLIENT_TO_ENCLAVE.value,
            )
        elif from_mode == "background" and kms_envelope:
            state_bytes = self.kms.envelope_decrypt(kms_envelope)
        else:
            return AgentRunResponse(success=False, error="No state to re-encrypt")

        # Re-encrypt to new mode
        if to_mode == "zero_trust":
            encrypted_state_out = encrypt_to_public_key(
                user_public_key,
                state_bytes,
                EncryptionContext.AGENT_STATE_STORAGE.value,
            )
            return AgentRunResponse(
                success=True,
                encrypted_state=encrypted_state_out,
            )
        else:  # background
            kms_envelope_out = self.kms.envelope_encrypt(state_bytes)
            return AgentRunResponse(
                success=True,
                kms_envelope=kms_envelope_out,
                encrypted_dek=kms_envelope_out["encrypted_dek"],
            )
    except Exception as e:
        return AgentRunResponse(success=False, error=str(e))
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/routers/test_agents.py::TestAgentModeSwitch -v`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add backend/routers/agents.py backend/schemas/agent.py backend/core/enclave/mock_enclave.py backend/tests/unit/routers/test_agents.py
git commit -m "feat(agent): add encryption mode switching endpoint

- Add PATCH /agents/{name}/encryption endpoint
- Add reencrypt_state to MockEnclave
- Support switching between zero_trust and background modes
- Re-encrypts state through enclave on mode change"
```

---

## Task 9: Run full test suite and verify

**Step 1: Run all agent-related tests**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -m pytest tests/unit/models/test_agent_state.py tests/unit/schemas/test_agent_schemas.py tests/unit/enclave/test_mock_enclave.py tests/unit/enclave/test_kms_encryption.py tests/unit/routers/test_agents.py -v`
Expected: All tests PASS

**Step 2: Run full backend test suite**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ./run_tests.sh`
Expected: All tests PASS

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "test: ensure all agent encryption tests pass"
```

---

## Summary

This plan implements dual encryption modes for agent state:

1. **Zero Trust (default)**: Agent state encrypted to user's public key. Client fetches encrypted state, decrypts with their key, re-encrypts to enclave transport key, sends with message. Isol8 cannot read agent data.

2. **Background (opt-in)**: Agent state encrypted with KMS envelope encryption. Enclave can decrypt autonomously using KMS. Enables scheduled tasks, channel bridges, proactive notifications. Requires trusting Isol8 infrastructure.

**Critical bug fixed**: Agent state was encrypted to enclave's ephemeral key (lost on reboot). Now encrypted to user's key (zero_trust) or KMS (background).

**Files modified:**
- `models/agent_state.py` - Add encryption_mode, encrypted_dek columns
- `schemas/agent.py` - Add encryption_mode to request/response
- `core/enclave/mock_enclave.py` - Fix encryption, add mode support
- `core/enclave/kms_encryption.py` - KMS envelope encryption
- `core/enclave/__init__.py` - Update AgentRunResponse
- `routers/agents.py` - Update endpoints for both modes
- Migration file for database changes
