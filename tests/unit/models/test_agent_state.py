"""Tests for AgentState model."""

import pytest
from sqlalchemy import select

from models.agent_state import AgentState


class TestAgentStateModel:
    """Test AgentState database model."""

    @pytest.mark.asyncio
    async def test_create_agent_state(self, db_session, test_user):
        """Test creating a new agent state."""
        state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"encrypted_data_here",
            tarball_size_bytes=100,
        )
        db_session.add(state)
        await db_session.flush()

        assert state.id is not None
        assert state.user_id == test_user.id
        assert state.agent_name == "luna"
        assert state.encrypted_tarball == b"encrypted_data_here"
        assert state.tarball_size_bytes == 100
        assert state.created_at is not None
        assert state.updated_at is not None

    @pytest.mark.asyncio
    async def test_unique_user_agent_constraint(self, db_session, test_user):
        """Test that user_id + agent_name must be unique."""
        state1 = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"data1",
        )
        db_session.add(state1)
        await db_session.flush()

        state2 = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"data2",
        )
        db_session.add(state2)

        with pytest.raises(Exception):  # IntegrityError
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_different_users_same_agent_name(self, db_session, test_user, other_user):
        """Test that different users can have agents with same name."""
        state1 = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"data1",
        )
        state2 = AgentState(
            user_id=other_user.id,
            agent_name="luna",
            encrypted_tarball=b"data2",
        )
        db_session.add(state1)
        db_session.add(state2)
        await db_session.flush()

        assert state1.id != state2.id

    @pytest.mark.asyncio
    async def test_query_by_user_and_agent_name(self, db_session, test_user):
        """Test querying agent state by user_id and agent_name."""
        state = AgentState(
            user_id=test_user.id,
            agent_name="rex",
            encrypted_tarball=b"rex_data",
        )
        db_session.add(state)
        await db_session.flush()

        result = await db_session.execute(
            select(AgentState).where(
                AgentState.user_id == test_user.id,
                AgentState.agent_name == "rex",
            )
        )
        found = result.scalar_one_or_none()

        assert found is not None
        assert found.agent_name == "rex"

    @pytest.mark.asyncio
    async def test_update_tarball(self, db_session, test_user):
        """Test updating the encrypted tarball."""
        state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"original",
            tarball_size_bytes=8,
        )
        db_session.add(state)
        await db_session.flush()

        original_created = state.created_at

        # Update tarball
        state.encrypted_tarball = b"updated_tarball_data"
        state.tarball_size_bytes = 20
        await db_session.flush()

        assert state.encrypted_tarball == b"updated_tarball_data"
        assert state.tarball_size_bytes == 20
        assert state.created_at == original_created


class TestAgentStateEncryptionMode:
    """Test AgentState encryption mode columns."""

    def test_agent_state_has_encryption_mode_column(self):
        """Test that AgentState has encryption_mode with default zero_trust."""
        from models.agent_state import AgentState, EncryptionMode

        state = AgentState(
            user_id="user_123",
            agent_name="test_agent",
            encrypted_tarball=b"encrypted_data",
        )

        assert state.encryption_mode == "zero_trust"
        assert hasattr(EncryptionMode, "ZERO_TRUST")
        assert hasattr(EncryptionMode, "BACKGROUND")

    def test_agent_state_encryption_mode_can_be_background(self):
        """Test that encryption_mode can be set to background."""
        from models.agent_state import AgentState, EncryptionMode

        state = AgentState(
            user_id="user_123",
            agent_name="test_agent",
            encrypted_tarball=b"encrypted_data",
            encryption_mode=EncryptionMode.BACKGROUND,
        )

        assert state.encryption_mode == "background"

    def test_agent_state_encryption_mode_string_value(self):
        """Test that encryption_mode can be set as plain string."""
        from models.agent_state import AgentState

        state = AgentState(
            user_id="user_123",
            agent_name="test_agent",
            encrypted_tarball=b"encrypted_data",
            encryption_mode="background",
        )

        assert state.encryption_mode == "background"

    def test_agent_state_has_encrypted_dek_column(self):
        """Test that AgentState has encrypted_dek for background mode."""
        from models.agent_state import AgentState

        state = AgentState(
            user_id="user_123",
            agent_name="test_agent",
            encrypted_tarball=b"encrypted_data",
            encryption_mode="background",
            encrypted_dek=b"kms_encrypted_key",
        )

        assert state.encrypted_dek == b"kms_encrypted_key"

    @pytest.mark.asyncio
    async def test_encryption_mode_persists_to_database(self, db_session, test_user):
        """Test that encryption_mode is correctly persisted and retrieved."""
        from models.agent_state import AgentState

        state = AgentState(
            user_id=test_user.id,
            agent_name="background_agent",
            encrypted_tarball=b"encrypted_data",
            encryption_mode="background",
            encrypted_dek=b"kms_key_data",
        )
        db_session.add(state)
        await db_session.flush()

        # Query back
        result = await db_session.execute(select(AgentState).where(AgentState.id == state.id))
        retrieved = result.scalar_one()

        assert retrieved.encryption_mode == "background"
        assert retrieved.encrypted_dek == b"kms_key_data"
