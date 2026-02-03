"""Tests for AgentService business logic."""

import pytest

from core.services.agent_service import AgentService


class TestAgentService:
    """Test AgentService CRUD operations."""

    @pytest.fixture
    def service(self, db_session):
        """Create service instance."""
        return AgentService(db_session)

    @pytest.mark.asyncio
    async def test_get_agent_state_not_found(self, service, test_user):
        """Test getting non-existent agent state returns None."""
        result = await service.get_agent_state(
            user_id=test_user.id,
            agent_name="nonexistent",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_create_agent_state(self, service, test_user):
        """Test creating new agent state."""
        state = await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"encrypted_data",
        )

        assert state is not None
        assert state.user_id == test_user.id
        assert state.agent_name == "luna"
        assert state.encrypted_tarball == b"encrypted_data"

    @pytest.mark.asyncio
    async def test_get_agent_state(self, service, test_user):
        """Test getting existing agent state."""
        # Create first
        await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"test_data",
        )

        # Then get
        state = await service.get_agent_state(
            user_id=test_user.id,
            agent_name="luna",
        )

        assert state is not None
        assert state.agent_name == "luna"

    @pytest.mark.asyncio
    async def test_update_agent_state(self, service, test_user):
        """Test updating agent state."""
        # Create first
        state = await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"original",
        )

        # Update
        updated = await service.update_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"updated_data",
        )

        assert updated is not None
        assert updated.encrypted_tarball == b"updated_data"
        assert updated.id == state.id

    @pytest.mark.asyncio
    async def test_list_user_agents(self, service, test_user):
        """Test listing all agents for a user."""
        await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"luna_data",
        )
        await service.create_agent_state(
            user_id=test_user.id,
            agent_name="rex",
            encrypted_tarball=b"rex_data",
        )

        agents = await service.list_user_agents(user_id=test_user.id)

        assert len(agents) == 2
        agent_names = [a.agent_name for a in agents]
        assert "luna" in agent_names
        assert "rex" in agent_names

    @pytest.mark.asyncio
    async def test_delete_agent_state(self, service, test_user):
        """Test deleting agent state."""
        await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"data",
        )

        deleted = await service.delete_agent_state(
            user_id=test_user.id,
            agent_name="luna",
        )

        assert deleted is True

        # Verify it's gone
        state = await service.get_agent_state(
            user_id=test_user.id,
            agent_name="luna",
        )
        assert state is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_agent(self, service, test_user):
        """Test deleting non-existent agent returns False."""
        deleted = await service.delete_agent_state(
            user_id=test_user.id,
            agent_name="nonexistent",
        )
        assert deleted is False

    @pytest.mark.asyncio
    async def test_user_isolation(self, service, test_user, other_user):
        """Test that users can't access each other's agents."""
        await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"test_user_data",
        )

        # Other user shouldn't see it
        state = await service.get_agent_state(
            user_id=other_user.id,
            agent_name="luna",
        )
        assert state is None

        # Other user's list should be empty
        agents = await service.list_user_agents(user_id=other_user.id)
        assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, service, test_user):
        """Test get_or_create for new agent."""
        state, created = await service.get_or_create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            default_tarball=b"default_data",
        )

        assert created is True
        assert state.encrypted_tarball == b"default_data"

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, service, test_user):
        """Test get_or_create for existing agent."""
        await service.create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            encrypted_tarball=b"existing_data",
        )

        state, created = await service.get_or_create_agent_state(
            user_id=test_user.id,
            agent_name="luna",
            default_tarball=b"default_data",
        )

        assert created is False
        assert state.encrypted_tarball == b"existing_data"
