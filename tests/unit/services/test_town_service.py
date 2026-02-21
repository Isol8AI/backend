"""Tests for TownService CRUD operations."""

import pytest

from core.services.town_service import TownService
from models.agent_state import AgentState


class TestTownServiceOptIn:
    """Test opt-in/opt-out operations."""

    @pytest.fixture
    def service(self, db_session):
        return TownService(db_session)

    @pytest.mark.asyncio
    async def test_opt_in_creates_town_agent_and_state(self, service, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        town_agent = await service.opt_in(
            user_id=test_user.id,
            agent_name="luna",
            display_name="Luna the Dreamer",
            personality_summary="Curious bookworm",
        )

        assert town_agent is not None
        assert town_agent.agent_name == "luna"
        assert town_agent.display_name == "Luna the Dreamer"
        assert town_agent.is_active is True

    @pytest.mark.asyncio
    async def test_opt_in_requires_background_mode(self, service, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="zero_trust",
        )
        db_session.add(agent_state)
        await db_session.flush()

        with pytest.raises(ValueError, match="background"):
            await service.opt_in(
                user_id=test_user.id,
                agent_name="luna",
                display_name="Luna",
            )

    @pytest.mark.asyncio
    async def test_opt_in_fails_if_agent_not_found(self, service, test_user):
        with pytest.raises(ValueError, match="not found"):
            await service.opt_in(
                user_id=test_user.id,
                agent_name="nonexistent",
                display_name="Ghost",
            )

    @pytest.mark.asyncio
    async def test_opt_out_deactivates_agent(self, service, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        await service.opt_in(
            user_id=test_user.id,
            agent_name="luna",
            display_name="Luna",
        )

        result = await service.opt_out(user_id=test_user.id, agent_name="luna")
        assert result is True

    @pytest.mark.asyncio
    async def test_opt_out_nonexistent_returns_false(self, service, test_user):
        result = await service.opt_out(user_id=test_user.id, agent_name="ghost")
        assert result is False


class TestTownServiceState:
    """Test state query operations."""

    @pytest.fixture
    def service(self, db_session):
        return TownService(db_session)

    @pytest.mark.asyncio
    async def test_get_all_active_agents(self, service, db_session, test_user, other_user):
        for user, name, display in [
            (test_user, "luna", "Luna"),
            (other_user, "rex", "Rex"),
        ]:
            agent_state = AgentState(
                user_id=user.id,
                agent_name=name,
                encryption_mode="background",
            )
            db_session.add(agent_state)
            await db_session.flush()

            await service.opt_in(
                user_id=user.id,
                agent_name=name,
                display_name=display,
            )

        agents = await service.get_active_agents()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_get_town_state(self, service, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        await service.opt_in(
            user_id=test_user.id,
            agent_name="luna",
            display_name="Luna",
        )

        states = await service.get_town_state()
        assert len(states) == 1
        assert states[0]["display_name"] == "Luna"
        assert states[0]["position_x"] == 0.0


class TestTownServiceSeedAgent:
    """Test seed_agent for default agents (bypasses encryption check)."""

    @pytest.fixture
    def service(self, db_session):
        return TownService(db_session)

    @pytest.mark.asyncio
    async def test_seed_agent_creates_agent_and_state(self, service, db_session):
        agent = await service.seed_agent(
            user_id="system",
            agent_name="lucky",
            display_name="Lucky",
            personality_summary="Happy and curious",
            position_x=8.0,
            position_y=6.0,
            home_location="cafe",
        )

        assert agent is not None
        assert agent.agent_name == "lucky"
        assert agent.display_name == "Lucky"
        assert agent.is_active is True
        assert agent.home_location == "cafe"

        states = await service.get_town_state()
        assert len(states) == 1
        assert states[0]["position_x"] == 8.0
        assert states[0]["position_y"] == 6.0

    @pytest.mark.asyncio
    async def test_seed_agent_idempotent(self, service, db_session):
        """Seeding same agent twice returns existing agent."""
        agent1 = await service.seed_agent(
            user_id="system",
            agent_name="lucky",
            display_name="Lucky",
            position_x=8.0,
            position_y=6.0,
        )
        agent2 = await service.seed_agent(
            user_id="system",
            agent_name="lucky",
            display_name="Lucky Updated",
            position_x=10.0,
            position_y=10.0,
        )

        assert agent1.id == agent2.id
        # Display name unchanged since agent was already active
        assert agent2.display_name == "Lucky"

    @pytest.mark.asyncio
    async def test_seed_agent_reactivates_inactive(self, service, db_session):
        """Seeding an inactive agent reactivates it."""
        agent = await service.seed_agent(
            user_id="system",
            agent_name="lucky",
            display_name="Lucky",
            position_x=8.0,
            position_y=6.0,
        )
        await db_session.flush()

        # Deactivate
        agent.is_active = False
        await db_session.flush()

        # Re-seed
        reactivated = await service.seed_agent(
            user_id="system",
            agent_name="lucky",
            display_name="Lucky Reactivated",
            position_x=12.0,
            position_y=20.0,
        )

        assert reactivated.id == agent.id
        assert reactivated.is_active is True
        assert reactivated.display_name == "Lucky Reactivated"


class TestTownServiceRelationships:
    """Test relationship operations."""

    @pytest.fixture
    def service(self, db_session):
        return TownService(db_session)

    @pytest.mark.asyncio
    async def test_get_or_create_relationship(self, service, db_session, test_user, other_user):
        for user, name, display in [
            (test_user, "luna", "Luna"),
            (other_user, "rex", "Rex"),
        ]:
            agent_state = AgentState(
                user_id=user.id,
                agent_name=name,
                encryption_mode="background",
            )
            db_session.add(agent_state)
            await db_session.flush()
            await service.opt_in(user_id=user.id, agent_name=name, display_name=display)

        agents = await service.get_active_agents()
        rel, created = await service.get_or_create_relationship(agents[0].id, agents[1].id)

        assert created is True
        assert rel.affinity_score == 0
        assert rel.relationship_type == "stranger"

    @pytest.mark.asyncio
    async def test_update_relationship(self, service, db_session, test_user, other_user):
        for user, name, display in [
            (test_user, "luna", "Luna"),
            (other_user, "rex", "Rex"),
        ]:
            agent_state = AgentState(
                user_id=user.id,
                agent_name=name,
                encryption_mode="background",
            )
            db_session.add(agent_state)
            await db_session.flush()
            await service.opt_in(user_id=user.id, agent_name=name, display_name=display)

        agents = await service.get_active_agents()
        rel, _ = await service.get_or_create_relationship(agents[0].id, agents[1].id)

        updated = await service.update_relationship(rel.id, affinity_delta=10, new_type="acquaintance")

        assert updated.affinity_score == 10
        assert updated.interaction_count == 1
        assert updated.relationship_type == "acquaintance"
