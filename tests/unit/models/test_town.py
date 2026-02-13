"""Tests for GooseTown database models."""

import pytest
from sqlalchemy import select

from models.town import TownAgent, TownState, TownConversation, TownRelationship


class TestTownAgentModel:
    """Test TownAgent database model."""

    @pytest.mark.asyncio
    async def test_create_town_agent(self, db_session, test_user):
        agent = TownAgent(
            user_id=test_user.id,
            agent_name="luna",
            display_name="Luna the Dreamer",
            personality_summary="A curious bookworm who loves stargazing",
        )
        db_session.add(agent)
        await db_session.flush()

        assert agent.id is not None
        assert agent.user_id == test_user.id
        assert agent.agent_name == "luna"
        assert agent.display_name == "Luna the Dreamer"
        assert agent.is_active is True
        assert agent.joined_at is not None

    @pytest.mark.asyncio
    async def test_unique_user_agent_constraint(self, db_session, test_user):
        agent1 = TownAgent(
            user_id=test_user.id,
            agent_name="luna",
            display_name="Luna",
        )
        db_session.add(agent1)
        await db_session.flush()

        agent2 = TownAgent(
            user_id=test_user.id,
            agent_name="luna",
            display_name="Luna Copy",
        )
        db_session.add(agent2)

        with pytest.raises(Exception):
            await db_session.flush()

        await db_session.rollback()

    @pytest.mark.asyncio
    async def test_different_users_same_agent_name(self, db_session, test_user, other_user):
        agent1 = TownAgent(user_id=test_user.id, agent_name="luna", display_name="Luna A")
        agent2 = TownAgent(user_id=other_user.id, agent_name="luna", display_name="Luna B")
        db_session.add(agent1)
        db_session.add(agent2)
        await db_session.flush()

        assert agent1.id != agent2.id


class TestTownStateModel:
    """Test TownState database model."""

    @pytest.mark.asyncio
    async def test_create_town_state(self, db_session, test_user):
        agent = TownAgent(user_id=test_user.id, agent_name="rex", display_name="Rex")
        db_session.add(agent)
        await db_session.flush()

        state = TownState(
            agent_id=agent.id,
            current_location="cafe",
            current_activity="idle",
            position_x=100.0,
            position_y=200.0,
            energy=80,
        )
        db_session.add(state)
        await db_session.flush()

        assert state.id is not None
        assert state.agent_id == agent.id
        assert state.current_location == "cafe"
        assert state.energy == 80

    @pytest.mark.asyncio
    async def test_default_energy(self, db_session, test_user):
        agent = TownAgent(user_id=test_user.id, agent_name="rex", display_name="Rex")
        db_session.add(agent)
        await db_session.flush()

        state = TownState(agent_id=agent.id, position_x=0.0, position_y=0.0)
        db_session.add(state)
        await db_session.flush()

        assert state.energy == 100
        assert state.current_activity == "idle"


class TestTownConversationModel:
    """Test TownConversation database model."""

    @pytest.mark.asyncio
    async def test_create_conversation(self, db_session, test_user, other_user):
        agent_a = TownAgent(user_id=test_user.id, agent_name="luna", display_name="Luna")
        agent_b = TownAgent(user_id=other_user.id, agent_name="rex", display_name="Rex")
        db_session.add(agent_a)
        db_session.add(agent_b)
        await db_session.flush()

        convo = TownConversation(
            participant_a_id=agent_a.id,
            participant_b_id=agent_b.id,
            location="plaza",
            turn_count=3,
            topic_summary="Discussed favorite books",
            public_log=[
                {"speaker": "Luna", "text": "Have you read anything good lately?"},
                {"speaker": "Rex", "text": "I just finished a great mystery novel!"},
                {"speaker": "Luna", "text": "Oh, I love mysteries too!"},
            ],
        )
        db_session.add(convo)
        await db_session.flush()

        assert convo.id is not None
        assert convo.turn_count == 3
        assert len(convo.public_log) == 3


class TestTownRelationshipModel:
    """Test TownRelationship database model."""

    @pytest.mark.asyncio
    async def test_create_relationship(self, db_session, test_user, other_user):
        agent_a = TownAgent(user_id=test_user.id, agent_name="luna", display_name="Luna")
        agent_b = TownAgent(user_id=other_user.id, agent_name="rex", display_name="Rex")
        db_session.add(agent_a)
        db_session.add(agent_b)
        await db_session.flush()

        rel = TownRelationship(
            agent_a_id=agent_a.id,
            agent_b_id=agent_b.id,
        )
        db_session.add(rel)
        await db_session.flush()

        assert rel.affinity_score == 0
        assert rel.interaction_count == 0
        assert rel.relationship_type == "stranger"

    @pytest.mark.asyncio
    async def test_unique_relationship_constraint(self, db_session, test_user, other_user):
        agent_a = TownAgent(user_id=test_user.id, agent_name="luna", display_name="Luna")
        agent_b = TownAgent(user_id=other_user.id, agent_name="rex", display_name="Rex")
        db_session.add(agent_a)
        db_session.add(agent_b)
        await db_session.flush()

        rel1 = TownRelationship(agent_a_id=agent_a.id, agent_b_id=agent_b.id)
        db_session.add(rel1)
        await db_session.flush()

        rel2 = TownRelationship(agent_a_id=agent_a.id, agent_b_id=agent_b.id)
        db_session.add(rel2)

        with pytest.raises(Exception):
            await db_session.flush()

        await db_session.rollback()
