"""Tests for GooseTown router endpoints."""

import pytest
from models.agent_state import AgentState


class TestTownOptIn:
    """Test POST /api/v1/town/opt-in."""

    @pytest.mark.asyncio
    async def test_opt_in_success(self, async_client, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        response = await async_client.post(
            "/api/v1/town/opt-in",
            json={
                "agent_name": "luna",
                "display_name": "Luna the Dreamer",
                "personality_summary": "A curious bookworm",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_name"] == "luna"
        assert data["display_name"] == "Luna the Dreamer"
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_opt_in_requires_background_mode(self, async_client, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="zero_trust",
        )
        db_session.add(agent_state)
        await db_session.flush()

        response = await async_client.post(
            "/api/v1/town/opt-in",
            json={"agent_name": "luna", "display_name": "Luna"},
        )

        assert response.status_code == 400
        assert "background" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_opt_in_agent_not_found(self, async_client, test_user):
        response = await async_client.post(
            "/api/v1/town/opt-in",
            json={"agent_name": "ghost", "display_name": "Ghost"},
        )

        assert response.status_code == 400


class TestTownOptOut:
    """Test POST /api/v1/town/opt-out."""

    @pytest.mark.asyncio
    async def test_opt_out_success(self, async_client, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        await async_client.post(
            "/api/v1/town/opt-in",
            json={"agent_name": "luna", "display_name": "Luna"},
        )

        response = await async_client.post(
            "/api/v1/town/opt-out",
            json={"agent_name": "luna"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_opt_out_not_found(self, async_client, test_user):
        response = await async_client.post(
            "/api/v1/town/opt-out",
            json={"agent_name": "ghost"},
        )

        assert response.status_code == 404


class TestTownState:
    """Test GET /api/v1/town/state."""

    @pytest.mark.asyncio
    async def test_get_state_empty(self, async_client):
        response = await async_client.get("/api/v1/town/state")

        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == []

    @pytest.mark.asyncio
    async def test_get_state_with_agents(self, async_client, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        await async_client.post(
            "/api/v1/town/opt-in",
            json={"agent_name": "luna", "display_name": "Luna"},
        )

        response = await async_client.get("/api/v1/town/state")

        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["display_name"] == "Luna"


class TestTownConversations:
    """Test GET /api/v1/town/conversations."""

    @pytest.mark.asyncio
    async def test_get_conversations_empty(self, async_client):
        response = await async_client.get("/api/v1/town/conversations")

        assert response.status_code == 200
        data = response.json()
        assert data["conversations"] == []
