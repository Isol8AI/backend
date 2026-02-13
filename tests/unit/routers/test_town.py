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


class TestTownStatus:
    """Test GET /api/v1/town/status (AI Town worldStatus)."""

    @pytest.mark.asyncio
    async def test_get_status(self, async_client):
        response = await async_client.get("/api/v1/town/status")

        assert response.status_code == 200
        data = response.json()
        assert data["worldId"] == "world_default"
        assert data["engineId"] == "engine_default"
        assert data["status"] == "running"
        assert data["isDefault"] is True


class TestTownState:
    """Test GET /api/v1/town/state (AI Town worldState format)."""

    @pytest.mark.asyncio
    async def test_get_state_empty(self, async_client):
        response = await async_client.get("/api/v1/town/state")

        assert response.status_code == 200
        data = response.json()
        assert data["world"]["players"] == []
        assert data["world"]["agents"] == []
        assert data["world"]["conversations"] == []
        assert "currentTime" in data["engine"]

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
        assert len(data["world"]["players"]) == 1
        assert data["world"]["players"][0]["id"] == "p:0"
        assert "position" in data["world"]["players"][0]
        assert len(data["world"]["agents"]) == 1
        assert data["world"]["agents"][0]["id"] == "a:0"
        assert data["world"]["agents"][0]["playerId"] == "p:0"


class TestTownDescriptions:
    """Test GET /api/v1/town/descriptions (AI Town gameDescriptions)."""

    @pytest.mark.asyncio
    async def test_get_descriptions_empty(self, async_client):
        response = await async_client.get("/api/v1/town/descriptions")

        assert response.status_code == 200
        data = response.json()
        assert "worldMap" in data
        assert data["worldMap"]["width"] > 0
        assert data["playerDescriptions"] == []
        assert data["agentDescriptions"] == []

    @pytest.mark.asyncio
    async def test_get_descriptions_with_agents(self, async_client, db_session, test_user):
        agent_state = AgentState(
            user_id=test_user.id,
            agent_name="luna",
            encryption_mode="background",
        )
        db_session.add(agent_state)
        await db_session.flush()

        await async_client.post(
            "/api/v1/town/opt-in",
            json={
                "agent_name": "luna",
                "display_name": "Luna",
                "personality_summary": "A dreamer",
            },
        )

        response = await async_client.get("/api/v1/town/descriptions")

        assert response.status_code == 200
        data = response.json()
        assert len(data["playerDescriptions"]) == 1
        assert data["playerDescriptions"][0]["name"] == "Luna"
        assert data["playerDescriptions"][0]["playerId"] == "p:0"
        assert len(data["agentDescriptions"]) == 1
        assert data["agentDescriptions"][0]["agentId"] == "a:0"


class TestTownConversations:
    """Test GET /api/v1/town/conversations."""

    @pytest.mark.asyncio
    async def test_get_conversations_empty(self, async_client):
        response = await async_client.get("/api/v1/town/conversations")

        assert response.status_code == 200
        data = response.json()
        assert data["conversations"] == []


class TestTownStubs:
    """Test stub endpoints return without error."""

    @pytest.mark.asyncio
    async def test_heartbeat(self, async_client):
        response = await async_client.post("/api/v1/town/heartbeat", json={})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_music(self, async_client):
        response = await async_client.get("/api/v1/town/music")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_user_status(self, async_client):
        response = await async_client.get("/api/v1/town/user-status")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_input_status(self, async_client):
        response = await async_client.get("/api/v1/town/input-status")
        assert response.status_code == 200
