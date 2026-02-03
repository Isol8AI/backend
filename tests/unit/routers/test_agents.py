"""Tests for Agent API endpoints."""

import pytest
from unittest.mock import MagicMock, patch

from core.crypto import generate_x25519_keypair


class TestAgentEndpoints:
    """Test agent REST API endpoints."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, async_client, test_user):
        """Test listing agents when user has none."""
        response = await async_client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == []

    @pytest.mark.asyncio
    @patch("routers.agents.get_enclave")
    async def test_create_agent(self, mock_get_enclave, async_client, test_user):
        """Test creating a new agent."""
        # Setup mock enclave
        mock_enclave = MagicMock()
        keypair = generate_x25519_keypair()
        mock_enclave.get_info.return_value = MagicMock(
            enclave_public_key=keypair.public_key
        )
        mock_enclave.encrypt_for_storage.return_value = MagicMock(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"z" * 100,
            auth_tag=b"a" * 16,
            hkdf_salt=b"b" * 32,
        )
        mock_get_enclave.return_value = mock_enclave

        response = await async_client.post(
            "/api/v1/agents",
            json={
                "agent_name": "luna",
                "soul_content": "# Luna\nA friendly companion.",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["agent_name"] == "luna"
        assert data["user_id"] == test_user.id

    @pytest.mark.asyncio
    @patch("routers.agents.get_enclave")
    async def test_create_duplicate_agent(self, mock_get_enclave, async_client, test_user):
        """Test creating duplicate agent fails."""
        # Setup mock enclave
        mock_enclave = MagicMock()
        keypair = generate_x25519_keypair()
        mock_enclave.get_info.return_value = MagicMock(
            enclave_public_key=keypair.public_key
        )
        mock_enclave.encrypt_for_storage.return_value = MagicMock(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"z" * 100,
            auth_tag=b"a" * 16,
            hkdf_salt=b"b" * 32,
        )
        mock_get_enclave.return_value = mock_enclave

        # Create first
        await async_client.post(
            "/api/v1/agents",
            json={"agent_name": "luna"},
        )

        # Try to create again
        response = await async_client.post(
            "/api/v1/agents",
            json={"agent_name": "luna"},
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    @patch("routers.agents.get_enclave")
    async def test_get_agent(self, mock_get_enclave, async_client, test_user):
        """Test getting agent details."""
        # Setup mock enclave
        mock_enclave = MagicMock()
        keypair = generate_x25519_keypair()
        mock_enclave.get_info.return_value = MagicMock(
            enclave_public_key=keypair.public_key
        )
        mock_enclave.encrypt_for_storage.return_value = MagicMock(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"z" * 100,
            auth_tag=b"a" * 16,
            hkdf_salt=b"b" * 32,
        )
        mock_get_enclave.return_value = mock_enclave

        # Create first
        await async_client.post(
            "/api/v1/agents",
            json={"agent_name": "luna"},
        )

        # Get details
        response = await async_client.get("/api/v1/agents/luna")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_name"] == "luna"

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, async_client, test_user):
        """Test getting non-existent agent returns 404."""
        response = await async_client.get("/api/v1/agents/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    @patch("routers.agents.get_enclave")
    async def test_delete_agent(self, mock_get_enclave, async_client, test_user):
        """Test deleting an agent."""
        # Setup mock enclave
        mock_enclave = MagicMock()
        keypair = generate_x25519_keypair()
        mock_enclave.get_info.return_value = MagicMock(
            enclave_public_key=keypair.public_key
        )
        mock_enclave.encrypt_for_storage.return_value = MagicMock(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"z" * 100,
            auth_tag=b"a" * 16,
            hkdf_salt=b"b" * 32,
        )
        mock_get_enclave.return_value = mock_enclave

        # Create first
        await async_client.post(
            "/api/v1/agents",
            json={"agent_name": "luna"},
        )

        # Delete
        response = await async_client.delete("/api/v1/agents/luna")
        assert response.status_code == 204

        # Verify gone
        response = await async_client.get("/api/v1/agents/luna")
        assert response.status_code == 404

    @pytest.mark.asyncio
    @patch("routers.agents.get_enclave")
    async def test_list_agents(self, mock_get_enclave, async_client, test_user):
        """Test listing all user's agents."""
        # Setup mock enclave
        mock_enclave = MagicMock()
        keypair = generate_x25519_keypair()
        mock_enclave.get_info.return_value = MagicMock(
            enclave_public_key=keypair.public_key
        )
        mock_enclave.encrypt_for_storage.return_value = MagicMock(
            ephemeral_public_key=b"x" * 32,
            iv=b"y" * 16,
            ciphertext=b"z" * 100,
            auth_tag=b"a" * 16,
            hkdf_salt=b"b" * 32,
        )
        mock_get_enclave.return_value = mock_enclave

        # Create multiple
        await async_client.post("/api/v1/agents", json={"agent_name": "luna"})
        await async_client.post("/api/v1/agents", json={"agent_name": "rex"})

        # List
        response = await async_client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 2
        agent_names = [a["agent_name"] for a in data["agents"]]
        assert "luna" in agent_names
        assert "rex" in agent_names


class TestAgentAuthorization:
    """Test agent authorization."""

    @pytest.mark.asyncio
    async def test_unauthenticated_list_agents(self, unauthenticated_async_client):
        """Test that unauthenticated users can't list agents."""
        response = await unauthenticated_async_client.get("/api/v1/agents")
        # Server returns 403 Forbidden for unauthenticated requests
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_unauthenticated_create_agent(self, unauthenticated_async_client):
        """Test that unauthenticated users can't create agents."""
        response = await unauthenticated_async_client.post(
            "/api/v1/agents",
            json={"agent_name": "luna"},
        )
        # Server returns 403 Forbidden for unauthenticated requests
        assert response.status_code in (401, 403)


class TestAgentValidation:
    """Test input validation for agent endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_agent_name_special_chars(self, async_client, test_user):
        """Test that agent names with invalid characters are rejected."""
        response = await async_client.post(
            "/api/v1/agents",
            json={"agent_name": "luna@bot!"},
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_agent_name_too_long(self, async_client, test_user):
        """Test that agent names that are too long are rejected."""
        response = await async_client.post(
            "/api/v1/agents",
            json={"agent_name": "a" * 51},  # Max is 50
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_agent_name_empty(self, async_client, test_user):
        """Test that empty agent names are rejected."""
        response = await async_client.post(
            "/api/v1/agents",
            json={"agent_name": ""},
        )
        assert response.status_code == 422  # Validation error
