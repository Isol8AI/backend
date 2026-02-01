"""Tests for ConnectionService - DynamoDB WebSocket connection state management."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from botocore.exceptions import ClientError

from core.services.connection_service import (
    ConnectionService,
    ConnectionServiceError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_dynamodb_client():
    """Create a mock DynamoDB client."""
    client = MagicMock()
    return client


@pytest.fixture
def connection_service(mock_dynamodb_client):
    """Create a ConnectionService with mocked DynamoDB client."""
    with patch("core.services.connection_service.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_dynamodb_client
        service = ConnectionService(table_name="test-connections")
        return service, mock_dynamodb_client


# =============================================================================
# Test store_connection
# =============================================================================


class TestStoreConnection:
    """Tests for store_connection method."""

    def test_stores_connection_with_org_id(self, connection_service):
        """Successfully stores connection with user_id and org_id."""
        service, mock_client = connection_service
        mock_client.put_item.return_value = {}

        service.store_connection(
            connection_id="conn_abc123",
            user_id="user_456",
            org_id="org_789",
        )

        mock_client.put_item.assert_called_once()
        call_args = mock_client.put_item.call_args
        item = call_args[1]["Item"]

        assert item["connectionId"]["S"] == "conn_abc123"
        assert item["userId"]["S"] == "user_456"
        assert item["orgId"]["S"] == "org_789"
        assert "connectedAt" in item
        # Verify connectedAt is ISO format
        datetime.fromisoformat(item["connectedAt"]["S"])

    def test_stores_connection_without_org_id(self, connection_service):
        """Successfully stores connection with None org_id as empty string."""
        service, mock_client = connection_service
        mock_client.put_item.return_value = {}

        service.store_connection(
            connection_id="conn_abc123",
            user_id="user_456",
            org_id=None,
        )

        mock_client.put_item.assert_called_once()
        call_args = mock_client.put_item.call_args
        item = call_args[1]["Item"]

        assert item["connectionId"]["S"] == "conn_abc123"
        assert item["userId"]["S"] == "user_456"
        assert item["orgId"]["S"] == ""  # Empty string for None

    def test_uses_correct_table_name(self, connection_service):
        """Verifies the correct table name is used."""
        service, mock_client = connection_service
        mock_client.put_item.return_value = {}

        service.store_connection(
            connection_id="conn_abc123",
            user_id="user_456",
            org_id=None,
        )

        call_args = mock_client.put_item.call_args
        assert call_args[1]["TableName"] == "test-connections"

    def test_raises_error_on_dynamodb_failure(self, connection_service):
        """Raises ConnectionServiceError on DynamoDB ClientError."""
        service, mock_client = connection_service
        mock_client.put_item.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "DynamoDB error"}},
            "PutItem",
        )

        with pytest.raises(ConnectionServiceError, match="Failed to store connection"):
            service.store_connection(
                connection_id="conn_abc123",
                user_id="user_456",
                org_id=None,
            )


# =============================================================================
# Test get_connection
# =============================================================================


class TestGetConnection:
    """Tests for get_connection method."""

    def test_returns_user_data_for_existing_connection(self, connection_service):
        """Returns user_id and org_id for existing connection."""
        service, mock_client = connection_service
        mock_client.get_item.return_value = {
            "Item": {
                "connectionId": {"S": "conn_abc123"},
                "userId": {"S": "user_456"},
                "orgId": {"S": "org_789"},
                "connectedAt": {"S": "2024-01-15T10:30:00"},
            }
        }

        result = service.get_connection("conn_abc123")

        assert result is not None
        assert result["user_id"] == "user_456"
        assert result["org_id"] == "org_789"

    def test_returns_none_org_id_for_empty_string(self, connection_service):
        """Returns None for org_id when stored as empty string."""
        service, mock_client = connection_service
        mock_client.get_item.return_value = {
            "Item": {
                "connectionId": {"S": "conn_abc123"},
                "userId": {"S": "user_456"},
                "orgId": {"S": ""},  # Empty string
                "connectedAt": {"S": "2024-01-15T10:30:00"},
            }
        }

        result = service.get_connection("conn_abc123")

        assert result is not None
        assert result["user_id"] == "user_456"
        assert result["org_id"] is None  # Should be None, not empty string

    def test_returns_none_for_nonexistent_connection(self, connection_service):
        """Returns None when connection is not found."""
        service, mock_client = connection_service
        mock_client.get_item.return_value = {}  # No Item key

        result = service.get_connection("conn_nonexistent")

        assert result is None

    def test_uses_correct_key(self, connection_service):
        """Verifies the correct key is used for lookup."""
        service, mock_client = connection_service
        mock_client.get_item.return_value = {}

        service.get_connection("conn_abc123")

        mock_client.get_item.assert_called_once()
        call_args = mock_client.get_item.call_args
        assert call_args[1]["TableName"] == "test-connections"
        assert call_args[1]["Key"] == {"connectionId": {"S": "conn_abc123"}}

    def test_raises_error_on_dynamodb_failure(self, connection_service):
        """Raises ConnectionServiceError on DynamoDB ClientError."""
        service, mock_client = connection_service
        mock_client.get_item.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "DynamoDB error"}},
            "GetItem",
        )

        with pytest.raises(ConnectionServiceError, match="Failed to get connection"):
            service.get_connection("conn_abc123")


# =============================================================================
# Test delete_connection
# =============================================================================


class TestDeleteConnection:
    """Tests for delete_connection method."""

    def test_deletes_connection_from_dynamodb(self, connection_service):
        """Successfully deletes connection from DynamoDB."""
        service, mock_client = connection_service
        mock_client.delete_item.return_value = {}

        service.delete_connection("conn_abc123")

        mock_client.delete_item.assert_called_once()
        call_args = mock_client.delete_item.call_args
        assert call_args[1]["TableName"] == "test-connections"
        assert call_args[1]["Key"] == {"connectionId": {"S": "conn_abc123"}}

    def test_noop_for_nonexistent_connection(self, connection_service):
        """Does not raise error for nonexistent connection."""
        service, mock_client = connection_service
        # DynamoDB delete_item doesn't error if item doesn't exist
        mock_client.delete_item.return_value = {}

        # Should not raise
        service.delete_connection("conn_nonexistent")

        mock_client.delete_item.assert_called_once()

    def test_raises_error_on_dynamodb_failure(self, connection_service):
        """Raises ConnectionServiceError on DynamoDB ClientError."""
        service, mock_client = connection_service
        mock_client.delete_item.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "DynamoDB error"}},
            "DeleteItem",
        )

        with pytest.raises(ConnectionServiceError, match="Failed to delete connection"):
            service.delete_connection("conn_abc123")


# =============================================================================
# Test Default Table Name
# =============================================================================


class TestDefaultTableName:
    """Tests for default table name from environment."""

    def test_uses_environment_variable(self):
        """Uses WS_CONNECTIONS_TABLE env var for default table name."""
        with patch("core.services.connection_service.os.environ.get") as mock_env:
            mock_env.return_value = "my-custom-table"
            with patch("core.services.connection_service.boto3"):
                service = ConnectionService()
                assert service.table_name == "my-custom-table"
                mock_env.assert_called_with(
                    "WS_CONNECTIONS_TABLE", "isol8-websocket-connections"
                )

    def test_uses_default_if_env_not_set(self):
        """Uses default table name if env var is not set."""
        with patch("core.services.connection_service.os.environ.get") as mock_env:
            mock_env.return_value = "isol8-websocket-connections"
            with patch("core.services.connection_service.boto3"):
                service = ConnectionService()
                assert service.table_name == "isol8-websocket-connections"
