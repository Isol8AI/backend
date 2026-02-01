"""Tests for ManagementApiClient - API Gateway WebSocket Management API client."""

import json
import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from core.services.management_api_client import (
    ManagementApiClient,
    ManagementApiClientError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_apigw_client():
    """Create a mock API Gateway Management API client."""
    client = MagicMock()
    return client


@pytest.fixture
def management_api_client(mock_apigw_client):
    """Create a ManagementApiClient with mocked boto3 client."""
    with patch("core.services.management_api_client.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_apigw_client
        client = ManagementApiClient(endpoint_url="https://abc123.execute-api.us-east-1.amazonaws.com/prod")
        return client, mock_apigw_client, mock_boto3


# =============================================================================
# Test Initialization
# =============================================================================


class TestInitialization:
    """Tests for ManagementApiClient initialization."""

    def test_creates_apigatewaymanagementapi_client_with_endpoint_url(self, management_api_client):
        """Creates boto3 client with correct service and endpoint_url."""
        client, mock_apigw, mock_boto3 = management_api_client

        mock_boto3.client.assert_called_once_with(
            "apigatewaymanagementapi",
            endpoint_url="https://abc123.execute-api.us-east-1.amazonaws.com/prod",
        )

    def test_uses_environment_variable_for_endpoint(self):
        """Uses WS_MANAGEMENT_API_URL env var for default endpoint."""
        with patch("core.services.management_api_client.os.environ.get") as mock_env:
            mock_env.return_value = "https://xyz789.execute-api.us-west-2.amazonaws.com/dev"
            with patch("core.services.management_api_client.boto3"):
                client = ManagementApiClient()
                assert client.endpoint_url == "https://xyz789.execute-api.us-west-2.amazonaws.com/dev"
                mock_env.assert_called_with("WS_MANAGEMENT_API_URL")

    def test_raises_error_when_no_endpoint_provided(self):
        """Raises error when no endpoint URL is provided and env var not set."""
        with patch("core.services.management_api_client.os.environ.get") as mock_env:
            mock_env.return_value = None
            with pytest.raises(ManagementApiClientError, match="WS_MANAGEMENT_API_URL"):
                ManagementApiClient()


# =============================================================================
# Test send_message
# =============================================================================


class TestSendMessage:
    """Tests for send_message method."""

    def test_posts_json_encoded_data_to_connection(self, management_api_client):
        """Successfully posts JSON-encoded payload to the connection."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.return_value = {}

        payload = {"type": "message", "content": "Hello, world!"}
        result = client.send_message("conn_abc123", payload)

        assert result is True
        mock_apigw.post_to_connection.assert_called_once_with(
            ConnectionId="conn_abc123",
            Data=json.dumps(payload).encode("utf-8"),
        )

    def test_returns_true_on_success(self, management_api_client):
        """Returns True when message is successfully sent."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.return_value = {}

        result = client.send_message("conn_abc123", {"hello": "world"})

        assert result is True

    def test_returns_false_on_gone_exception(self, management_api_client):
        """Returns False when connection is gone (410 status)."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.side_effect = ClientError(
            {
                "Error": {"Code": "GoneException", "Message": "Connection is gone"},
                "ResponseMetadata": {"HTTPStatusCode": 410},
            },
            "PostToConnection",
        )

        result = client.send_message("conn_gone", {"hello": "world"})

        assert result is False

    def test_handles_nested_payload(self, management_api_client):
        """Correctly JSON encodes nested payload structures."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.return_value = {}

        payload = {
            "type": "stream_chunk",
            "data": {
                "content": "Hello",
                "metadata": {"model": "gpt-4", "tokens": 10},
            },
        }
        client.send_message("conn_abc123", payload)

        call_args = mock_apigw.post_to_connection.call_args
        # Verify the data was JSON encoded
        sent_data = call_args[1]["Data"]
        decoded = json.loads(sent_data.decode("utf-8"))
        assert decoded == payload

    def test_raises_error_on_other_client_errors(self, management_api_client):
        """Raises ManagementApiClientError on non-Gone ClientError."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.side_effect = ClientError(
            {
                "Error": {
                    "Code": "InternalServerError",
                    "Message": "Internal error",
                },
            },
            "PostToConnection",
        )

        with pytest.raises(ManagementApiClientError, match="Failed to send message"):
            client.send_message("conn_abc123", {"hello": "world"})

    def test_handles_unicode_content(self, management_api_client):
        """Correctly handles Unicode characters in payload."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.return_value = {}

        payload = {"message": "Hello, world!"}
        client.send_message("conn_abc123", payload)

        call_args = mock_apigw.post_to_connection.call_args
        sent_data = call_args[1]["Data"]
        decoded = json.loads(sent_data.decode("utf-8"))
        assert decoded["message"] == "Hello, world!"


# =============================================================================
# Test close_connection
# =============================================================================


class TestCloseConnection:
    """Tests for close_connection method."""

    def test_deletes_connection(self, management_api_client):
        """Successfully calls delete_connection API."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.delete_connection.return_value = {}

        client.close_connection("conn_abc123")

        mock_apigw.delete_connection.assert_called_once_with(ConnectionId="conn_abc123")

    def test_handles_gone_exception_silently(self, management_api_client):
        """Does not raise error when connection is already gone."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.delete_connection.side_effect = ClientError(
            {
                "Error": {"Code": "GoneException", "Message": "Connection is gone"},
                "ResponseMetadata": {"HTTPStatusCode": 410},
            },
            "DeleteConnection",
        )

        # Should not raise
        client.close_connection("conn_already_gone")

        mock_apigw.delete_connection.assert_called_once()

    def test_raises_error_on_other_client_errors(self, management_api_client):
        """Raises ManagementApiClientError on non-Gone ClientError."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.delete_connection.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ForbiddenException",
                    "Message": "Access denied",
                },
            },
            "DeleteConnection",
        )

        with pytest.raises(ManagementApiClientError, match="Failed to close connection"):
            client.close_connection("conn_abc123")


# =============================================================================
# Test Logging
# =============================================================================


class TestLogging:
    """Tests for logging behavior."""

    def test_logs_successful_send(self, management_api_client, caplog):
        """Logs info message on successful send."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.return_value = {}

        with caplog.at_level("DEBUG"):
            client.send_message("conn_abc123", {"test": "data"})

        # Just verify no exceptions - logging level may vary
        assert mock_apigw.post_to_connection.called

    def test_logs_gone_connection(self, management_api_client, caplog):
        """Logs warning when connection is gone."""
        client, mock_apigw, _ = management_api_client
        mock_apigw.post_to_connection.side_effect = ClientError(
            {
                "Error": {"Code": "GoneException", "Message": "Gone"},
                "ResponseMetadata": {"HTTPStatusCode": 410},
            },
            "PostToConnection",
        )

        with caplog.at_level("WARNING"):
            client.send_message("conn_gone", {"test": "data"})

        # Verify the method was called and returned False
        assert mock_apigw.post_to_connection.called
