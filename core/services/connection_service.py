"""
Connection Service - manages WebSocket connection state in DynamoDB.

This service stores the mapping between API Gateway WebSocket connectionIds
and authenticated user identities. The API Gateway Management API only knows
connectionId, so we need to track user_id and org_id separately.

Architecture:
- INBOUND: Client WebSocket -> API Gateway -> HTTP POST -> VPC Link -> ALB -> EC2
- OUTBOUND: EC2 -> POST to Management API -> API Gateway -> Client WebSocket

DynamoDB Table Schema:
- connectionId (S) - partition key
- userId (S) - the authenticated user's ID
- orgId (S) - organization ID (empty string if None)
- connectedAt (S) - ISO timestamp of connection

Note: Table creation is handled by Terraform, not this service.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ConnectionServiceError(Exception):
    """Base exception for connection service errors."""

    pass


class ConnectionService:
    """
    Service for managing WebSocket connection state in DynamoDB.

    Stores the mapping between API Gateway connectionId and user identity,
    enabling the backend to know which user a WebSocket connection belongs to.
    """

    def __init__(self, table_name: Optional[str] = None):
        """
        Initialize the connection service.

        Args:
            table_name: DynamoDB table name. If not provided, uses
                       WS_CONNECTIONS_TABLE env var (default: "isol8-websocket-connections")
        """
        self.table_name = table_name or os.environ.get(
            "WS_CONNECTIONS_TABLE", "isol8-websocket-connections"
        )
        self._client = boto3.client("dynamodb")

    def store_connection(
        self,
        connection_id: str,
        user_id: str,
        org_id: Optional[str],
    ) -> None:
        """
        Store a new WebSocket connection mapping.

        Called on $connect when a user establishes a WebSocket connection.

        Args:
            connection_id: API Gateway WebSocket connection ID
            user_id: Authenticated user's ID from Clerk JWT
            org_id: Organization ID from Clerk JWT (None for personal context)

        Raises:
            ConnectionServiceError: If DynamoDB operation fails
        """
        connected_at = datetime.now(timezone.utc).isoformat()

        item = {
            "connectionId": {"S": connection_id},
            "userId": {"S": user_id},
            "orgId": {"S": org_id or ""},  # Empty string for None
            "connectedAt": {"S": connected_at},
        }

        try:
            self._client.put_item(
                TableName=self.table_name,
                Item=item,
            )
            logger.info(
                "Stored connection %s for user %s (org: %s)",
                connection_id,
                user_id,
                org_id or "personal",
            )
        except ClientError as e:
            logger.error(
                "Failed to store connection %s: %s",
                connection_id,
                e.response["Error"]["Message"],
            )
            raise ConnectionServiceError(
                f"Failed to store connection {connection_id}: {e.response['Error']['Message']}"
            ) from e

    def get_connection(self, connection_id: str) -> Optional[dict]:
        """
        Get the user identity for a WebSocket connection.

        Args:
            connection_id: API Gateway WebSocket connection ID

        Returns:
            Dict with user_id and org_id (org_id is None for personal context),
            or None if connection not found

        Raises:
            ConnectionServiceError: If DynamoDB operation fails
        """
        try:
            response = self._client.get_item(
                TableName=self.table_name,
                Key={"connectionId": {"S": connection_id}},
            )
        except ClientError as e:
            logger.error(
                "Failed to get connection %s: %s",
                connection_id,
                e.response["Error"]["Message"],
            )
            raise ConnectionServiceError(
                f"Failed to get connection {connection_id}: {e.response['Error']['Message']}"
            ) from e

        if "Item" not in response:
            logger.debug("Connection %s not found", connection_id)
            return None

        item = response["Item"]
        org_id_value = item["orgId"]["S"]

        return {
            "user_id": item["userId"]["S"],
            "org_id": org_id_value if org_id_value else None,  # Convert empty string to None
        }

    def delete_connection(self, connection_id: str) -> None:
        """
        Delete a WebSocket connection mapping.

        Called on $disconnect when a user's WebSocket connection is closed.

        Args:
            connection_id: API Gateway WebSocket connection ID

        Raises:
            ConnectionServiceError: If DynamoDB operation fails

        Note:
            This is a no-op if the connection doesn't exist (DynamoDB behavior).
        """
        try:
            self._client.delete_item(
                TableName=self.table_name,
                Key={"connectionId": {"S": connection_id}},
            )
            logger.info("Deleted connection %s", connection_id)
        except ClientError as e:
            logger.error(
                "Failed to delete connection %s: %s",
                connection_id,
                e.response["Error"]["Message"],
            )
            raise ConnectionServiceError(
                f"Failed to delete connection {connection_id}: {e.response['Error']['Message']}"
            ) from e
