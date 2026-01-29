"""
Bedrock client factory that supports multiple credential sources.
"""

import boto3
from botocore.config import Config as BotoConfig

from core.services.credential_service import AWSCredentials


class BedrockClientFactory:
    """Factory for creating Bedrock clients with appropriate credentials."""

    @staticmethod
    def create_client(
        credentials: AWSCredentials,
        timeout: float = 120.0,
    ):
        """
        Create a Bedrock runtime client.

        Args:
            credentials: Resolved AWS credentials
            timeout: Request timeout in seconds

        Returns:
            boto3 bedrock-runtime client
        """
        boto_config = BotoConfig(
            read_timeout=int(timeout),
            connect_timeout=10,
            retries={"max_attempts": 2},
        )

        if credentials.is_custom:
            # Use explicit credentials
            return boto3.client(
                "bedrock-runtime",
                region_name=credentials.region,
                aws_access_key_id=credentials.access_key_id,
                aws_secret_access_key=credentials.secret_access_key,
                config=boto_config,
            )
        else:
            # Use IAM role (no explicit credentials)
            return boto3.client(
                "bedrock-runtime",
                region_name=credentials.region,
                config=boto_config,
            )
