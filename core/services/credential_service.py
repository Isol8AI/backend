"""
Credential resolution service for AWS Bedrock access.

Resolves credentials in order:
1. User's custom credentials (from Clerk privateMetadata)
2. Org's custom credentials (from Clerk privateMetadata)
3. Default backend IAM role
"""

import base64
import logging
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AWSCredentials:
    """Resolved AWS credentials for Bedrock access."""

    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: str = "us-east-1"
    is_custom: bool = False  # True if user/org provided, False if IAM role


class CredentialService:
    """Service for resolving and decrypting AWS credentials."""

    _encryption_key: Optional[bytes] = None

    @classmethod
    def _get_encryption_key(cls) -> bytes:
        """Get the credential encryption key."""
        if cls._encryption_key is None:
            if not settings.CREDENTIAL_ENCRYPTION_KEY:
                raise RuntimeError("CREDENTIAL_ENCRYPTION_KEY not configured")
            cls._encryption_key = base64.b64decode(settings.CREDENTIAL_ENCRYPTION_KEY)
        return cls._encryption_key

    @classmethod
    def decrypt_secret(cls, encrypted_value: str) -> str:
        """Decrypt a secret stored in Clerk metadata."""
        if not encrypted_value.startswith("encrypted:"):
            return encrypted_value  # Not encrypted (legacy)

        encrypted_data = base64.b64decode(encrypted_value[10:])
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        aesgcm = AESGCM(cls._get_encryption_key())
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    @classmethod
    def encrypt_secret(cls, plaintext: str) -> str:
        """Encrypt a secret for storage in Clerk metadata."""
        import os

        nonce = os.urandom(12)
        aesgcm = AESGCM(cls._get_encryption_key())
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        encrypted_data = base64.b64encode(nonce + ciphertext).decode("utf-8")
        return f"encrypted:{encrypted_data}"

    @classmethod
    def resolve_credentials(
        cls,
        user_metadata: Optional[dict] = None,
        org_metadata: Optional[dict] = None,
    ) -> AWSCredentials:
        """
        Resolve AWS credentials from user/org metadata or fall back to IAM role.

        Priority:
        1. User's custom credentials (if enabled)
        2. Org's custom credentials (if enabled)
        3. Default IAM role (no explicit credentials)
        """
        # Check user credentials
        if user_metadata:
            user_creds = user_metadata.get("aws_credentials", {})
            if user_creds.get("enabled"):
                logger.info("Using user's custom AWS credentials")
                return AWSCredentials(
                    access_key_id=user_creds.get("access_key_id"),
                    secret_access_key=cls.decrypt_secret(
                        user_creds.get("secret_access_key", "")
                    ),
                    region=user_creds.get("region", "us-east-1"),
                    is_custom=True,
                )

        # Check org credentials
        if org_metadata:
            org_creds = org_metadata.get("aws_credentials", {})
            if org_creds.get("enabled"):
                logger.info("Using org's custom AWS credentials")
                return AWSCredentials(
                    access_key_id=org_creds.get("access_key_id"),
                    secret_access_key=cls.decrypt_secret(
                        org_creds.get("secret_access_key", "")
                    ),
                    region=org_creds.get("region", "us-east-1"),
                    is_custom=True,
                )

        # Fall back to IAM role
        logger.debug("Using default IAM role credentials")
        return AWSCredentials(
            region=settings.AWS_REGION,
            is_custom=False,
        )
