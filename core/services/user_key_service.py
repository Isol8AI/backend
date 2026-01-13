"""
User Key Service - manages user encryption keys.

Security Note:
- This service NEVER sees plaintext private keys
- All private key data is encrypted client-side before reaching the server
- The server stores encrypted blobs and metadata only
"""
import logging
from typing import Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.audit_log import AuditLog, AuditEventType

logger = logging.getLogger(__name__)


class UserKeyServiceError(Exception):
    """Base exception for user key service errors."""
    pass


class KeysAlreadyExistError(UserKeyServiceError):
    """User already has encryption keys."""
    pass


class KeysNotFoundError(UserKeyServiceError):
    """User does not have encryption keys."""
    pass


class UserKeyService:
    """
    Service for managing user encryption keys.

    All operations work with encrypted key material only.
    The server cannot decrypt private keys.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_encryption_status(self, user_id: str) -> dict:
        """
        Get user's encryption status.

        Returns:
            Dict with has_encryption_keys, public_key, encryption_created_at

        Raises:
            UserKeyServiceError: If user not found
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserKeyServiceError(f"User {user_id} not found")

        return {
            "has_encryption_keys": user.has_encryption_keys,
            "public_key": user.public_key,
            "encryption_created_at": user.encryption_created_at,
        }

    async def store_encryption_keys(
        self,
        user_id: str,
        public_key: str,
        encrypted_private_key: str,
        iv: str,
        tag: str,
        salt: str,
        recovery_encrypted_private_key: str,
        recovery_iv: str,
        recovery_tag: str,
        recovery_salt: str,
        allow_overwrite: bool = False,
    ) -> User:
        """
        Store user's encryption keys.

        All private key material is already encrypted client-side.
        Server stores but cannot decrypt.

        Args:
            user_id: The user's ID
            public_key: Hex-encoded X25519 public key
            encrypted_private_key: Passcode-encrypted private key
            iv: AES-GCM IV for passcode encryption
            tag: AES-GCM auth tag
            salt: Argon2id salt for passcode derivation
            recovery_*: Same but encrypted with recovery code
            allow_overwrite: If True, allow replacing existing keys

        Returns:
            Updated User object

        Raises:
            UserKeyServiceError: If user not found or invalid format
            KeysAlreadyExistError: If user has keys and allow_overwrite is False
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserKeyServiceError(f"User {user_id} not found")

        if user.has_encryption_keys and not allow_overwrite:
            raise KeysAlreadyExistError(
                f"User {user_id} already has encryption keys"
            )

        try:
            user.set_encryption_keys(
                public_key=public_key,
                encrypted_private_key=encrypted_private_key,
                iv=iv,
                tag=tag,
                salt=salt,
                recovery_encrypted_private_key=recovery_encrypted_private_key,
                recovery_iv=recovery_iv,
                recovery_tag=recovery_tag,
                recovery_salt=recovery_salt,
            )
        except ValueError as e:
            raise UserKeyServiceError(f"Invalid key format: {e}")

        # Choose audit event type based on whether this is initial setup or recovery
        audit_event = AuditEventType.USER_KEYS_CREATED
        if allow_overwrite:
            audit_event = AuditEventType.USER_KEYS_RECOVERED

        audit_log = AuditLog.create(
            id=str(uuid4()),
            event_type=audit_event,
            actor_user_id=user_id,
        )
        self.db.add(audit_log)

        await self.db.commit()
        await self.db.refresh(user)

        logger.info("Stored encryption keys for user %s", user_id)
        return user

    async def get_encryption_keys(
        self,
        user_id: str,
        include_recovery: bool = False,
    ) -> dict:
        """
        Get user's encrypted keys for client-side decryption.

        Args:
            user_id: The user's ID
            include_recovery: If True, include recovery-encrypted keys

        Returns:
            Dict with encrypted key material

        Raises:
            UserKeyServiceError: If user not found
            KeysNotFoundError: If user has no encryption keys
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserKeyServiceError(f"User {user_id} not found")

        if not user.has_encryption_keys:
            raise KeysNotFoundError(f"User {user_id} has no encryption keys")

        result = {
            "public_key": user.public_key,
            "encrypted_private_key": user.encrypted_private_key,
            "iv": user.encrypted_private_key_iv,
            "tag": user.encrypted_private_key_tag,
            "salt": user.key_salt,
        }

        if include_recovery:
            result.update({
                "recovery_encrypted_private_key": user.recovery_encrypted_private_key,
                "recovery_iv": user.recovery_iv,
                "recovery_tag": user.recovery_tag,
                "recovery_salt": user.recovery_salt,
            })

        return result

    async def get_recovery_keys(self, user_id: str) -> dict:
        """
        Get user's recovery-encrypted keys.

        This is logged as it may indicate the user lost their passcode.

        Args:
            user_id: The user's ID

        Returns:
            Dict with recovery-encrypted key material

        Raises:
            UserKeyServiceError: If user not found
            KeysNotFoundError: If user has no encryption keys
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserKeyServiceError(f"User {user_id} not found")

        if not user.has_encryption_keys:
            raise KeysNotFoundError(f"User {user_id} has no encryption keys")

        # Log recovery attempt for audit trail
        audit_log = AuditLog.create(
            id=str(uuid4()),
            event_type=AuditEventType.USER_KEYS_RECOVERED,
            actor_user_id=user_id,
            event_data={"method": "recovery_code"},
        )
        self.db.add(audit_log)
        await self.db.commit()

        return {
            "public_key": user.public_key,
            "encrypted_private_key": user.recovery_encrypted_private_key,
            "iv": user.recovery_iv,
            "tag": user.recovery_tag,
            "salt": user.recovery_salt,
        }

    async def delete_encryption_keys(self, user_id: str) -> None:
        """
        Delete user's encryption keys.

        WARNING: This makes all encrypted messages unrecoverable!

        Args:
            user_id: The user's ID

        Raises:
            UserKeyServiceError: If user not found
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserKeyServiceError(f"User {user_id} not found")

        if not user.has_encryption_keys:
            # Nothing to delete
            return

        user.clear_encryption_keys()

        audit_log = AuditLog.create(
            id=str(uuid4()),
            event_type=AuditEventType.USER_KEYS_DELETED,
            actor_user_id=user_id,
        )
        self.db.add(audit_log)
        await self.db.commit()

        logger.warning("Deleted encryption keys for user %s", user_id)

    async def get_public_key(self, user_id: str) -> Optional[str]:
        """
        Get user's public key for encrypting data TO them.

        This is used for key distribution (e.g., sharing org keys with members).

        Args:
            user_id: The user's ID

        Returns:
            Hex-encoded public key, or None if user has no keys
        """
        user = await self.get_user(user_id)
        if not user or not user.has_encryption_keys:
            return None
        return user.public_key
