"""Tests for UserKeyService."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.services.user_key_service import (
    UserKeyService,
    UserKeyServiceError,
    KeysAlreadyExistError,
    KeysNotFoundError,
)
from models.user import User


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def user_without_keys():
    """User that hasn't set up encryption keys."""
    user = User(id="user_123")
    user.has_encryption_keys = False  # Explicitly set (DB would set default)
    return user


@pytest.fixture
def user_with_keys():
    """User with encryption keys already set up."""
    user = User(id="user_456")
    user.set_encryption_keys(
        public_key="aa" * 32,
        encrypted_private_key="bb" * 48,
        iv="cc" * 16,
        tag="dd" * 16,
        salt="ee" * 32,
        recovery_encrypted_private_key="ff" * 48,
        recovery_iv="11" * 16,
        recovery_tag="22" * 16,
        recovery_salt="33" * 32,
    )
    return user


@pytest.fixture
def valid_key_data():
    """Valid encryption key data for tests."""
    return {
        "public_key": "aa" * 32,
        "encrypted_private_key": "bb" * 48,
        "iv": "cc" * 16,
        "tag": "dd" * 16,
        "salt": "ee" * 32,
        "recovery_encrypted_private_key": "ff" * 48,
        "recovery_iv": "11" * 16,
        "recovery_tag": "22" * 16,
        "recovery_salt": "33" * 32,
    }


def mock_execute_result(user):
    """Create a mock execute result that returns the given user."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=user)
    return mock_result


# =============================================================================
# Test GetEncryptionStatus
# =============================================================================


class TestGetEncryptionStatus:
    """Tests for get_encryption_status method."""

    @pytest.mark.asyncio
    async def test_returns_status_for_user_without_keys(self, mock_db, user_without_keys):
        """Returns status showing no keys for user without encryption setup."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        status = await service.get_encryption_status("user_123")

        assert status["has_encryption_keys"] is False
        assert status["public_key"] is None
        assert status["encryption_created_at"] is None

    @pytest.mark.asyncio
    async def test_returns_status_for_user_with_keys(self, mock_db, user_with_keys):
        """Returns status showing keys exist for user with encryption setup."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        status = await service.get_encryption_status("user_456")

        assert status["has_encryption_keys"] is True
        assert status["public_key"] == "aa" * 32
        assert status["encryption_created_at"] is not None

    @pytest.mark.asyncio
    async def test_returns_default_for_nonexistent_user(self, mock_db):
        """Returns default status when user doesn't exist (handles race condition)."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = UserKeyService(mock_db)
        status = await service.get_encryption_status("nonexistent_user")

        # Should return default "not set up" status instead of raising error
        # This handles race condition where encryption status is checked
        # before user sync completes
        assert status["has_encryption_keys"] is False
        assert status["public_key"] is None
        assert status["encryption_created_at"] is None


# =============================================================================
# Test StoreEncryptionKeys
# =============================================================================


class TestStoreEncryptionKeys:
    """Tests for store_encryption_keys method."""

    @pytest.mark.asyncio
    async def test_stores_keys_for_new_user(self, mock_db, user_without_keys, valid_key_data):
        """Successfully stores encryption keys for user without existing keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        result = await service.store_encryption_keys(user_id="user_123", **valid_key_data)

        assert result.has_encryption_keys is True
        assert result.public_key == "aa" * 32
        mock_db.commit.assert_called_once()
        mock_db.add.assert_called()  # Audit log was added

    @pytest.mark.asyncio
    async def test_raises_error_if_keys_exist(self, mock_db, user_with_keys, valid_key_data):
        """Raises KeysAlreadyExistError when user already has keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        with pytest.raises(KeysAlreadyExistError):
            await service.store_encryption_keys(user_id="user_456", **valid_key_data)

    @pytest.mark.asyncio
    async def test_allows_overwrite_when_flag_set(self, mock_db, user_with_keys, valid_key_data):
        """Allows overwriting existing keys when allow_overwrite=True."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        result = await service.store_encryption_keys(user_id="user_456", allow_overwrite=True, **valid_key_data)

        assert result.has_encryption_keys is True
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_validates_public_key_length(self, mock_db, user_without_keys):
        """Raises error when public key has invalid length."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        with pytest.raises(UserKeyServiceError, match="Invalid key format"):
            await service.store_encryption_keys(
                user_id="user_123",
                public_key="aa" * 16,  # Too short
                encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 32,
                recovery_encrypted_private_key="ff" * 48,
                recovery_iv="11" * 16,
                recovery_tag="22" * 16,
                recovery_salt="33" * 32,
            )

    @pytest.mark.asyncio
    async def test_validates_iv_length(self, mock_db, user_without_keys):
        """Raises error when IV has invalid length."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        with pytest.raises(UserKeyServiceError, match="Invalid key format"):
            await service.store_encryption_keys(
                user_id="user_123",
                public_key="aa" * 32,
                encrypted_private_key="bb" * 48,
                iv="cc" * 8,  # Too short
                tag="dd" * 16,
                salt="ee" * 32,
                recovery_encrypted_private_key="ff" * 48,
                recovery_iv="11" * 16,
                recovery_tag="22" * 16,
                recovery_salt="33" * 32,
            )

    @pytest.mark.asyncio
    async def test_raises_error_for_nonexistent_user(self, mock_db, valid_key_data):
        """Raises error when user doesn't exist."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = UserKeyService(mock_db)
        with pytest.raises(UserKeyServiceError, match="not found"):
            await service.store_encryption_keys(user_id="nonexistent_user", **valid_key_data)

    @pytest.mark.asyncio
    async def test_creates_audit_log(self, mock_db, user_without_keys, valid_key_data):
        """Creates audit log entry when storing keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        await service.store_encryption_keys(user_id="user_123", **valid_key_data)

        # Verify audit log was added
        assert mock_db.add.call_count >= 1
        added_objects = [call.args[0] for call in mock_db.add.call_args_list]
        from models.audit_log import AuditLog

        audit_logs = [obj for obj in added_objects if isinstance(obj, AuditLog)]
        assert len(audit_logs) == 1


# =============================================================================
# Test GetEncryptionKeys
# =============================================================================


class TestGetEncryptionKeys:
    """Tests for get_encryption_keys method."""

    @pytest.mark.asyncio
    async def test_returns_encrypted_keys(self, mock_db, user_with_keys):
        """Returns encrypted key material for client-side decryption."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        keys = await service.get_encryption_keys("user_456")

        assert keys["public_key"] == "aa" * 32
        assert keys["encrypted_private_key"] == "bb" * 48
        assert keys["iv"] == "cc" * 16
        assert keys["tag"] == "dd" * 16
        assert keys["salt"] == "ee" * 32
        # Recovery keys not included by default
        assert "recovery_encrypted_private_key" not in keys

    @pytest.mark.asyncio
    async def test_includes_recovery_when_requested(self, mock_db, user_with_keys):
        """Includes recovery keys when include_recovery=True."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        keys = await service.get_encryption_keys("user_456", include_recovery=True)

        assert "recovery_encrypted_private_key" in keys
        assert keys["recovery_encrypted_private_key"] == "ff" * 48

    @pytest.mark.asyncio
    async def test_raises_error_if_no_keys(self, mock_db, user_without_keys):
        """Raises KeysNotFoundError when user has no encryption keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        with pytest.raises(KeysNotFoundError):
            await service.get_encryption_keys("user_123")

    @pytest.mark.asyncio
    async def test_raises_error_for_nonexistent_user(self, mock_db):
        """Raises error when user doesn't exist."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = UserKeyService(mock_db)
        with pytest.raises(UserKeyServiceError, match="not found"):
            await service.get_encryption_keys("nonexistent_user")


# =============================================================================
# Test GetRecoveryKeys
# =============================================================================


class TestGetRecoveryKeys:
    """Tests for get_recovery_keys method."""

    @pytest.mark.asyncio
    async def test_returns_recovery_encrypted_keys(self, mock_db, user_with_keys):
        """Returns recovery-encrypted keys for client-side recovery."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        keys = await service.get_recovery_keys("user_456")

        assert keys["public_key"] == "aa" * 32
        assert keys["encrypted_private_key"] == "ff" * 48  # Recovery version
        assert keys["iv"] == "11" * 16  # Recovery IV
        assert keys["tag"] == "22" * 16  # Recovery tag
        assert keys["salt"] == "33" * 32  # Recovery salt

    @pytest.mark.asyncio
    async def test_creates_audit_log(self, mock_db, user_with_keys):
        """Creates audit log entry for recovery attempt."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        await service.get_recovery_keys("user_456")

        # Verify audit log was added
        assert mock_db.add.call_count >= 1
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_error_if_no_keys(self, mock_db, user_without_keys):
        """Raises KeysNotFoundError when user has no encryption keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        with pytest.raises(KeysNotFoundError):
            await service.get_recovery_keys("user_123")


# =============================================================================
# Test DeleteEncryptionKeys
# =============================================================================


class TestDeleteEncryptionKeys:
    """Tests for delete_encryption_keys method."""

    @pytest.mark.asyncio
    async def test_clears_all_key_fields(self, mock_db, user_with_keys):
        """Clears all encryption key fields from user."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        await service.delete_encryption_keys("user_456")

        assert user_with_keys.has_encryption_keys is False
        assert user_with_keys.public_key is None
        assert user_with_keys.encrypted_private_key is None
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_audit_log(self, mock_db, user_with_keys):
        """Creates audit log entry for key deletion."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        await service.delete_encryption_keys("user_456")

        # Verify audit log was added
        assert mock_db.add.call_count >= 1

    @pytest.mark.asyncio
    async def test_noop_if_no_keys(self, mock_db, user_without_keys):
        """Does nothing if user has no keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        await service.delete_encryption_keys("user_123")

        # Should not have committed anything (no changes)
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_for_nonexistent_user(self, mock_db):
        """Raises error when user doesn't exist."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = UserKeyService(mock_db)
        with pytest.raises(UserKeyServiceError, match="not found"):
            await service.delete_encryption_keys("nonexistent_user")


# =============================================================================
# Test GetPublicKey
# =============================================================================


class TestGetPublicKey:
    """Tests for get_public_key method."""

    @pytest.mark.asyncio
    async def test_returns_public_key(self, mock_db, user_with_keys):
        """Returns public key for user with encryption setup."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_with_keys))

        service = UserKeyService(mock_db)
        public_key = await service.get_public_key("user_456")

        assert public_key == "aa" * 32

    @pytest.mark.asyncio
    async def test_returns_none_if_no_keys(self, mock_db, user_without_keys):
        """Returns None for user without encryption keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user_without_keys))

        service = UserKeyService(mock_db)
        public_key = await service.get_public_key("user_123")

        assert public_key is None

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent_user(self, mock_db):
        """Returns None for nonexistent user."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = UserKeyService(mock_db)
        public_key = await service.get_public_key("nonexistent_user")

        assert public_key is None
