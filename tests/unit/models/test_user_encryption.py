"""Tests for user encryption model fields and methods."""
import pytest

from models.user import User


class TestUserEncryptionFields:
    """Tests for User model encryption field validation."""

    def test_set_encryption_keys_valid(self):
        """User can set all encryption keys with valid data."""
        user = User(id="user_123")

        user.set_encryption_keys(
            public_key="aa" * 32,  # 64 hex chars
            encrypted_private_key="bb" * 48,  # Variable
            iv="cc" * 16,  # 32 hex chars
            tag="dd" * 16,  # 32 hex chars
            salt="ee" * 32,  # 64 hex chars
            recovery_encrypted_private_key="ff" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )

        assert user.has_encryption_keys is True
        assert user.public_key == "aa" * 32
        assert user.encrypted_private_key == "bb" * 48
        assert user.encryption_created_at is not None

    def test_set_encryption_keys_validates_public_key_length(self):
        """Public key must be exactly 64 hex characters."""
        user = User(id="user_123")

        with pytest.raises(ValueError, match="64 hex characters"):
            user.set_encryption_keys(
                public_key="aa" * 31,  # Too short
                encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 32,
                recovery_encrypted_private_key="ff" * 48,
                recovery_iv="11" * 16,
                recovery_tag="22" * 16,
                recovery_salt="33" * 32,
            )

    def test_set_encryption_keys_validates_iv_length(self):
        """IV must be exactly 32 hex characters."""
        user = User(id="user_123")

        with pytest.raises(ValueError, match="32 hex characters"):
            user.set_encryption_keys(
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

    def test_set_encryption_keys_validates_tag_length(self):
        """Auth tag must be exactly 32 hex characters."""
        user = User(id="user_123")

        with pytest.raises(ValueError, match="32 hex characters"):
            user.set_encryption_keys(
                public_key="aa" * 32,
                encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 8,  # Too short
                salt="ee" * 32,
                recovery_encrypted_private_key="ff" * 48,
                recovery_iv="11" * 16,
                recovery_tag="22" * 16,
                recovery_salt="33" * 32,
            )

    def test_set_encryption_keys_validates_salt_length(self):
        """Salt must be exactly 64 hex characters."""
        user = User(id="user_123")

        with pytest.raises(ValueError, match="64 hex characters"):
            user.set_encryption_keys(
                public_key="aa" * 32,
                encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 16,  # Too short
                recovery_encrypted_private_key="ff" * 48,
                recovery_iv="11" * 16,
                recovery_tag="22" * 16,
                recovery_salt="33" * 32,
            )

    def test_set_encryption_keys_validates_hex_format(self):
        """All fields must be valid hex strings."""
        user = User(id="user_123")

        with pytest.raises(ValueError, match="valid hex string"):
            user.set_encryption_keys(
                public_key="gg" * 32,  # Not valid hex
                encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 32,
                recovery_encrypted_private_key="ff" * 48,
                recovery_iv="11" * 16,
                recovery_tag="22" * 16,
                recovery_salt="33" * 32,
            )

    def test_set_encryption_keys_normalizes_to_lowercase(self):
        """All hex strings are normalized to lowercase."""
        user = User(id="user_123")

        user.set_encryption_keys(
            public_key="AA" * 32,  # Uppercase
            encrypted_private_key="BB" * 48,
            iv="CC" * 16,
            tag="DD" * 16,
            salt="EE" * 32,
            recovery_encrypted_private_key="FF" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )

        assert user.public_key == "aa" * 32
        assert user.encrypted_private_key == "bb" * 48


class TestUserEncryptionClear:
    """Tests for clearing user encryption keys."""

    def test_clear_encryption_keys(self):
        """User can clear all encryption keys."""
        user = User(id="user_123")
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

        user.clear_encryption_keys()

        assert user.has_encryption_keys is False
        assert user.public_key is None
        assert user.encrypted_private_key is None
        assert user.encrypted_private_key_iv is None
        assert user.encrypted_private_key_tag is None
        assert user.key_salt is None
        assert user.recovery_encrypted_private_key is None
        assert user.recovery_iv is None
        assert user.recovery_tag is None
        assert user.recovery_salt is None
        assert user.encryption_created_at is None


class TestUserEncryptionProperties:
    """Tests for user encryption helper properties."""

    def test_can_receive_encrypted_messages_without_key(self):
        """User cannot receive encrypted messages without public key."""
        user = User(id="user_123")

        assert user.can_receive_encrypted_messages is False

    def test_can_receive_encrypted_messages_with_key(self):
        """User can receive encrypted messages with public key."""
        user = User(id="user_123")
        user.public_key = "aa" * 32

        assert user.can_receive_encrypted_messages is True

    def test_encryption_key_info_excludes_private_keys(self):
        """encryption_key_info never includes private key data."""
        user = User(id="user_123")
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

        info = user.encryption_key_info

        assert "public_key" in info
        assert info["public_key"] == "aa" * 32
        assert info["has_encryption_keys"] is True
        assert "encryption_created_at" in info

        # These should NOT be in the info
        assert "encrypted_private_key" not in info
        assert "recovery_encrypted_private_key" not in info
        assert "salt" not in info
        assert "iv" not in info
        assert "tag" not in info

    def test_encryption_key_info_without_keys(self):
        """encryption_key_info works for user without keys."""
        user = User(id="user_123")

        info = user.encryption_key_info

        assert info["has_encryption_keys"] is False
        assert info["public_key"] is None
        assert info["encryption_created_at"] is None


class TestUserEncryptionGetters:
    """Tests for encryption key getter methods."""

    def test_get_encrypted_keys_for_unlock(self):
        """get_encrypted_keys_for_unlock returns key data for client unlock."""
        user = User(id="user_123")
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

        keys = user.get_encrypted_keys_for_unlock()

        assert keys is not None
        assert keys["public_key"] == "aa" * 32
        assert keys["encrypted_private_key"] == "bb" * 48
        assert keys["iv"] == "cc" * 16
        assert keys["tag"] == "dd" * 16
        assert keys["salt"] == "ee" * 32

    def test_get_encrypted_keys_for_unlock_without_keys(self):
        """get_encrypted_keys_for_unlock returns None without keys."""
        user = User(id="user_123")

        keys = user.get_encrypted_keys_for_unlock()

        assert keys is None

    def test_get_recovery_keys(self):
        """get_recovery_keys returns recovery key data."""
        user = User(id="user_123")
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

        recovery = user.get_recovery_keys()

        assert recovery is not None
        assert recovery["recovery_encrypted_private_key"] == "ff" * 48
        assert recovery["recovery_iv"] == "11" * 16
        assert recovery["recovery_tag"] == "22" * 16
        assert recovery["recovery_salt"] == "33" * 32

    def test_get_recovery_keys_without_keys(self):
        """get_recovery_keys returns None without keys."""
        user = User(id="user_123")

        recovery = user.get_recovery_keys()

        assert recovery is None


class TestUserEncryptionPersistence:
    """Tests for user encryption database persistence."""

    @pytest.mark.asyncio
    async def test_user_encryption_persistence(self, db_session):
        """User encryption keys persist to database."""
        from sqlalchemy import select

        user = User(id="user_persist_encryption")
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
        db_session.add(user)
        await db_session.flush()

        result = await db_session.execute(
            select(User).where(User.id == "user_persist_encryption")
        )
        fetched_user = result.scalar_one()

        assert fetched_user.has_encryption_keys is True
        assert fetched_user.public_key == "aa" * 32
        assert fetched_user.encrypted_private_key == "bb" * 48
        assert fetched_user.encryption_created_at is not None

    @pytest.mark.asyncio
    async def test_user_with_keys_fixture(self, test_user_with_keys):
        """test_user_with_keys fixture creates user with encryption keys."""
        assert test_user_with_keys.has_encryption_keys is True
        assert test_user_with_keys.public_key is not None
        assert test_user_with_keys.can_receive_encrypted_messages is True
