"""Tests for ChatService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from core.services.chat_service import ChatService, StorageKeyNotFoundError
from core.crypto import EncryptedPayload
from models.user import User
from models.session import Session
from models.message import MessageRole
from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = AsyncMock()
    db.commit = AsyncMock()
    db.add = MagicMock()
    db.refresh = AsyncMock()
    return db


def mock_execute_result(item):
    """Create a mock execute result that returns the given item."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=item)
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[] if item is None else [item])))
    return mock_result


def mock_execute_result_list(items):
    """Create a mock execute result that returns a list."""
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all = MagicMock(return_value=items)
    mock_result.scalars = MagicMock(return_value=mock_scalars)
    return mock_result


def create_test_encrypted_payload():
    """Create a valid test encrypted payload using bytes."""
    return EncryptedPayload(
        ephemeral_public_key=bytes.fromhex("aa" * 32),
        iv=bytes.fromhex("bb" * 16),
        ciphertext=bytes.fromhex("cc" * 32),
        auth_tag=bytes.fromhex("dd" * 16),
        hkdf_salt=bytes.fromhex("ee" * 32),
    )


# =============================================================================
# Test Enclave Info
# =============================================================================

class TestEnclaveInfo:
    """Tests for enclave information methods."""

    def test_get_enclave_public_key(self, mock_db):
        """Returns enclave's public key as hex."""
        with patch('core.services.chat_service.get_enclave') as mock_get:
            mock_enclave = MagicMock()
            mock_enclave.get_transport_public_key.return_value = "ff" * 32
            mock_get.return_value = mock_enclave

            service = ChatService(mock_db)
            key = service.get_enclave_public_key()

            assert key == "ff" * 32
            mock_enclave.get_transport_public_key.assert_called_once()

    def test_get_enclave_info(self, mock_db):
        """Returns enclave info dict."""
        with patch('core.services.chat_service.get_enclave') as mock_get:
            mock_enclave = MagicMock()
            mock_info = MagicMock()
            mock_info.to_hex_dict.return_value = {
                "enclave_public_key": "ff" * 32,
                "attestation_document": None,
            }
            mock_enclave.get_info.return_value = mock_info
            mock_get.return_value = mock_enclave

            service = ChatService(mock_db)
            info = service.get_enclave_info()

            assert info["enclave_public_key"] == "ff" * 32
            assert info["attestation_document"] is None


# =============================================================================
# Test Session Management
# =============================================================================

class TestSessionManagement:
    """Tests for session creation and retrieval."""

    @pytest.mark.asyncio
    async def test_create_session_personal(self, mock_db):
        """Creates personal session (no org_id)."""
        user = User(id="user_123")
        user.has_encryption_keys = True
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user))

        service = ChatService(mock_db)
        session = await service.create_session(
            user_id="user_123",
            name="Test Chat",
        )

        assert session.user_id == "user_123"
        assert session.name == "Test Chat"
        assert session.org_id is None
        mock_db.add.assert_called()
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_create_session_org(self, mock_db):
        """Creates org session."""
        user = User(id="user_123")
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),  # User lookup
            mock_execute_result(membership),  # Membership lookup
        ])

        service = ChatService(mock_db)
        session = await service.create_session(
            user_id="user_123",
            name="Org Chat",
            org_id="org_456",
        )

        assert session.user_id == "user_123"
        assert session.org_id == "org_456"

    @pytest.mark.asyncio
    async def test_create_session_user_not_found(self, mock_db):
        """Raises error when user not found."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ChatService(mock_db)
        with pytest.raises(ValueError, match="User .* not found"):
            await service.create_session(user_id="nonexistent", name="Test")

    @pytest.mark.asyncio
    async def test_create_session_not_member(self, mock_db):
        """Raises error when user not member of org."""
        user = User(id="user_123")
        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),  # User exists
            mock_execute_result(None),  # No membership
        ])

        service = ChatService(mock_db)
        with pytest.raises(ValueError, match="not a member"):
            await service.create_session(
                user_id="user_123",
                name="Test",
                org_id="org_456",
            )

    @pytest.mark.asyncio
    async def test_get_session(self, mock_db):
        """Gets session with ownership verification."""
        session = Session(
            id="sess_123",
            user_id="user_123",
            org_id=None,
            name="Test",
        )
        mock_db.execute = AsyncMock(return_value=mock_execute_result(session))

        service = ChatService(mock_db)
        result = await service.get_session(
            session_id="sess_123",
            user_id="user_123",
        )

        assert result.id == "sess_123"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_db):
        """Returns None when session not found."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ChatService(mock_db)
        result = await service.get_session(
            session_id="nonexistent",
            user_id="user_123",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_db):
        """Lists user's sessions."""
        sessions = [
            Session(id="s1", user_id="user_123", name="Chat 1"),
            Session(id="s2", user_id="user_123", name="Chat 2"),
        ]
        mock_db.execute = AsyncMock(return_value=mock_execute_result_list(sessions))

        service = ChatService(mock_db)
        result = await service.list_sessions(user_id="user_123")

        assert len(result) == 2


# =============================================================================
# Test Message Operations
# =============================================================================

class TestMessageOperations:
    """Tests for encrypted message storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_encrypted_message(self, mock_db):
        """Stores encrypted message."""
        payload = create_test_encrypted_payload()

        service = ChatService(mock_db)
        message = await service.store_encrypted_message(
            session_id="sess_123",
            role=MessageRole.USER,
            encrypted_payload=payload,
            model_used="test-model",
        )

        assert message.session_id == "sess_123"
        assert message.role == "user"
        assert message.ephemeral_public_key == "aa" * 32
        mock_db.add.assert_called()
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_store_assistant_message_with_tokens(self, mock_db):
        """Stores assistant message with token counts."""
        payload = create_test_encrypted_payload()

        service = ChatService(mock_db)
        message = await service.store_encrypted_message(
            session_id="sess_123",
            role=MessageRole.ASSISTANT,
            encrypted_payload=payload,
            model_used="qwen/qwen-2.5-72b",
            input_tokens=100,
            output_tokens=50,
        )

        assert message.role == "assistant"
        assert message.model_used == "qwen/qwen-2.5-72b"
        assert message.input_tokens == 100
        assert message.output_tokens == 50

    @pytest.mark.asyncio
    async def test_get_session_messages(self, mock_db):
        """Gets encrypted messages for session."""
        session = Session(
            id="sess_123",
            user_id="user_123",
            name="Test",
        )
        messages = [
            MagicMock(
                id="msg1",
                session_id="sess_123",
                role="user",
                ephemeral_public_key="aa" * 32,
                iv="bb" * 16,
                ciphertext="cc" * 32,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
                created_at=datetime.utcnow(),
            ),
        ]

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(session),  # Session lookup
            mock_execute_result_list(messages),  # Messages
        ])

        service = ChatService(mock_db)
        result = await service.get_session_messages(
            session_id="sess_123",
            user_id="user_123",
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_session_messages_not_found(self, mock_db):
        """Raises error when session not found."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ChatService(mock_db)
        with pytest.raises(ValueError, match="not found"):
            await service.get_session_messages(
                session_id="nonexistent",
                user_id="user_123",
            )


# =============================================================================
# Test Key Resolution
# =============================================================================

class TestKeyResolution:
    """Tests for getting public keys for encryption."""

    @pytest.mark.asyncio
    async def test_get_storage_key_personal(self, mock_db):
        """Gets user's key for personal session."""
        user = User(id="user_123")
        user.set_encryption_keys(
            public_key="ff" * 32,
            encrypted_private_key="aa" * 48,
            iv="bb" * 16,
            tag="cc" * 16,
            salt="dd" * 32,
            recovery_encrypted_private_key="ee" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user))

        service = ChatService(mock_db)
        key = await service.get_storage_public_key(user_id="user_123")

        assert key == bytes.fromhex("ff" * 32)

    @pytest.mark.asyncio
    async def test_get_storage_key_org(self, mock_db):
        """Gets org's key for org session."""
        org = Organization(id="org_123", name="Test Org")
        org.set_encryption_keys(
            org_public_key="ab" * 32,
            admin_encrypted_private_key="cd" * 48,
            iv="ef" * 16,
            tag="12" * 16,
            salt="34" * 32,
            created_by="admin_123",
        )
        mock_db.execute = AsyncMock(return_value=mock_execute_result(org))

        service = ChatService(mock_db)
        key = await service.get_storage_public_key(
            user_id="user_123",
            org_id="org_123",
        )

        assert key == bytes.fromhex("ab" * 32)

    @pytest.mark.asyncio
    async def test_get_storage_key_no_keys(self, mock_db):
        """Raises StorageKeyNotFoundError when user has no keys."""
        user = User(id="user_123")
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user))

        service = ChatService(mock_db)
        with pytest.raises(StorageKeyNotFoundError) as exc_info:
            await service.get_storage_public_key(user_id="user_123")

        assert "User user_123 has not set up encryption keys" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_user_public_key(self, mock_db):
        """Gets user's public key for transport."""
        user = User(id="user_123")
        user.set_encryption_keys(
            public_key="ff" * 32,
            encrypted_private_key="aa" * 48,
            iv="bb" * 16,
            tag="cc" * 16,
            salt="dd" * 32,
            recovery_encrypted_private_key="ee" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user))

        service = ChatService(mock_db)
        key = await service.get_user_public_key("user_123")

        assert key == bytes.fromhex("ff" * 32)


# =============================================================================
# Test Encryption Verification
# =============================================================================

class TestEncryptionVerification:
    """Tests for verifying encryption capability."""

    @pytest.mark.asyncio
    async def test_verify_personal_can_send(self, mock_db):
        """User with keys can send personal messages."""
        user = User(id="user_123")
        user.set_encryption_keys(
            public_key="ff" * 32,
            encrypted_private_key="aa" * 48,
            iv="bb" * 16,
            tag="cc" * 16,
            salt="dd" * 32,
            recovery_encrypted_private_key="ee" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user))

        service = ChatService(mock_db)
        can_send, error = await service.verify_can_send_encrypted("user_123")

        assert can_send is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_verify_no_keys_cannot_send(self, mock_db):
        """User without keys cannot send."""
        user = User(id="user_123")
        mock_db.execute = AsyncMock(return_value=mock_execute_result(user))

        service = ChatService(mock_db)
        can_send, error = await service.verify_can_send_encrypted("user_123")

        assert can_send is False
        assert "encryption keys" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_org_can_send(self, mock_db):
        """User with org key can send org messages."""
        user = User(id="user_123")
        user.set_encryption_keys(
            public_key="ff" * 32,
            encrypted_private_key="aa" * 48,
            iv="bb" * 16,
            tag="cc" * 16,
            salt="dd" * 32,
            recovery_encrypted_private_key="ee" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )

        org = Organization(id="org_123", name="Test Org")
        org.set_encryption_keys(
            org_public_key="ab" * 32,
            admin_encrypted_private_key="cd" * 48,
            iv="ef" * 16,
            tag="12" * 16,
            salt="34" * 32,
            created_by="admin_123",
        )

        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_123",
            role=MemberRole.MEMBER,
        )
        membership.set_encrypted_org_key(
            ephemeral_public_key="11" * 32,
            iv="22" * 16,
            ciphertext="33" * 32,
            auth_tag="44" * 16,
            hkdf_salt="55" * 32,
            distributed_by_user_id="admin_789",
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),
            mock_execute_result(org),
            mock_execute_result(membership),
        ])

        service = ChatService(mock_db)
        can_send, error = await service.verify_can_send_encrypted(
            "user_123",
            org_id="org_123",
        )

        assert can_send is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_verify_org_no_key_cannot_send(self, mock_db):
        """User without org key cannot send org messages."""
        user = User(id="user_123")
        user.set_encryption_keys(
            public_key="ff" * 32,
            encrypted_private_key="aa" * 48,
            iv="bb" * 16,
            tag="cc" * 16,
            salt="dd" * 32,
            recovery_encrypted_private_key="ee" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )

        org = Organization(id="org_123", name="Test Org")
        org.set_encryption_keys(
            org_public_key="ab" * 32,
            admin_encrypted_private_key="cd" * 48,
            iv="ef" * 16,
            tag="12" * 16,
            salt="34" * 32,
            created_by="admin_123",
        )

        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_123",
            role=MemberRole.MEMBER,
        )
        # No org key set

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),
            mock_execute_result(org),
            mock_execute_result(membership),
        ])

        service = ChatService(mock_db)
        can_send, error = await service.verify_can_send_encrypted(
            "user_123",
            org_id="org_123",
        )

        assert can_send is False
        assert "organization key distributed" in error.lower()
