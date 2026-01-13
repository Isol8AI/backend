"""Unit tests for encrypted Message model."""
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from models.message import Message, MessageRole
from tests.factories.message_factory import generate_encrypted_payload


class TestMessageModel:
    """Tests for the encrypted Message model."""

    def test_message_creation_with_encrypted_fields(self):
        """Message can be created with encrypted payload fields."""
        payload = generate_encrypted_payload("Hello, world!")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        assert message.role == "user"
        assert message.ephemeral_public_key == payload["ephemeral_public_key"]
        assert message.ciphertext == payload["ciphertext"]

    def test_message_tablename(self):
        """Message model uses correct table name."""
        assert Message.__tablename__ == "messages"

    def test_message_model_used_nullable(self):
        """model_used field defaults to None for user messages."""
        payload = generate_encrypted_payload("User message")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        assert message.model_used is None

    def test_message_with_model_used(self):
        """Assistant message can include model attribution."""
        payload = generate_encrypted_payload("Response")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            role="assistant",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
            model_used="Qwen/Qwen2.5-72B-Instruct",
        )
        assert message.model_used == "Qwen/Qwen2.5-72B-Instruct"

    def test_message_encrypted_payload_property(self):
        """encrypted_payload property returns correct structure."""
        payload = generate_encrypted_payload("Test content")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )

        result = message.encrypted_payload
        assert result["ephemeral_public_key"] == payload["ephemeral_public_key"]
        assert result["iv"] == payload["iv"]
        assert result["ciphertext"] == payload["ciphertext"]
        assert result["auth_tag"] == payload["auth_tag"]
        assert result["hkdf_salt"] == payload["hkdf_salt"]

    def test_message_to_api_response(self):
        """to_api_response returns expected format."""
        payload = generate_encrypted_payload("Test content")
        message = Message(
            id="msg_123",
            session_id="session_456",
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        message.created_at = datetime.utcnow()

        response = message.to_api_response()

        assert response["id"] == "msg_123"
        assert response["session_id"] == "session_456"
        assert response["role"] == "user"
        assert "encrypted_content" in response
        assert response["encrypted_content"]["ciphertext"] == payload["ciphertext"]


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_role_values(self):
        """MessageRole enum has expected string values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_role_is_string_enum(self):
        """MessageRole values can be used as strings."""
        assert isinstance(MessageRole.USER, str)
        assert MessageRole.USER == "user"


class TestMessageCreateEncrypted:
    """Tests for Message.create_encrypted factory method."""

    def test_create_encrypted_valid(self):
        """create_encrypted works with valid hex strings."""
        message = Message.create_encrypted(
            id="msg_123",
            session_id="session_456",
            role=MessageRole.USER,
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 50,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
        )
        assert message.ephemeral_public_key == "aa" * 32
        assert message.iv == "bb" * 16

    def test_create_encrypted_validates_ephemeral_key_length(self):
        """create_encrypted rejects invalid ephemeral_public_key length."""
        with pytest.raises(ValueError, match="64 hex characters"):
            Message.create_encrypted(
                id="msg_123",
                session_id="session_456",
                role=MessageRole.USER,
                ephemeral_public_key="aa" * 16,  # Too short
                iv="bb" * 16,
                ciphertext="cc" * 50,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
            )

    def test_create_encrypted_validates_iv_length(self):
        """create_encrypted rejects invalid iv length."""
        with pytest.raises(ValueError, match="32 hex characters"):
            Message.create_encrypted(
                id="msg_123",
                session_id="session_456",
                role=MessageRole.USER,
                ephemeral_public_key="aa" * 32,
                iv="bb" * 8,  # Too short
                ciphertext="cc" * 50,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
            )

    def test_create_encrypted_validates_ciphertext_not_empty(self):
        """create_encrypted rejects empty ciphertext."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Message.create_encrypted(
                id="msg_123",
                session_id="session_456",
                role=MessageRole.USER,
                ephemeral_public_key="aa" * 32,
                iv="bb" * 16,
                ciphertext="",  # Empty
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
            )

    def test_create_encrypted_validates_hex_format(self):
        """create_encrypted rejects invalid hex strings."""
        with pytest.raises(ValueError, match="valid hex string"):
            Message.create_encrypted(
                id="msg_123",
                session_id="session_456",
                role=MessageRole.USER,
                ephemeral_public_key="gg" * 32,  # Not hex
                iv="bb" * 16,
                ciphertext="cc" * 50,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
            )

    def test_create_encrypted_with_token_counts(self):
        """create_encrypted stores token counts."""
        message = Message.create_encrypted(
            id="msg_123",
            session_id="session_456",
            role=MessageRole.ASSISTANT,
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 50,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            model_used="test-model",
            input_tokens=100,
            output_tokens=50,
        )
        assert message.input_tokens == 100
        assert message.output_tokens == 50
        assert message.model_used == "test-model"


class TestMessagePersistence:
    """Tests for Message database operations."""

    @pytest.mark.asyncio
    async def test_message_persistence(self, db_session, test_session):
        """Encrypted message can be persisted and retrieved from database."""
        payload = generate_encrypted_payload("Persisted message")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        db_session.add(message)
        await db_session.flush()

        result = await db_session.execute(select(Message).where(Message.id == message.id))
        fetched_message = result.scalar_one()

        assert fetched_message.ciphertext == payload["ciphertext"]
        assert fetched_message.session_id == test_session.id

    @pytest.mark.asyncio
    async def test_message_session_foreign_key(self, db_session):
        """Message requires valid session_id foreign key."""
        payload = generate_encrypted_payload("Invalid")
        message = Message(
            id=str(uuid.uuid4()),
            session_id="nonexistent_session",
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        db_session.add(message)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_message_created_at_default(self, db_session, test_session):
        """Message created_at defaults to current time on creation."""
        before = datetime.utcnow()

        payload = generate_encrypted_payload("Timestamped")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        db_session.add(message)
        await db_session.flush()

        after = datetime.utcnow()

        assert message.created_at is not None
        assert before <= message.created_at <= after

    @pytest.mark.asyncio
    async def test_message_session_relationship(self, db_session, test_session):
        """Message has navigable session relationship."""
        payload = generate_encrypted_payload("With relationship")
        message = Message(
            id=str(uuid.uuid4()),
            session_id=test_session.id,
            role="user",
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
        )
        db_session.add(message)
        await db_session.flush()

        result = await db_session.execute(
            select(Message).options(selectinload(Message.session)).where(Message.id == message.id)
        )
        fetched_message = result.scalar_one()

        assert fetched_message.session is not None
        assert fetched_message.session.id == test_session.id
