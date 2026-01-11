"""Unit tests for Message model."""
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from models.message import Message, MessageRole


class TestMessageModel:
    """Tests for the Message model."""

    def test_message_creation(self):
        """Message can be created with required fields."""
        message = Message(id=str(uuid.uuid4()), session_id=str(uuid.uuid4()), role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_message_tablename(self):
        """Message model uses correct table name."""
        assert Message.__tablename__ == "messages"

    def test_message_model_used_nullable(self):
        """model_used field defaults to None for user messages."""
        message = Message(id=str(uuid.uuid4()), session_id=str(uuid.uuid4()), role="user", content="User message")
        assert message.model_used is None

    def test_message_with_model_used(self):
        """Assistant message can include model attribution."""
        message = Message(
            id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            role="assistant",
            content="Response",
            model_used="Qwen/Qwen2.5-72B-Instruct",
        )
        assert message.model_used == "Qwen/Qwen2.5-72B-Instruct"


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_role_values(self):
        """MessageRole enum has expected string values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_role_is_string_enum(self):
        """MessageRole values can be used as strings."""
        assert isinstance(MessageRole.USER, str)
        assert MessageRole.USER == "user"


class TestMessagePersistence:
    """Tests for Message database operations."""

    @pytest.mark.asyncio
    async def test_message_persistence(self, db_session, test_session):
        """Message can be persisted and retrieved from database."""
        message = Message(id=str(uuid.uuid4()), session_id=test_session.id, role="user", content="Persisted message")
        db_session.add(message)
        await db_session.flush()

        result = await db_session.execute(select(Message).where(Message.id == message.id))
        fetched_message = result.scalar_one()

        assert fetched_message.content == "Persisted message"
        assert fetched_message.session_id == test_session.id

    @pytest.mark.asyncio
    async def test_message_session_foreign_key(self, db_session):
        """Message requires valid session_id foreign key."""
        message = Message(id=str(uuid.uuid4()), session_id="nonexistent_session", role="user", content="Invalid")
        db_session.add(message)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_message_timestamp_default(self, db_session, test_session):
        """Message timestamp defaults to current time on creation."""
        before = datetime.utcnow()

        message = Message(id=str(uuid.uuid4()), session_id=test_session.id, role="user", content="Timestamped")
        db_session.add(message)
        await db_session.flush()

        after = datetime.utcnow()

        assert message.timestamp is not None
        assert before <= message.timestamp <= after

    @pytest.mark.asyncio
    async def test_message_session_relationship(self, db_session, test_session):
        """Message has navigable session relationship."""
        message = Message(id=str(uuid.uuid4()), session_id=test_session.id, role="user", content="With relationship")
        db_session.add(message)
        await db_session.flush()

        result = await db_session.execute(
            select(Message).options(selectinload(Message.session)).where(Message.id == message.id)
        )
        fetched_message = result.scalar_one()

        assert fetched_message.session is not None
        assert fetched_message.session.id == test_session.id
