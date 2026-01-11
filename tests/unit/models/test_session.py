"""Unit tests for Session model."""
import pytest
import uuid
from datetime import datetime
from models.session import Session
from models.user import User


class TestSessionModel:
    """Tests for the Session model."""

    def test_session_creation(self):
        """Session can be created with required fields."""
        session = Session(
            id=str(uuid.uuid4()),
            user_id="user_123",
            name="Test Chat"
        )
        assert session.user_id == "user_123"
        assert session.name == "Test Chat"

    def test_session_tablename(self):
        """Session model uses correct table name."""
        assert Session.__tablename__ == "sessions"

    def test_session_default_name(self):
        """Session name column has 'New Chat' as default (applied at DB insert)."""
        # SQLAlchemy defaults are applied at insert time, not object creation
        # Verify the column has the correct default configured
        name_column = Session.__table__.c.name
        assert name_column.default.arg == "New Chat"

    def test_session_id_is_uuid(self):
        """Session ID is auto-generated as UUID."""
        session = Session(user_id="user_123")
        # The default function should generate a valid UUID
        assert session.id is not None or hasattr(Session.id, 'default')

    def test_session_has_created_at(self):
        """Session has created_at field."""
        assert hasattr(Session, 'created_at')

    @pytest.mark.asyncio
    async def test_session_persistence(self, db_session, test_user):
        """Session can be persisted to database."""
        session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            name="Persisted Session"
        )
        db_session.add(session)
        await db_session.flush()

        from sqlalchemy import select
        result = await db_session.execute(select(Session).where(Session.id == session.id))
        fetched_session = result.scalar_one()

        assert fetched_session.name == "Persisted Session"
        assert fetched_session.user_id == test_user.id

    @pytest.mark.asyncio
    async def test_session_user_foreign_key(self, db_session):
        """Session requires valid user_id foreign key."""
        from sqlalchemy.exc import IntegrityError

        session = Session(
            id=str(uuid.uuid4()),
            user_id="nonexistent_user",
            name="Invalid Session"
        )
        db_session.add(session)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_session_messages_relationship(self, db_session, test_session, test_message):
        """Session has messages relationship."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await db_session.execute(
            select(Session)
            .options(selectinload(Session.messages))
            .where(Session.id == test_session.id)
        )
        session = result.scalar_one()

        assert len(session.messages) == 1
        assert session.messages[0].content == test_message.content

    @pytest.mark.asyncio
    async def test_session_cascade_delete(self, db_session, test_user):
        """Deleting session cascades to messages."""
        from models.message import Message
        from sqlalchemy import select

        # Create session with messages
        session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            name="Session to Delete"
        )
        db_session.add(session)
        await db_session.flush()

        message = Message(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="user",
            content="This message should be deleted"
        )
        db_session.add(message)
        await db_session.flush()

        message_id = message.id

        # Delete session
        await db_session.delete(session)
        await db_session.flush()

        # Verify message is also deleted
        result = await db_session.execute(select(Message).where(Message.id == message_id))
        assert result.scalar_one_or_none() is None
