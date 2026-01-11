"""Unit tests for Session model."""
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from models.message import Message
from models.session import Session


class TestSessionModel:
    """Tests for the Session model."""

    def test_session_creation(self):
        """Session can be created with required fields."""
        session = Session(id=str(uuid.uuid4()), user_id="user_123", name="Test Chat")
        assert session.user_id == "user_123"
        assert session.name == "Test Chat"

    def test_session_has_org_id(self):
        """Session has optional org_id field."""
        session = Session(id=str(uuid.uuid4()), user_id="user_123", org_id="org_123")
        assert session.org_id == "org_123"

    def test_session_org_id_default_none(self):
        """Session org_id defaults to None (personal session)."""
        session = Session(id=str(uuid.uuid4()), user_id="user_123")
        assert session.org_id is None

    def test_session_tablename(self):
        """Session model uses correct table name."""
        assert Session.__tablename__ == "sessions"

    def test_session_default_name(self):
        """Session name column defaults to 'New Chat'."""
        name_column = Session.__table__.c.name
        assert name_column.default.arg == "New Chat"

    def test_session_has_created_at(self):
        """Session has created_at field."""
        assert hasattr(Session, "created_at")

    @pytest.mark.asyncio
    async def test_session_persistence(self, db_session, test_user):
        """Session can be persisted and retrieved from database."""
        session = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="Persisted Session")
        db_session.add(session)
        await db_session.flush()

        result = await db_session.execute(select(Session).where(Session.id == session.id))
        fetched_session = result.scalar_one()

        assert fetched_session.name == "Persisted Session"
        assert fetched_session.user_id == test_user.id

    @pytest.mark.asyncio
    async def test_session_user_foreign_key(self, db_session):
        """Session requires valid user_id foreign key."""
        session = Session(id=str(uuid.uuid4()), user_id="nonexistent_user", name="Invalid Session")
        db_session.add(session)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_session_messages_relationship(self, db_session, test_session, test_message):
        """Session has messages relationship that loads correctly."""
        result = await db_session.execute(
            select(Session).options(selectinload(Session.messages)).where(Session.id == test_session.id)
        )
        session = result.scalar_one()

        assert len(session.messages) == 1
        assert session.messages[0].content == test_message.content

    @pytest.mark.asyncio
    async def test_session_cascade_delete(self, db_session, test_user):
        """Deleting session cascades to delete associated messages."""
        session = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="Session to Delete")
        db_session.add(session)
        await db_session.flush()

        message = Message(id=str(uuid.uuid4()), session_id=session.id, role="user", content="To be deleted")
        db_session.add(message)
        await db_session.flush()
        message_id = message.id

        await db_session.delete(session)
        await db_session.flush()

        result = await db_session.execute(select(Message).where(Message.id == message_id))
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_session_with_org_persistence(self, db_session, test_user, test_organization):
        """Session with org_id can be persisted and retrieved."""
        session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            org_id=test_organization.id,
            name="Org Session"
        )
        db_session.add(session)
        await db_session.flush()

        result = await db_session.execute(select(Session).where(Session.id == session.id))
        fetched = result.scalar_one()

        assert fetched.org_id == test_organization.id
        assert fetched.user_id == test_user.id

    @pytest.mark.asyncio
    async def test_session_org_foreign_key(self, db_session, test_user):
        """Session requires valid org_id foreign key if provided."""
        session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            org_id="nonexistent_org",
            name="Invalid Org Session"
        )
        db_session.add(session)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_filter_personal_sessions(self, db_session, test_user, test_organization):
        """Can filter sessions by org_id IS NULL for personal sessions."""
        personal = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="Personal")
        org_session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            org_id=test_organization.id,
            name="Org"
        )
        db_session.add(personal)
        db_session.add(org_session)
        await db_session.flush()

        result = await db_session.execute(
            select(Session).where(Session.user_id == test_user.id, Session.org_id == None)
        )
        personal_sessions = result.scalars().all()

        assert len(personal_sessions) == 1
        assert personal_sessions[0].name == "Personal"

    @pytest.mark.asyncio
    async def test_filter_org_sessions(self, db_session, test_user, test_organization):
        """Can filter sessions by specific org_id."""
        personal = Session(id=str(uuid.uuid4()), user_id=test_user.id, name="Personal")
        org_session = Session(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            org_id=test_organization.id,
            name="Org"
        )
        db_session.add(personal)
        db_session.add(org_session)
        await db_session.flush()

        result = await db_session.execute(
            select(Session).where(
                Session.user_id == test_user.id,
                Session.org_id == test_organization.id
            )
        )
        org_sessions = result.scalars().all()

        assert len(org_sessions) == 1
        assert org_sessions[0].name == "Org"
