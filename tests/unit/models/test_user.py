"""Unit tests for User model."""
import pytest
from models.user import User


class TestUserModel:
    """Tests for the User model."""

    def test_user_creation_with_clerk_id(self):
        """User can be created with a Clerk ID."""
        user = User(id="user_clerk_123")
        assert user.id == "user_clerk_123"

    def test_user_tablename(self):
        """User model uses correct table name."""
        assert User.__tablename__ == "users"

    def test_user_id_is_primary_key(self):
        """User ID is the primary key."""
        pk_columns = [col.name for col in User.__table__.primary_key.columns]
        assert "id" in pk_columns

    @pytest.mark.asyncio
    async def test_user_persistence(self, db_session):
        """User can be persisted to database."""
        user = User(id="user_persist_test")
        db_session.add(user)
        await db_session.flush()

        # Query it back
        from sqlalchemy import select
        result = await db_session.execute(select(User).where(User.id == "user_persist_test"))
        fetched_user = result.scalar_one()

        assert fetched_user.id == "user_persist_test"

    @pytest.mark.asyncio
    async def test_user_unique_id(self, db_session):
        """User IDs must be unique."""
        from sqlalchemy.exc import IntegrityError

        user1 = User(id="duplicate_id")
        db_session.add(user1)
        await db_session.flush()

        user2 = User(id="duplicate_id")
        db_session.add(user2)

        with pytest.raises(IntegrityError):
            await db_session.flush()
