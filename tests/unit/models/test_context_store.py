"""Unit tests for ContextStore model."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.context_store import ContextStore


class TestContextStoreModel:
    """Tests for the ContextStore model."""

    def test_context_store_creation(self):
        """ContextStore can be created with required fields."""
        store = ContextStore(id="ctx_123", owner_type="user", owner_id="user_123")
        assert store.owner_type == "user"
        assert store.owner_id == "user_123"

    def test_context_store_tablename(self):
        """ContextStore model uses correct table name."""
        assert ContextStore.__tablename__ == "context_stores"

    def test_context_store_default_store_type(self):
        """ContextStore defaults to postgres store type."""
        store_type_column = ContextStore.__table__.c.store_type
        assert store_type_column.default.arg == "postgres"

    def test_context_store_has_context_data(self):
        """ContextStore has context_data JSONB field."""
        store = ContextStore(
            id="ctx_123", owner_type="user", owner_id="user_123", context_data={"preferences": {"theme": "dark"}}
        )
        assert store.context_data == {"preferences": {"theme": "dark"}}

    def test_context_store_has_timestamps(self):
        """ContextStore has created_at and updated_at fields."""
        assert hasattr(ContextStore, "created_at")
        assert hasattr(ContextStore, "updated_at")

    @pytest.mark.asyncio
    async def test_context_store_for_user(self, db_session, test_user):
        """ContextStore can be created for a user."""
        store = ContextStore(
            id=f"ctx_user_{uuid.uuid4()}", owner_type="user", owner_id=test_user.id, context_data={"test": "value"}
        )
        db_session.add(store)
        await db_session.flush()

        result = await db_session.execute(select(ContextStore).where(ContextStore.id == store.id))
        fetched = result.scalar_one()

        assert fetched.owner_type == "user"
        assert fetched.owner_id == test_user.id
        assert fetched.context_data == {"test": "value"}
        assert fetched.store_type == "postgres"

    @pytest.mark.asyncio
    async def test_context_store_for_organization(self, db_session, test_organization):
        """ContextStore can be created for an organization."""
        store = ContextStore(
            id=f"ctx_org_{uuid.uuid4()}",
            owner_type="org",
            owner_id=test_organization.id,
            context_data={"shared": "context"},
        )
        db_session.add(store)
        await db_session.flush()

        result = await db_session.execute(select(ContextStore).where(ContextStore.id == store.id))
        fetched = result.scalar_one()

        assert fetched.owner_type == "org"
        assert fetched.owner_id == test_organization.id

    @pytest.mark.asyncio
    async def test_context_store_unique_owner(self, db_session, test_user):
        """Only one context store per owner_type/owner_id combination."""
        store1 = ContextStore(id="ctx_dup_1", owner_type="user", owner_id=test_user.id)
        store2 = ContextStore(id="ctx_dup_2", owner_type="user", owner_id=test_user.id)

        db_session.add(store1)
        await db_session.flush()

        db_session.add(store2)
        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_context_store_different_owner_types(self, db_session, test_user, test_organization):
        """Same owner_id with different owner_types are allowed."""
        # This tests that "user_123" as a user and "user_123" as an org are separate
        store1 = ContextStore(id="ctx_type_1", owner_type="user", owner_id="shared_id_123")
        store2 = ContextStore(id="ctx_type_2", owner_type="org", owner_id="shared_id_123")

        db_session.add(store1)
        db_session.add(store2)
        await db_session.flush()  # Should not raise

        result = await db_session.execute(select(ContextStore).where(ContextStore.owner_id == "shared_id_123"))
        stores = result.scalars().all()
        assert len(stores) == 2

    @pytest.mark.asyncio
    async def test_context_store_vector_type(self, db_session, test_user):
        """ContextStore can be configured for vector storage."""
        store = ContextStore(
            id=f"ctx_vector_{uuid.uuid4()}",
            owner_type="user",
            owner_id=test_user.id,
            store_type="vector",
            context_data={"vector_config": {"provider": "pinecone"}},
        )
        db_session.add(store)
        await db_session.flush()

        result = await db_session.execute(select(ContextStore).where(ContextStore.id == store.id))
        fetched = result.scalar_one()

        assert fetched.store_type == "vector"
