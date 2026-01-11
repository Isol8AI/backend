"""Unit tests for Organization model."""
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.organization import Organization


class TestOrganizationModel:
    """Tests for the Organization model."""

    def test_organization_creation(self):
        """Organization can be created with required fields."""
        org = Organization(id="org_123", name="Test Org")
        assert org.id == "org_123"
        assert org.name == "Test Org"

    def test_organization_tablename(self):
        """Organization model uses correct table name."""
        assert Organization.__tablename__ == "organizations"

    def test_organization_has_slug(self):
        """Organization has optional slug field."""
        org = Organization(id="org_123", name="Test Org", slug="test-org")
        assert org.slug == "test-org"

    def test_organization_has_future_fields(self):
        """Organization has fields for future model endpoints."""
        org = Organization(
            id="org_123",
            name="Test Org",
            custom_model_endpoint="https://models.example.com",
            fine_tuned_model_id="ft_model_123"
        )
        assert org.custom_model_endpoint == "https://models.example.com"
        assert org.fine_tuned_model_id == "ft_model_123"

    def test_organization_has_timestamps(self):
        """Organization has created_at and updated_at fields."""
        assert hasattr(Organization, "created_at")
        assert hasattr(Organization, "updated_at")

    @pytest.mark.asyncio
    async def test_organization_persistence(self, db_session):
        """Organization can be persisted and retrieved from database."""
        org = Organization(id="org_persist_123", name="Persisted Org", slug="persisted-org")
        db_session.add(org)
        await db_session.flush()

        result = await db_session.execute(select(Organization).where(Organization.id == org.id))
        fetched_org = result.scalar_one()

        assert fetched_org.name == "Persisted Org"
        assert fetched_org.slug == "persisted-org"
        assert fetched_org.created_at is not None

    @pytest.mark.asyncio
    async def test_organization_slug_unique(self, db_session):
        """Organization slug must be unique."""
        org1 = Organization(id="org_1", name="Org 1", slug="unique-slug")
        org2 = Organization(id="org_2", name="Org 2", slug="unique-slug")

        db_session.add(org1)
        await db_session.flush()

        db_session.add(org2)
        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_organization_null_slug_allowed(self, db_session):
        """Multiple organizations can have null slugs."""
        org1 = Organization(id="org_null_1", name="Org 1", slug=None)
        org2 = Organization(id="org_null_2", name="Org 2", slug=None)

        db_session.add(org1)
        db_session.add(org2)
        await db_session.flush()  # Should not raise

        result = await db_session.execute(select(Organization))
        orgs = result.scalars().all()
        assert len([o for o in orgs if o.id.startswith("org_null_")]) == 2
