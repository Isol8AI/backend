"""Unit tests for OrganizationMembership model."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole


class TestOrganizationMembershipModel:
    """Tests for the OrganizationMembership model."""

    def test_membership_creation(self):
        """OrganizationMembership can be created with required fields."""
        membership = OrganizationMembership(id="mem_123", user_id="user_123", org_id="org_123", role=MemberRole.MEMBER)
        assert membership.user_id == "user_123"
        assert membership.org_id == "org_123"
        assert membership.role == MemberRole.MEMBER

    def test_membership_tablename(self):
        """OrganizationMembership model uses correct table name."""
        assert OrganizationMembership.__tablename__ == "organization_memberships"

    def test_membership_default_role(self):
        """OrganizationMembership defaults to MEMBER role."""
        role_column = OrganizationMembership.__table__.c.role
        assert role_column.default.arg == MemberRole.MEMBER

    def test_membership_has_joined_at(self):
        """OrganizationMembership has joined_at field."""
        assert hasattr(OrganizationMembership, "joined_at")

    @pytest.mark.asyncio
    async def test_membership_persistence(self, db_session, test_user, test_organization):
        """OrganizationMembership can be persisted and retrieved from database."""
        membership = OrganizationMembership(
            id=f"mem_{uuid.uuid4()}", user_id=test_user.id, org_id=test_organization.id, role=MemberRole.ADMIN
        )
        db_session.add(membership)
        await db_session.flush()

        result = await db_session.execute(
            select(OrganizationMembership).where(OrganizationMembership.id == membership.id)
        )
        fetched = result.scalar_one()

        assert fetched.user_id == test_user.id
        assert fetched.org_id == test_organization.id
        assert fetched.role == MemberRole.ADMIN
        assert fetched.joined_at is not None

    @pytest.mark.asyncio
    async def test_membership_user_foreign_key(self, db_session, test_organization):
        """OrganizationMembership requires valid user_id foreign key."""
        membership = OrganizationMembership(
            id="mem_invalid_user", user_id="nonexistent_user", org_id=test_organization.id, role=MemberRole.MEMBER
        )
        db_session.add(membership)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_membership_org_foreign_key(self, db_session, test_user):
        """OrganizationMembership requires valid org_id foreign key."""
        membership = OrganizationMembership(
            id="mem_invalid_org", user_id=test_user.id, org_id="nonexistent_org", role=MemberRole.MEMBER
        )
        db_session.add(membership)

        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_membership_unique_user_org_pair(self, db_session, test_user, test_organization):
        """User can only have one membership per organization."""
        mem1 = OrganizationMembership(
            id="mem_dup_1", user_id=test_user.id, org_id=test_organization.id, role=MemberRole.MEMBER
        )
        mem2 = OrganizationMembership(
            id="mem_dup_2", user_id=test_user.id, org_id=test_organization.id, role=MemberRole.ADMIN
        )

        db_session.add(mem1)
        await db_session.flush()

        db_session.add(mem2)
        with pytest.raises(IntegrityError):
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_user_can_be_in_multiple_orgs(self, db_session, test_user):
        """User can be a member of multiple organizations."""
        org1 = Organization(id="org_multi_1", name="Org 1")
        org2 = Organization(id="org_multi_2", name="Org 2")
        db_session.add(org1)
        db_session.add(org2)
        await db_session.flush()

        mem1 = OrganizationMembership(id="mem_multi_1", user_id=test_user.id, org_id=org1.id, role=MemberRole.MEMBER)
        mem2 = OrganizationMembership(id="mem_multi_2", user_id=test_user.id, org_id=org2.id, role=MemberRole.MEMBER)

        db_session.add(mem1)
        db_session.add(mem2)
        await db_session.flush()  # Should not raise

        result = await db_session.execute(
            select(OrganizationMembership).where(OrganizationMembership.user_id == test_user.id)
        )
        memberships = result.scalars().all()
        assert len(memberships) == 2
