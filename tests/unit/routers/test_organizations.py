"""Unit tests for organizations router."""
import pytest
from sqlalchemy import select

from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole


class TestSyncOrganization:
    """Tests for POST /api/v1/organizations/sync endpoint."""

    @pytest.mark.asyncio
    async def test_sync_creates_new_org(self, async_client_org, db_session, test_user):
        """Sync creates new organization when not exists."""
        response = await async_client_org.post(
            "/api/v1/organizations/sync",
            json={"org_id": "org_test_123", "name": "Test Organization", "slug": "test-org"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["org_id"] == "org_test_123"

    @pytest.mark.asyncio
    async def test_sync_updates_existing_org(self, async_client_org, db_session, test_user, test_organization):
        """Sync updates existing organization."""
        response = await async_client_org.post(
            "/api/v1/organizations/sync",
            json={"org_id": "org_test_123", "name": "Updated Organization", "slug": "test-org"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert data["org_id"] == "org_test_123"

        # Verify name was updated
        result = await db_session.execute(
            select(Organization).where(Organization.id == "org_test_123")
        )
        org = result.scalar_one()
        assert org.name == "Updated Organization"

    @pytest.mark.asyncio
    async def test_sync_creates_membership(self, async_client_org, db_session, test_user):
        """Sync creates membership for user in organization."""
        response = await async_client_org.post(
            "/api/v1/organizations/sync",
            json={"org_id": "org_test_123", "name": "Test Organization", "slug": "test-org"}
        )

        assert response.status_code == 200

        # Verify membership was created
        result = await db_session.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == "user_test_123",
                OrganizationMembership.org_id == "org_test_123"
            )
        )
        membership = result.scalar_one_or_none()
        assert membership is not None
        assert membership.role == MemberRole.MEMBER

    @pytest.mark.asyncio
    async def test_sync_updates_membership_role(self, async_client_org_admin, db_session, test_user, test_membership):
        """Sync updates membership role when role changes."""
        response = await async_client_org_admin.post(
            "/api/v1/organizations/sync",
            json={"org_id": "org_test_123", "name": "Test Organization", "slug": "test-org"}
        )

        assert response.status_code == 200

        # Verify membership role was updated to admin
        result = await db_session.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == "user_test_123",
                OrganizationMembership.org_id == "org_test_123"
            )
        )
        membership = result.scalar_one()
        assert membership.role == MemberRole.ADMIN

    @pytest.mark.asyncio
    async def test_sync_works_from_personal_context(self, async_client, db_session, test_user):
        """Sync works from personal context (org_id from request body)."""
        response = await async_client.post(
            "/api/v1/organizations/sync",
            json={"org_id": "org_new_789", "name": "New Organization", "slug": "new-org"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["org_id"] == "org_new_789"

        # Verify organization was created
        result = await db_session.execute(
            select(Organization).where(Organization.id == "org_new_789")
        )
        org = result.scalar_one_or_none()
        assert org is not None
        assert org.name == "New Organization"

    @pytest.mark.asyncio
    async def test_sync_requires_authentication(self, unauthenticated_async_client):
        """Sync requires authentication."""
        response = await unauthenticated_async_client.post(
            "/api/v1/organizations/sync",
            json={"org_id": "org_test_123", "name": "Test Organization", "slug": "test-org"}
        )
        assert response.status_code in [401, 403]


class TestGetCurrentOrg:
    """Tests for GET /api/v1/organizations/current endpoint."""

    @pytest.mark.asyncio
    async def test_get_current_returns_none_in_personal_mode(self, async_client, db_session, test_user):
        """Get current org returns None when in personal mode."""
        response = await async_client.get("/api/v1/organizations/current")

        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] is None
        assert data["is_personal_context"] is True

    @pytest.mark.asyncio
    async def test_get_current_returns_org_in_org_mode(
        self, async_client_org, db_session, test_user, test_organization, test_membership
    ):
        """Get current org returns organization details when in org mode."""
        response = await async_client_org.get("/api/v1/organizations/current")

        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] == "org_test_123"
        assert data["org_name"] == "Test Organization"
        assert data["org_slug"] == "test-org"
        assert data["org_role"] == "org:member"  # From Clerk JWT claims, not DB
        assert data["is_personal_context"] is False

    @pytest.mark.asyncio
    async def test_get_current_shows_admin_role(
        self, async_client_org_admin, db_session, test_user, test_organization, test_admin_membership
    ):
        """Get current org shows admin role correctly."""
        response = await async_client_org_admin.get("/api/v1/organizations/current")

        assert response.status_code == 200
        data = response.json()
        assert data["org_role"] == "org:admin"  # From Clerk JWT claims, not DB
        assert data["is_org_admin"] is True

    @pytest.mark.asyncio
    async def test_get_current_requires_authentication(self, unauthenticated_async_client):
        """Get current org requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/organizations/current")
        assert response.status_code in [401, 403]


class TestListOrganizations:
    """Tests for GET /api/v1/organizations/ endpoint."""

    @pytest.mark.asyncio
    async def test_list_returns_user_orgs(self, async_client, db_session, test_user, test_organization, test_membership):
        """List organizations returns all user's organizations."""
        response = await async_client.get("/api/v1/organizations/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["organizations"]) == 1
        assert data["organizations"][0]["id"] == "org_test_123"
        assert data["organizations"][0]["name"] == "Test Organization"

    @pytest.mark.asyncio
    async def test_list_returns_multiple_orgs(
        self, async_client, db_session, test_user, test_organization, other_organization, test_membership
    ):
        """List organizations returns all organizations user belongs to."""
        # Add membership to other org
        other_membership = OrganizationMembership(
            id="mem_other",
            user_id="user_test_123",
            org_id="org_other_456",
            role=MemberRole.MEMBER
        )
        db_session.add(other_membership)
        await db_session.flush()

        response = await async_client.get("/api/v1/organizations/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["organizations"]) == 2
        org_ids = [org["id"] for org in data["organizations"]]
        assert "org_test_123" in org_ids
        assert "org_other_456" in org_ids

    @pytest.mark.asyncio
    async def test_list_returns_empty_for_no_memberships(self, async_client, db_session, test_user):
        """List organizations returns empty array when user has no memberships."""
        response = await async_client.get("/api/v1/organizations/")

        assert response.status_code == 200
        data = response.json()
        assert data["organizations"] == []

    @pytest.mark.asyncio
    async def test_list_does_not_return_other_users_orgs(
        self, async_client, db_session, test_user, other_user, test_organization
    ):
        """List organizations does not return orgs user is not a member of."""
        # Other user is a member, but test user is not
        other_membership = OrganizationMembership(
            id="mem_other_user",
            user_id=other_user.id,
            org_id=test_organization.id,
            role=MemberRole.MEMBER
        )
        db_session.add(other_membership)
        await db_session.flush()

        response = await async_client.get("/api/v1/organizations/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["organizations"]) == 0

    @pytest.mark.asyncio
    async def test_list_includes_role_info(self, async_client, db_session, test_user, test_organization, test_admin_membership):
        """List organizations includes role information."""
        response = await async_client.get("/api/v1/organizations/")

        assert response.status_code == 200
        data = response.json()
        assert data["organizations"][0]["role"] == "org:admin"

    @pytest.mark.asyncio
    async def test_list_requires_authentication(self, unauthenticated_async_client):
        """List organizations requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/organizations/")
        assert response.status_code in [401, 403]


class TestOrganizationContext:
    """Tests for organization context endpoints (GET/PUT /api/v1/organizations/context)."""

    @pytest.mark.asyncio
    async def test_get_org_context_returns_empty_initially(
        self, async_client_org, db_session, test_user, test_organization, test_membership
    ):
        """Get org context returns empty context when none set."""
        response = await async_client_org.get("/api/v1/organizations/context")

        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == {} or data["context_data"] is None

    @pytest.mark.asyncio
    async def test_get_org_context_requires_org_context(self, async_client, db_session, test_user):
        """Get org context requires organization context."""
        response = await async_client.get("/api/v1/organizations/context")

        assert response.status_code == 403
        assert "organization context" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_update_org_context_as_admin(
        self, async_client_org_admin, db_session, test_user, test_organization, test_admin_membership
    ):
        """Admin can update organization context."""
        context_data = {"shared_settings": {"theme": "dark"}, "custom_prompt": "Be helpful"}

        response = await async_client_org_admin.put(
            "/api/v1/organizations/context",
            json={"context_data": context_data}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == context_data

    @pytest.mark.asyncio
    async def test_update_org_context_denied_for_member(
        self, async_client_org, db_session, test_user, test_organization, test_membership
    ):
        """Non-admin member cannot update organization context."""
        context_data = {"shared_settings": {"theme": "dark"}}

        response = await async_client_org.put(
            "/api/v1/organizations/context",
            json={"context_data": context_data}
        )

        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_org_context_after_update(
        self, async_client_org_admin, db_session, test_user, test_organization, test_admin_membership
    ):
        """Context can be read after being updated."""
        # Admin updates context
        context_data = {"knowledge_base": {"topic": "sales"}}
        await async_client_org_admin.put(
            "/api/v1/organizations/context",
            json={"context_data": context_data}
        )

        # Read context back (with same client to verify persistence)
        response = await async_client_org_admin.get("/api/v1/organizations/context")

        assert response.status_code == 200
        data = response.json()
        assert data["context_data"] == context_data

    @pytest.mark.asyncio
    async def test_update_org_context_requires_org_context(self, async_client, db_session, test_user):
        """Update org context requires organization context."""
        response = await async_client.put(
            "/api/v1/organizations/context",
            json={"context_data": {"test": "data"}}
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_org_context_requires_authentication(self, unauthenticated_async_client):
        """Get org context requires authentication."""
        response = await unauthenticated_async_client.get("/api/v1/organizations/context")
        assert response.status_code in [401, 403]
