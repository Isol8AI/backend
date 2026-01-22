"""Tests for ClerkSyncService."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.services.clerk_sync_service import ClerkSyncService
from models.user import User
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
    db.delete = AsyncMock()
    return db


def mock_execute_result(item):
    """Create a mock execute result that returns the given item."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=item)
    return mock_result


def mock_execute_result_scalars(items):
    """Create a mock execute result that returns multiple items."""
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all = MagicMock(return_value=items)
    mock_result.scalars = MagicMock(return_value=mock_scalars)
    return mock_result


# =============================================================================
# Test User Sync
# =============================================================================

class TestUserSync:
    """Tests for user sync operations."""

    @pytest.mark.asyncio
    async def test_create_user(self, mock_db):
        """Creates new user from webhook data."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ClerkSyncService(mock_db)
        user = await service.create_user({"id": "user_123"})

        assert user.id == "user_123"
        mock_db.add.assert_called()
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_create_user_already_exists(self, mock_db):
        """Updates existing user when creating duplicate."""
        existing_user = User(id="user_123")
        mock_db.execute = AsyncMock(return_value=mock_execute_result(existing_user))

        service = ClerkSyncService(mock_db)
        user = await service.create_user({"id": "user_123"})

        assert user.id == "user_123"

    @pytest.mark.asyncio
    async def test_update_user(self, mock_db):
        """Updates existing user."""
        existing_user = User(id="user_123")
        mock_db.execute = AsyncMock(return_value=mock_execute_result(existing_user))

        service = ClerkSyncService(mock_db)
        user = await service.update_user({"id": "user_123"})

        assert user.id == "user_123"
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_user_not_found_creates(self, mock_db):
        """Creates user if not found during update."""
        # First call returns None (not found), second returns None (for create check)
        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(None),
            mock_execute_result(None),
        ])

        service = ClerkSyncService(mock_db)
        user = await service.update_user({"id": "user_123"})

        assert user.id == "user_123"
        mock_db.add.assert_called()

    @pytest.mark.asyncio
    async def test_delete_user(self, mock_db):
        """Deletes user and clears encryption keys."""
        user = User(id="user_123")
        user.has_encryption_keys = False

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),
            mock_execute_result_scalars([]),  # No memberships
        ])

        service = ClerkSyncService(mock_db)
        await service.delete_user({"id": "user_123"})

        mock_db.delete.assert_called_with(user)
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_delete_user_clears_keys(self, mock_db):
        """Clears encryption keys when user is deleted."""
        user = User(id="user_123")
        user.set_encryption_keys(
            public_key="aa" * 32,
            encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            recovery_encrypted_private_key="ff" * 48,
            recovery_iv="11" * 16,
            recovery_tag="22" * 16,
            recovery_salt="33" * 32,
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),
            mock_execute_result_scalars([]),  # No memberships
        ])

        service = ClerkSyncService(mock_db)
        await service.delete_user({"id": "user_123"})

        assert user.has_encryption_keys is False
        # Audit log should be added
        assert mock_db.add.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_user_clears_memberships(self, mock_db):
        """Deletes user's memberships when user is deleted."""
        user = User(id="user_123")
        user.has_encryption_keys = False

        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(user),
            mock_execute_result_scalars([membership]),
        ])

        service = ClerkSyncService(mock_db)
        await service.delete_user({"id": "user_123"})

        # Both membership and user deleted
        assert mock_db.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, mock_db):
        """Handles deletion of non-existent user."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ClerkSyncService(mock_db)
        await service.delete_user({"id": "user_123"})

        mock_db.delete.assert_not_called()


# =============================================================================
# Test Organization Sync
# =============================================================================

class TestOrganizationSync:
    """Tests for organization sync operations."""

    @pytest.mark.asyncio
    async def test_create_organization(self, mock_db):
        """Creates new organization from webhook data."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ClerkSyncService(mock_db)
        org = await service.create_organization({
            "id": "org_123",
            "name": "Test Org",
            "slug": "test-org",
        })

        assert org.id == "org_123"
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        mock_db.add.assert_called()

    @pytest.mark.asyncio
    async def test_create_organization_already_exists(self, mock_db):
        """Updates existing organization when creating duplicate."""
        existing_org = Organization(id="org_123", name="Old Name")
        mock_db.execute = AsyncMock(return_value=mock_execute_result(existing_org))

        service = ClerkSyncService(mock_db)
        org = await service.create_organization({
            "id": "org_123",
            "name": "New Name",
        })

        assert org.name == "New Name"

    @pytest.mark.asyncio
    async def test_update_organization(self, mock_db):
        """Updates existing organization."""
        existing_org = Organization(id="org_123", name="Old Name")
        mock_db.execute = AsyncMock(return_value=mock_execute_result(existing_org))

        service = ClerkSyncService(mock_db)
        org = await service.update_organization({
            "id": "org_123",
            "name": "New Name",
            "slug": "new-slug",
        })

        assert org.name == "New Name"
        assert org.slug == "new-slug"

    @pytest.mark.asyncio
    async def test_delete_organization(self, mock_db):
        """Deletes organization and clears all org keys."""
        org = Organization(id="org_123", name="Test Org")

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(org),
            mock_execute_result_scalars([]),  # No memberships
        ])

        service = ClerkSyncService(mock_db)
        await service.delete_organization({"id": "org_123"})

        mock_db.delete.assert_called_with(org)

    @pytest.mark.asyncio
    async def test_delete_organization_clears_member_keys(self, mock_db):
        """Clears org keys from all memberships when org is deleted."""
        org = Organization(id="org_123", name="Test Org")

        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_123",
            role=MemberRole.MEMBER,
        )
        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 32,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="admin_123",
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(org),
            mock_execute_result_scalars([membership]),
        ])

        service = ClerkSyncService(mock_db)
        await service.delete_organization({"id": "org_123"})

        assert membership.has_org_key is False


# =============================================================================
# Test Membership Sync
# =============================================================================

class TestMembershipSync:
    """Tests for membership sync operations."""

    @pytest.mark.asyncio
    async def test_create_membership(self, mock_db):
        """Creates membership with pending key distribution."""
        org = Organization(id="org_456", name="Test Org")
        user = User(id="user_123")

        # Returns: org, user, no existing membership
        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(org),
            mock_execute_result(user),
            mock_execute_result(None),  # No existing membership
        ])

        service = ClerkSyncService(mock_db)
        membership = await service.create_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
            "role": "org:member",
        })

        assert membership.id == "mem_123"
        assert not membership.has_org_key  # Pending distribution (None or False)
        assert membership.role == MemberRole.MEMBER

    @pytest.mark.asyncio
    async def test_create_membership_creates_org_if_needed(self, mock_db):
        """Creates organization if it doesn't exist."""
        user = User(id="user_123")

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(None),  # No org
            mock_execute_result(user),
            mock_execute_result(None),  # No existing membership
        ])

        service = ClerkSyncService(mock_db)
        membership = await service.create_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456", "name": "New Org"},
            "role": "org:admin",
        })

        assert membership.role == MemberRole.ADMIN
        # Org should have been added
        added_objects = [call.args[0] for call in mock_db.add.call_args_list]
        orgs = [obj for obj in added_objects if isinstance(obj, Organization)]
        assert len(orgs) == 1

    @pytest.mark.asyncio
    async def test_create_membership_creates_user_if_needed(self, mock_db):
        """Creates user if it doesn't exist."""
        org = Organization(id="org_456", name="Test Org")

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(org),
            mock_execute_result(None),  # No user
            mock_execute_result(None),  # No existing membership
        ])

        service = ClerkSyncService(mock_db)
        await service.create_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
            "role": "org:member",
        })

        # User should have been added
        added_objects = [call.args[0] for call in mock_db.add.call_args_list]
        users = [obj for obj in added_objects if isinstance(obj, User)]
        assert len(users) == 1

    @pytest.mark.asyncio
    async def test_create_membership_updates_role_if_exists(self, mock_db):
        """Updates role if membership already exists."""
        org = Organization(id="org_456", name="Test Org")
        user = User(id="user_123")
        existing_membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(org),
            mock_execute_result(user),
            mock_execute_result(existing_membership),
        ])

        service = ClerkSyncService(mock_db)
        membership = await service.create_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
            "role": "org:admin",  # Changed to admin
        })

        assert membership.role == MemberRole.ADMIN

    @pytest.mark.asyncio
    async def test_create_membership_missing_fields(self, mock_db):
        """Handles missing required fields gracefully."""
        service = ClerkSyncService(mock_db)
        membership = await service.create_membership({
            "id": "mem_123",
            # Missing public_user_data and organization
        })

        assert membership is None
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_membership_role_change(self, mock_db):
        """Updates membership role."""
        existing_membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )

        mock_db.execute = AsyncMock(return_value=mock_execute_result(existing_membership))

        service = ClerkSyncService(mock_db)
        membership = await service.update_membership({
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
            "role": "org:admin",
        })

        assert membership.role == MemberRole.ADMIN
        # Audit log should be created
        assert mock_db.add.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_membership(self, mock_db):
        """Deletes membership."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )

        mock_db.execute = AsyncMock(return_value=mock_execute_result(membership))

        service = ClerkSyncService(mock_db)
        await service.delete_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
        })

        mock_db.delete.assert_called_with(membership)

    @pytest.mark.asyncio
    async def test_delete_membership_revokes_org_key(self, mock_db):
        """Revokes org key when membership is deleted."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )
        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 32,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="admin_123",
        )

        mock_db.execute = AsyncMock(return_value=mock_execute_result(membership))

        service = ClerkSyncService(mock_db)
        await service.delete_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
        })

        # Key should be cleared before deletion
        assert membership.has_org_key is False
        # Audit logs should be created (revocation + leaving)
        assert mock_db.add.call_count >= 2

    @pytest.mark.asyncio
    async def test_delete_membership_not_found(self, mock_db):
        """Handles deletion of non-existent membership."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = ClerkSyncService(mock_db)
        await service.delete_membership({
            "id": "mem_123",
            "public_user_data": {"user_id": "user_123"},
            "organization": {"id": "org_456"},
        })

        mock_db.delete.assert_not_called()
