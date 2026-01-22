"""Tests for OrgKeyService."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from core.services.org_key_service import (
    OrgKeyService,
    OrgKeyServiceError,
    OrgKeysAlreadyExistError,
    OrgKeysNotFoundError,
    MembershipNotFoundError,
    NotAdminError,
)
from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole
from models.user import User


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def admin_user():
    """Admin user with encryption keys."""
    user = User(id="admin_123")
    user.has_encryption_keys = True
    user.public_key = "aa" * 32
    return user


@pytest.fixture
def member_user():
    """Regular member with encryption keys."""
    user = User(id="member_456")
    user.has_encryption_keys = True
    user.public_key = "bb" * 32
    return user


@pytest.fixture
def member_without_keys():
    """Member without encryption keys."""
    user = User(id="member_no_keys")
    user.has_encryption_keys = False
    return user


@pytest.fixture
def org_without_keys():
    """Organization without encryption keys."""
    org = Organization(id="org_123", name="Test Org", slug="test-org")
    org.has_encryption_keys = False
    return org


@pytest.fixture
def org_with_keys():
    """Organization with encryption keys."""
    org = Organization(id="org_456", name="Encrypted Org", slug="encrypted-org")
    org.set_encryption_keys(
        org_public_key="cc" * 32,
        admin_encrypted_private_key="dd" * 48,
        iv="ee" * 16,
        tag="ff" * 16,
        salt="11" * 32,
        created_by="admin_123",
    )
    return org


@pytest.fixture
def admin_membership(admin_user, org_without_keys):
    """Admin membership."""
    membership = OrganizationMembership(
        id="mem_admin",
        user_id=admin_user.id,
        org_id=org_without_keys.id,
        role=MemberRole.ADMIN,
        has_org_key=False,
    )
    membership.user = admin_user
    membership.created_at = datetime.utcnow()
    return membership


@pytest.fixture
def member_membership(member_user, org_with_keys):
    """Member membership without org key."""
    membership = OrganizationMembership(
        id="mem_member",
        user_id=member_user.id,
        org_id=org_with_keys.id,
        role=MemberRole.MEMBER,
        has_org_key=False,
    )
    membership.user = member_user
    membership.created_at = datetime.utcnow()
    return membership


@pytest.fixture
def valid_org_key_data():
    """Valid org key creation data."""
    return {
        "org_public_key": "aa" * 32,
        "admin_encrypted_private_key": "bb" * 48,
        "admin_iv": "cc" * 16,
        "admin_tag": "dd" * 16,
        "admin_salt": "ee" * 32,
        "admin_member_key_ephemeral": "ff" * 32,
        "admin_member_key_iv": "11" * 16,
        "admin_member_key_ciphertext": "22" * 48,
        "admin_member_key_tag": "33" * 16,
        "admin_member_key_hkdf_salt": "44" * 32,
    }


@pytest.fixture
def valid_distribution_data():
    """Valid key distribution data."""
    return {
        "ephemeral_public_key": "aa" * 32,
        "iv": "bb" * 16,
        "ciphertext": "cc" * 48,
        "auth_tag": "dd" * 16,
        "hkdf_salt": "ee" * 32,
    }


def mock_execute_result(item):
    """Create a mock execute result."""
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=item)
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[item] if item else [])))
    return result


# =============================================================================
# Test Verify Admin
# =============================================================================

class TestVerifyAdmin:
    """Tests for verify_admin method."""

    @pytest.mark.asyncio
    async def test_returns_membership_for_admin(self, mock_db, admin_membership):
        """Returns membership when user is admin."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(admin_membership))

        service = OrgKeyService(mock_db)
        result = await service.verify_admin("admin_123", "org_123")

        assert result == admin_membership

    @pytest.mark.asyncio
    async def test_raises_error_for_non_member(self, mock_db):
        """Raises MembershipNotFoundError for non-member."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = OrgKeyService(mock_db)
        with pytest.raises(MembershipNotFoundError):
            await service.verify_admin("nonexistent", "org_123")

    @pytest.mark.asyncio
    async def test_raises_error_for_non_admin(self, mock_db, member_membership):
        """Raises NotAdminError for non-admin member."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(member_membership))

        service = OrgKeyService(mock_db)
        with pytest.raises(NotAdminError):
            await service.verify_admin("member_456", "org_456")


# =============================================================================
# Test Create Org Keys
# =============================================================================

class TestCreateOrgKeys:
    """Tests for create_org_keys method."""

    @pytest.mark.asyncio
    async def test_creates_keys_for_org(
        self, mock_db, org_without_keys, admin_membership, valid_org_key_data
    ):
        """Successfully creates org keys."""
        org_without_keys.memberships = [admin_membership]

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),  # verify_admin
            mock_execute_result(org_without_keys),  # get_organization
        ])

        service = OrgKeyService(mock_db)
        result = await service.create_org_keys(
            org_id="org_123",
            admin_user_id="admin_123",
            **valid_org_key_data,
        )

        assert result.has_encryption_keys is True
        assert result.org_public_key == "aa" * 32
        mock_db.commit.assert_called_once()
        mock_db.add.assert_called()  # Audit log

    @pytest.mark.asyncio
    async def test_raises_error_if_keys_exist(
        self, mock_db, org_with_keys, admin_membership, valid_org_key_data
    ):
        """Raises OrgKeysAlreadyExistError if org has keys."""
        admin_membership.org_id = org_with_keys.id

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(org_with_keys),
        ])

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeysAlreadyExistError):
            await service.create_org_keys(
                org_id="org_456",
                admin_user_id="admin_123",
                **valid_org_key_data,
            )

    @pytest.mark.asyncio
    async def test_raises_error_for_non_admin(
        self, mock_db, member_membership, valid_org_key_data
    ):
        """Raises NotAdminError for non-admin."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(member_membership))

        service = OrgKeyService(mock_db)
        with pytest.raises(NotAdminError):
            await service.create_org_keys(
                org_id="org_456",
                admin_user_id="member_456",
                **valid_org_key_data,
            )


# =============================================================================
# Test Get Pending Distributions
# =============================================================================

class TestGetPendingDistributions:
    """Tests for get_pending_distributions method."""

    @pytest.mark.asyncio
    async def test_returns_members_needing_distribution(
        self, mock_db, org_with_keys, admin_membership, member_membership
    ):
        """Returns members who need org key distribution in ready_for_distribution."""
        admin_membership.org_id = org_with_keys.id
        admin_membership.has_org_key = True
        org_with_keys.memberships = [admin_membership, member_membership]

        # Mock the pending members query
        pending_result = MagicMock()
        pending_result.scalars = MagicMock(return_value=MagicMock(
            all=MagicMock(return_value=[member_membership])
        ))

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),  # verify_admin
            mock_execute_result(org_with_keys),  # get_organization
            pending_result,  # pending members query
        ])

        service = OrgKeyService(mock_db)
        result = await service.get_pending_distributions("org_456", "admin_123")

        assert result["ready_count"] == 1
        assert result["needs_setup_count"] == 0
        assert len(result["ready_for_distribution"]) == 1
        assert result["ready_for_distribution"][0]["user_id"] == "member_456"
        assert result["ready_for_distribution"][0]["user_public_key"] == "bb" * 32

    @pytest.mark.asyncio
    async def test_excludes_members_without_personal_keys(
        self, mock_db, org_with_keys, admin_membership, member_without_keys
    ):
        """Members without personal keys go in needs_personal_setup."""
        admin_membership.org_id = org_with_keys.id
        admin_membership.has_org_key = True

        member_membership_no_keys = OrganizationMembership(
            id="mem_no_keys",
            user_id=member_without_keys.id,
            org_id=org_with_keys.id,
            role=MemberRole.MEMBER,
            has_org_key=False,
        )
        member_membership_no_keys.user = member_without_keys
        member_membership_no_keys.created_at = datetime.utcnow()

        pending_result = MagicMock()
        pending_result.scalars = MagicMock(return_value=MagicMock(
            all=MagicMock(return_value=[member_membership_no_keys])
        ))

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(org_with_keys),
            pending_result,
        ])

        service = OrgKeyService(mock_db)
        result = await service.get_pending_distributions("org_456", "admin_123")

        # Member without personal keys should be in needs_personal_setup, not ready_for_distribution
        assert result["ready_count"] == 0
        assert result["needs_setup_count"] == 1
        assert len(result["ready_for_distribution"]) == 0
        assert len(result["needs_personal_setup"]) == 1
        assert result["needs_personal_setup"][0]["user_id"] == member_without_keys.id

    @pytest.mark.asyncio
    async def test_raises_error_if_org_has_no_keys(
        self, mock_db, org_without_keys, admin_membership
    ):
        """Raises OrgKeysNotFoundError if org has no encryption keys."""
        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(org_without_keys),
        ])

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeysNotFoundError):
            await service.get_pending_distributions("org_123", "admin_123")


# =============================================================================
# Test Distribute Org Key
# =============================================================================

class TestDistributeOrgKey:
    """Tests for distribute_org_key method."""

    @pytest.mark.asyncio
    async def test_distributes_key_to_member(
        self, mock_db, admin_membership, member_membership, valid_distribution_data
    ):
        """Successfully distributes org key to member."""
        admin_membership.org_id = "org_456"
        admin_membership.has_org_key = True

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),  # verify_admin
            mock_execute_result(member_membership),  # get membership
        ])

        service = OrgKeyService(mock_db)
        result = await service.distribute_org_key(
            org_id="org_456",
            admin_user_id="admin_123",
            membership_id="mem_member",
            **valid_distribution_data,
        )

        assert result.has_org_key is True
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_error_if_membership_wrong_org(
        self, mock_db, admin_membership, member_membership, valid_distribution_data
    ):
        """Raises error if membership belongs to different org."""
        admin_membership.org_id = "org_456"
        member_membership.org_id = "org_different"

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(member_membership),
        ])

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeyServiceError, match="does not belong"):
            await service.distribute_org_key(
                org_id="org_456",
                admin_user_id="admin_123",
                membership_id="mem_member",
                **valid_distribution_data,
            )

    @pytest.mark.asyncio
    async def test_raises_error_if_already_has_key(
        self, mock_db, admin_membership, member_membership, valid_distribution_data
    ):
        """Raises error if member already has org key."""
        admin_membership.org_id = "org_456"
        member_membership.has_org_key = True

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(member_membership),
        ])

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeyServiceError, match="already has"):
            await service.distribute_org_key(
                org_id="org_456",
                admin_user_id="admin_123",
                membership_id="mem_member",
                **valid_distribution_data,
            )


# =============================================================================
# Test Get Org Encryption Status
# =============================================================================

class TestGetOrgEncryptionStatus:
    """Tests for get_org_encryption_status method."""

    @pytest.mark.asyncio
    async def test_returns_status_for_org_without_keys(self, mock_db, org_without_keys):
        """Returns status showing no keys."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(org_without_keys))

        service = OrgKeyService(mock_db)
        status = await service.get_org_encryption_status("org_123")

        assert status["has_encryption_keys"] is False
        assert status["org_public_key"] is None

    @pytest.mark.asyncio
    async def test_returns_status_for_org_with_keys(self, mock_db, org_with_keys):
        """Returns status showing keys exist."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(org_with_keys))

        service = OrgKeyService(mock_db)
        status = await service.get_org_encryption_status("org_456")

        assert status["has_encryption_keys"] is True
        assert status["org_public_key"] == "cc" * 32

    @pytest.mark.asyncio
    async def test_raises_error_for_nonexistent_org(self, mock_db):
        """Raises error for nonexistent org."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeyServiceError, match="not found"):
            await service.get_org_encryption_status("nonexistent")


# =============================================================================
# Test Get Member Org Key
# =============================================================================

class TestGetMemberOrgKey:
    """Tests for get_member_org_key method."""

    @pytest.mark.asyncio
    async def test_returns_encrypted_key(self, mock_db, member_membership):
        """Returns encrypted org key for member."""
        member_membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="admin_123",
        )

        mock_db.execute = AsyncMock(return_value=mock_execute_result(member_membership))

        service = OrgKeyService(mock_db)
        key = await service.get_member_org_key("member_456", "org_456")

        assert key["ephemeral_public_key"] == "aa" * 32
        assert key["ciphertext"] == "cc" * 48

    @pytest.mark.asyncio
    async def test_raises_error_if_no_org_key(self, mock_db, member_membership):
        """Raises OrgKeysNotFoundError if member has no org key."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(member_membership))

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeysNotFoundError):
            await service.get_member_org_key("member_456", "org_456")

    @pytest.mark.asyncio
    async def test_raises_error_for_non_member(self, mock_db):
        """Raises MembershipNotFoundError for non-member."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(None))

        service = OrgKeyService(mock_db)
        with pytest.raises(MembershipNotFoundError):
            await service.get_member_org_key("nonexistent", "org_456")


# =============================================================================
# Test Revoke Member Org Key
# =============================================================================

class TestRevokeMemberOrgKey:
    """Tests for revoke_member_org_key method."""

    @pytest.mark.asyncio
    async def test_revokes_member_key(self, mock_db, admin_membership, member_membership):
        """Successfully revokes member's org key."""
        admin_membership.org_id = "org_456"
        member_membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="admin_123",
        )

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),  # verify_admin
            mock_execute_result(member_membership),  # get_membership
        ])

        service = OrgKeyService(mock_db)
        await service.revoke_member_org_key(
            org_id="org_456",
            admin_user_id="admin_123",
            member_user_id="member_456",
            reason="Test revocation",
        )

        assert member_membership.has_org_key is False
        mock_db.commit.assert_called_once()
        mock_db.add.assert_called()  # Audit log

    @pytest.mark.asyncio
    async def test_noop_if_member_has_no_key(self, mock_db, admin_membership, member_membership):
        """Does nothing if member has no org key."""
        admin_membership.org_id = "org_456"
        member_membership.has_org_key = False

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(member_membership),
        ])

        service = OrgKeyService(mock_db)
        await service.revoke_member_org_key(
            org_id="org_456",
            admin_user_id="admin_123",
            member_user_id="member_456",
        )

        # Should not commit (no changes)
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_for_non_admin(self, mock_db, member_membership):
        """Raises NotAdminError for non-admin."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(member_membership))

        service = OrgKeyService(mock_db)
        with pytest.raises(NotAdminError):
            await service.revoke_member_org_key(
                org_id="org_456",
                admin_user_id="member_456",
                member_user_id="another_member",
            )


# =============================================================================
# Test Get Admin Recovery Key
# =============================================================================

class TestGetAdminRecoveryKey:
    """Tests for get_admin_recovery_key method."""

    @pytest.mark.asyncio
    async def test_returns_admin_encrypted_keys(self, mock_db, admin_membership, org_with_keys):
        """Returns admin-encrypted org keys for recovery."""
        admin_membership.org_id = org_with_keys.id

        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(org_with_keys),
        ])

        service = OrgKeyService(mock_db)
        keys = await service.get_admin_recovery_key("admin_123", "org_456")

        assert keys["org_public_key"] == "cc" * 32
        assert keys["admin_encrypted_private_key"] == "dd" * 48

    @pytest.mark.asyncio
    async def test_raises_error_for_non_admin(self, mock_db, member_membership):
        """Raises NotAdminError for non-admin."""
        mock_db.execute = AsyncMock(return_value=mock_execute_result(member_membership))

        service = OrgKeyService(mock_db)
        with pytest.raises(NotAdminError):
            await service.get_admin_recovery_key("member_456", "org_456")

    @pytest.mark.asyncio
    async def test_raises_error_if_org_has_no_keys(
        self, mock_db, admin_membership, org_without_keys
    ):
        """Raises OrgKeysNotFoundError if org has no keys."""
        mock_db.execute = AsyncMock(side_effect=[
            mock_execute_result(admin_membership),
            mock_execute_result(org_without_keys),
        ])

        service = OrgKeyService(mock_db)
        with pytest.raises(OrgKeysNotFoundError):
            await service.get_admin_recovery_key("admin_123", "org_123")
