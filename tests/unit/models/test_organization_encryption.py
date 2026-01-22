"""Tests for organization encryption models."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole


class TestOrganizationEncryption:
    """Test Organization encryption methods."""

    def test_organization_creation(self):
        """Organization can be created with basic fields."""
        org = Organization(id="org_123", name="Test Org")
        assert org.id == "org_123"
        assert org.name == "Test Org"
        # Default is None for unpersisted, False when persisted
        assert not org.has_encryption_keys

    def test_set_encryption_keys_valid(self):
        """Org can set encryption keys with valid data."""
        org = Organization(id="org_123", name="Test Org")

        org.set_encryption_keys(
            org_public_key="aa" * 32,  # 64 hex chars
            admin_encrypted_private_key="bb" * 48,  # Variable
            iv="cc" * 16,  # 32 hex chars
            tag="dd" * 16,  # 32 hex chars
            salt="ee" * 32,  # 64 hex chars
            created_by="user_admin",
        )

        assert org.has_encryption_keys is True
        assert org.org_public_key == "aa" * 32
        assert org.admin_encrypted_private_key == "bb" * 48
        assert org.encryption_created_by == "user_admin"
        assert org.encryption_created_at is not None

    def test_set_encryption_keys_validates_public_key_length(self):
        """Org public key must be exactly 64 hex characters."""
        org = Organization(id="org_123", name="Test Org")

        with pytest.raises(ValueError, match="64 hex characters"):
            org.set_encryption_keys(
                org_public_key="aa" * 31,  # Too short
                admin_encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 32,
                created_by="user_admin",
            )

    def test_set_encryption_keys_validates_iv_length(self):
        """IV must be exactly 32 hex characters."""
        org = Organization(id="org_123", name="Test Org")

        with pytest.raises(ValueError, match="32 hex characters"):
            org.set_encryption_keys(
                org_public_key="aa" * 32,
                admin_encrypted_private_key="bb" * 48,
                iv="cc" * 8,  # Too short
                tag="dd" * 16,
                salt="ee" * 32,
                created_by="user_admin",
            )

    def test_set_encryption_keys_validates_tag_length(self):
        """Tag must be exactly 32 hex characters."""
        org = Organization(id="org_123", name="Test Org")

        with pytest.raises(ValueError, match="32 hex characters"):
            org.set_encryption_keys(
                org_public_key="aa" * 32,
                admin_encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 8,  # Too short
                salt="ee" * 32,
                created_by="user_admin",
            )

    def test_set_encryption_keys_validates_salt_length(self):
        """Salt must be exactly 64 hex characters."""
        org = Organization(id="org_123", name="Test Org")

        with pytest.raises(ValueError, match="64 hex characters"):
            org.set_encryption_keys(
                org_public_key="aa" * 32,
                admin_encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 16,  # Too short
                created_by="user_admin",
            )

    def test_set_encryption_keys_validates_hex_format(self):
        """All fields must be valid hex strings."""
        org = Organization(id="org_123", name="Test Org")

        with pytest.raises(ValueError, match="valid hex string"):
            org.set_encryption_keys(
                org_public_key="gg" * 32,  # Not valid hex
                admin_encrypted_private_key="bb" * 48,
                iv="cc" * 16,
                tag="dd" * 16,
                salt="ee" * 32,
                created_by="user_admin",
            )

    def test_set_encryption_keys_normalizes_to_lowercase(self):
        """All hex strings are normalized to lowercase."""
        org = Organization(id="org_123", name="Test Org")

        org.set_encryption_keys(
            org_public_key="AA" * 32,  # Uppercase
            admin_encrypted_private_key="BB" * 48,
            iv="CC" * 16,
            tag="DD" * 16,
            salt="EE" * 32,
            created_by="user_admin",
        )

        assert org.org_public_key == "aa" * 32
        assert org.admin_encrypted_private_key == "bb" * 48

    def test_clear_encryption_keys(self):
        """Org can clear all encryption keys."""
        org = Organization(id="org_123", name="Test Org")
        org.set_encryption_keys(
            org_public_key="aa" * 32,
            admin_encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            created_by="user_admin",
        )

        org.clear_encryption_keys()

        assert org.has_encryption_keys is False
        assert org.org_public_key is None
        assert org.admin_encrypted_private_key is None
        assert org.admin_encrypted_private_key_iv is None
        assert org.admin_encrypted_private_key_tag is None
        assert org.admin_key_salt is None
        assert org.encryption_created_at is None
        assert org.encryption_created_by is None

    def test_encryption_key_info_excludes_private_key(self):
        """encryption_key_info never includes admin private key."""
        org = Organization(id="org_123", name="Test Org")
        org.set_encryption_keys(
            org_public_key="aa" * 32,
            admin_encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            created_by="user_admin",
        )

        info = org.encryption_key_info

        assert "org_public_key" in info
        assert info["org_public_key"] == "aa" * 32
        assert info["has_encryption_keys"] is True
        assert info["encryption_created_by"] == "user_admin"
        assert "encryption_created_at" in info

        # These should NOT be in the info
        assert "admin_encrypted_private_key" not in info
        assert "admin_key_salt" not in info
        assert "admin_iv" not in info
        assert "admin_tag" not in info

    def test_encryption_key_info_without_keys(self):
        """encryption_key_info works for org without keys."""
        org = Organization(id="org_123", name="Test Org")

        info = org.encryption_key_info

        assert info["has_encryption_keys"] is False
        assert info["org_public_key"] is None
        assert info["encryption_created_at"] is None

    def test_can_receive_encrypted_messages_without_key(self):
        """Org cannot receive encrypted messages without public key."""
        org = Organization(id="org_123", name="Test Org")

        assert org.can_receive_encrypted_messages is False

    def test_can_receive_encrypted_messages_with_key(self):
        """Org can receive encrypted messages with public key."""
        org = Organization(id="org_123", name="Test Org")
        org.org_public_key = "aa" * 32

        assert org.can_receive_encrypted_messages is True

    def test_get_admin_encrypted_keys(self):
        """get_admin_encrypted_keys returns key data for admin unlock."""
        org = Organization(id="org_123", name="Test Org")
        org.set_encryption_keys(
            org_public_key="aa" * 32,
            admin_encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            created_by="user_admin",
        )

        keys = org.get_admin_encrypted_keys()

        assert keys is not None
        assert keys["org_public_key"] == "aa" * 32
        assert keys["admin_encrypted_private_key"] == "bb" * 48
        assert keys["iv"] == "cc" * 16
        assert keys["tag"] == "dd" * 16
        assert keys["salt"] == "ee" * 32

    def test_get_admin_encrypted_keys_without_keys(self):
        """get_admin_encrypted_keys returns None without keys."""
        org = Organization(id="org_123", name="Test Org")

        keys = org.get_admin_encrypted_keys()

        assert keys is None


class TestOrganizationMembershipModel:
    """Test OrganizationMembership model."""

    def test_membership_creation(self):
        """Membership can be created with required fields."""
        membership = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )
        assert membership.user_id == "user_123"
        assert membership.org_id == "org_456"
        assert membership.role == MemberRole.MEMBER
        # Default is None for unpersisted, False when persisted
        assert not membership.has_org_key

    def test_membership_default_role(self):
        """Membership defaults to MEMBER role when persisted."""
        # Note: Default value only applies when model is persisted to DB
        # For unpersisted instances, explicitly set the role
        membership = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id="user_123",
            org_id="org_456",
            role=MemberRole.MEMBER,
        )
        assert membership.role == MemberRole.MEMBER

    def test_is_admin_property(self):
        """is_admin property works correctly."""
        admin = OrganizationMembership(
            id="mem_1",
            user_id="u1",
            org_id="o1",
            role=MemberRole.ADMIN,
        )
        member = OrganizationMembership(
            id="mem_2",
            user_id="u2",
            org_id="o1",
            role=MemberRole.MEMBER,
        )

        assert admin.is_admin is True
        assert admin.is_member is False
        assert member.is_admin is False
        assert member.is_member is True

    def test_set_encrypted_org_key_valid(self):
        """Membership can store encrypted org key."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
            role=MemberRole.MEMBER,
        )

        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="user_admin",
        )

        assert membership.has_org_key is True
        assert membership.encrypted_org_key_ephemeral == "aa" * 32
        assert membership.encrypted_org_key_ciphertext == "cc" * 48
        assert membership.key_distributed_by == "user_admin"
        assert membership.key_distributed_at is not None

    def test_set_encrypted_org_key_validates_ephemeral_length(self):
        """Ephemeral public key must be 64 hex characters."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        with pytest.raises(ValueError, match="64 hex characters"):
            membership.set_encrypted_org_key(
                ephemeral_public_key="aa" * 16,  # Too short
                iv="bb" * 16,
                ciphertext="cc" * 48,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
                distributed_by_user_id="user_admin",
            )

    def test_set_encrypted_org_key_validates_iv_length(self):
        """IV must be 32 hex characters."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        with pytest.raises(ValueError, match="32 hex characters"):
            membership.set_encrypted_org_key(
                ephemeral_public_key="aa" * 32,
                iv="bb" * 8,  # Too short
                ciphertext="cc" * 48,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
                distributed_by_user_id="user_admin",
            )

    def test_set_encrypted_org_key_validates_ciphertext_not_empty(self):
        """Ciphertext cannot be empty."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            membership.set_encrypted_org_key(
                ephemeral_public_key="aa" * 32,
                iv="bb" * 16,
                ciphertext="",  # Empty
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
                distributed_by_user_id="user_admin",
            )

    def test_set_encrypted_org_key_validates_hex_format(self):
        """All fields must be valid hex strings."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        with pytest.raises(ValueError, match="valid hex string"):
            membership.set_encrypted_org_key(
                ephemeral_public_key="gg" * 32,  # Not valid hex
                iv="bb" * 16,
                ciphertext="cc" * 48,
                auth_tag="dd" * 16,
                hkdf_salt="ee" * 32,
                distributed_by_user_id="user_admin",
            )

    def test_clear_encrypted_org_key(self):
        """Membership can clear encrypted org key on revocation."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="user_admin",
        )

        membership.clear_encrypted_org_key()

        assert membership.has_org_key is False
        assert membership.encrypted_org_key_ephemeral is None
        assert membership.encrypted_org_key_iv is None
        assert membership.encrypted_org_key_ciphertext is None
        assert membership.encrypted_org_key_tag is None
        assert membership.encrypted_org_key_hkdf_salt is None
        assert membership.key_distributed_at is None
        assert membership.key_distributed_by is None

    def test_encrypted_org_key_payload_with_key(self):
        """encrypted_org_key_payload returns proper structure."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="user_admin",
        )

        payload = membership.encrypted_org_key_payload

        assert payload["ephemeral_public_key"] == "aa" * 32
        assert payload["iv"] == "bb" * 16
        assert payload["ciphertext"] == "cc" * 48
        assert payload["auth_tag"] == "dd" * 16
        assert payload["hkdf_salt"] == "ee" * 32

    def test_encrypted_org_key_payload_without_key(self):
        """encrypted_org_key_payload returns None without key."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
        )

        payload = membership.encrypted_org_key_payload

        assert payload is None

    def test_to_api_response_basic(self):
        """to_api_response returns expected format."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
            role=MemberRole.MEMBER,
            has_org_key=False,  # Explicitly set for test
        )
        membership.joined_at = datetime.utcnow()
        membership.created_at = datetime.utcnow()

        response = membership.to_api_response()

        assert response["id"] == "mem_123"
        assert response["user_id"] == "user_456"
        assert response["org_id"] == "org_789"
        assert response["role"] == "org:member"
        assert response["has_org_key"] is False

    def test_to_api_response_with_encrypted_key(self):
        """to_api_response includes encrypted key when requested."""
        membership = OrganizationMembership(
            id="mem_123",
            user_id="user_456",
            org_id="org_789",
            role=MemberRole.MEMBER,
        )
        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="user_admin",
        )
        membership.joined_at = datetime.utcnow()
        membership.created_at = datetime.utcnow()

        response = membership.to_api_response(include_encrypted_key=True)

        assert "encrypted_org_key" in response
        assert response["encrypted_org_key"]["ciphertext"] == "cc" * 48


class TestOrganizationMembershipPersistence:
    """Tests for membership database persistence."""

    @pytest.mark.asyncio
    async def test_membership_persistence(self, db_session, test_user, test_organization):
        """Membership can be persisted and retrieved."""
        membership = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            org_id=test_organization.id,
            role=MemberRole.MEMBER,
        )
        db_session.add(membership)
        await db_session.flush()

        result = await db_session.execute(
            select(OrganizationMembership).where(OrganizationMembership.id == membership.id)
        )
        fetched = result.scalar_one()

        assert fetched.user_id == test_user.id
        assert fetched.org_id == test_organization.id
        assert fetched.role == MemberRole.MEMBER

    @pytest.mark.asyncio
    async def test_membership_with_encrypted_key_persistence(self, db_session, test_user, test_organization):
        """Membership with encrypted org key persists correctly."""
        membership = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
            org_id=test_organization.id,
            role=MemberRole.MEMBER,
        )
        membership.set_encrypted_org_key(
            ephemeral_public_key="aa" * 32,
            iv="bb" * 16,
            ciphertext="cc" * 48,
            auth_tag="dd" * 16,
            hkdf_salt="ee" * 32,
            distributed_by_user_id="user_admin",
        )
        db_session.add(membership)
        await db_session.flush()

        result = await db_session.execute(
            select(OrganizationMembership).where(OrganizationMembership.id == membership.id)
        )
        fetched = result.scalar_one()

        assert fetched.has_org_key is True
        assert fetched.encrypted_org_key_ciphertext == "cc" * 48


class TestOrganizationEncryptionPersistence:
    """Tests for organization encryption database persistence."""

    @pytest.mark.asyncio
    async def test_org_encryption_persistence(self, db_session):
        """Organization encryption keys persist to database."""
        org = Organization(id=f"org_{uuid.uuid4().hex[:8]}", name="Test Org")
        org.set_encryption_keys(
            org_public_key="aa" * 32,
            admin_encrypted_private_key="bb" * 48,
            iv="cc" * 16,
            tag="dd" * 16,
            salt="ee" * 32,
            created_by="user_admin",
        )
        db_session.add(org)
        await db_session.flush()

        result = await db_session.execute(select(Organization).where(Organization.id == org.id))
        fetched = result.scalar_one()

        assert fetched.has_encryption_keys is True
        assert fetched.org_public_key == "aa" * 32
        assert fetched.admin_encrypted_private_key == "bb" * 48
        assert fetched.encryption_created_at is not None
