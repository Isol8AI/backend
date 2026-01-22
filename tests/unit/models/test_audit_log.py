"""Tests for audit log model."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from models.audit_log import AuditLog, AuditEventType


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_event_type_values(self):
        """AuditEventType enum has expected string values."""
        assert AuditEventType.USER_KEYS_CREATED.value == "user_keys_created"
        assert AuditEventType.ORG_KEYS_CREATED.value == "org_keys_created"
        assert AuditEventType.ORG_KEY_DISTRIBUTED.value == "org_key_distributed"
        assert AuditEventType.MEMBER_ROLE_CHANGED.value == "member_role_changed"

    def test_event_type_is_string_enum(self):
        """AuditEventType values can be used as strings."""
        assert isinstance(AuditEventType.USER_KEYS_CREATED, str)
        assert AuditEventType.USER_KEYS_CREATED == "user_keys_created"


class TestAuditLogCreate:
    """Tests for AuditLog creation."""

    def test_create_basic_log(self):
        """Can create basic audit log entry."""
        log = AuditLog.create(
            id="log_123",
            event_type=AuditEventType.USER_KEYS_CREATED,
            actor_user_id="user_456",
        )

        assert log.id == "log_123"
        assert log.event_type == AuditEventType.USER_KEYS_CREATED
        assert log.actor_user_id == "user_456"
        assert log.target_user_id is None
        assert log.org_id is None
        assert log.event_data is None

    def test_create_log_with_all_fields(self):
        """Can create audit log with all fields."""
        log = AuditLog.create(
            id="log_123",
            event_type=AuditEventType.ORG_KEY_DISTRIBUTED,
            actor_user_id="admin_456",
            target_user_id="member_789",
            org_id="org_111",
            event_data={"extra": "data"},
        )

        assert log.actor_user_id == "admin_456"
        assert log.target_user_id == "member_789"
        assert log.org_id == "org_111"
        assert log.event_data == {"extra": "data"}


class TestAuditLogFactoryMethods:
    """Tests for AuditLog factory methods."""

    def test_log_user_keys_created(self):
        """Factory method for user key creation."""
        log = AuditLog.log_user_keys_created(
            id="log_123",
            user_id="user_456",
        )

        assert log.event_type == AuditEventType.USER_KEYS_CREATED
        assert log.actor_user_id == "user_456"
        assert log.target_user_id is None
        assert log.org_id is None

    def test_log_user_keys_recovered(self):
        """Factory method for user key recovery."""
        log = AuditLog.log_user_keys_recovered(
            id="log_123",
            user_id="user_456",
        )

        assert log.event_type == AuditEventType.USER_KEYS_RECOVERED
        assert log.actor_user_id == "user_456"

    def test_log_user_keys_deleted(self):
        """Factory method for user key deletion."""
        log = AuditLog.log_user_keys_deleted(
            id="log_123",
            user_id="user_456",
        )

        assert log.event_type == AuditEventType.USER_KEYS_DELETED
        assert log.actor_user_id == "user_456"

    def test_log_org_keys_created(self):
        """Factory method for org key creation."""
        log = AuditLog.log_org_keys_created(
            id="log_123",
            admin_user_id="admin_789",
            org_id="org_111",
        )

        assert log.event_type == AuditEventType.ORG_KEYS_CREATED
        assert log.actor_user_id == "admin_789"
        assert log.org_id == "org_111"
        assert log.target_user_id is None

    def test_log_org_keys_rotated(self):
        """Factory method for org key rotation."""
        log = AuditLog.log_org_keys_rotated(
            id="log_123",
            admin_user_id="admin_789",
            org_id="org_111",
        )

        assert log.event_type == AuditEventType.ORG_KEYS_ROTATED
        assert log.actor_user_id == "admin_789"
        assert log.org_id == "org_111"

    def test_log_org_key_distributed(self):
        """Factory method for org key distribution."""
        log = AuditLog.log_org_key_distributed(
            id="log_123",
            admin_user_id="admin_789",
            member_user_id="member_456",
            org_id="org_111",
        )

        assert log.event_type == AuditEventType.ORG_KEY_DISTRIBUTED
        assert log.actor_user_id == "admin_789"
        assert log.target_user_id == "member_456"
        assert log.org_id == "org_111"

    def test_log_org_key_revoked_without_reason(self):
        """Factory method for org key revocation without reason."""
        log = AuditLog.log_org_key_revoked(
            id="log_123",
            admin_user_id="admin_789",
            member_user_id="member_456",
            org_id="org_111",
        )

        assert log.event_type == AuditEventType.ORG_KEY_REVOKED
        assert log.actor_user_id == "admin_789"
        assert log.target_user_id == "member_456"
        assert log.org_id == "org_111"
        assert log.event_data is None

    def test_log_org_key_revoked_with_reason(self):
        """Factory method for org key revocation with reason."""
        log = AuditLog.log_org_key_revoked(
            id="log_123",
            admin_user_id="admin_789",
            member_user_id="member_456",
            org_id="org_111",
            reason="Left the company",
        )

        assert log.event_type == AuditEventType.ORG_KEY_REVOKED
        assert log.event_data == {"reason": "Left the company"}

    def test_log_member_joined(self):
        """Factory method for member joining."""
        log = AuditLog.log_member_joined(
            id="log_123",
            member_user_id="member_456",
            org_id="org_111",
            role="member",
        )

        assert log.event_type == AuditEventType.MEMBER_JOINED
        assert log.actor_user_id == "member_456"
        assert log.target_user_id == "member_456"
        assert log.org_id == "org_111"
        assert log.event_data == {"role": "member"}

    def test_log_member_left(self):
        """Factory method for member leaving."""
        log = AuditLog.log_member_left(
            id="log_123",
            member_user_id="member_456",
            org_id="org_111",
            reason="Resigned",
        )

        assert log.event_type == AuditEventType.MEMBER_LEFT
        assert log.actor_user_id == "member_456"
        assert log.target_user_id == "member_456"
        assert log.event_data == {"reason": "Resigned"}

    def test_log_member_role_changed(self):
        """Factory method for member role change."""
        log = AuditLog.log_member_role_changed(
            id="log_123",
            admin_user_id="admin_789",
            member_user_id="member_456",
            org_id="org_111",
            old_role="org:member",
            new_role="org:admin",
        )

        assert log.event_type == AuditEventType.MEMBER_ROLE_CHANGED
        assert log.actor_user_id == "admin_789"
        assert log.target_user_id == "member_456"
        assert log.org_id == "org_111"
        assert log.event_data["old_role"] == "org:member"
        assert log.event_data["new_role"] == "org:admin"

    def test_log_encrypted_session_created(self):
        """Factory method for encrypted session creation."""
        log = AuditLog.log_encrypted_session_created(
            id="log_123",
            user_id="user_456",
            session_id="session_789",
            org_id="org_111",
        )

        assert log.event_type == AuditEventType.ENCRYPTED_SESSION_CREATED
        assert log.actor_user_id == "user_456"
        assert log.org_id == "org_111"
        assert log.event_data == {"session_id": "session_789"}

    def test_log_encrypted_session_created_personal(self):
        """Factory method for personal encrypted session."""
        log = AuditLog.log_encrypted_session_created(
            id="log_123",
            user_id="user_456",
            session_id="session_789",
        )

        assert log.event_type == AuditEventType.ENCRYPTED_SESSION_CREATED
        assert log.org_id is None
        assert log.event_data == {"session_id": "session_789"}

    def test_log_user_signed_in(self):
        """Factory method for user sign in."""
        log = AuditLog.log_user_signed_in(
            id="log_123",
            user_id="user_456",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert log.event_type == AuditEventType.USER_SIGNED_IN
        assert log.actor_user_id == "user_456"
        assert log.event_data["ip_address"] == "192.168.1.1"
        assert log.event_data["user_agent"] == "Mozilla/5.0"

    def test_log_user_signed_in_minimal(self):
        """Factory method for user sign in without optional fields."""
        log = AuditLog.log_user_signed_in(
            id="log_123",
            user_id="user_456",
        )

        assert log.event_type == AuditEventType.USER_SIGNED_IN
        assert log.event_data is None

    def test_log_user_signed_out(self):
        """Factory method for user sign out."""
        log = AuditLog.log_user_signed_out(
            id="log_123",
            user_id="user_456",
        )

        assert log.event_type == AuditEventType.USER_SIGNED_OUT
        assert log.actor_user_id == "user_456"


class TestAuditLogApiResponse:
    """Tests for AuditLog API response formatting."""

    def test_to_api_response(self):
        """to_api_response formats correctly."""
        log = AuditLog.create(
            id="log_123",
            event_type=AuditEventType.ORG_KEYS_CREATED,
            actor_user_id="user_456",
            org_id="org_789",
            event_data={"key": "value"},
        )
        log.created_at = datetime(2024, 1, 15, 10, 30, 0)

        response = log.to_api_response()

        assert response["id"] == "log_123"
        assert response["event_type"] == "org_keys_created"
        assert response["actor_user_id"] == "user_456"
        assert response["target_user_id"] is None
        assert response["org_id"] == "org_789"
        assert response["event_data"] == {"key": "value"}
        assert response["created_at"] == "2024-01-15T10:30:00"

    def test_to_api_response_minimal(self):
        """to_api_response works with minimal data."""
        log = AuditLog.create(
            id="log_123",
            event_type=AuditEventType.USER_KEYS_CREATED,
        )
        log.created_at = datetime.utcnow()

        response = log.to_api_response()

        assert response["id"] == "log_123"
        assert response["event_type"] == "user_keys_created"
        assert response["actor_user_id"] is None
        assert response["event_data"] is None


class TestAuditLogPersistence:
    """Tests for AuditLog database persistence."""

    @pytest.mark.asyncio
    async def test_audit_log_persistence(self, db_session, test_user):
        """Audit log can be persisted and retrieved."""
        log = AuditLog.log_user_keys_created(
            id=str(uuid.uuid4()),
            user_id=test_user.id,
        )
        db_session.add(log)
        await db_session.flush()

        result = await db_session.execute(select(AuditLog).where(AuditLog.id == log.id))
        fetched = result.scalar_one()

        assert fetched.event_type == AuditEventType.USER_KEYS_CREATED
        assert fetched.actor_user_id == test_user.id
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    async def test_audit_log_with_metadata_persistence(self, db_session, test_user, test_organization):
        """Audit log with metadata persists correctly."""
        log = AuditLog.log_member_role_changed(
            id=str(uuid.uuid4()),
            admin_user_id=test_user.id,
            member_user_id=test_user.id,
            org_id=test_organization.id,
            old_role="org:member",
            new_role="org:admin",
        )
        db_session.add(log)
        await db_session.flush()

        result = await db_session.execute(select(AuditLog).where(AuditLog.id == log.id))
        fetched = result.scalar_one()

        assert fetched.event_data["old_role"] == "org:member"
        assert fetched.event_data["new_role"] == "org:admin"

    @pytest.mark.asyncio
    async def test_audit_log_org_relationship(self, db_session, test_user, test_organization):
        """Audit log has correct organization relationship."""
        log = AuditLog.log_org_keys_created(
            id=str(uuid.uuid4()),
            admin_user_id=test_user.id,
            org_id=test_organization.id,
        )
        db_session.add(log)
        await db_session.flush()

        result = await db_session.execute(select(AuditLog).where(AuditLog.id == log.id))
        fetched = result.scalar_one()

        assert fetched.org_id == test_organization.id

    @pytest.mark.asyncio
    async def test_multiple_audit_logs_query(self, db_session, test_user, test_organization):
        """Can query multiple audit logs by org_id."""
        log1 = AuditLog.log_org_keys_created(
            id=str(uuid.uuid4()),
            admin_user_id=test_user.id,
            org_id=test_organization.id,
        )
        log2 = AuditLog.log_org_key_distributed(
            id=str(uuid.uuid4()),
            admin_user_id=test_user.id,
            member_user_id=test_user.id,
            org_id=test_organization.id,
        )
        db_session.add(log1)
        db_session.add(log2)
        await db_session.flush()

        result = await db_session.execute(
            select(AuditLog).where(AuditLog.org_id == test_organization.id).order_by(AuditLog.created_at)
        )
        logs = result.scalars().all()

        assert len(logs) == 2
        assert logs[0].event_type == AuditEventType.ORG_KEYS_CREATED
        assert logs[1].event_type == AuditEventType.ORG_KEY_DISTRIBUTED
