"""
Audit log model for security-relevant events.

Security Note:
- Logs are append-only (never updated or deleted in normal operation)
- Contains NO encrypted content or keys - only metadata
- Used for compliance, debugging, and security monitoring
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, DateTime, ForeignKey, Index, Enum, JSON
from sqlalchemy.orm import relationship

from .base import Base


class AuditEventType(str, PyEnum):
    """Types of security-relevant events."""

    # User key events
    USER_KEYS_CREATED = "user_keys_created"
    USER_KEYS_RECOVERED = "user_keys_recovered"  # Recovery code used to restore keys
    RECOVERY_KEYS_FETCHED = "recovery_keys_fetched"  # Recovery keys fetched (before actual recovery)
    USER_KEYS_DELETED = "user_keys_deleted"

    # Org key events
    ORG_KEYS_CREATED = "org_keys_created"
    ORG_KEYS_ROTATED = "org_keys_rotated"

    # Key distribution events
    ORG_KEY_DISTRIBUTED = "org_key_distributed"
    ORG_KEY_REVOKED = "org_key_revoked"

    # Membership events
    MEMBER_JOINED = "member_joined"
    MEMBER_LEFT = "member_left"
    MEMBER_ROLE_CHANGED = "member_role_changed"

    # Session events
    ENCRYPTED_SESSION_CREATED = "encrypted_session_created"
    SESSION_DELETED = "session_deleted"  # For GDPR compliance

    # Authentication events (from Clerk webhooks)
    USER_SIGNED_IN = "user_signed_in"
    USER_SIGNED_OUT = "user_signed_out"

    # Admin actions
    ADMIN_RECOVERY_INITIATED = "admin_recovery_initiated"
    ADMIN_PASSCODE_CHANGED = "admin_passcode_changed"


class AuditLog(Base):
    """
    Immutable audit log for security events.

    Every security-relevant action is logged here for:
    - Compliance requirements
    - Security incident investigation
    - Debugging key distribution issues

    Fields:
        event_type: What happened
        actor_user_id: Who did it
        target_user_id: Who was affected (if applicable)
        org_id: Which org (if applicable)
        metadata: Additional context (JSON)
    """

    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True)  # UUID

    # Event type
    event_type = Column(Enum(AuditEventType), nullable=False)

    # Actor - who performed the action
    actor_user_id = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Target - who was affected (for member operations)
    target_user_id = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Organization context
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True
    )

    # Additional event details (flexible JSON)
    # Examples:
    # - {"session_id": "xxx"} for session events
    # - {"old_role": "member", "new_role": "admin"} for role changes
    # - {"ip_address": "x.x.x.x"} for auth events
    # Note: Named 'event_data' instead of 'metadata' (reserved by SQLAlchemy)
    event_data = Column(JSON, nullable=True)

    # Timestamp (never modified)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # =========================================================================
    # Relationships
    # =========================================================================

    actor = relationship("User", foreign_keys=[actor_user_id])
    target = relationship("User", foreign_keys=[target_user_id])
    organization = relationship("Organization", back_populates="audit_logs")

    # =========================================================================
    # Indexes
    # =========================================================================

    __table_args__ = (
        Index("ix_audit_logs_event_type", "event_type"),
        Index("ix_audit_logs_actor", "actor_user_id"),
        Index("ix_audit_logs_target", "target_user_id"),
        Index("ix_audit_logs_org", "org_id"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_org_created", "org_id", "created_at"),
        Index("ix_audit_logs_event_created", "event_type", "created_at"),
    )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create(
        cls,
        id: str,
        event_type: AuditEventType,
        actor_user_id: Optional[str] = None,
        target_user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> "AuditLog":
        """
        Create an audit log entry.

        Args:
            id: UUID for the log entry
            event_type: Type of event
            actor_user_id: User who performed the action
            target_user_id: User affected by the action (optional)
            org_id: Organization context (optional)
            event_data: Additional context (optional)
        """
        return cls(
            id=id,
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_user_id=target_user_id,
            org_id=org_id,
            event_data=event_data,
        )

    @classmethod
    def log_user_keys_created(cls, id: str, user_id: str) -> "AuditLog":
        """Log user encryption key creation."""
        return cls.create(
            id=id,
            event_type=AuditEventType.USER_KEYS_CREATED,
            actor_user_id=user_id,
        )

    @classmethod
    def log_user_keys_recovered(cls, id: str, user_id: str) -> "AuditLog":
        """Log user encryption key recovery using recovery code."""
        return cls.create(
            id=id,
            event_type=AuditEventType.USER_KEYS_RECOVERED,
            actor_user_id=user_id,
        )

    @classmethod
    def log_recovery_keys_fetched(cls, id: str, user_id: str) -> "AuditLog":
        """Log when recovery keys are fetched (before actual recovery)."""
        return cls.create(
            id=id,
            event_type=AuditEventType.RECOVERY_KEYS_FETCHED,
            actor_user_id=user_id,
        )

    @classmethod
    def log_user_keys_deleted(cls, id: str, user_id: str) -> "AuditLog":
        """Log user encryption key deletion."""
        return cls.create(
            id=id,
            event_type=AuditEventType.USER_KEYS_DELETED,
            actor_user_id=user_id,
        )

    @classmethod
    def log_org_keys_created(
        cls,
        id: str,
        admin_user_id: str,
        org_id: str,
    ) -> "AuditLog":
        """Log org encryption key creation."""
        return cls.create(
            id=id,
            event_type=AuditEventType.ORG_KEYS_CREATED,
            actor_user_id=admin_user_id,
            org_id=org_id,
        )

    @classmethod
    def log_org_keys_rotated(
        cls,
        id: str,
        admin_user_id: str,
        org_id: str,
    ) -> "AuditLog":
        """Log org encryption key rotation."""
        return cls.create(
            id=id,
            event_type=AuditEventType.ORG_KEYS_ROTATED,
            actor_user_id=admin_user_id,
            org_id=org_id,
        )

    @classmethod
    def log_org_key_distributed(
        cls,
        id: str,
        admin_user_id: str,
        member_user_id: str,
        org_id: str,
    ) -> "AuditLog":
        """Log org key distribution to a member."""
        return cls.create(
            id=id,
            event_type=AuditEventType.ORG_KEY_DISTRIBUTED,
            actor_user_id=admin_user_id,
            target_user_id=member_user_id,
            org_id=org_id,
        )

    @classmethod
    def log_org_key_revoked(
        cls,
        id: str,
        admin_user_id: str,
        member_user_id: str,
        org_id: str,
        reason: Optional[str] = None,
    ) -> "AuditLog":
        """Log org key revocation from a member."""
        return cls.create(
            id=id,
            event_type=AuditEventType.ORG_KEY_REVOKED,
            actor_user_id=admin_user_id,
            target_user_id=member_user_id,
            org_id=org_id,
            event_data={"reason": reason} if reason else None,
        )

    @classmethod
    def log_member_joined(
        cls,
        id: str,
        member_user_id: str,
        org_id: str,
        role: str,
    ) -> "AuditLog":
        """Log member joining an organization."""
        return cls.create(
            id=id,
            event_type=AuditEventType.MEMBER_JOINED,
            actor_user_id=member_user_id,
            target_user_id=member_user_id,
            org_id=org_id,
            event_data={"role": role},
        )

    @classmethod
    def log_member_left(
        cls,
        id: str,
        member_user_id: str,
        org_id: str,
        reason: Optional[str] = None,
    ) -> "AuditLog":
        """Log member leaving an organization."""
        return cls.create(
            id=id,
            event_type=AuditEventType.MEMBER_LEFT,
            actor_user_id=member_user_id,
            target_user_id=member_user_id,
            org_id=org_id,
            event_data={"reason": reason} if reason else None,
        )

    @classmethod
    def log_member_role_changed(
        cls,
        id: str,
        admin_user_id: str,
        member_user_id: str,
        org_id: str,
        old_role: str,
        new_role: str,
    ) -> "AuditLog":
        """Log member role change."""
        return cls.create(
            id=id,
            event_type=AuditEventType.MEMBER_ROLE_CHANGED,
            actor_user_id=admin_user_id,
            target_user_id=member_user_id,
            org_id=org_id,
            event_data={"old_role": old_role, "new_role": new_role},
        )

    @classmethod
    def log_encrypted_session_created(
        cls,
        id: str,
        user_id: str,
        session_id: str,
        org_id: Optional[str] = None,
    ) -> "AuditLog":
        """Log encrypted chat session creation."""
        return cls.create(
            id=id,
            event_type=AuditEventType.ENCRYPTED_SESSION_CREATED,
            actor_user_id=user_id,
            org_id=org_id,
            event_data={"session_id": session_id},
        )

    @classmethod
    def log_session_deleted(
        cls,
        id: str,
        user_id: str,
        session_id: str,
        org_id: Optional[str] = None,
    ) -> "AuditLog":
        """Log session deletion (GDPR compliance)."""
        return cls.create(
            id=id,
            event_type=AuditEventType.SESSION_DELETED,
            actor_user_id=user_id,
            org_id=org_id,
            event_data={"session_id": session_id},
        )

    @classmethod
    def log_user_signed_in(
        cls,
        id: str,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> "AuditLog":
        """Log user sign in event."""
        data = {}
        if ip_address:
            data["ip_address"] = ip_address
        if user_agent:
            data["user_agent"] = user_agent
        return cls.create(
            id=id,
            event_type=AuditEventType.USER_SIGNED_IN,
            actor_user_id=user_id,
            event_data=data if data else None,
        )

    @classmethod
    def log_user_signed_out(
        cls,
        id: str,
        user_id: str,
    ) -> "AuditLog":
        """Log user sign out event."""
        return cls.create(
            id=id,
            event_type=AuditEventType.USER_SIGNED_OUT,
            actor_user_id=user_id,
        )

    def to_api_response(self) -> dict:
        """Convert to API response format."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "actor_user_id": self.actor_user_id,
            "target_user_id": self.target_user_id,
            "org_id": self.org_id,
            "event_data": self.event_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
