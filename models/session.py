"""
Chat session model.

Sessions can be personal (org_id=None) or organization-scoped (org_id set).
All messages in sessions are encrypted - this is a zero-trust platform.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import relationship

from .base import Base


class Session(Base):
    """
    Chat session containing encrypted messages.

    All messages in sessions are encrypted. The encryption key used depends on context:
    - Personal sessions (org_id=NULL): Encrypted to user's public key
    - Organization sessions (org_id set): Encrypted to org's public key
    """
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    org_id = Column(
        String,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    name = Column(String, default="New Chat")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index for common query pattern (user's sessions in an org context)
    __table_args__ = (
        Index("ix_sessions_user_org", "user_id", "org_id"),
    )

    # Relationships
    user = relationship("User", back_populates="sessions")
    organization = relationship("Organization", back_populates="sessions")
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

    # =========================================================================
    # Helper Properties
    # =========================================================================

    @property
    def encryption_key_type(self) -> str:
        """Determine which key type is used for this session's messages."""
        return "org" if self.org_id else "user"

    @property
    def is_org_session(self) -> bool:
        """Check if this is an organization session."""
        return self.org_id is not None

    @property
    def is_personal_session(self) -> bool:
        """Check if this is a personal (non-org) session."""
        return self.org_id is None
