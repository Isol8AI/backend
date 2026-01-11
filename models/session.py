import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship

from .base import Base

class Session(Base):
    """Chat session model.

    Sessions can be personal (org_id=None) or organization-scoped (org_id set).
    All sessions are private to the owner user, even within an organization.
    """

    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=True)  # None = personal session
    name = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    organization = relationship("Organization")
