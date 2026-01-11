"""OrganizationMembership model for user-organization relationships."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship

from .base import Base


class OrganizationMembership(Base):
    """Membership model linking users to organizations with roles.

    Each user can be a member of multiple organizations, but only one
    membership per user-org pair is allowed.
    """

    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "org_id", name="uq_user_org"),
    )

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    role = Column(String, default="org:member")  # org:admin or org:member
    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="memberships")
