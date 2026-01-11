from sqlalchemy import String, Column
from sqlalchemy.orm import relationship

from .base import Base


class User(Base):
    """User model synced from Clerk authentication."""

    __tablename__ = "users"

    id = Column(String, primary_key=True)  # Clerk User ID

    # Relationships
    memberships = relationship(
        "OrganizationMembership",
        back_populates="user",
        cascade="all, delete-orphan"
    )
