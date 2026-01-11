"""Organization model for multi-tenant support."""
from datetime import datetime

from sqlalchemy import Column, DateTime, String
from sqlalchemy.orm import relationship

from .base import Base


class Organization(Base):
    """Organization model representing a team/company workspace.

    Organizations are synced from Clerk and provide multi-tenant isolation.
    Future features include custom model endpoints and fine-tuning.
    """

    __tablename__ = "organizations"

    id = Column(String, primary_key=True)  # Clerk org_id
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=True)

    # Future: model endpoint configuration
    custom_model_endpoint = Column(String, nullable=True)  # For self-hosted models
    fine_tuned_model_id = Column(String, nullable=True)  # For fine-tuned models

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    memberships = relationship(
        "OrganizationMembership",
        back_populates="organization",
        cascade="all, delete-orphan"
    )
