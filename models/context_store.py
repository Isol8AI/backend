"""ContextStore model for RAG-ready context storage."""
from datetime import datetime

from sqlalchemy import Column, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base


class ContextStore(Base):
    """Context store model for user and organization context.

    This model provides the database layer for the context abstraction.
    It supports different storage types (postgres, vector) to allow
    future migration to vector databases for RAG functionality.
    """

    __tablename__ = "context_stores"
    __table_args__ = (
        UniqueConstraint("owner_type", "owner_id", name="uq_owner_type_id"),
    )

    id = Column(String, primary_key=True)
    owner_type = Column(String, nullable=False)  # 'user' or 'org'
    owner_id = Column(String, nullable=False)  # user_id or org_id
    store_type = Column(String, default="postgres")  # 'postgres', 'vector', etc.
    context_data = Column(JSONB, nullable=True)  # Config and simple key-value context

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
