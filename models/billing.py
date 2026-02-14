"""Database models for billing and usage tracking."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from models.base import Base


class ModelPricing(Base):
    """Per-model token pricing, mirroring AWS Bedrock costs."""

    __tablename__ = "model_pricing"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    input_cost_per_token = Column(Numeric(20, 12), nullable=False)
    output_cost_per_token = Column(Numeric(20, 12), nullable=False)
    effective_from = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (Index("idx_model_pricing_active", "model_id", unique=True, postgresql_where=is_active.is_(True)),)

    def __repr__(self) -> str:
        return f"<ModelPricing(model_id={self.model_id}, active={self.is_active})>"


class BillingAccount(Base):
    """Maps Clerk users/orgs to Stripe customers."""

    __tablename__ = "billing_account"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clerk_user_id = Column(String, nullable=True)
    clerk_org_id = Column(String, nullable=True)
    stripe_customer_id = Column(String, nullable=False, unique=True)
    stripe_subscription_id = Column(String, nullable=True)
    plan_tier = Column(String, nullable=False, default="free", server_default="free")
    markup_multiplier = Column(Numeric(5, 3), nullable=False, default=Decimal("1.400"), server_default="1.400")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "(clerk_user_id IS NOT NULL AND clerk_org_id IS NULL) OR "
            "(clerk_user_id IS NULL AND clerk_org_id IS NOT NULL)",
            name="chk_billing_entity",
        ),
        Index("idx_billing_clerk_user", "clerk_user_id", unique=True, postgresql_where=clerk_user_id.isnot(None)),
        Index("idx_billing_clerk_org", "clerk_org_id", unique=True, postgresql_where=clerk_org_id.isnot(None)),
    )

    def __repr__(self) -> str:
        entity = self.clerk_user_id or self.clerk_org_id
        return f"<BillingAccount(entity={entity}, plan={self.plan_tier})>"


class UsageEvent(Base):
    """Every billable LLM interaction. Immutable audit trail."""

    __tablename__ = "usage_event"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    billing_account_id = Column(UUID(as_uuid=True), ForeignKey("billing_account.id"), nullable=False)
    clerk_user_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    input_cost = Column(Numeric(20, 12), nullable=False)
    output_cost = Column(Numeric(20, 12), nullable=False)
    total_cost = Column(Numeric(20, 12), nullable=False)
    billable_amount = Column(Numeric(20, 12), nullable=False)
    source = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    agent_name = Column(String, nullable=True)
    stripe_meter_event_id = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    month_partition = Column(String, nullable=False)

    __table_args__ = (
        CheckConstraint("source IN ('chat', 'agent')", name="chk_usage_source"),
        Index("idx_usage_event_account_month", "billing_account_id", "month_partition"),
        Index("idx_usage_event_created", "created_at"),
        Index(
            "idx_usage_event_stripe_null",
            "id",
            postgresql_where=stripe_meter_event_id.is_(None),
        ),
    )

    def __repr__(self) -> str:
        return f"<UsageEvent(model={self.model_id}, tokens={self.input_tokens}+{self.output_tokens})>"


class UsageDaily(Base):
    """Pre-aggregated daily rollups for fast billing dashboard queries."""

    __tablename__ = "usage_daily"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    billing_account_id = Column(UUID(as_uuid=True), ForeignKey("billing_account.id"), nullable=False)
    date = Column(Date, nullable=False)
    model_id = Column(String, nullable=False)
    source = Column(String, nullable=False)
    total_input_tokens = Column(BigInteger, nullable=False, default=0, server_default="0")
    total_output_tokens = Column(BigInteger, nullable=False, default=0, server_default="0")
    total_cost = Column(Numeric(20, 12), nullable=False, default=Decimal("0"), server_default="0")
    total_billable = Column(Numeric(20, 12), nullable=False, default=Decimal("0"), server_default="0")
    request_count = Column(Integer, nullable=False, default=0, server_default="0")

    __table_args__ = (
        CheckConstraint("source IN ('chat', 'agent')", name="chk_daily_source"),
        UniqueConstraint("billing_account_id", "date", "model_id", "source", name="uq_usage_daily"),
        Index("idx_usage_daily_account_date", "billing_account_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<UsageDaily(date={self.date}, model={self.model_id}, requests={self.request_count})>"
