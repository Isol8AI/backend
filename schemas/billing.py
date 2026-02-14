"""Pydantic schemas for billing API endpoints."""

from datetime import date
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class PlanTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    USAGE_ONLY = "usage_only"


class UsagePeriod(BaseModel):
    start: date
    end: date
    included_budget: float
    used: float
    overage: float
    percent_used: float


class ModelUsage(BaseModel):
    model: str
    cost: float
    requests: int


class DailyUsage(BaseModel):
    date: date
    cost: float


class BillingAccountResponse(BaseModel):
    plan_tier: str
    has_subscription: bool
    current_period: UsagePeriod

    model_config = {"from_attributes": True}


class CheckoutRequest(BaseModel):
    tier: PlanTier = Field(..., description="Plan tier to subscribe to")

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: PlanTier) -> PlanTier:
        if v == PlanTier.FREE:
            raise ValueError("Cannot checkout for free tier")
        return v


class CheckoutResponse(BaseModel):
    checkout_url: str


class PortalResponse(BaseModel):
    portal_url: str


class UsageResponse(BaseModel):
    period: UsagePeriod
    total_cost: float
    total_requests: int
    by_model: list[ModelUsage]
    by_day: list[DailyUsage]
