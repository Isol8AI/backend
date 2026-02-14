"""Tests for billing Pydantic schemas."""

from datetime import date

import pytest

from schemas.billing import (
    BillingAccountResponse,
    CheckoutRequest,
    UsageResponse,
    UsagePeriod,
    ModelUsage,
    DailyUsage,
)


class TestBillingSchemas:
    """Test billing schema validation."""

    def test_billing_account_response(self):
        """Should serialize billing account data."""
        resp = BillingAccountResponse(
            plan_tier="starter",
            has_subscription=True,
            current_period=UsagePeriod(
                start=date(2026, 2, 1),
                end=date(2026, 2, 28),
                included_budget=15.00,
                used=11.42,
                overage=0.0,
                percent_used=76.1,
            ),
        )
        data = resp.model_dump()
        assert data["plan_tier"] == "starter"
        assert data["current_period"]["included_budget"] == 15.00

    def test_checkout_request_validation(self):
        """Should validate tier is one of allowed values."""
        req = CheckoutRequest(tier="starter")
        assert req.tier == "starter"

    def test_checkout_request_invalid_tier(self):
        """Should reject invalid tier."""
        with pytest.raises(ValueError):
            CheckoutRequest(tier="diamond")

    def test_usage_response(self):
        """Should serialize usage breakdown."""
        resp = UsageResponse(
            period=UsagePeriod(
                start=date(2026, 2, 1),
                end=date(2026, 2, 28),
                included_budget=15.0,
                used=11.42,
                overage=0.0,
                percent_used=76.1,
            ),
            total_cost=11.42,
            total_requests=847,
            by_model=[
                ModelUsage(model="Claude 3.5 Sonnet", cost=8.21, requests=312),
            ],
            by_day=[
                DailyUsage(date=date(2026, 2, 1), cost=0.89),
            ],
        )
        data = resp.model_dump()
        assert data["total_requests"] == 847
        assert len(data["by_model"]) == 1
