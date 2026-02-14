"""Tests for UsageService."""

from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from core.services.usage_service import UsageService, UsageServiceError
from models.billing import BillingAccount, ModelPricing, UsageEvent, UsageDaily


class TestUsageServiceRecordUsage:
    """Test UsageService.record_usage."""

    @pytest.fixture
    async def pricing(self, db_session):
        """Create test model pricing."""
        p = ModelPricing(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet",
            input_cost_per_token=Decimal("0.000003"),
            output_cost_per_token=Decimal("0.000015"),
        )
        db_session.add(p)
        await db_session.commit()
        return p

    @pytest.fixture
    async def billing_account(self, db_session):
        """Create test billing account."""
        account = BillingAccount(
            clerk_user_id="user_usage_svc",
            stripe_customer_id="cus_usage_svc",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    def service(self, db_session):
        return UsageService(db_session)

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_creates_event(
        self, mock_stripe, service, billing_account, pricing, db_session
    ):
        """Should create a UsageEvent with calculated costs."""
        await service.record_usage(
            billing_account_id=billing_account.id,
            clerk_user_id="user_usage_svc",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=1000,
            output_tokens=500,
            source="chat",
        )

        result = await db_session.execute(
            select(UsageEvent).where(UsageEvent.billing_account_id == billing_account.id)
        )
        event = result.scalar_one()
        assert event.input_tokens == 1000
        assert event.output_tokens == 500
        assert event.input_cost == Decimal("0.000003") * 1000  # 0.003
        assert event.output_cost == Decimal("0.000015") * 500  # 0.0075
        assert event.total_cost == event.input_cost + event.output_cost
        # Billable = total_cost * 1.4 (default markup)
        expected_billable = event.total_cost * Decimal("1.400")
        assert event.billable_amount == expected_billable
        assert event.source == "chat"

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_upserts_daily(
        self, mock_stripe, service, billing_account, pricing, db_session
    ):
        """Should create/update a UsageDaily rollup."""
        account_id = billing_account.id  # Capture before any expiry

        await service.record_usage(
            billing_account_id=account_id,
            clerk_user_id="user_usage_svc",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=1000,
            output_tokens=500,
            source="chat",
        )

        result = await db_session.execute(
            select(UsageDaily).where(UsageDaily.billing_account_id == account_id)
        )
        daily = result.scalar_one()
        assert daily.total_input_tokens == 1000
        assert daily.total_output_tokens == 500
        assert daily.request_count == 1

        # Second call should increment, not duplicate
        await service.record_usage(
            billing_account_id=account_id,
            clerk_user_id="user_usage_svc",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=2000,
            output_tokens=1000,
            source="chat",
        )

        # Expire cached objects so SQLAlchemy re-fetches the updated row from DB
        db_session.expire_all()

        result = await db_session.execute(
            select(UsageDaily).where(UsageDaily.billing_account_id == account_id)
        )
        daily = result.scalar_one()
        assert daily.total_input_tokens == 3000
        assert daily.total_output_tokens == 1500
        assert daily.request_count == 2

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_unknown_model_uses_fallback(
        self, mock_stripe, service, billing_account, db_session
    ):
        """Should use fallback pricing when model not in pricing table."""
        await service.record_usage(
            billing_account_id=billing_account.id,
            clerk_user_id="user_usage_svc",
            model_id="unknown.model.v1",
            input_tokens=100,
            output_tokens=50,
            source="chat",
        )

        result = await db_session.execute(
            select(UsageEvent).where(UsageEvent.billing_account_id == billing_account.id)
        )
        event = result.scalar_one()
        # Fallback pricing should still produce non-zero costs
        assert event.total_cost > 0


class TestUsageServiceQueries:
    """Test usage query methods."""

    @pytest.fixture
    async def pricing(self, db_session):
        p = ModelPricing(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet",
            input_cost_per_token=Decimal("0.000003"),
            output_cost_per_token=Decimal("0.000015"),
        )
        db_session.add(p)
        await db_session.commit()
        return p

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_query_test",
            stripe_customer_id="cus_query_test",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    def service(self, db_session):
        return UsageService(db_session)

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_get_monthly_billable(
        self, mock_stripe, service, billing_account, pricing, db_session
    ):
        """Should return total billable in microdollars for current month."""
        await service.record_usage(
            billing_account_id=billing_account.id,
            clerk_user_id="user_query_test",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=1000,
            output_tokens=500,
            source="chat",
        )

        monthly = await service.get_monthly_billable(billing_account.id)
        assert monthly > 0
        assert isinstance(monthly, int)

    @pytest.mark.asyncio
    async def test_get_monthly_billable_empty(self, service, billing_account):
        """Should return 0 when no usage exists."""
        monthly = await service.get_monthly_billable(billing_account.id)
        assert monthly == 0

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_get_usage_breakdown(
        self, mock_stripe, service, billing_account, pricing, db_session
    ):
        """Should return per-model and per-day breakdown."""
        await service.record_usage(
            billing_account_id=billing_account.id,
            clerk_user_id="user_query_test",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=1000,
            output_tokens=500,
            source="chat",
        )

        breakdown = await service.get_usage_breakdown(billing_account.id)
        assert len(breakdown["by_model"]) >= 1
        assert len(breakdown["by_day"]) >= 1
        assert breakdown["total_cost"] > 0
        assert breakdown["total_requests"] >= 1


class TestStripeReporting:
    """Test Stripe meter event reporting."""

    @pytest.fixture
    async def pricing(self, db_session):
        p = ModelPricing(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet",
            input_cost_per_token=Decimal("0.000003"),
            output_cost_per_token=Decimal("0.000015"),
        )
        db_session.add(p)
        await db_session.commit()
        return p

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_stripe_test",
            stripe_customer_id="cus_stripe_test",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    def service(self, db_session):
        return UsageService(db_session)

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_reports_to_stripe(
        self, mock_stripe, service, billing_account, pricing, db_session
    ):
        """Should report usage to Stripe meter after recording."""
        event = await service.record_usage(
            billing_account_id=billing_account.id,
            clerk_user_id="user_stripe_test",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=1000,
            output_tokens=500,
            source="chat",
        )

        mock_stripe.billing.MeterEvent.create.assert_called_once()
        call_kwargs = mock_stripe.billing.MeterEvent.create.call_args
        # Handle both keyword and positional call styles
        if call_kwargs.kwargs:
            payload = call_kwargs.kwargs.get("payload")
        else:
            payload = call_kwargs[1].get("payload")
        assert payload["stripe_customer_id"] == "cus_stripe_test"
        assert int(payload["value"]) > 0

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_stripe_failure_does_not_block(
        self, mock_stripe, service, billing_account, pricing, db_session
    ):
        """Should still record usage locally even if Stripe fails."""
        mock_stripe.billing.MeterEvent.create.side_effect = Exception("Stripe down")

        # Should not raise
        event = await service.record_usage(
            billing_account_id=billing_account.id,
            clerk_user_id="user_stripe_test",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=100,
            output_tokens=50,
            source="chat",
        )

        # Event should still be in DB
        assert event.id is not None
        assert event.stripe_meter_event_id is None
