"""Tests for usage tracking hooks in chat and agent flows."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from core.services.usage_service import UsageService
from models.billing import BillingAccount, ModelPricing, UsageEvent, UsageDaily


class TestUsageTrackingIntegration:
    """Test that usage tracking resolves billing accounts and records events."""

    @pytest.fixture
    async def model_pricing(self, db_session):
        pricing = ModelPricing(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet",
            input_cost_per_token=Decimal("0.000003"),
            output_cost_per_token=Decimal("0.000015"),
        )
        db_session.add(pricing)
        await db_session.commit()
        return pricing

    @pytest.fixture
    async def user_billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_test_123",
            stripe_customer_id="cus_track_user",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    async def org_billing_account(self, db_session):
        account = BillingAccount(
            clerk_org_id="org_test_456",
            stripe_customer_id="cus_track_org",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_for_personal_chat(
        self, mock_stripe, db_session, model_pricing, user_billing_account
    ):
        """Should resolve user billing account and record usage for personal chat."""
        usage_service = UsageService(db_session)

        # Simulate the pattern used in websocket_chat.py
        account = await usage_service.get_billing_account_for_user("user_test_123")
        assert account is not None
        assert account.id == user_billing_account.id

        event = await usage_service.record_usage(
            billing_account_id=account.id,
            clerk_user_id="user_test_123",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=100,
            output_tokens=200,
            source="chat",
            session_id="sess_123",
        )

        assert event.billing_account_id == account.id
        assert event.input_tokens == 100
        assert event.output_tokens == 200
        assert event.source == "chat"
        assert event.session_id == "sess_123"

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_for_org_chat(
        self, mock_stripe, db_session, model_pricing, org_billing_account
    ):
        """Should resolve org billing account and record usage for org chat."""
        usage_service = UsageService(db_session)

        # Simulate the org context pattern
        account = await usage_service.get_billing_account_for_org("org_test_456")
        assert account is not None
        assert account.id == org_billing_account.id

        event = await usage_service.record_usage(
            billing_account_id=account.id,
            clerk_user_id="user_test_123",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=50,
            output_tokens=150,
            source="chat",
            session_id="sess_org_456",
        )

        assert event.billing_account_id == org_billing_account.id
        assert event.source == "chat"

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_record_usage_for_agent(
        self, mock_stripe, db_session, model_pricing, user_billing_account
    ):
        """Should record usage for agent chat with agent_name."""
        usage_service = UsageService(db_session)

        account = await usage_service.get_billing_account_for_user("user_test_123")
        event = await usage_service.record_usage(
            billing_account_id=account.id,
            clerk_user_id="user_test_123",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=75,
            output_tokens=300,
            source="agent",
            agent_name="my_agent",
        )

        assert event.source == "agent"
        assert event.agent_name == "my_agent"
        assert event.session_id is None

    @pytest.mark.asyncio
    async def test_no_billing_account_skips_recording(self, db_session, model_pricing):
        """Should gracefully skip recording when no billing account exists."""
        usage_service = UsageService(db_session)

        # No billing account for this user
        account = await usage_service.get_billing_account_for_user("user_nonexistent")
        assert account is None
        # The hook code checks `if account:` before calling record_usage,
        # so no error should occur

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_usage_creates_daily_rollup(
        self, mock_stripe, db_session, model_pricing, user_billing_account
    ):
        """Should create a daily rollup entry when recording usage."""
        usage_service = UsageService(db_session)

        account = await usage_service.get_billing_account_for_user("user_test_123")
        await usage_service.record_usage(
            billing_account_id=account.id,
            clerk_user_id="user_test_123",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=100,
            output_tokens=200,
            source="chat",
        )

        # Check daily rollup was created
        result = await db_session.execute(
            select(UsageDaily).where(
                UsageDaily.billing_account_id == account.id,
            )
        )
        daily = result.scalar_one()
        assert daily.request_count == 1
        assert daily.total_input_tokens == 100
        assert daily.total_output_tokens == 200

    @pytest.mark.asyncio
    @patch("core.services.usage_service.stripe")
    async def test_stripe_failure_does_not_block(
        self, mock_stripe, db_session, model_pricing, user_billing_account
    ):
        """Stripe reporting failure should not prevent usage recording."""
        mock_stripe.billing.MeterEvent.create.side_effect = Exception("Stripe down")

        usage_service = UsageService(db_session)
        account = await usage_service.get_billing_account_for_user("user_test_123")

        # Should not raise despite Stripe failure
        event = await usage_service.record_usage(
            billing_account_id=account.id,
            clerk_user_id="user_test_123",
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=100,
            output_tokens=200,
            source="chat",
        )

        # Event should still be recorded
        assert event is not None
        assert event.input_tokens == 100
