"""Tests for billing API endpoints."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from models.billing import BillingAccount, ModelPricing


class TestGetBillingAccount:
    """Test GET /api/v1/billing/account."""

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_test_123",
            stripe_customer_id="cus_billing_test",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.mark.asyncio
    async def test_get_billing_account(self, async_client, billing_account):
        """Should return billing account for authenticated user."""
        response = await async_client.get("/api/v1/billing/account")
        assert response.status_code == 200
        data = response.json()
        assert data["plan_tier"] == "free"
        assert "current_period" in data

    @pytest.mark.asyncio
    async def test_get_billing_account_not_found(self, async_client):
        """Should return 404 when no billing account exists."""
        response = await async_client.get("/api/v1/billing/account")
        assert response.status_code == 404


class TestGetUsage:
    """Test GET /api/v1/billing/usage."""

    @pytest.fixture
    async def billing_account_with_usage(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_test_123",
            stripe_customer_id="cus_usage_endpoint",
        )
        db_session.add(account)

        pricing = ModelPricing(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet",
            input_cost_per_token=Decimal("0.000003"),
            output_cost_per_token=Decimal("0.000015"),
        )
        db_session.add(pricing)
        await db_session.commit()
        return account

    @pytest.mark.asyncio
    async def test_get_usage_empty(self, async_client, billing_account_with_usage):
        """Should return empty usage for new account."""
        response = await async_client.get("/api/v1/billing/usage")
        assert response.status_code == 200
        data = response.json()
        assert data["total_cost"] == 0
        assert data["total_requests"] == 0


class TestCheckout:
    """Test POST /api/v1/billing/checkout."""

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_test_123",
            stripe_customer_id="cus_checkout_test",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.mark.asyncio
    @patch("core.services.billing_service.stripe")
    async def test_create_checkout(self, mock_stripe, async_client, billing_account):
        """Should return Stripe checkout URL."""
        mock_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe.com/test_session")

        response = await async_client.post(
            "/api/v1/billing/checkout",
            json={"tier": "starter"},
        )
        assert response.status_code == 200
        assert "checkout_url" in response.json()


class TestStripeWebhook:
    """Test POST /api/v1/billing/webhooks/stripe."""

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_webhook_test",
            stripe_customer_id="cus_webhook_test",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.mark.asyncio
    @patch("routers.billing.stripe")
    async def test_subscription_created_webhook(self, mock_stripe, async_client, billing_account, db_session):
        """Should update billing account on subscription.created."""
        mock_stripe.Webhook.construct_event.return_value = {
            "type": "customer.subscription.created",
            "data": {
                "object": {
                    "id": "sub_test_123",
                    "customer": "cus_webhook_test",
                    "items": {"data": [{"price": {"lookup_key": "starter_fixed"}}]},
                    "metadata": {"plan_tier": "starter"},
                }
            },
        }

        response = await async_client.post(
            "/api/v1/billing/webhooks/stripe",
            content=b'{"test": true}',
            headers={
                "stripe-signature": "test_sig",
                "content-type": "application/json",
            },
        )
        assert response.status_code == 200
