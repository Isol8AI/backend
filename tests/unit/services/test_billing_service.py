"""Tests for BillingService."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from core.services.billing_service import BillingService
from models.billing import BillingAccount


class TestBillingServiceCreateCustomer:
    """Test Stripe customer creation."""

    @pytest.fixture
    def service(self, db_session):
        return BillingService(db_session)

    @pytest.mark.asyncio
    @patch("core.services.billing_service.stripe")
    async def test_create_customer_for_user(self, mock_stripe, service, db_session):
        """Should create Stripe customer and billing account for user."""
        mock_stripe.Customer.create.return_value = MagicMock(id="cus_new_123")

        account = await service.create_customer_for_user(
            clerk_user_id="user_new_123",
            email="test@example.com",
        )

        mock_stripe.Customer.create.assert_called_once()
        assert account.clerk_user_id == "user_new_123"
        assert account.stripe_customer_id == "cus_new_123"
        assert account.plan_tier == "free"

    @pytest.mark.asyncio
    @patch("core.services.billing_service.stripe")
    async def test_create_customer_for_org(self, mock_stripe, service, db_session):
        """Should create Stripe customer and billing account for org."""
        mock_stripe.Customer.create.return_value = MagicMock(id="cus_org_456")

        account = await service.create_customer_for_org(
            clerk_org_id="org_new_456",
            org_name="Test Org",
        )

        assert account.clerk_org_id == "org_new_456"
        assert account.stripe_customer_id == "cus_org_456"

    @pytest.mark.asyncio
    @patch("core.services.billing_service.stripe")
    async def test_create_customer_idempotent(self, mock_stripe, service, db_session):
        """Should return existing account if already created."""
        # Create first
        mock_stripe.Customer.create.return_value = MagicMock(id="cus_idem")
        first = await service.create_customer_for_user(
            clerk_user_id="user_idem",
            email="idem@example.com",
        )

        # Second call should return same account
        second = await service.create_customer_for_user(
            clerk_user_id="user_idem",
            email="idem@example.com",
        )
        assert first.id == second.id
        # Stripe should only be called once
        assert mock_stripe.Customer.create.call_count == 1


class TestBillingServiceCheckout:
    """Test Stripe Checkout session creation."""

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_checkout",
            stripe_customer_id="cus_checkout",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    def service(self, db_session):
        return BillingService(db_session)

    @pytest.mark.asyncio
    @patch("core.services.billing_service.stripe")
    async def test_create_checkout_session(self, mock_stripe, service, billing_account):
        """Should create Stripe Checkout session."""
        mock_stripe.checkout.Session.create.return_value = MagicMock(
            url="https://checkout.stripe.com/test"
        )

        url = await service.create_checkout_session(
            billing_account=billing_account,
            tier="starter",
        )

        assert url == "https://checkout.stripe.com/test"
        mock_stripe.checkout.Session.create.assert_called_once()


class TestBillingServicePortal:
    """Test Stripe Customer Portal session."""

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_portal",
            stripe_customer_id="cus_portal",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    def service(self, db_session):
        return BillingService(db_session)

    @pytest.mark.asyncio
    @patch("core.services.billing_service.stripe")
    async def test_create_portal_session(self, mock_stripe, service, billing_account):
        """Should create Stripe Portal session."""
        mock_stripe.billing_portal.Session.create.return_value = MagicMock(
            url="https://billing.stripe.com/test"
        )

        url = await service.create_portal_session(billing_account=billing_account)

        assert url == "https://billing.stripe.com/test"


class TestBillingServiceSubscription:
    """Test subscription management."""

    @pytest.fixture
    async def billing_account(self, db_session):
        account = BillingAccount(
            clerk_user_id="user_sub",
            stripe_customer_id="cus_sub",
        )
        db_session.add(account)
        await db_session.commit()
        return account

    @pytest.fixture
    def service(self, db_session):
        return BillingService(db_session)

    @pytest.mark.asyncio
    async def test_update_subscription(self, service, billing_account, db_session):
        """Should update billing account with subscription details."""
        account_id = billing_account.id  # Capture before expiry
        await service.update_subscription(billing_account, "sub_123", "starter")

        db_session.expire_all()
        result = await db_session.execute(
            select(BillingAccount).where(BillingAccount.id == account_id)
        )
        updated = result.scalar_one()
        assert updated.stripe_subscription_id == "sub_123"
        assert updated.plan_tier == "starter"

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, service, billing_account, db_session):
        """Should revert to free tier on cancellation."""
        account_id = billing_account.id  # Capture before expiry
        billing_account.stripe_subscription_id = "sub_to_cancel"
        billing_account.plan_tier = "pro"
        await db_session.commit()

        await service.cancel_subscription(billing_account)

        db_session.expire_all()
        result = await db_session.execute(
            select(BillingAccount).where(BillingAccount.id == account_id)
        )
        updated = result.scalar_one()
        assert updated.stripe_subscription_id is None
        assert updated.plan_tier == "free"
