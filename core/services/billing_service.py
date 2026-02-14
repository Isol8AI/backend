"""Service for Stripe billing operations."""

import logging
import os
import stripe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.billing import BillingAccount

logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Stripe Price IDs per tier — set via env vars after creating products in Stripe Dashboard.
PLAN_PRICES = {
    "starter": {
        "fixed": os.getenv("STRIPE_STARTER_FIXED_PRICE_ID", ""),
        "metered": os.getenv("STRIPE_STARTER_METERED_PRICE_ID", ""),
    },
    "pro": {
        "fixed": os.getenv("STRIPE_PRO_FIXED_PRICE_ID", ""),
        "metered": os.getenv("STRIPE_PRO_METERED_PRICE_ID", ""),
    },
    "usage_only": {
        "fixed": "",
        "metered": os.getenv("STRIPE_USAGE_METERED_PRICE_ID", ""),
    },
}

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


class BillingServiceError(Exception):
    """Base exception for billing service errors."""

    pass


class BillingService:
    """Manages Stripe customers, subscriptions, and checkout flows."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_customer_for_user(self, clerk_user_id: str, email: str) -> BillingAccount:
        """Create Stripe customer + billing account for a personal user.

        Idempotent: returns existing account if already created.
        """
        existing = await self.db.execute(select(BillingAccount).where(BillingAccount.clerk_user_id == clerk_user_id))
        account = existing.scalar_one_or_none()
        if account:
            return account

        customer = stripe.Customer.create(
            email=email,
            metadata={"clerk_user_id": clerk_user_id},
        )

        account = BillingAccount(
            clerk_user_id=clerk_user_id,
            stripe_customer_id=customer.id,
        )
        self.db.add(account)
        await self.db.commit()
        return account

    async def create_customer_for_org(self, clerk_org_id: str, org_name: str) -> BillingAccount:
        """Create Stripe customer + billing account for an organization.

        Idempotent: returns existing account if already created.
        """
        existing = await self.db.execute(select(BillingAccount).where(BillingAccount.clerk_org_id == clerk_org_id))
        account = existing.scalar_one_or_none()
        if account:
            return account

        customer = stripe.Customer.create(
            name=org_name,
            metadata={"clerk_org_id": clerk_org_id},
        )

        account = BillingAccount(
            clerk_org_id=clerk_org_id,
            stripe_customer_id=customer.id,
        )
        self.db.add(account)
        await self.db.commit()
        return account

    async def create_checkout_session(self, billing_account: BillingAccount, tier: str) -> str:
        """Create a Stripe Checkout session for subscribing to a plan.

        Returns the checkout URL.
        """
        prices = PLAN_PRICES.get(tier)
        if not prices:
            raise BillingServiceError(f"Unknown tier: {tier}")

        line_items = []
        if prices.get("fixed"):
            line_items.append({"price": prices["fixed"], "quantity": 1})
        if prices.get("metered"):
            line_items.append({"price": prices["metered"]})

        session = stripe.checkout.Session.create(
            customer=billing_account.stripe_customer_id,
            mode="subscription",
            line_items=line_items,
            success_url=f"{FRONTEND_URL}/settings/billing?success=true",
            cancel_url=f"{FRONTEND_URL}/settings/billing?canceled=true",
        )
        return session.url

    async def create_portal_session(self, billing_account: BillingAccount) -> str:
        """Create a Stripe Customer Portal session.

        Returns the portal URL for managing payment methods and invoices.
        """
        session = stripe.billing_portal.Session.create(
            customer=billing_account.stripe_customer_id,
            return_url=f"{FRONTEND_URL}/settings/billing",
        )
        return session.url

    async def update_subscription(self, billing_account: BillingAccount, subscription_id: str, tier: str) -> None:
        """Update billing account after subscription change."""
        billing_account.stripe_subscription_id = subscription_id
        billing_account.plan_tier = tier
        await self.db.commit()

    async def cancel_subscription(self, billing_account: BillingAccount) -> None:
        """Revert to free tier after subscription cancellation."""
        billing_account.stripe_subscription_id = None
        billing_account.plan_tier = "free"
        await self.db.commit()
