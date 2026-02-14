"""Service for tracking LLM usage and calculating costs."""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional
from uuid import UUID

import stripe
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.billing import BillingAccount, ModelPricing, UsageDaily, UsageEvent

logger = logging.getLogger(__name__)

stripe.api_key = settings.STRIPE_SECRET_KEY

# Fallback pricing when model not found in pricing table.
# Generous estimate to avoid undercharging.
FALLBACK_INPUT_COST = Decimal("0.000003")  # $3/1M tokens
FALLBACK_OUTPUT_COST = Decimal("0.000015")  # $15/1M tokens


class UsageServiceError(Exception):
    """Base exception for usage service errors."""

    pass


class UsageService:
    """Tracks LLM usage and calculates costs per request."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_active_model_pricing(self, model_id: str) -> Optional[ModelPricing]:
        """Look up active pricing for a model."""
        result = await self.db.execute(
            select(ModelPricing).where(
                ModelPricing.model_id == model_id,
                ModelPricing.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def get_billing_account(self, billing_account_id: UUID) -> Optional[BillingAccount]:
        """Fetch a billing account by ID."""
        result = await self.db.execute(select(BillingAccount).where(BillingAccount.id == billing_account_id))
        return result.scalar_one_or_none()

    async def get_billing_account_for_user(self, clerk_user_id: str) -> Optional[BillingAccount]:
        """Fetch billing account for a personal user."""
        result = await self.db.execute(select(BillingAccount).where(BillingAccount.clerk_user_id == clerk_user_id))
        return result.scalar_one_or_none()

    async def get_billing_account_for_org(self, clerk_org_id: str) -> Optional[BillingAccount]:
        """Fetch billing account for an organization."""
        result = await self.db.execute(select(BillingAccount).where(BillingAccount.clerk_org_id == clerk_org_id))
        return result.scalar_one_or_none()

    async def record_usage(
        self,
        billing_account_id: UUID,
        clerk_user_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        source: str,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> UsageEvent:
        """Record a billable LLM usage event.

        Calculates cost from the model pricing table, applies markup,
        writes a UsageEvent, and upserts UsageDaily rollup.

        This method commits its own transaction.
        """
        # 1. Look up pricing
        pricing = await self.get_active_model_pricing(model_id)
        if not pricing:
            logger.warning("No active pricing for model %s, using fallback", model_id)
            input_cost_per_token = FALLBACK_INPUT_COST
            output_cost_per_token = FALLBACK_OUTPUT_COST
        else:
            input_cost_per_token = pricing.input_cost_per_token
            output_cost_per_token = pricing.output_cost_per_token

        # 2. Calculate cost
        input_cost = Decimal(str(input_tokens)) * input_cost_per_token
        output_cost = Decimal(str(output_tokens)) * output_cost_per_token
        total_cost = input_cost + output_cost

        # 3. Apply markup
        account = await self.get_billing_account(billing_account_id)
        if not account:
            raise UsageServiceError(f"Billing account {billing_account_id} not found")
        billable = total_cost * account.markup_multiplier

        # 4. Write usage event
        today = date.today()
        event = UsageEvent(
            billing_account_id=billing_account_id,
            clerk_user_id=clerk_user_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            billable_amount=billable,
            source=source,
            session_id=session_id,
            agent_name=agent_name,
            month_partition=today.strftime("%Y-%m"),
        )
        self.db.add(event)

        # 5. Upsert daily rollup
        await self._upsert_daily_rollup(
            billing_account_id=billing_account_id,
            day=today,
            model_id=model_id,
            source=source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            billable=billable,
        )

        await self.db.commit()

        # 6. Report to Stripe (non-blocking — don't fail if Stripe is down)
        await self._report_to_stripe(account.stripe_customer_id, billable, event)

        return event

    async def _upsert_daily_rollup(
        self,
        billing_account_id: UUID,
        day: date,
        model_id: str,
        source: str,
        input_tokens: int,
        output_tokens: int,
        total_cost: Decimal,
        billable: Decimal,
    ) -> None:
        """Insert or update the daily usage rollup row."""
        stmt = pg_insert(UsageDaily).values(
            billing_account_id=billing_account_id,
            date=day,
            model_id=model_id,
            source=source,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=total_cost,
            total_billable=billable,
            request_count=1,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_usage_daily",
            set_={
                "total_input_tokens": UsageDaily.total_input_tokens + input_tokens,
                "total_output_tokens": UsageDaily.total_output_tokens + output_tokens,
                "total_cost": UsageDaily.total_cost + total_cost,
                "total_billable": UsageDaily.total_billable + billable,
                "request_count": UsageDaily.request_count + 1,
            },
        )
        await self.db.execute(stmt)

    async def _report_to_stripe(self, stripe_customer_id: str, billable: Decimal, event: UsageEvent) -> None:
        """Report usage to Stripe Meters API. Non-blocking."""
        try:
            value = int(billable * 1_000_000)  # Convert to microdollars
            if value <= 0:
                return

            stripe.billing.MeterEvent.create(
                event_name="llm_usage",
                payload={
                    "stripe_customer_id": stripe_customer_id,
                    "value": str(value),
                },
                idempotency_key=str(event.id),
            )

            event.stripe_meter_event_id = str(event.id)
            await self.db.commit()
        except Exception as e:
            logger.error("Stripe meter event failed for %s: %s", event.id, e)
            # Left as NULL — retry job will pick it up

    async def get_monthly_billable(self, billing_account_id: UUID) -> int:
        """Get total billable amount for the current month in microdollars.

        Returns an integer (microdollars) for comparison with plan budgets.
        """
        today = date.today()
        month_start = today.replace(day=1)
        result = await self.db.execute(
            select(func.coalesce(func.sum(UsageDaily.total_billable), 0)).where(
                UsageDaily.billing_account_id == billing_account_id,
                UsageDaily.date >= month_start,
            )
        )
        total = result.scalar_one()
        # Convert from dollars to microdollars
        return int(Decimal(str(total)) * 1_000_000)

    async def get_usage_breakdown(self, billing_account_id: UUID) -> dict:
        """Get current month usage breakdown by model and day.

        Returns dict with total_cost, total_requests, by_model, by_day.
        """
        today = date.today()
        month_start = today.replace(day=1)

        # Per-model aggregation
        model_result = await self.db.execute(
            select(
                UsageDaily.model_id,
                func.sum(UsageDaily.total_billable).label("cost"),
                func.sum(UsageDaily.request_count).label("requests"),
            )
            .where(
                UsageDaily.billing_account_id == billing_account_id,
                UsageDaily.date >= month_start,
            )
            .group_by(UsageDaily.model_id)
            .order_by(func.sum(UsageDaily.total_billable).desc())
        )
        by_model = [
            {"model": row.model_id, "cost": float(row.cost), "requests": int(row.requests)}
            for row in model_result.all()
        ]

        # Per-day aggregation
        day_result = await self.db.execute(
            select(
                UsageDaily.date,
                func.sum(UsageDaily.total_billable).label("cost"),
            )
            .where(
                UsageDaily.billing_account_id == billing_account_id,
                UsageDaily.date >= month_start,
            )
            .group_by(UsageDaily.date)
            .order_by(UsageDaily.date)
        )
        by_day = [{"date": row.date, "cost": float(row.cost)} for row in day_result.all()]

        total_cost = sum(m["cost"] for m in by_model)
        total_requests = sum(m["requests"] for m in by_model)

        return {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "by_model": by_model,
            "by_day": by_day,
        }
