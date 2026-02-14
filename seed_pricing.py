"""Seed the model_pricing table with current AWS Bedrock prices."""

import asyncio
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.config import settings
from models.billing import ModelPricing

BEDROCK_PRICES = [
    {
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "display_name": "Claude 3.5 Sonnet",
        "input_cost_per_token": Decimal("0.000003"),
        "output_cost_per_token": Decimal("0.000015"),
    },
    {
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "display_name": "Claude 3.5 Haiku",
        "input_cost_per_token": Decimal("0.0000008"),
        "output_cost_per_token": Decimal("0.000004"),
    },
    {
        "model_id": "us.anthropic.claude-3-opus-20240229-v1:0",
        "display_name": "Claude 3 Opus",
        "input_cost_per_token": Decimal("0.000015"),
        "output_cost_per_token": Decimal("0.000075"),
    },
    {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
        "display_name": "Llama 3.3 70B",
        "input_cost_per_token": Decimal("0.00000099"),
        "output_cost_per_token": Decimal("0.00000099"),
    },
    {
        "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
        "display_name": "Llama 3.1 70B",
        "input_cost_per_token": Decimal("0.00000099"),
        "output_cost_per_token": Decimal("0.00000099"),
    },
    {
        "model_id": "us.amazon.nova-pro-v1:0",
        "display_name": "Amazon Nova Pro",
        "input_cost_per_token": Decimal("0.0000008"),
        "output_cost_per_token": Decimal("0.0000032"),
    },
    {
        "model_id": "us.amazon.nova-lite-v1:0",
        "display_name": "Amazon Nova Lite",
        "input_cost_per_token": Decimal("0.00000006"),
        "output_cost_per_token": Decimal("0.00000024"),
    },
]


async def seed():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        for price_data in BEDROCK_PRICES:
            existing = await db.execute(
                select(ModelPricing).where(
                    ModelPricing.model_id == price_data["model_id"],
                    ModelPricing.is_active.is_(True),
                )
            )
            if existing.scalar_one_or_none():
                print(f"  Skipping {price_data['display_name']} (already exists)")
                continue

            pricing = ModelPricing(**price_data)
            db.add(pricing)
            print(f"  Added {price_data['display_name']}")

        await db.commit()
    await engine.dispose()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(seed())
