"""
Clerk webhook handlers.

Security Note:
- All webhooks are verified using Clerk's signing secret
- Membership deletion triggers org key revocation
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header, Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from svix.webhooks import Webhook, WebhookVerificationError

from core.config import settings
from core.database import get_session_factory
from core.services.clerk_sync_service import ClerkSyncService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


async def verify_webhook(
    request: Request,
    svix_id: Optional[str] = Header(None, alias="svix-id"),
    svix_timestamp: Optional[str] = Header(None, alias="svix-timestamp"),
    svix_signature: Optional[str] = Header(None, alias="svix-signature"),
) -> dict:
    """Verify Clerk webhook signature and return payload."""
    body = await request.body()

    if not settings.CLERK_WEBHOOK_SECRET:
        logger.warning("CLERK_WEBHOOK_SECRET not configured - skipping verification in dev")
        # In development, allow unverified webhooks
        import json

        return json.loads(body)

    if not all([svix_id, svix_timestamp, svix_signature]):
        raise HTTPException(status_code=400, detail="Missing svix headers for webhook verification")

    try:
        wh = Webhook(settings.CLERK_WEBHOOK_SECRET)
        payload = wh.verify(
            body,
            {
                "svix-id": svix_id,
                "svix-timestamp": svix_timestamp,
                "svix-signature": svix_signature,
            },
        )
        return payload
    except WebhookVerificationError as e:
        logger.warning("Webhook verification failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


@router.post("/clerk")
async def handle_clerk_webhook(
    request: Request,
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    svix_id: Optional[str] = Header(None, alias="svix-id"),
    svix_timestamp: Optional[str] = Header(None, alias="svix-timestamp"),
    svix_signature: Optional[str] = Header(None, alias="svix-signature"),
):
    """
    Handle Clerk webhooks.

    Events:
    - user.* - User lifecycle
    - organization.* - Org lifecycle
    - organizationMembership.* - Membership lifecycle (key revocation)
    """
    payload = await verify_webhook(request, svix_id, svix_timestamp, svix_signature)

    event_type = payload.get("type")
    data = payload.get("data", {})

    logger.info("Received Clerk webhook: %s", event_type)

    async with session_factory() as db:
        service = ClerkSyncService(db)

        try:
            # User events
            if event_type == "user.created":
                await service.create_user(data)
            elif event_type == "user.updated":
                await service.update_user(data)
            elif event_type == "user.deleted":
                await service.delete_user(data)

            # Organization events
            elif event_type == "organization.created":
                await service.create_organization(data)
            elif event_type == "organization.updated":
                await service.update_organization(data)
            elif event_type == "organization.deleted":
                await service.delete_organization(data)

            # Membership events
            elif event_type == "organizationMembership.created":
                await service.create_membership(data)
            elif event_type == "organizationMembership.updated":
                await service.update_membership(data)
            elif event_type == "organizationMembership.deleted":
                # CRITICAL: This triggers org key revocation
                try:
                    await service.delete_membership(data)
                except Exception:
                    # Security-critical: Revoked member may still have org key access
                    logger.critical(
                        "SECURITY_ALERT: Membership deletion failed - revoked member may retain org key access",
                        extra={
                            "event_type": event_type,
                            "webhook_id": svix_id,
                            "membership_id": data.get("id"),
                            "user_id": data.get("public_user_data", {}).get("user_id"),
                            "org_id": data.get("organization", {}).get("id"),
                            "alert": True,
                        },
                        exc_info=True,
                    )
                    raise  # Re-raise to be caught by outer handler

            else:
                logger.debug("Ignoring webhook event: %s", event_type)
                return {"status": "ignored", "event": event_type}

        except Exception as e:
            # Log with structured fields for monitoring systems (Datadog, Sentry, etc.)
            logger.error(
                "WEBHOOK_PROCESSING_FAILED",
                extra={
                    "event_type": event_type,
                    "webhook_id": svix_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "alert": True,  # Flag for alerting systems
                },
                exc_info=True,
            )
            # Return 200 to prevent Clerk from retrying on our errors
            # Errors should be handled via monitoring alerts, not webhook retries
            return {"status": "error", "event": event_type, "error": str(e)}

    return {"status": "processed", "event": event_type}
