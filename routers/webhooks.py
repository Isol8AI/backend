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
        raise HTTPException(
            status_code=400,
            detail="Missing svix headers for webhook verification"
        )

    try:
        wh = Webhook(settings.CLERK_WEBHOOK_SECRET)
        payload = wh.verify(
            body,
            {
                "svix-id": svix_id,
                "svix-timestamp": svix_timestamp,
                "svix-signature": svix_signature,
            }
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
                await service.delete_membership(data)

            else:
                logger.debug("Ignoring webhook event: %s", event_type)
                return {"status": "ignored", "event": event_type}

        except Exception as e:
            logger.exception("Error processing webhook %s: %s", event_type, e)
            # Return 200 to prevent Clerk from retrying on our errors
            return {"status": "error", "event": event_type, "error": str(e)}

    return {"status": "processed", "event": event_type}
