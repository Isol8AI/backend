"""Tests for Clerk webhooks router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport


# =============================================================================
# Test Webhook Endpoint
# =============================================================================


class TestClerkWebhookEndpoint:
    """Tests for the clerk webhook endpoint."""

    @pytest.mark.asyncio
    async def test_processes_user_created_event(self):
        """Processes user.created webhook event."""
        from main import app

        payload = {
            "type": "user.created",
            "data": {"id": "user_123"},
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.create_user = AsyncMock()
                MockService.return_value = mock_service

                with patch("routers.webhooks.BillingService") as MockBilling:
                    mock_billing = MagicMock()
                    mock_billing.create_customer_for_user = AsyncMock()
                    MockBilling.return_value = mock_billing

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            "/api/v1/webhooks/clerk",
                            json=payload,
                            headers={
                                "svix-id": "test",
                                "svix-timestamp": "123",
                                "svix-signature": "test",
                            },
                        )

                    assert response.status_code == 200
                    assert response.json()["status"] == "processed"
                    assert response.json()["event"] == "user.created"

    @pytest.mark.asyncio
    async def test_processes_user_deleted_event(self):
        """Processes user.deleted webhook event."""
        from main import app

        payload = {
            "type": "user.deleted",
            "data": {"id": "user_123"},
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.delete_user = AsyncMock()
                MockService.return_value = mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/webhooks/clerk",
                        json=payload,
                        headers={
                            "svix-id": "test",
                            "svix-timestamp": "123",
                            "svix-signature": "test",
                        },
                    )

                assert response.status_code == 200
                assert response.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_processes_organization_created_event(self):
        """Processes organization.created webhook event."""
        from main import app

        payload = {
            "type": "organization.created",
            "data": {"id": "org_123", "name": "Test Org"},
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.create_organization = AsyncMock()
                MockService.return_value = mock_service

                with patch("routers.webhooks.BillingService") as MockBilling:
                    mock_billing = MagicMock()
                    mock_billing.create_customer_for_org = AsyncMock()
                    MockBilling.return_value = mock_billing

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            "/api/v1/webhooks/clerk",
                            json=payload,
                            headers={
                                "svix-id": "test",
                                "svix-timestamp": "123",
                                "svix-signature": "test",
                            },
                        )

                    assert response.status_code == 200
                    assert response.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_processes_membership_created_event(self):
        """Processes organizationMembership.created webhook event."""
        from main import app

        payload = {
            "type": "organizationMembership.created",
            "data": {
                "id": "mem_123",
                "public_user_data": {"user_id": "user_123"},
                "organization": {"id": "org_456"},
                "role": "org:member",
            },
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.create_membership = AsyncMock()
                MockService.return_value = mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/webhooks/clerk",
                        json=payload,
                        headers={
                            "svix-id": "test",
                            "svix-timestamp": "123",
                            "svix-signature": "test",
                        },
                    )

                assert response.status_code == 200
                assert response.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_processes_membership_deleted_event(self):
        """Processes organizationMembership.deleted webhook event (key revocation)."""
        from main import app

        payload = {
            "type": "organizationMembership.deleted",
            "data": {
                "id": "mem_123",
                "public_user_data": {"user_id": "user_123"},
                "organization": {"id": "org_456"},
            },
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.delete_membership = AsyncMock()
                MockService.return_value = mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/webhooks/clerk",
                        json=payload,
                        headers={
                            "svix-id": "test",
                            "svix-timestamp": "123",
                            "svix-signature": "test",
                        },
                    )

                assert response.status_code == 200
                assert response.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_ignores_unhandled_events(self):
        """Ignores webhook events that are not handled."""
        from main import app

        payload = {
            "type": "session.created",  # Not handled
            "data": {"id": "sess_123"},
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                MockService.return_value = mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/webhooks/clerk",
                        json=payload,
                        headers={
                            "svix-id": "test",
                            "svix-timestamp": "123",
                            "svix-signature": "test",
                        },
                    )

                assert response.status_code == 200
                assert response.json()["status"] == "ignored"


# =============================================================================
# Test Webhook Verification
# =============================================================================


class TestWebhookVerification:
    """Tests for webhook signature verification."""

    @pytest.mark.asyncio
    async def test_verification_skipped_without_secret(self):
        """Skips verification when CLERK_WEBHOOK_SECRET is not set."""
        from routers.webhooks import verify_webhook
        from unittest.mock import MagicMock

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b'{"type": "test", "data": {}}')

        with patch("routers.webhooks.settings") as mock_settings:
            mock_settings.CLERK_WEBHOOK_SECRET = None

            payload = await verify_webhook(mock_request, None, None, None)
            assert payload["type"] == "test"

    @pytest.mark.asyncio
    async def test_verification_fails_without_headers(self):
        """Returns 400 when svix headers are missing."""
        from routers.webhooks import verify_webhook
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b'{"type": "test"}')

        with patch("routers.webhooks.settings") as mock_settings:
            mock_settings.CLERK_WEBHOOK_SECRET = "whsec_test123"

            with pytest.raises(HTTPException) as exc_info:
                await verify_webhook(mock_request, None, None, None)

            assert exc_info.value.status_code == 400
            assert "Missing svix headers" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verification_fails_with_invalid_signature(self):
        """Returns 401 when signature is invalid."""
        from routers.webhooks import verify_webhook
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b'{"type": "test"}')

        with patch("routers.webhooks.settings") as mock_settings:
            # Svix expects base64-encoded secrets
            mock_settings.CLERK_WEBHOOK_SECRET = "whsec_dGVzdHNlY3JldHRlc3RzZWNyZXQ="

            with pytest.raises(HTTPException) as exc_info:
                await verify_webhook(mock_request, "msg_test123", "1234567890", "v1,invalid_signature")

            assert exc_info.value.status_code == 401
            assert "Invalid webhook signature" in exc_info.value.detail


# =============================================================================
# Test Billing Account Creation on Webhooks
# =============================================================================


class TestBillingAccountCreationOnWebhook:
    """Tests that billing accounts are created when users/orgs are created."""

    @pytest.mark.asyncio
    async def test_user_created_creates_billing_account(self):
        """user.created webhook should call BillingService.create_customer_for_user."""
        from main import app

        payload = {
            "type": "user.created",
            "data": {
                "id": "user_billing_new",
                "email_addresses": [{"email_address": "new@test.com"}],
            },
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.create_user = AsyncMock()
                MockService.return_value = mock_service

                with patch("routers.webhooks.BillingService") as MockBilling:
                    mock_billing = MagicMock()
                    mock_billing.create_customer_for_user = AsyncMock()
                    MockBilling.return_value = mock_billing

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            "/api/v1/webhooks/clerk",
                            json=payload,
                            headers={
                                "svix-id": "test",
                                "svix-timestamp": "123",
                                "svix-signature": "test",
                            },
                        )

                    assert response.status_code == 200
                    mock_billing.create_customer_for_user.assert_called_once_with(
                        clerk_user_id="user_billing_new",
                        email="new@test.com",
                    )

    @pytest.mark.asyncio
    async def test_org_created_creates_billing_account(self):
        """organization.created webhook should call BillingService.create_customer_for_org."""
        from main import app

        payload = {
            "type": "organization.created",
            "data": {"id": "org_billing_new", "name": "New Org"},
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.create_organization = AsyncMock()
                MockService.return_value = mock_service

                with patch("routers.webhooks.BillingService") as MockBilling:
                    mock_billing = MagicMock()
                    mock_billing.create_customer_for_org = AsyncMock()
                    MockBilling.return_value = mock_billing

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            "/api/v1/webhooks/clerk",
                            json=payload,
                            headers={
                                "svix-id": "test",
                                "svix-timestamp": "123",
                                "svix-signature": "test",
                            },
                        )

                    assert response.status_code == 200
                    mock_billing.create_customer_for_org.assert_called_once_with(
                        clerk_org_id="org_billing_new",
                        org_name="New Org",
                    )

    @pytest.mark.asyncio
    async def test_billing_failure_does_not_fail_webhook(self):
        """Billing account creation failure should not fail the webhook."""
        from main import app

        payload = {
            "type": "user.created",
            "data": {"id": "user_billing_fail"},
        }

        with patch("routers.webhooks.verify_webhook", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = payload

            with patch("routers.webhooks.ClerkSyncService") as MockService:
                mock_service = MagicMock()
                mock_service.create_user = AsyncMock()
                MockService.return_value = mock_service

                with patch("routers.webhooks.BillingService") as MockBilling:
                    mock_billing = MagicMock()
                    mock_billing.create_customer_for_user = AsyncMock(
                        side_effect=Exception("Stripe down")
                    )
                    MockBilling.return_value = mock_billing

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            "/api/v1/webhooks/clerk",
                            json=payload,
                            headers={
                                "svix-id": "test",
                                "svix-timestamp": "123",
                                "svix-signature": "test",
                            },
                        )

                    # Webhook should still succeed even if billing fails
                    assert response.status_code == 200
                    assert response.json()["status"] == "processed"
