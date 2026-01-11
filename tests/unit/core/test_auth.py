"""Unit tests for authentication module."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
from jose import jwt
import time

from core.auth import get_current_user


class TestGetCurrentUser:
    """Tests for get_current_user authentication function."""

    @pytest.fixture
    def mock_credentials(self):
        """Create mock HTTP authorization credentials."""
        mock = MagicMock()
        mock.credentials = "test_token"
        return mock

    @pytest.fixture
    def valid_jwks(self):
        """Valid JWKS response with test key."""
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "test-key-id",
                    "use": "sig",
                    "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
                    "e": "AQAB"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_valid_token_returns_payload(self, mock_credentials, valid_jwks):
        """Valid JWT token returns decoded payload."""
        expected_payload = {
            "sub": "user_123",
            "email": "test@example.com",
            "iss": "https://test.clerk.accounts.dev",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time())
        }

        with patch("core.auth.httpx.AsyncClient") as mock_client_class, \
             patch("core.auth.jwt.get_unverified_header") as mock_header, \
             patch("core.auth.jwt.decode") as mock_decode, \
             patch("core.auth.settings") as mock_settings:

            mock_settings.CLERK_ISSUER = "https://test.clerk.accounts.dev"
            mock_settings.CLERK_AUDIENCE = None

            # Mock JWKS fetch - use MagicMock with async context manager methods
            mock_response = MagicMock()
            mock_response.json.return_value = valid_jwks
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Mock JWT header extraction
            mock_header.return_value = {"kid": "test-key-id", "alg": "RS256"}

            # Mock JWT decode
            mock_decode.return_value = expected_payload

            result = await get_current_user(mock_credentials)

            assert result == expected_payload
            mock_decode.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_token_raises_401(self, mock_credentials, valid_jwks):
        """Expired token raises 401 HTTPException."""
        with patch("core.auth.httpx.AsyncClient") as mock_client_class, \
             patch("core.auth.jwt.get_unverified_header") as mock_header, \
             patch("core.auth.jwt.decode") as mock_decode, \
             patch("core.auth.settings") as mock_settings:

            mock_settings.CLERK_ISSUER = "https://test.clerk.accounts.dev"
            mock_settings.CLERK_AUDIENCE = None

            # Mock JWKS fetch
            mock_response = MagicMock()
            mock_response.json.return_value = valid_jwks
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_header.return_value = {"kid": "test-key-id", "alg": "RS256"}

            # Mock expired token - use jwt.ExpiredSignatureError to match auth.py exception handling
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_credentials)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Token expired"

    @pytest.mark.asyncio
    async def test_invalid_claims_raises_401(self, mock_credentials, valid_jwks):
        """Invalid claims raises 401 HTTPException."""
        with patch("core.auth.httpx.AsyncClient") as mock_client_class, \
             patch("core.auth.jwt.get_unverified_header") as mock_header, \
             patch("core.auth.jwt.decode") as mock_decode, \
             patch("core.auth.settings") as mock_settings:

            mock_settings.CLERK_ISSUER = "https://test.clerk.accounts.dev"
            mock_settings.CLERK_AUDIENCE = None

            mock_response = MagicMock()
            mock_response.json.return_value = valid_jwks
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_header.return_value = {"kid": "test-key-id", "alg": "RS256"}
            # Use jwt.JWTClaimsError to match auth.py exception handling
            mock_decode.side_effect = jwt.JWTClaimsError("Invalid claims")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_credentials)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Invalid claims"

    @pytest.mark.asyncio
    async def test_invalid_kid_raises_401(self, mock_credentials, valid_jwks):
        """Token with unknown key ID raises 401 HTTPException."""
        with patch("core.auth.httpx.AsyncClient") as mock_client_class, \
             patch("core.auth.jwt.get_unverified_header") as mock_header, \
             patch("core.auth.settings") as mock_settings:

            mock_settings.CLERK_ISSUER = "https://test.clerk.accounts.dev"
            mock_settings.CLERK_AUDIENCE = None

            mock_response = MagicMock()
            mock_response.json.return_value = valid_jwks
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Return a kid that doesn't match any key in JWKS
            mock_header.return_value = {"kid": "unknown-key-id", "alg": "RS256"}

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_credentials)

            assert exc_info.value.status_code == 401
            # Note: "Invalid token headers" is raised but caught by generic except block
            # which re-raises as "Could not validate credentials"
            assert exc_info.value.detail == "Could not validate credentials"

    @pytest.mark.asyncio
    async def test_jwks_fetch_failure_raises_401(self, mock_credentials):
        """JWKS fetch failure raises 401 HTTPException."""
        with patch("core.auth.httpx.AsyncClient") as mock_client_class, \
             patch("core.auth.settings") as mock_settings:

            mock_settings.CLERK_ISSUER = "https://test.clerk.accounts.dev"

            # Mock network failure
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_credentials)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Could not validate credentials"

    @pytest.mark.asyncio
    async def test_generic_exception_raises_401(self, mock_credentials, valid_jwks):
        """Generic exception during validation raises 401."""
        with patch("core.auth.httpx.AsyncClient") as mock_client_class, \
             patch("core.auth.jwt.get_unverified_header") as mock_header, \
             patch("core.auth.settings") as mock_settings:

            mock_settings.CLERK_ISSUER = "https://test.clerk.accounts.dev"

            mock_response = MagicMock()
            mock_response.json.return_value = valid_jwks
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_header.side_effect = Exception("Unexpected error")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_credentials)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Could not validate credentials"
