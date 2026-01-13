import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt

from core.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()

# JWKS cache with TTL
_jwks_cache: dict = {"data": None, "expires_at": None}
JWKS_CACHE_TTL = timedelta(hours=1)


async def _get_cached_jwks(jwks_url: str) -> dict:
    """Fetch JWKS with TTL-based caching to avoid hitting Clerk on every request."""
    now = datetime.utcnow()

    # Return cached data if still valid
    if _jwks_cache["data"] and _jwks_cache["expires_at"] and now < _jwks_cache["expires_at"]:
        return _jwks_cache["data"]

    # Fetch fresh JWKS
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url, timeout=10.0)
            response.raise_for_status()
            jwks = response.json()

        # Update cache
        _jwks_cache["data"] = jwks
        _jwks_cache["expires_at"] = now + JWKS_CACHE_TTL
        logger.info("JWKS cache refreshed")
        return jwks
    except httpx.HTTPError as e:
        # If fetch fails but we have stale cached data, use it as fallback
        if _jwks_cache["data"]:
            logger.warning(f"JWKS fetch failed, using stale cache: {e}")
            return _jwks_cache["data"]
        raise


@dataclass
class AuthContext:
    """Structured auth context from JWT claims.

    Provides convenient properties for checking user context:
    - is_org_context: True when user has active organization selected
    - is_personal_context: True when user is in personal mode
    - is_org_admin: True when user has admin role in current org
    """

    user_id: str
    org_id: str | None = None
    org_role: str | None = None
    org_slug: str | None = None
    org_permissions: list[str] = field(default_factory=list)

    @property
    def is_org_context(self) -> bool:
        """True when user has active organization selected."""
        return self.org_id is not None

    @property
    def is_personal_context(self) -> bool:
        """True when user is in personal mode (no active org)."""
        return self.org_id is None

    @property
    def is_org_admin(self) -> bool:
        """True when user has admin role in current org."""
        return self.org_role == "org:admin"


def _find_rsa_key(jwks: dict, kid: str) -> dict | None:
    """Find RSA key in JWKS by key ID."""
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
    return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthContext:
    """Validate JWT and return AuthContext with user and org claims."""
    token = credentials.credentials
    jwks_url = f"{settings.CLERK_ISSUER}/.well-known/jwks.json"

    try:
        # Fetch JWKS with caching
        jwks = await _get_cached_jwks(jwks_url)

        # Find RSA key matching the token's key ID
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = _find_rsa_key(jwks, unverified_header["kid"])

        if not rsa_key:
            raise HTTPException(status_code=401, detail="Invalid token headers")

        # Verify the token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=settings.CLERK_AUDIENCE,
            issuer=settings.CLERK_ISSUER,
        )

        return AuthContext(
            user_id=payload["sub"],
            org_id=payload.get("org_id"),
            org_role=payload.get("org_role"),
            org_slug=payload.get("org_slug"),
            org_permissions=payload.get("org_permissions", []),
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTClaimsError:
        raise HTTPException(status_code=401, detail="Invalid claims")
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")


async def require_org_context(auth: AuthContext = Depends(get_current_user)) -> AuthContext:
    """Dependency that requires an active organization context."""
    if not auth.is_org_context:
        raise HTTPException(
            status_code=403,
            detail="This action requires an active organization context"
        )
    return auth


async def require_org_admin(auth: AuthContext = Depends(get_current_user)) -> AuthContext:
    """Dependency that requires org admin role."""
    if not auth.is_org_admin:
        raise HTTPException(
            status_code=403,
            detail="This action requires organization admin privileges"
        )
    return auth

