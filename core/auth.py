from dataclasses import dataclass, field

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt

from core.config import settings

security = HTTPBearer()


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
        # Fetch JWKS (In production, cache this!)
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url)
            jwks = response.json()

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
    except HTTPException:
        raise
    except Exception:
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

