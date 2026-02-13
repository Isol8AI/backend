"""Pydantic response schemas for user endpoints."""
from pydantic import BaseModel, Field


class SyncUserResponse(BaseModel):
    """Response from POST /users/sync."""
    status: str = Field(..., description="'created' or 'exists'")
    user_id: str = Field(..., description="Clerk user ID")


class UserPublicKeyResponse(BaseModel):
    """Response from GET /users/{user_id}/public-key."""
    user_id: str = Field(..., description="Clerk user ID")
    public_key: str = Field(..., description="X25519 public key (32 bytes hex)")


class CreateKeysResponse(BaseModel):
    """Response from POST /users/me/keys."""
    status: str = Field(..., description="'created'")
    public_key: str = Field(..., description="X25519 public key (32 bytes hex)")
