"""User API endpoints including encryption key management."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.database import get_db
from core.auth import get_current_user, AuthContext
from core.services.user_key_service import (
    UserKeyService,
    UserKeyServiceError,
    KeysAlreadyExistError,
    KeysNotFoundError,
)
from models.user import User
from models.organization import Organization
from models.organization_membership import OrganizationMembership
from schemas.encryption import (
    CreateUserKeysRequest,
    UserKeysResponse,
    EncryptionStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sync")
async def sync_user(auth: AuthContext = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Ensures the logged-in user exists in the database."""
    user_id = auth.user_id

    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()

    if not user:
        new_user = User(id=user_id)
        db.add(new_user)
        try:
            await db.commit()
            return {"status": "created", "user_id": user_id}
        except IntegrityError:
            # Race condition: another request created the user first
            # This is fine - treat as success (idempotent operation)
            # Rollback is REQUIRED: IntegrityError leaves the transaction in a
            # failed state. Must rollback to reset the session, release locks,
            # and return the connection to the pool in a clean state.
            await db.rollback()
            logger.debug("User sync race condition handled: %s", user_id)
            return {"status": "exists", "user_id": user_id}
        except Exception as e:
            logger.error("Database error on user sync for %s: %s", user_id, e)
            await db.rollback()
            raise HTTPException(status_code=500, detail="Database operation failed")

    return {"status": "exists", "user_id": user_id}


# =============================================================================
# Encryption Key Endpoints
# =============================================================================


@router.get("/me/encryption-status", response_model=EncryptionStatusResponse)
async def get_encryption_status(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current user's encryption status.

    Returns whether the user has set up encryption keys.
    """
    service = UserKeyService(db)
    try:
        status_data = await service.get_encryption_status(auth.user_id)
        return EncryptionStatusResponse(**status_data)
    except UserKeyServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/me/keys", status_code=status.HTTP_201_CREATED)
async def create_encryption_keys(
    request: CreateUserKeysRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Store user's encryption keys (already encrypted client-side).

    The client generates a keypair, encrypts the private key with the user's
    passcode, and sends the encrypted blob to the server. The server never
    sees the plaintext private key.
    """
    service = UserKeyService(db)
    try:
        await service.store_encryption_keys(
            user_id=auth.user_id,
            public_key=request.public_key,
            encrypted_private_key=request.encrypted_private_key,
            iv=request.iv,
            tag=request.tag,
            salt=request.salt,
            recovery_encrypted_private_key=request.recovery_encrypted_private_key,
            recovery_iv=request.recovery_iv,
            recovery_tag=request.recovery_tag,
            recovery_salt=request.recovery_salt,
        )
        return {"status": "created", "public_key": request.public_key}
    except KeysAlreadyExistError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already has encryption keys")
    except UserKeyServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me/keys", response_model=UserKeysResponse)
async def get_encryption_keys(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's encrypted keys for client-side decryption.

    The client uses these with the user's passcode to decrypt the private key.
    """
    service = UserKeyService(db)
    try:
        keys = await service.get_encryption_keys(auth.user_id)
        return UserKeysResponse(**keys)
    except KeysNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User has no encryption keys")
    except UserKeyServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me/keys/recovery", response_model=UserKeysResponse)
async def get_recovery_keys(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's recovery-encrypted keys.

    Used when the user has lost their passcode and needs to recover
    using their recovery code. This action is logged for audit purposes.
    """
    service = UserKeyService(db)
    try:
        keys = await service.get_recovery_keys(auth.user_id)
        return UserKeysResponse(
            public_key=keys["public_key"],
            encrypted_private_key=keys["encrypted_private_key"],
            iv=keys["iv"],
            tag=keys["tag"],
            salt=keys["salt"],
        )
    except KeysNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User has no encryption keys")
    except UserKeyServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/me/keys", status_code=status.HTTP_204_NO_CONTENT)
async def delete_encryption_keys(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete user's encryption keys.

    WARNING: This makes all encrypted messages unrecoverable!
    The client should confirm this action with the user first.
    """
    service = UserKeyService(db)
    try:
        await service.delete_encryption_keys(auth.user_id)
    except UserKeyServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{user_id}/public-key")
async def get_user_public_key(
    user_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get another user's public key (for key distribution).

    Used when distributing org keys to members - the admin needs
    the member's public key to encrypt the org key for them.
    """
    service = UserKeyService(db)
    public_key = await service.get_public_key(user_id)
    if not public_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User has no public key")
    return {"user_id": user_id, "public_key": public_key}


# =============================================================================
# Membership Endpoints
# =============================================================================


class MembershipItem(BaseModel):
    """Membership item for list response."""

    id: str
    org_id: str
    org_name: str | None
    role: str
    has_org_key: bool
    key_distributed_at: str | None
    joined_at: str | None


class MembershipsResponse(BaseModel):
    """Response for user's memberships list."""

    memberships: List[MembershipItem]


@router.get("/me/memberships", response_model=MembershipsResponse)
async def get_my_memberships(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current user's organization memberships.

    Returns all organizations the user is a member of, including
    their role and org key distribution status.
    """
    result = await db.execute(
        select(OrganizationMembership, Organization)
        .outerjoin(Organization, OrganizationMembership.org_id == Organization.id)
        .where(OrganizationMembership.user_id == auth.user_id)
    )
    rows = result.all()

    memberships = [
        MembershipItem(
            id=membership.id,
            org_id=membership.org_id,
            org_name=org.name if org else None,
            role=membership.role.value if hasattr(membership.role, "value") else str(membership.role),
            has_org_key=membership.has_org_key,
            key_distributed_at=membership.key_distributed_at.isoformat() if membership.key_distributed_at else None,
            joined_at=membership.joined_at.isoformat() if membership.joined_at else None,
        )
        for membership, org in rows
    ]

    return MembershipsResponse(memberships=memberships)
