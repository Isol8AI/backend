"""
Clerk Sync Service - syncs Clerk data to our database.

Security Note:
- Membership deletion clears encrypted org keys (revocation)
- User deletion clears all encryption keys
- All changes are audit logged
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.organization import Organization
from models.organization_membership import OrganizationMembership, MemberRole
from models.audit_log import AuditLog, AuditEventType

logger = logging.getLogger(__name__)


class ClerkSyncService:
    """Service for syncing Clerk data to our database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # =========================================================================
    # User Sync
    # =========================================================================

    async def create_user(self, data: dict) -> User:
        """Create user from Clerk webhook data."""
        user_id = data.get("id")

        # Check if user already exists
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info("User %s already exists, updating", user_id)
            return await self.update_user(data)

        user = User(id=user_id)

        self.db.add(user)
        await self.db.commit()

        logger.info("Created user %s", user_id)
        return user

    async def update_user(self, data: dict) -> Optional[User]:
        """Update user from Clerk webhook data."""
        user_id = data.get("id")

        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            logger.warning("User %s not found for update, creating", user_id)
            return await self.create_user(data)

        # User model is minimal - just update timestamp if needed
        await self.db.commit()

        logger.info("Updated user %s", user_id)
        return user

    async def delete_user(self, data: dict) -> None:
        """
        Delete user and their encryption keys.

        WARNING: This makes all their encrypted messages unrecoverable!
        """
        user_id = data.get("id")

        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            logger.warning("User %s not found for deletion", user_id)
            return

        # Audit log if they had encryption keys
        if user.has_encryption_keys:
            user.clear_encryption_keys()
            audit_log = AuditLog.create(
                id=str(uuid4()),
                event_type=AuditEventType.USER_KEYS_DELETED,
                actor_user_id=user_id,
                event_data={"reason": "user_deleted_from_clerk"},
            )
            self.db.add(audit_log)

        # Delete user's memberships first (clear org keys)
        membership_result = await self.db.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == user_id
            )
        )
        memberships = membership_result.scalars().all()

        for membership in memberships:
            if membership.has_org_key:
                membership.clear_encrypted_org_key()
            await self.db.delete(membership)

        # Delete user
        await self.db.delete(user)
        await self.db.commit()

        logger.warning("Deleted user %s and all their data", user_id)

    # =========================================================================
    # Organization Sync
    # =========================================================================

    async def create_organization(self, data: dict) -> Organization:
        """Create organization from Clerk webhook data."""
        org_id = data.get("id")
        name = data.get("name", "Unknown Organization")
        slug = data.get("slug")

        # Check if org already exists
        result = await self.db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info("Organization %s already exists, updating", org_id)
            return await self.update_organization(data)

        org = Organization(
            id=org_id,
            name=name,
            slug=slug,
        )

        self.db.add(org)
        await self.db.commit()

        logger.info("Created organization %s (%s)", org_id, name)
        return org

    async def update_organization(self, data: dict) -> Optional[Organization]:
        """Update organization from Clerk webhook data."""
        org_id = data.get("id")

        result = await self.db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        org = result.scalar_one_or_none()

        if not org:
            logger.warning("Organization %s not found for update, creating", org_id)
            return await self.create_organization(data)

        org.name = data.get("name", org.name)
        if data.get("slug"):
            org.slug = data.get("slug")

        await self.db.commit()

        logger.info("Updated organization %s", org_id)
        return org

    async def delete_organization(self, data: dict) -> None:
        """Delete organization and all encryption keys."""
        org_id = data.get("id")

        result = await self.db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        org = result.scalar_one_or_none()

        if not org:
            logger.warning("Organization %s not found for deletion", org_id)
            return

        # Clear org keys from all memberships first
        membership_result = await self.db.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.org_id == org_id
            )
        )
        memberships = membership_result.scalars().all()

        for membership in memberships:
            if membership.has_org_key:
                membership.clear_encrypted_org_key()
            await self.db.delete(membership)

        # Audit log if org had encryption keys
        if org.has_encryption_keys:
            audit_log = AuditLog.create(
                id=str(uuid4()),
                event_type=AuditEventType.ORG_KEYS_ROTATED,  # Using rotated as closest match
                org_id=org_id,
                event_data={"reason": "organization_deleted_from_clerk"},
            )
            self.db.add(audit_log)

        # Delete org
        await self.db.delete(org)
        await self.db.commit()

        logger.warning("Deleted organization %s and all its data", org_id)

    # =========================================================================
    # Membership Sync
    # =========================================================================

    async def create_membership(self, data: dict) -> Optional[OrganizationMembership]:
        """
        Create membership from Clerk webhook data.

        New members start with has_org_key=False.
        An admin must distribute the org key to them.
        """
        membership_id = data.get("id")
        user_id = data.get("public_user_data", {}).get("user_id")
        org_id = data.get("organization", {}).get("id")
        role_str = data.get("role", "org:member")

        if not all([membership_id, user_id, org_id]):
            logger.warning(
                "Membership webhook missing required fields: "
                "membership_id=%s, user_id=%s, org_id=%s",
                membership_id, user_id, org_id
            )
            return None

        # Map Clerk role to our enum (Clerk uses "org:admin", "org:member")
        role = MemberRole(role_str)

        # Ensure organization exists
        org_result = await self.db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        org = org_result.scalar_one_or_none()
        if not org:
            org_name = data.get("organization", {}).get("name", "Unknown Organization")
            org_slug = data.get("organization", {}).get("slug")
            org = Organization(id=org_id, name=org_name, slug=org_slug)
            self.db.add(org)
            logger.info("Created organization %s from membership webhook", org_id)

        # Ensure user exists
        user_result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        if not user:
            user = User(id=user_id)
            self.db.add(user)
            logger.info("Created user %s from membership webhook", user_id)

        # Check if membership already exists
        result = await self.db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.user_id == user_id,
                    OrganizationMembership.org_id == org_id,
                )
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info("Membership already exists for user %s in org %s", user_id, org_id)
            # Update role if changed
            if existing.role != role:
                old_role = existing.role
                existing.role = role
                # Audit log for role change
                audit_log = AuditLog.log_member_role_changed(
                    id=str(uuid4()),
                    admin_user_id="system",
                    member_user_id=user_id,
                    org_id=org_id,
                    old_role=old_role.value,
                    new_role=role.value,
                )
                self.db.add(audit_log)
                await self.db.commit()
            return existing

        membership = OrganizationMembership(
            id=membership_id,
            user_id=user_id,
            org_id=org_id,
            role=role,
            # has_org_key=False by default - key needs to be distributed by admin
        )

        self.db.add(membership)

        # Audit log
        audit_log = AuditLog.log_member_joined(
            id=str(uuid4()),
            member_user_id=user_id,
            org_id=org_id,
            role=role_str,
        )
        self.db.add(audit_log)

        await self.db.commit()

        logger.info(
            "Created membership for user %s in org %s (role: %s, pending key distribution)",
            user_id, org_id, role_str
        )
        return membership

    async def update_membership(self, data: dict) -> Optional[OrganizationMembership]:
        """Update membership (mainly role changes)."""
        user_id = data.get("public_user_data", {}).get("user_id")
        org_id = data.get("organization", {}).get("id")
        new_role_str = data.get("role", "org:member")

        if not all([user_id, org_id]):
            logger.warning("Membership update webhook missing required fields")
            return None

        result = await self.db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.user_id == user_id,
                    OrganizationMembership.org_id == org_id,
                )
            )
        )
        membership = result.scalar_one_or_none()

        if not membership:
            logger.warning("Membership not found for user %s in org %s, creating", user_id, org_id)
            return await self.create_membership(data)

        old_role = membership.role
        new_role = MemberRole(new_role_str)

        if old_role != new_role:
            membership.role = new_role

            # Audit log for role change
            audit_log = AuditLog.log_member_role_changed(
                id=str(uuid4()),
                admin_user_id="system",  # System change from Clerk
                member_user_id=user_id,
                org_id=org_id,
                old_role=old_role.value,
                new_role=new_role.value,
            )
            self.db.add(audit_log)

            await self.db.commit()
            logger.info(
                "Updated role for user %s in org %s: %s -> %s",
                user_id, org_id, old_role.value, new_role.value
            )

        return membership

    async def delete_membership(self, data: dict) -> None:
        """
        Delete membership and REVOKE org key access.

        CRITICAL: This clears the encrypted org key from the membership.
        The member can no longer decrypt organization messages.
        """
        user_id = data.get("public_user_data", {}).get("user_id")
        org_id = data.get("organization", {}).get("id")
        membership_id = data.get("id")

        # Try to find by ID first, then by user_id + org_id
        membership = None
        if membership_id:
            result = await self.db.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.id == membership_id
                )
            )
            membership = result.scalar_one_or_none()

        if not membership and user_id and org_id:
            result = await self.db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.user_id == user_id,
                        OrganizationMembership.org_id == org_id,
                    )
                )
            )
            membership = result.scalar_one_or_none()

        if not membership:
            logger.warning("Membership not found for deletion")
            return

        had_org_key = membership.has_org_key
        actual_user_id = membership.user_id
        actual_org_id = membership.org_id

        # Clear encrypted org key before deletion (revocation)
        if had_org_key:
            membership.clear_encrypted_org_key()

            # Audit log for key revocation
            audit_log = AuditLog.log_org_key_revoked(
                id=str(uuid4()),
                admin_user_id="system",  # System revocation from Clerk
                member_user_id=actual_user_id,
                org_id=actual_org_id,
                reason="membership_deleted_from_clerk",
            )
            self.db.add(audit_log)

        # Audit log for leaving
        audit_log = AuditLog.log_member_left(
            id=str(uuid4()),
            member_user_id=actual_user_id,
            org_id=actual_org_id,
            reason="removed_via_clerk_webhook",
        )
        self.db.add(audit_log)

        # Delete membership
        await self.db.delete(membership)
        await self.db.commit()

        logger.warning(
            "Deleted membership for user %s in org %s (had_org_key: %s)",
            actual_user_id, actual_org_id, had_org_key
        )
