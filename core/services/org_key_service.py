"""
Organization Key Service - manages org encryption keys and distribution.

Security Note:
- Org private keys are encrypted client-side
- Admin-encrypted copy uses org passcode (only admins know)
- Member copies are encrypted TO each member's personal public key
- Server cannot decrypt any of these
"""

import logging
from dataclasses import dataclass
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.organization import Organization
from models.organization_membership import OrganizationMembership
from models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class OrgKeyServiceError(Exception):
    """Base exception for org key service errors."""

    pass


class OrgKeysAlreadyExistError(OrgKeyServiceError):
    """Organization already has encryption keys."""

    pass


class OrgKeysNotFoundError(OrgKeyServiceError):
    """Organization does not have encryption keys."""

    pass


class MembershipNotFoundError(OrgKeyServiceError):
    """Membership not found."""

    pass


class NotAdminError(OrgKeyServiceError):
    """User is not an admin of the organization."""

    pass


class MemberNotReadyError(OrgKeyServiceError):
    """Member has not set up personal encryption keys."""

    pass


@dataclass
class BulkDistributionResult:
    """Result for a single distribution in a bulk operation."""

    membership_id: str
    user_id: str
    success: bool
    error: Optional[str] = None


class OrgKeyService:
    """
    Service for managing organization encryption keys.

    All operations work with encrypted key material only.
    The server cannot decrypt org private keys.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID with memberships loaded."""
        result = await self.db.execute(
            select(Organization).where(Organization.id == org_id).options(selectinload(Organization.memberships))
        )
        return result.scalar_one_or_none()

    async def get_membership(
        self,
        user_id: str,
        org_id: str,
    ) -> Optional[OrganizationMembership]:
        """Get user's membership in an organization."""
        result = await self.db.execute(
            select(OrganizationMembership)
            .where(
                and_(
                    OrganizationMembership.user_id == user_id,
                    OrganizationMembership.org_id == org_id,
                )
            )
            .options(selectinload(OrganizationMembership.user))
        )
        return result.scalar_one_or_none()

    async def verify_admin(self, user_id: str, org_id: str) -> OrganizationMembership:
        """
        Verify user is an admin of the organization.

        Args:
            user_id: User ID to verify
            org_id: Organization ID

        Returns:
            OrganizationMembership if user is admin

        Raises:
            MembershipNotFoundError: If user is not a member
            NotAdminError: If user is not an admin
        """
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            raise MembershipNotFoundError(f"User {user_id} is not a member of org {org_id}")
        if not membership.is_admin:
            raise NotAdminError(f"User {user_id} is not an admin of org {org_id}")
        return membership

    # =========================================================================
    # Org Key Creation
    # =========================================================================

    async def create_org_keys(
        self,
        org_id: str,
        admin_user_id: str,
        org_public_key: str,
        admin_encrypted_private_key: str,
        admin_iv: str,
        admin_tag: str,
        admin_salt: str,
        admin_member_key_ephemeral: str,
        admin_member_key_iv: str,
        admin_member_key_ciphertext: str,
        admin_member_key_tag: str,
        admin_member_key_hkdf_salt: str,
    ) -> Organization:
        """
        Create organization encryption keys.

        Args:
            org_id: Organization ID
            admin_user_id: Admin creating the keys
            org_public_key: Org's X25519 public key (hex)
            admin_encrypted_private_key: Org private key encrypted with org passcode
            admin_iv, admin_tag, admin_salt: AES-GCM params for passcode encryption
            admin_member_key_*: Org key encrypted TO admin's personal public key

        Returns:
            Updated Organization

        Raises:
            OrgKeyServiceError: If org not found or invalid format
            OrgKeysAlreadyExistError: If org already has keys
            NotAdminError: If user is not an admin
        """
        # Verify user is admin
        admin_membership = await self.verify_admin(admin_user_id, org_id)

        # Get organization
        org = await self.get_organization(org_id)
        if not org:
            raise OrgKeyServiceError(f"Organization {org_id} not found")

        if org.has_encryption_keys:
            raise OrgKeysAlreadyExistError(f"Organization {org_id} already has encryption keys")

        # Store org keys
        try:
            org.set_encryption_keys(
                org_public_key=org_public_key,
                admin_encrypted_private_key=admin_encrypted_private_key,
                iv=admin_iv,
                tag=admin_tag,
                salt=admin_salt,
                created_by=admin_user_id,
            )
        except ValueError as e:
            raise OrgKeyServiceError(f"Invalid key format: {e}")

        # Also store admin's member copy (so admin can use org encryption)
        try:
            admin_membership.set_encrypted_org_key(
                ephemeral_public_key=admin_member_key_ephemeral,
                iv=admin_member_key_iv,
                ciphertext=admin_member_key_ciphertext,
                auth_tag=admin_member_key_tag,
                hkdf_salt=admin_member_key_hkdf_salt,
                distributed_by_user_id=admin_user_id,
            )
        except ValueError as e:
            raise OrgKeyServiceError(f"Invalid admin member key format: {e}")

        # Audit log
        audit_log = AuditLog.log_org_keys_created(
            id=str(uuid4()),
            admin_user_id=admin_user_id,
            org_id=org_id,
        )
        self.db.add(audit_log)

        await self.db.commit()
        await self.db.refresh(org)

        logger.info("Created org keys for %s by admin %s", org_id, admin_user_id)
        return org

    # =========================================================================
    # Key Distribution
    # =========================================================================

    async def get_pending_distributions(
        self,
        org_id: str,
        admin_user_id: str,
    ) -> dict:
        """
        Get members who need org key distribution.

        Returns two categories:
        - ready_for_distribution: Members with personal keys, ready to receive org key
        - needs_personal_setup: Members without personal keys (cannot receive org key yet)

        Args:
            org_id: Organization ID
            admin_user_id: Admin requesting the list

        Returns:
            Dict with ready_for_distribution, needs_personal_setup, ready_count, needs_setup_count

        Raises:
            NotAdminError: If user is not an admin
            OrgKeysNotFoundError: If org has no encryption keys
        """
        # Verify admin
        await self.verify_admin(admin_user_id, org_id)

        # Get org
        org = await self.get_organization(org_id)
        if not org:
            raise OrgKeyServiceError(f"Organization {org_id} not found")

        if not org.has_encryption_keys:
            raise OrgKeysNotFoundError(f"Organization {org_id} has no encryption keys")

        # Find pending members (those without org key)
        result = await self.db.execute(
            select(OrganizationMembership)
            .where(
                and_(
                    OrganizationMembership.org_id == org_id,
                    OrganizationMembership.has_org_key.is_(False),
                )
            )
            .options(selectinload(OrganizationMembership.user))
        )
        memberships = result.scalars().all()

        ready_for_distribution = []
        needs_personal_setup = []

        for m in memberships:
            if m.user and m.user.has_encryption_keys:
                # Member has personal keys - ready for distribution
                ready_for_distribution.append(
                    {
                        "membership_id": m.id,
                        "user_id": m.user_id,
                        "user_public_key": m.user.public_key,
                        "role": m.role.value,
                        "joined_at": m.joined_at or m.created_at,
                    }
                )
            elif m.user:
                # Member exists but has no personal keys - needs setup first
                needs_personal_setup.append(
                    {
                        "membership_id": m.id,
                        "user_id": m.user_id,
                        "role": m.role.value,
                        "joined_at": m.joined_at or m.created_at,
                    }
                )

        return {
            "ready_for_distribution": ready_for_distribution,
            "needs_personal_setup": needs_personal_setup,
            "ready_count": len(ready_for_distribution),
            "needs_setup_count": len(needs_personal_setup),
        }

    async def distribute_org_key(
        self,
        org_id: str,
        admin_user_id: str,
        membership_id: str,
        ephemeral_public_key: str,
        iv: str,
        ciphertext: str,
        auth_tag: str,
        hkdf_salt: str,
    ) -> OrganizationMembership:
        """
        Distribute org key to a member.

        The admin has already:
        1. Decrypted the org key using their personal private key
        2. Re-encrypted it TO the member's public key

        This stores that encrypted blob in the membership record.

        Args:
            org_id: Organization ID
            admin_user_id: Admin distributing the key
            membership_id: Target membership ID
            ephemeral_public_key: For ECDH (64 hex)
            iv: AES-GCM IV (32 hex)
            ciphertext: Encrypted org key (variable hex)
            auth_tag: AES-GCM auth tag (32 hex)
            hkdf_salt: HKDF salt (64 hex)

        Returns:
            Updated OrganizationMembership

        Raises:
            NotAdminError: If user is not an admin
            MembershipNotFoundError: If membership not found
            OrgKeyServiceError: If membership doesn't belong to org or already has key
        """
        # Verify admin
        await self.verify_admin(admin_user_id, org_id)

        # Get membership
        result = await self.db.execute(
            select(OrganizationMembership)
            .where(OrganizationMembership.id == membership_id)
            .options(selectinload(OrganizationMembership.user))
        )
        membership = result.scalar_one_or_none()

        if not membership:
            raise MembershipNotFoundError(f"Membership {membership_id} not found")

        if membership.org_id != org_id:
            raise OrgKeyServiceError("Membership does not belong to this organization")

        if membership.has_org_key:
            raise OrgKeyServiceError("Member already has org key")

        # Verify member has personal encryption keys
        if not membership.user or not membership.user.has_encryption_keys:
            raise MemberNotReadyError(f"Member {membership.user_id} has not set up personal encryption keys")

        # Store encrypted org key
        try:
            membership.set_encrypted_org_key(
                ephemeral_public_key=ephemeral_public_key,
                iv=iv,
                ciphertext=ciphertext,
                auth_tag=auth_tag,
                hkdf_salt=hkdf_salt,
                distributed_by_user_id=admin_user_id,
            )
        except ValueError as e:
            raise OrgKeyServiceError(f"Invalid key format: {e}")

        # Audit log
        audit_log = AuditLog.log_org_key_distributed(
            id=str(uuid4()),
            admin_user_id=admin_user_id,
            member_user_id=membership.user_id,
            org_id=org_id,
        )
        self.db.add(audit_log)

        await self.db.commit()
        await self.db.refresh(membership)

        logger.info("Distributed org key to user %s in org %s by admin %s", membership.user_id, org_id, admin_user_id)
        return membership

    async def bulk_distribute_org_keys(
        self,
        org_id: str,
        admin_user_id: str,
        distributions: List[dict],
    ) -> List["BulkDistributionResult"]:
        """
        Distribute org key to multiple members at once.

        Each distribution dict should contain:
        - membership_id: Target membership ID
        - ephemeral_public_key, iv, ciphertext, auth_tag, hkdf_salt: Encrypted key data

        Args:
            org_id: Organization ID
            admin_user_id: Admin distributing the keys
            distributions: List of distribution requests

        Returns:
            List of BulkDistributionResult for each distribution attempt

        Raises:
            NotAdminError: If user is not an admin
        """
        # Verify admin once
        await self.verify_admin(admin_user_id, org_id)

        results = []

        for dist in distributions:
            membership_id = dist.get("membership_id", "unknown")
            try:
                # Get membership
                result = await self.db.execute(
                    select(OrganizationMembership)
                    .where(OrganizationMembership.id == dist["membership_id"])
                    .options(selectinload(OrganizationMembership.user))
                )
                membership = result.scalar_one_or_none()

                if not membership:
                    results.append(
                        BulkDistributionResult(
                            membership_id=membership_id,
                            user_id="unknown",
                            success=False,
                            error="Membership not found",
                        )
                    )
                    continue

                if membership.org_id != org_id:
                    results.append(
                        BulkDistributionResult(
                            membership_id=membership_id,
                            user_id=membership.user_id,
                            success=False,
                            error="Membership does not belong to this organization",
                        )
                    )
                    continue

                if membership.has_org_key:
                    results.append(
                        BulkDistributionResult(
                            membership_id=membership_id,
                            user_id=membership.user_id,
                            success=False,
                            error="Member already has org key",
                        )
                    )
                    continue

                if not membership.user or not membership.user.has_encryption_keys:
                    results.append(
                        BulkDistributionResult(
                            membership_id=membership_id,
                            user_id=membership.user_id if membership.user else "unknown",
                            success=False,
                            error="Member has not set up personal encryption keys",
                        )
                    )
                    continue

                # Store encrypted org key
                membership.set_encrypted_org_key(
                    ephemeral_public_key=dist["ephemeral_public_key"],
                    iv=dist["iv"],
                    ciphertext=dist["ciphertext"],
                    auth_tag=dist["auth_tag"],
                    hkdf_salt=dist["hkdf_salt"],
                    distributed_by_user_id=admin_user_id,
                )

                # Audit log
                audit_log = AuditLog.log_org_key_distributed(
                    id=str(uuid4()),
                    admin_user_id=admin_user_id,
                    member_user_id=membership.user_id,
                    org_id=org_id,
                )
                self.db.add(audit_log)

                results.append(
                    BulkDistributionResult(
                        membership_id=membership_id,
                        user_id=membership.user_id,
                        success=True,
                    )
                )

            except Exception as e:
                results.append(
                    BulkDistributionResult(
                        membership_id=membership_id,
                        user_id="unknown",
                        success=False,
                        error=str(e),
                    )
                )

        # Commit all successful distributions
        await self.db.commit()

        success_count = sum(1 for r in results if r.success)
        logger.info(
            "Bulk distributed org keys in org %s by admin %s: %d/%d successful",
            org_id,
            admin_user_id,
            success_count,
            len(results),
        )

        return results

    # =========================================================================
    # Key Retrieval
    # =========================================================================

    async def get_org_encryption_status(self, org_id: str) -> dict:
        """
        Get organization's encryption status.

        Args:
            org_id: Organization ID

        Returns:
            Dict with has_encryption_keys, org_public_key, etc.

        Raises:
            OrgKeyServiceError: If org not found
        """
        org = await self.get_organization(org_id)
        if not org:
            raise OrgKeyServiceError(f"Organization {org_id} not found")

        return org.encryption_key_info

    async def get_member_org_key(
        self,
        user_id: str,
        org_id: str,
    ) -> dict:
        """
        Get member's encrypted org key.

        Used by member to decrypt org key using their personal private key.

        Args:
            user_id: User ID
            org_id: Organization ID

        Returns:
            Dict with encrypted org key payload

        Raises:
            MembershipNotFoundError: If user is not a member
            OrgKeysNotFoundError: If org key not yet distributed to member
        """
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            raise MembershipNotFoundError(f"User {user_id} is not a member of org {org_id}")

        if not membership.has_org_key:
            raise OrgKeysNotFoundError("Org key not yet distributed to this member")

        return membership.encrypted_org_key_payload

    async def get_admin_recovery_key(
        self,
        admin_user_id: str,
        org_id: str,
    ) -> dict:
        """
        Get admin-encrypted org key for recovery.

        Used when admin needs to recover org key using org passcode.

        Args:
            admin_user_id: Admin requesting recovery keys
            org_id: Organization ID

        Returns:
            Dict with admin-encrypted org key data

        Raises:
            NotAdminError: If user is not an admin
            OrgKeysNotFoundError: If org has no encryption keys
        """
        await self.verify_admin(admin_user_id, org_id)

        org = await self.get_organization(org_id)
        if not org or not org.has_encryption_keys:
            raise OrgKeysNotFoundError(f"Organization {org_id} has no encryption keys")

        return org.get_admin_encrypted_keys()

    # =========================================================================
    # Key Revocation
    # =========================================================================

    async def revoke_member_org_key(
        self,
        org_id: str,
        admin_user_id: str,
        member_user_id: str,
        reason: Optional[str] = None,
    ) -> None:
        """
        Revoke a member's org key.

        This clears the encrypted org key from their membership.
        They can no longer decrypt org messages.

        Args:
            org_id: Organization ID
            admin_user_id: Admin revoking the key
            member_user_id: Member whose key is being revoked
            reason: Optional reason for revocation (audit log)

        Raises:
            NotAdminError: If user is not an admin
            MembershipNotFoundError: If member not found
        """
        await self.verify_admin(admin_user_id, org_id)

        membership = await self.get_membership(member_user_id, org_id)
        if not membership:
            raise MembershipNotFoundError(f"User {member_user_id} is not a member of org {org_id}")

        if not membership.has_org_key:
            return  # Nothing to revoke

        # Clear org key
        membership.clear_encrypted_org_key()

        # Audit log
        audit_log = AuditLog.log_org_key_revoked(
            id=str(uuid4()),
            admin_user_id=admin_user_id,
            member_user_id=member_user_id,
            org_id=org_id,
            reason=reason,
        )
        self.db.add(audit_log)

        await self.db.commit()

        logger.warning(
            "Revoked org key from user %s in org %s by admin %s: %s", member_user_id, org_id, admin_user_id, reason
        )
