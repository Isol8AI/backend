"""
Chat service for encrypted message handling.

Security Note:
- Server acts as BLIND RELAY - never sees plaintext content
- All messages encrypted client-side before sending
- Enclave decrypts, processes, and re-encrypts for storage
- Messages stored encrypted to user's or org's public key
"""
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Tuple
from uuid import uuid4

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.crypto import EncryptedPayload
from core.enclave import get_enclave
from core.enclave.mock_enclave import StreamChunk
from models.audit_log import AuditLog
from models.message import Message, MessageRole
from models.session import Session
from models.user import User
from models.organization import Organization
from models.organization_membership import OrganizationMembership

logger = logging.getLogger(__name__)


class StorageKeyNotFoundError(ValueError):
    """Raised when encryption key cannot be found for storage."""
    pass


class ChatService:
    """
    Service for encrypted chat operations.

    This service:
    1. Manages chat sessions (personal and org)
    2. Stores encrypted messages (server cannot read)
    3. Retrieves encrypted messages for client decryption
    4. Coordinates with enclave for message processing
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # =========================================================================
    # Enclave Information
    # =========================================================================

    def get_enclave_public_key(self) -> str:
        """
        Get enclave's public key for client-side encryption.

        Returns:
            Enclave's X25519 public key as hex string
        """
        enclave = get_enclave()
        return enclave.get_transport_public_key()

    def get_enclave_info(self) -> dict:
        """
        Get enclave info for client.

        Returns:
            Dict with enclave_public_key and attestation (if available)
        """
        enclave = get_enclave()
        info = enclave.get_info()
        return info.to_hex_dict()

    # =========================================================================
    # Session Management
    # =========================================================================

    async def create_session(
        self,
        user_id: str,
        name: str = "New Chat",
        org_id: Optional[str] = None,
    ) -> Session:
        """
        Create a new chat session.

        Args:
            user_id: User creating the session
            name: Display name for the session
            org_id: Organization ID (None for personal session)

        Returns:
            Created Session object

        Raises:
            ValueError: If org_id provided but user not a member
        """
        # Verify user exists
        user = await self._get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # If org session, verify membership
        if org_id:
            membership = await self._get_membership(user_id, org_id)
            if not membership:
                raise ValueError(f"User {user_id} is not a member of org {org_id}")

        session = Session(
            id=str(uuid4()),
            user_id=user_id,
            org_id=org_id,
            name=name,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)

        logger.info(
            "Created session %s for user %s (org: %s)",
            session.id, user_id, org_id
        )
        return session

    async def get_session(
        self,
        session_id: str,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> Optional[Session]:
        """
        Get session with ownership verification.

        Args:
            session_id: Session to retrieve
            user_id: User requesting the session
            org_id: Current org context (must match session's org_id)

        Returns:
            Session if found and authorized, None otherwise
        """
        query = select(Session).where(
            and_(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )

        # Verify org context matches
        if org_id:
            query = query.where(Session.org_id == org_id)
        else:
            query = query.where(Session.org_id.is_(None))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_sessions(
        self,
        user_id: str,
        org_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[list[Session], int]:
        """
        List sessions for user in current context with pagination.

        Args:
            user_id: User's sessions to list
            org_id: Filter by org (None for personal sessions)
            limit: Maximum number of sessions to return (default 50)
            offset: Number of sessions to skip (default 0)

        Returns:
            Tuple of (sessions list, total count)
        """
        # Build base query conditions
        base_conditions = [Session.user_id == user_id]
        if org_id:
            base_conditions.append(Session.org_id == org_id)
        else:
            base_conditions.append(Session.org_id.is_(None))

        # Get total count
        count_query = select(func.count()).select_from(Session).where(and_(*base_conditions))
        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        # Get paginated results
        query = (
            select(Session)
            .where(and_(*base_conditions))
            .order_by(Session.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(query)
        sessions = list(result.scalars().all())

        return sessions, total

    async def update_session_timestamp(self, session_id: str) -> None:
        """Update session's updated_at timestamp."""
        result = await self.db.execute(
            select(Session).where(Session.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.updated_at = datetime.utcnow()
            await self.db.commit()

    async def delete_session(
        self,
        session_id: str,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a session and all its messages (GDPR compliance).

        The Session model has cascade="all, delete-orphan" for messages,
        so deleting a session automatically deletes all its messages.

        Args:
            session_id: Session to delete
            user_id: User requesting deletion (for ownership check)
            org_id: Current org context

        Returns:
            True if session was deleted, False if not found/unauthorized
        """
        session = await self.get_session(session_id, user_id, org_id)
        if not session:
            return False

        # Audit log the deletion
        audit_log = AuditLog.log_session_deleted(
            id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
        )
        self.db.add(audit_log)

        # Delete session (messages cascade)
        await self.db.delete(session)
        await self.db.commit()

        logger.info("Deleted session %s for user %s", session_id, user_id)
        return True

    async def delete_all_sessions(
        self,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> int:
        """
        Delete all sessions for user in current context (GDPR compliance).

        Args:
            user_id: User whose sessions to delete
            org_id: Organization context (None for personal sessions)

        Returns:
            Number of sessions deleted
        """
        # Get all sessions first (for audit logging)
        sessions, _ = await self.list_sessions(user_id, org_id, limit=10000, offset=0)

        if not sessions:
            return 0

        for session in sessions:
            # Audit log each deletion
            audit_log = AuditLog.log_session_deleted(
                id=str(uuid4()),
                user_id=user_id,
                session_id=session.id,
                org_id=org_id,
            )
            self.db.add(audit_log)
            await self.db.delete(session)

        await self.db.commit()

        logger.warning(
            "Deleted %d sessions for user %s (org: %s)",
            len(sessions), user_id, org_id
        )
        return len(sessions)

    # =========================================================================
    # Message Operations
    # =========================================================================

    async def get_session_messages(
        self,
        session_id: str,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> list[Message]:
        """
        Get all messages for a session (encrypted).

        Args:
            session_id: Session to get messages from
            user_id: User requesting (for ownership check)
            org_id: Current org context (must match session)

        Returns:
            List of encrypted messages, oldest first

        Raises:
            ValueError: If session not found or access denied
        """
        # Verify session ownership
        session = await self.get_session(session_id, user_id, org_id)
        if not session:
            raise ValueError("Session not found or access denied")

        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        return list(result.scalars().all())

    async def store_encrypted_message(
        self,
        session_id: str,
        role: MessageRole,
        encrypted_payload: EncryptedPayload,
        model_used: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Message:
        """
        Store an encrypted message.

        The server stores the encrypted content as-is, without any ability
        to read it. Only the client (with user's private key) can decrypt.

        Args:
            session_id: Session to add message to
            role: USER or ASSISTANT
            encrypted_payload: Pre-encrypted message content (bytes from crypto layer)
            model_used: Model ID (for assistant messages)
            input_tokens: Token usage (for billing)
            output_tokens: Token usage (for billing)

        Returns:
            Created Message object
        """
        # Convert bytes to hex strings for database storage
        # The crypto layer uses bytes, but the database stores hex strings
        def to_hex(value) -> str:
            if isinstance(value, bytes):
                return value.hex()
            return value

        message = Message.create_encrypted(
            id=str(uuid4()),
            session_id=session_id,
            role=role,
            ephemeral_public_key=to_hex(encrypted_payload.ephemeral_public_key),
            iv=to_hex(encrypted_payload.iv),
            ciphertext=to_hex(encrypted_payload.ciphertext),
            auth_tag=to_hex(encrypted_payload.auth_tag),
            hkdf_salt=to_hex(encrypted_payload.hkdf_salt),
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)

        logger.debug(
            "Stored encrypted %s message %s in session %s",
            role.value if isinstance(role, MessageRole) else role,
            message.id,
            session_id
        )
        return message

    # =========================================================================
    # Key Resolution
    # =========================================================================

    async def get_storage_public_key(
        self,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> bytes:
        """
        Get the public key for encrypting stored messages.

        For personal sessions: User's public key
        For org sessions: Org's public key (so all members can decrypt)

        Args:
            user_id: User ID
            org_id: Organization ID (None for personal)

        Returns:
            Public key bytes

        Raises:
            StorageKeyNotFoundError: If encryption keys not found
        """
        if org_id:
            # Org session - use org's key
            result = await self.db.execute(
                select(Organization).where(Organization.id == org_id)
            )
            org = result.scalar_one_or_none()
            if not org:
                raise StorageKeyNotFoundError(f"Organization {org_id} not found")
            if not org.has_encryption_keys:
                raise StorageKeyNotFoundError(
                    f"Organization {org_id} has no encryption keys configured"
                )
            return bytes.fromhex(org.org_public_key)
        else:
            # Personal session - use user's key
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise StorageKeyNotFoundError(f"User {user_id} not found")
            if not user.has_encryption_keys:
                raise StorageKeyNotFoundError(
                    f"User {user_id} has not set up encryption keys"
                )
            return bytes.fromhex(user.public_key)

    async def get_user_public_key(self, user_id: str) -> Optional[bytes]:
        """
        Get user's public key for transport encryption.

        Always the user's key, regardless of session type.
        Response is always encrypted to the requesting user.

        Args:
            user_id: User ID

        Returns:
            Public key bytes, or None if not found
        """
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user and user.has_encryption_keys:
            return bytes.fromhex(user.public_key)
        return None

    # =========================================================================
    # Streaming Chat with Enclave
    # =========================================================================

    async def process_encrypted_message_stream(
        self,
        session_id: str,
        user_id: str,
        org_id: Optional[str],
        encrypted_message: EncryptedPayload,
        encrypted_history: list[EncryptedPayload],
        model: str,
        client_transport_public_key: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process an encrypted message through the enclave with streaming.

        This is the main entry point for sending a message. It:
        1. Gets storage key (user or org)
        2. Uses client's ephemeral transport key for response encryption
        3. Forwards encrypted message to enclave
        4. Yields encrypted response chunks
        5. Stores final encrypted messages

        Args:
            session_id: Target session
            user_id: User sending the message
            org_id: Organization context (if any)
            encrypted_message: User's message encrypted to enclave
            encrypted_history: Previous messages re-encrypted to enclave
            model: LLM model to use
            client_transport_public_key: Client's ephemeral key for response encryption

        Yields:
            StreamChunk objects with encrypted content

        Raises:
            ValueError: If keys not found
        """
        # Get storage key (user or org public key for storing messages)
        # Raises StorageKeyNotFoundError if keys not configured
        storage_key = await self.get_storage_public_key(user_id, org_id)

        # Use the client's ephemeral transport key for response encryption
        # Convert from hex string to bytes
        client_key = bytes.fromhex(client_transport_public_key)

        enclave = get_enclave()

        # Stream through enclave
        async for chunk in enclave.process_message_streaming(
            encrypted_message=encrypted_message,
            encrypted_history=encrypted_history,
            storage_public_key=storage_key,
            client_public_key=client_key,
            session_id=session_id,
            model=model,
        ):
            # On final chunk, store the messages
            if chunk.is_final and not chunk.error:
                if chunk.stored_user_message and chunk.stored_assistant_message:
                    # Store user message
                    await self.store_encrypted_message(
                        session_id=session_id,
                        role=MessageRole.USER,
                        encrypted_payload=chunk.stored_user_message,
                        model_used=model,
                    )

                    # Store assistant message
                    await self.store_encrypted_message(
                        session_id=session_id,
                        role=MessageRole.ASSISTANT,
                        encrypted_payload=chunk.stored_assistant_message,
                        model_used=chunk.model_used,
                        input_tokens=chunk.input_tokens,
                        output_tokens=chunk.output_tokens,
                    )

                    # Update session timestamp
                    await self.update_session_timestamp(session_id)

            yield chunk

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def _get_membership(
        self,
        user_id: str,
        org_id: str,
    ) -> Optional[OrganizationMembership]:
        """Get membership for user in org."""
        result = await self.db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.user_id == user_id,
                    OrganizationMembership.org_id == org_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def verify_can_send_encrypted(
        self,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Verify user can send encrypted messages.

        Checks:
        1. User has encryption keys set up
        2. If org context, org has encryption keys

        Returns:
            Tuple of (can_send, error_message)
        """
        # Check user has keys
        user = await self._get_user(user_id)
        if not user:
            return False, "User not found"
        if not user.has_encryption_keys:
            return False, "User has not set up encryption keys"

        # Check org has keys (if org context)
        if org_id:
            result = await self.db.execute(
                select(Organization).where(Organization.id == org_id)
            )
            org = result.scalar_one_or_none()
            if not org:
                return False, "Organization not found"
            if not org.has_encryption_keys:
                return False, "Organization encryption not set up"

            # Verify user is member with org key
            membership = await self._get_membership(user_id, org_id)
            if not membership:
                return False, "User is not a member of this organization"
            if not membership.has_org_key:
                return False, "User does not have the organization key distributed"

        return True, ""
