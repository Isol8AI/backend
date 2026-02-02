# Fix Empty Sessions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure sessions are only persisted when messages are successfully stored, eliminating orphan empty sessions.

**Architecture:** Refactor to use deferred commit - create session object but don't commit until messages are stored. Both session creation and message storage happen in a single atomic transaction. Add cleanup for existing empty sessions.

**Tech Stack:** FastAPI, SQLAlchemy async, PostgreSQL

---

## Task 1: Add Transactional Session Creation in ChatService

**Files:**
- Modify: `backend/core/services/chat_service.py:82-124`

**Step 1: Add new method for creating session without commit**

Add this new method after `create_session`:

```python
async def create_session_deferred(
    self,
    user_id: str,
    name: str = "New Chat",
    org_id: Optional[str] = None,
) -> Session:
    """
    Create a new chat session WITHOUT committing.

    The session is added to the session but not committed.
    Call db.commit() after messages are stored to persist atomically.

    Args:
        user_id: User creating the session
        name: Display name for the session
        org_id: Organization ID (None for personal session)

    Returns:
        Created Session object (not yet committed)

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
    # NOTE: No commit here - caller must commit after storing messages

    logger.info("Created deferred session %s for user %s (org: %s)", session.id, user_id, org_id)
    return session
```

**Step 2: Modify store_encrypted_message to not commit individually**

Change `store_encrypted_message` (lines 322-380) to remove individual commits:

```python
async def store_encrypted_message(
    self,
    session_id: str,
    role: MessageRole,
    encrypted_payload: EncryptedPayload,
    model_used: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    commit: bool = True,  # Add parameter to control commit
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
        commit: Whether to commit after adding (default True for backward compat)

    Returns:
        Created Message object
    """

    # Convert bytes to hex strings for database storage
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
    if commit:
        await self.db.commit()
        await self.db.refresh(message)

    logger.debug(
        "Stored encrypted %s message %s in session %s",
        role.value if isinstance(role, MessageRole) else role,
        message.id,
        session_id,
    )
    return message
```

**Step 3: Add atomic commit method**

Add this method after `store_encrypted_message`:

```python
async def commit_session_with_messages(self) -> None:
    """
    Commit the current transaction (session + messages together).

    Call this after create_session_deferred and store_encrypted_message(commit=False)
    to persist everything atomically.
    """
    await self.db.commit()
    logger.debug("Committed session with messages atomically")
```

**Step 4: Run tests to verify no regressions**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird && ./run_tests.sh`
Expected: All existing tests pass (backward compatible changes)

**Step 5: Commit**

```bash
git add backend/core/services/chat_service.py
git commit -m "$(cat <<'EOF'
feat(backend): add deferred session creation for atomic commits

- Add create_session_deferred() that doesn't commit immediately
- Add commit parameter to store_encrypted_message for batch commits
- Add commit_session_with_messages() for atomic transaction commit

This enables session + messages to be committed together, preventing
orphan empty sessions when streaming fails.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update REST Endpoint to Use Atomic Commits

**Files:**
- Modify: `backend/routers/chat.py:341-358` and `395-426`

**Step 1: Change session creation to deferred**

In `chat_stream_encrypted` function, change session creation (around lines 341-358):

```python
        # Get or create session
        session_id = request.session_id
        is_new_session = False
        if session_id:
            session = await service.get_session(
                session_id=session_id,
                user_id=auth.user_id,
                org_id=auth.org_id,
            )
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or access denied")
        else:
            # Create new session (deferred - not committed yet)
            session = await service.create_session_deferred(
                user_id=auth.user_id,
                name="New Chat",
                org_id=auth.org_id,
            )
            session_id = session.id
            is_new_session = True
```

**Step 2: Update process_encrypted_message_stream call**

Update the streaming section to pass `is_new_session` context and handle atomic commit. Modify `process_encrypted_message_stream` in `chat_service.py` (lines 449-538):

```python
async def process_encrypted_message_stream(
    self,
    session_id: str,
    user_id: str,
    org_id: Optional[str],
    encrypted_message: EncryptedPayload,
    encrypted_history: list[EncryptedPayload],
    facts_context: Optional[str],
    model: str,
    client_transport_public_key: str,
    user_metadata: Optional[dict] = None,
    org_metadata: Optional[dict] = None,
    is_new_session: bool = False,  # Add this parameter
) -> AsyncGenerator[StreamChunk, None]:
    """
    Process an encrypted message through the enclave with streaming.

    This is the main entry point for sending a message. It:
    1. Gets storage key (user or org)
    2. Uses client's ephemeral transport key for response encryption
    3. Forwards encrypted message to enclave
    4. Yields encrypted response chunks
    5. Stores final encrypted messages (atomically with session if new)

    Args:
        session_id: Target session
        user_id: User sending the message
        org_id: Organization context (if any)
        encrypted_message: User's message encrypted to enclave
        encrypted_history: Previous messages re-encrypted to enclave
        facts_context: Client-side formatted facts context (already decrypted, plaintext)
        model: LLM model to use
        client_transport_public_key: Client's ephemeral key for response encryption
        user_metadata: User's Clerk privateMetadata (for AWS credentials)
        org_metadata: Org's Clerk privateMetadata (for AWS credentials)
        is_new_session: If True, session is uncommitted and will be committed with messages

    Yields:
        StreamChunk objects with encrypted content

    Raises:
        ValueError: If keys not found
    """
    # Get storage key (user or org public key for storing messages)
    storage_key = await self.get_storage_public_key(user_id, org_id)

    # Use the client's ephemeral transport key for response encryption
    client_key = bytes.fromhex(client_transport_public_key)

    enclave = get_enclave()

    # Stream through enclave
    async for chunk in enclave.process_message_streaming(
        encrypted_message=encrypted_message,
        encrypted_history=encrypted_history,
        facts_context=facts_context,
        storage_public_key=storage_key,
        client_public_key=client_key,
        session_id=session_id,
        model=model,
        user_id=user_id,
        org_id=org_id,
        user_metadata=user_metadata,
        org_metadata=org_metadata,
    ):
        # On final chunk, store the messages
        if chunk.is_final and not chunk.error:
            if chunk.stored_user_message and chunk.stored_assistant_message:
                # Store user message (don't commit yet if new session)
                await self.store_encrypted_message(
                    session_id=session_id,
                    role=MessageRole.USER,
                    encrypted_payload=chunk.stored_user_message,
                    model_used=model,
                    commit=False,  # Batch commit
                )

                # Store assistant message (don't commit yet)
                await self.store_encrypted_message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    encrypted_payload=chunk.stored_assistant_message,
                    model_used=chunk.model_used,
                    input_tokens=chunk.input_tokens,
                    output_tokens=chunk.output_tokens,
                    commit=False,  # Batch commit
                )

                # Update session timestamp (added to transaction)
                await self.update_session_timestamp_deferred(session_id)

                # Commit everything atomically (session if new + messages)
                await self.commit_session_with_messages()

        yield chunk
```

**Step 3: Add deferred timestamp update**

Add this method to `chat_service.py` after `update_session_timestamp`:

```python
async def update_session_timestamp_deferred(self, session_id: str) -> None:
    """Update session's updated_at timestamp without committing."""
    result = await self.db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if session:
        session.updated_at = datetime.utcnow()
        # NOTE: No commit - will be committed with messages
```

**Step 4: Update the router to pass is_new_session**

In `backend/routers/chat.py`, update the streaming call (around line 395):

```python
                async for chunk in stream_service.process_encrypted_message_stream(
                    session_id=session_id,
                    user_id=auth.user_id,
                    org_id=auth.org_id,
                    encrypted_message=encrypted_msg,
                    encrypted_history=encrypted_history,
                    facts_context=request.facts_context,
                    model=request.model,
                    client_transport_public_key=request.client_transport_public_key,
                    user_metadata=user_metadata,
                    org_metadata=org_metadata,
                    is_new_session=is_new_session,  # Add this
                ):
```

**Step 5: Run tests**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird && ./run_tests.sh`
Expected: All tests pass

**Step 6: Commit**

```bash
git add backend/routers/chat.py backend/core/services/chat_service.py
git commit -m "$(cat <<'EOF'
feat(backend): atomic session+message commits in REST endpoint

- Use create_session_deferred for new sessions
- Pass is_new_session flag to process_encrypted_message_stream
- Commit session and messages together atomically
- Add update_session_timestamp_deferred for batch operations

Sessions are only persisted when messages are successfully stored.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update WebSocket Endpoint to Use Atomic Commits

**Files:**
- Modify: `backend/routers/websocket_chat.py:317-324` and `356-410`

**Step 1: Change session creation to deferred**

In `_process_chat_message_background` function, change session creation (around lines 317-324):

```python
            session_id = request.session_id
            is_new_session = False
            if session_id:
                session = await service.get_session(
                    session_id=session_id,
                    user_id=user_id,
                    org_id=org_id,
                )
                if not session:
                    management_api.send_message(
                        connection_id,
                        {"type": "error", "message": "Session not found or access denied"},
                    )
                    return
            else:
                # Create new session (deferred - not committed yet)
                session = await service.create_session_deferred(
                    user_id=user_id,
                    name="New Chat",
                    org_id=org_id,
                )
                session_id = session.id
                is_new_session = True
```

**Step 2: Update the streaming call to pass is_new_session**

Around line 359:

```python
            async for chunk in stream_service.process_encrypted_message_stream(
                session_id=session_id,
                user_id=user_id,
                org_id=org_id,
                encrypted_message=encrypted_msg,
                encrypted_history=encrypted_history,
                facts_context=request.facts_context,
                model=request.model,
                client_transport_public_key=request.client_transport_public_key,
                user_metadata=user_metadata,
                org_metadata=org_metadata,
                is_new_session=is_new_session,  # Add this
            ):
```

**Step 3: Run tests**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird && ./run_tests.sh`
Expected: All tests pass

**Step 4: Commit**

```bash
git add backend/routers/websocket_chat.py
git commit -m "$(cat <<'EOF'
feat(backend): atomic session+message commits in WebSocket endpoint

- Use create_session_deferred for new sessions
- Pass is_new_session flag to process_encrypted_message_stream
- Sessions only persisted when messages successfully stored

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add Cleanup Script for Existing Empty Sessions

**Files:**
- Create: `backend/scripts/cleanup_empty_sessions.py`

**Step 1: Create cleanup script**

```python
#!/usr/bin/env python3
"""
Cleanup script to delete sessions with no messages.

These orphan sessions were created before the atomic commit fix
when streaming failed between session creation and message storage.

Usage:
    # Dry run (show what would be deleted)
    python scripts/cleanup_empty_sessions.py --dry-run

    # Actually delete
    python scripts/cleanup_empty_sessions.py
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session_factory
from models.session import Session
from models.message import Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def find_empty_sessions(db: AsyncSession) -> list[Session]:
    """Find all sessions that have no messages."""
    # Subquery to get session IDs that have at least one message
    sessions_with_messages = (
        select(Message.session_id)
        .distinct()
        .scalar_subquery()
    )

    # Find sessions NOT in that list
    query = select(Session).where(
        Session.id.not_in(sessions_with_messages)
    ).order_by(Session.created_at.asc())

    result = await db.execute(query)
    return list(result.scalars().all())


async def delete_empty_sessions(db: AsyncSession, sessions: list[Session]) -> int:
    """Delete the given sessions."""
    for session in sessions:
        await db.delete(session)
    await db.commit()
    return len(sessions)


async def main(dry_run: bool = True) -> None:
    """Main cleanup function."""
    async with async_session_factory() as db:
        empty_sessions = await find_empty_sessions(db)

        if not empty_sessions:
            logger.info("No empty sessions found. Database is clean!")
            return

        logger.info(f"Found {len(empty_sessions)} empty session(s):")
        for session in empty_sessions:
            logger.info(
                f"  - {session.id} (user: {session.user_id}, org: {session.org_id or 'personal'}, "
                f"created: {session.created_at})"
            )

        if dry_run:
            logger.info("\nDry run mode - no sessions deleted.")
            logger.info("Run with --delete to actually remove these sessions.")
        else:
            count = await delete_empty_sessions(db, empty_sessions)
            logger.info(f"\nDeleted {count} empty session(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup empty sessions")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete empty sessions (default is dry-run)",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=not args.delete))
```

**Step 2: Test the script in dry-run mode**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python scripts/cleanup_empty_sessions.py --dry-run`
Expected: Lists any empty sessions found

**Step 3: Commit**

```bash
git add backend/scripts/cleanup_empty_sessions.py
git commit -m "$(cat <<'EOF'
feat(backend): add cleanup script for orphan empty sessions

Identifies and removes sessions with no messages, which were created
before the atomic commit fix when streaming failed mid-process.

Usage:
  python scripts/cleanup_empty_sessions.py --dry-run  # Preview
  python scripts/cleanup_empty_sessions.py --delete   # Execute

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Run Cleanup and Verify

**Step 1: Run cleanup in dry-run mode first**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python scripts/cleanup_empty_sessions.py --dry-run`
Expected: Shows list of empty sessions (if any)

**Step 2: Run actual cleanup (if empty sessions found)**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python scripts/cleanup_empty_sessions.py --delete`
Expected: "Deleted N empty session(s)."

**Step 3: Verify no empty sessions remain**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python scripts/cleanup_empty_sessions.py --dry-run`
Expected: "No empty sessions found. Database is clean!"

---

## Task 6: Integration Test - Verify Atomic Behavior

**Step 1: Start dev environment**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird && ./start_dev.sh`

**Step 2: Manual test - simulate failure**

1. Open browser to dev.isol8.co
2. Start a new chat message
3. Quickly close browser/tab during streaming (before "done" event)
4. Re-open and check sidebar - the interrupted session should NOT appear

**Step 3: Verify in database**

Run cleanup script to confirm no new orphan sessions:
```bash
cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python scripts/cleanup_empty_sessions.py --dry-run
```
Expected: "No empty sessions found."

---

## Task 7: Push Changes

**Step 1: Run all tests one final time**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird && ./run_tests.sh`
Expected: All tests pass

**Step 2: Push to remote**

```bash
git push origin main
```

**Step 3: Monitor deployment**

Watch for CI/CD pipeline to complete successfully.

---

## Summary of Changes

1. **ChatService** (`chat_service.py`):
   - Added `create_session_deferred()` - creates session without commit
   - Added `commit` parameter to `store_encrypted_message()`
   - Added `update_session_timestamp_deferred()` - updates without commit
   - Added `commit_session_with_messages()` - atomic commit helper
   - Modified `process_encrypted_message_stream()` to accept `is_new_session` flag

2. **REST Endpoint** (`chat.py`):
   - Uses `create_session_deferred()` for new sessions
   - Passes `is_new_session` to streaming function

3. **WebSocket Endpoint** (`websocket_chat.py`):
   - Uses `create_session_deferred()` for new sessions
   - Passes `is_new_session` to streaming function

4. **Cleanup Script** (`scripts/cleanup_empty_sessions.py`):
   - Finds and deletes orphan sessions with no messages
   - Supports dry-run mode for safety
