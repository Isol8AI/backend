#!/usr/bin/env python3
"""
Cleanup script to delete sessions with no messages.

These orphan sessions were created before the atomic commit fix
when streaming failed between session creation and message storage.

Usage:
    # Dry run (show what would be deleted)
    python scripts/cleanup_empty_sessions.py --dry-run

    # Actually delete
    python scripts/cleanup_empty_sessions.py --delete
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session_factory
from models.session import Session
from models.message import Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def find_empty_sessions(db: AsyncSession) -> list[Session]:
    """Find all sessions that have no messages."""
    # Subquery to get session IDs that have at least one message
    sessions_with_messages = select(Message.session_id).distinct().scalar_subquery()

    # Find sessions NOT in that list
    query = select(Session).where(Session.id.not_in(sessions_with_messages)).order_by(Session.created_at.asc())

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
