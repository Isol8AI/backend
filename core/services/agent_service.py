"""
Agent service for managing encrypted agent state.

Handles CRUD operations for agent state stored in PostgreSQL.
The service only works with encrypted blobs - it cannot read
the actual agent data.
"""

import logging
from typing import List, Optional, Tuple

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from models.agent_state import AgentState

logger = logging.getLogger(__name__)


class AgentService:
    """
    Service for managing agent state.

    All operations work with encrypted tarballs. The service
    cannot read or modify the actual agent content - only the
    enclave can do that.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the service.

        Args:
            db: Async database session
        """
        self.db = db

    async def get_agent_state(
        self,
        user_id: str,
        agent_name: str,
    ) -> Optional[AgentState]:
        """
        Get agent state for a user.

        Args:
            user_id: Clerk user ID
            agent_name: Agent name/identifier

        Returns:
            AgentState if found, None otherwise
        """
        result = await self.db.execute(
            select(AgentState).where(
                AgentState.user_id == user_id,
                AgentState.agent_name == agent_name,
            )
        )
        return result.scalar_one_or_none()

    async def create_agent_state(
        self,
        user_id: str,
        agent_name: str,
        encrypted_tarball: bytes,
    ) -> AgentState:
        """
        Create new agent state.

        Args:
            user_id: Clerk user ID
            agent_name: Agent name/identifier
            encrypted_tarball: Encrypted tarball bytes

        Returns:
            Created AgentState
        """
        state = AgentState(
            user_id=user_id,
            agent_name=agent_name,
            encrypted_tarball=encrypted_tarball,
            tarball_size_bytes=len(encrypted_tarball),
        )
        self.db.add(state)
        await self.db.flush()
        logger.info(f"Created agent state for user={user_id}, agent={agent_name}")
        return state

    async def update_agent_state(
        self,
        user_id: str,
        agent_name: str,
        encrypted_tarball: bytes,
    ) -> Optional[AgentState]:
        """
        Update existing agent state.

        Args:
            user_id: Clerk user ID
            agent_name: Agent name/identifier
            encrypted_tarball: New encrypted tarball bytes

        Returns:
            Updated AgentState if found, None otherwise
        """
        state = await self.get_agent_state(user_id, agent_name)
        if state is None:
            return None

        state.encrypted_tarball = encrypted_tarball
        state.tarball_size_bytes = len(encrypted_tarball)
        await self.db.flush()
        logger.info(f"Updated agent state for user={user_id}, agent={agent_name}")
        return state

    async def get_or_create_agent_state(
        self,
        user_id: str,
        agent_name: str,
        default_tarball: bytes,
    ) -> Tuple[AgentState, bool]:
        """
        Get existing agent state or create new one.

        Args:
            user_id: Clerk user ID
            agent_name: Agent name/identifier
            default_tarball: Tarball to use if creating new

        Returns:
            Tuple of (AgentState, was_created)
        """
        state = await self.get_agent_state(user_id, agent_name)
        if state is not None:
            return state, False

        state = await self.create_agent_state(user_id, agent_name, default_tarball)
        return state, True

    async def list_user_agents(
        self,
        user_id: str,
    ) -> List[AgentState]:
        """
        List all agents for a user.

        Args:
            user_id: Clerk user ID

        Returns:
            List of AgentState objects
        """
        result = await self.db.execute(
            select(AgentState).where(AgentState.user_id == user_id).order_by(AgentState.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete_agent_state(
        self,
        user_id: str,
        agent_name: str,
    ) -> bool:
        """
        Delete agent state.

        Args:
            user_id: Clerk user ID
            agent_name: Agent name/identifier

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(AgentState).where(
                AgentState.user_id == user_id,
                AgentState.agent_name == agent_name,
            )
        )
        deleted = result.rowcount > 0
        if deleted:
            await self.db.flush()
            logger.info(f"Deleted agent state for user={user_id}, agent={agent_name}")
        return deleted
