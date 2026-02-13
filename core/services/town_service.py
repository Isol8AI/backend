"""Town service for managing GooseTown agent registration and state."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.agent_state import AgentState
from models.town import TownAgent, TownState, TownConversation, TownRelationship

logger = logging.getLogger(__name__)


class TownService:
    """Service for GooseTown CRUD operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def opt_in(
        self,
        user_id: str,
        agent_name: str,
        display_name: str,
        personality_summary: Optional[str] = None,
        avatar_config: Optional[dict] = None,
    ) -> TownAgent:
        """Register an agent in GooseTown. Requires background encryption mode."""
        result = await self.db.execute(
            select(AgentState).where(
                AgentState.user_id == user_id,
                AgentState.agent_name == agent_name,
            )
        )
        agent_state = result.scalar_one_or_none()

        if not agent_state:
            raise ValueError(f"Agent '{agent_name}' not found for user {user_id}")

        if agent_state.encryption_mode != "background":
            raise ValueError(
                f"Agent '{agent_name}' must be in background encryption mode to join GooseTown. "
                f"Current mode: {agent_state.encryption_mode}"
            )

        existing = await self._get_town_agent(user_id, agent_name)
        if existing:
            if not existing.is_active:
                existing.is_active = True
                existing.display_name = display_name
                existing.personality_summary = personality_summary
                existing.avatar_config = avatar_config
                existing.last_active_at = datetime.now(timezone.utc)
                await self.db.flush()
                return existing
            return existing

        town_agent = TownAgent(
            user_id=user_id,
            agent_name=agent_name,
            display_name=display_name,
            personality_summary=personality_summary,
            avatar_config=avatar_config,
        )
        self.db.add(town_agent)
        await self.db.flush()

        state = TownState(
            agent_id=town_agent.id,
            position_x=0.0,
            position_y=0.0,
        )
        self.db.add(state)
        await self.db.flush()

        logger.info(f"Agent '{agent_name}' opted into GooseTown as '{display_name}'")
        return town_agent

    async def opt_out(self, user_id: str, agent_name: str) -> bool:
        """Remove an agent from GooseTown (deactivate, preserve data)."""
        agent = await self._get_town_agent(user_id, agent_name)
        if not agent:
            return False

        agent.is_active = False
        await self.db.flush()
        logger.info(f"Agent '{agent_name}' opted out of GooseTown")
        return True

    async def get_active_agents(self) -> List[TownAgent]:
        """Get all active town agents."""
        result = await self.db.execute(
            select(TownAgent).where(TownAgent.is_active.is_(True)).order_by(TownAgent.joined_at)
        )
        return list(result.scalars().all())

    async def get_town_state(self) -> List[dict]:
        """Get current state of all active agents."""
        result = await self.db.execute(
            select(TownAgent, TownState)
            .join(TownState, TownState.agent_id == TownAgent.id)
            .where(TownAgent.is_active.is_(True))
        )
        rows = result.all()

        return [
            {
                "agent_id": agent.id,
                "display_name": agent.display_name,
                "agent_name": agent.agent_name,
                "personality_summary": agent.personality_summary,
                "current_location": state.current_location,
                "current_activity": state.current_activity,
                "target_location": state.target_location,
                "position_x": state.position_x,
                "position_y": state.position_y,
                "mood": state.mood,
                "energy": state.energy,
                "status_message": state.status_message,
            }
            for agent, state in rows
        ]

    async def get_or_create_relationship(self, agent_a_id: UUID, agent_b_id: UUID) -> Tuple[TownRelationship, bool]:
        """Get or create a relationship between two agents."""
        a_id, b_id = sorted([agent_a_id, agent_b_id], key=str)

        result = await self.db.execute(
            select(TownRelationship).where(
                TownRelationship.agent_a_id == a_id,
                TownRelationship.agent_b_id == b_id,
            )
        )
        rel = result.scalar_one_or_none()

        if rel:
            return rel, False

        rel = TownRelationship(agent_a_id=a_id, agent_b_id=b_id)
        self.db.add(rel)
        await self.db.flush()
        return rel, True

    async def update_relationship(
        self,
        relationship_id: UUID,
        affinity_delta: int = 0,
        new_type: Optional[str] = None,
    ) -> TownRelationship:
        """Update relationship after an interaction."""
        result = await self.db.execute(select(TownRelationship).where(TownRelationship.id == relationship_id))
        rel = result.scalar_one()

        rel.affinity_score = max(-100, min(100, rel.affinity_score + affinity_delta))
        rel.interaction_count += 1
        rel.last_interaction_at = datetime.now(timezone.utc)
        if new_type:
            rel.relationship_type = new_type

        await self.db.flush()
        return rel

    async def store_conversation(
        self,
        participant_a_id: UUID,
        participant_b_id: UUID,
        location: str,
        public_log: list,
        topic_summary: Optional[str] = None,
    ) -> TownConversation:
        """Store a completed conversation."""
        convo = TownConversation(
            participant_a_id=participant_a_id,
            participant_b_id=participant_b_id,
            location=location,
            turn_count=len(public_log),
            topic_summary=topic_summary,
            public_log=public_log,
            ended_at=datetime.now(timezone.utc),
        )
        self.db.add(convo)
        await self.db.flush()
        return convo

    async def get_recent_conversations(self, limit: int = 20) -> List[TownConversation]:
        """Get recent town conversations."""
        result = await self.db.execute(
            select(TownConversation).order_by(TownConversation.started_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def _get_town_agent(self, user_id: str, agent_name: str) -> Optional[TownAgent]:
        """Get a town agent by user_id and agent_name."""
        result = await self.db.execute(
            select(TownAgent).where(
                TownAgent.user_id == user_id,
                TownAgent.agent_name == agent_name,
            )
        )
        return result.scalar_one_or_none()

    async def get_town_agent_by_id(self, agent_id: UUID) -> Optional[TownAgent]:
        """Get a town agent by its UUID."""
        result = await self.db.execute(select(TownAgent).where(TownAgent.id == agent_id))
        return result.scalar_one_or_none()

    async def update_agent_state(
        self,
        agent_id: UUID,
        **kwargs,
    ) -> Optional[TownState]:
        """Update an agent's town state."""
        result = await self.db.execute(select(TownState).where(TownState.agent_id == agent_id))
        state = result.scalar_one_or_none()
        if not state:
            return None

        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        await self.db.flush()
        return state
