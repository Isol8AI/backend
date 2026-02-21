"""GooseTown simulation engine.

Runs as an asyncio background task inside FastAPI. Manages agent movement,
decision scheduling, and conversation triggers.
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from core.town_constants import (
    DEFAULT_CHARACTERS,
    SYSTEM_USER_ID,
    TOWN_LOCATIONS,
)

logger = logging.getLogger(__name__)

TICK_INTERVAL = 2.0  # seconds between simulation ticks
AGENT_SPEED = 0.6  # tiles per tick (~0.3 tiles/sec, natural walking pace)
ARRIVAL_THRESHOLD = 0.5  # tiles to consider "arrived"
DECISION_COOLDOWN = 10.0  # seconds idle before picking new destination
CONVERSATION_COOLDOWN = 120.0  # seconds between conversations
PROXIMITY_THRESHOLD = 3.0  # tiles to be "nearby"


class TownSimulation:
    """Manages the GooseTown simulation loop."""

    def __init__(self, db_factory, notify_fn: Optional[Callable] = None):
        """Initialize with a database session factory.

        Args:
            db_factory: Callable that returns an async context manager for DB sessions
            notify_fn: Optional callback to push state to WebSocket viewers
        """
        self._db_factory = db_factory
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._notify_fn = notify_fn

    async def start(self):
        """Start the simulation background task."""
        if self._running:
            return
        self._running = True
        await self._seed_default_agents()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("GooseTown simulation started")

    async def stop(self):
        """Stop the simulation."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GooseTown simulation stopped")

    def set_notify_fn(self, fn: Callable):
        """Set the WebSocket push callback (called after state changes)."""
        self._notify_fn = fn

    async def _seed_default_agents(self):
        """Seed default agents into the DB if they don't already exist."""
        from core.services.town_service import TownService

        try:
            async with self._db_factory() as db:
                service = TownService(db)
                for agent in DEFAULT_CHARACTERS:
                    await service.seed_agent(
                        user_id=SYSTEM_USER_ID,
                        agent_name=agent["agent_name"],
                        display_name=agent["name"],
                        personality_summary=agent["identity"][:200],
                        position_x=agent["spawn"]["x"],
                        position_y=agent["spawn"]["y"],
                        home_location=agent["home"],
                    )
                await db.commit()
            logger.info(f"Seeded {len(DEFAULT_CHARACTERS)} default agents")
        except Exception as e:
            logger.error(f"Failed to seed default agents: {e}", exc_info=True)

    async def _run_loop(self):
        """Main simulation tick loop."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"GooseTown tick error: {e}", exc_info=True)

            await asyncio.sleep(TICK_INTERVAL)

    async def _tick(self):
        """Execute one simulation tick."""
        from core.services.town_service import TownService

        async with self._db_factory() as db:
            service = TownService(db)
            states = await service.get_town_state()

            if not states:
                return

            now = datetime.now(timezone.utc)

            for agent_state in states:
                agent_id = agent_state["agent_id"]

                # Move agents toward their targets
                if agent_state["target_location"]:
                    target = TOWN_LOCATIONS.get(agent_state["target_location"])
                    if target:
                        new_x, new_y, arrived = self._move_toward(
                            agent_state["position_x"],
                            agent_state["position_y"],
                            target["x"],
                            target["y"],
                            AGENT_SPEED,
                        )

                        update = {
                            "position_x": new_x,
                            "position_y": new_y,
                        }

                        if arrived:
                            update["current_location"] = agent_state["target_location"]
                            update["target_location"] = None
                            update["current_activity"] = "idle"

                        await service.update_agent_state(agent_id, **update)

                # If idle and no target, pick a new destination after cooldown
                elif agent_state["current_activity"] == "idle":
                    last_decision = agent_state.get("last_decision_at")
                    if not last_decision or (now - last_decision).total_seconds() > DECISION_COOLDOWN:
                        target_loc = self._pick_random_location(exclude=agent_state["current_location"])
                        await service.update_agent_state(
                            agent_id,
                            target_location=target_loc,
                            current_activity="walking",
                            last_decision_at=now,
                        )

            await db.commit()

        # Push updated state to WebSocket viewers
        if self._notify_fn:
            try:
                self._notify_fn()
            except Exception as e:
                logger.debug(f"WS notify failed: {e}")

    def _pick_random_location(self, exclude: Optional[str] = None) -> str:
        """Pick a random town location, excluding current."""
        choices = [loc for loc in TOWN_LOCATIONS if loc != exclude]
        return random.choice(choices)

    @staticmethod
    def _calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def _move_toward(
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
        speed: float,
    ) -> Tuple[float, float, bool]:
        """Move toward target at given speed. Returns (new_x, new_y, arrived)."""
        dx = target_x - current_x
        dy = target_y - current_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= speed:
            return target_x, target_y, True

        ratio = speed / dist
        return current_x + dx * ratio, current_y + dy * ratio, False

    @staticmethod
    def _conversation_probability(affinity: int) -> float:
        """Calculate conversation probability based on relationship affinity.

        Strangers (0): 15%
        Acquaintances (25): ~30%
        Friends (50): ~45%
        Close friends (75): ~60%
        Best friends (100): ~70%
        """
        base = 0.15
        bonus = (affinity / 100.0) * 0.55
        return min(0.70, base + bonus)
