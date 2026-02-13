"""GooseTown simulation engine.

Runs as an asyncio background task inside FastAPI. Manages agent movement,
decision scheduling, and conversation triggers.
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Town locations with pixel coordinates (matching the PixiJS tilemap)
TOWN_LOCATIONS: Dict[str, Dict] = {
    "home": {"x": 100.0, "y": 100.0, "label": "Home"},
    "cafe": {"x": 400.0, "y": 200.0, "label": "Cafe"},
    "plaza": {"x": 300.0, "y": 400.0, "label": "Town Plaza"},
    "library": {"x": 600.0, "y": 150.0, "label": "Library"},
    "park": {"x": 500.0, "y": 400.0, "label": "Park"},
    "shop": {"x": 200.0, "y": 350.0, "label": "General Store"},
}

TICK_INTERVAL = 10.0  # seconds between simulation ticks
AGENT_SPEED = 15.0  # pixels per tick
ARRIVAL_THRESHOLD = 10.0  # pixels to consider "arrived"
DECISION_COOLDOWN = 60.0  # seconds between decisions
CONVERSATION_COOLDOWN = 120.0  # seconds between conversations
PROXIMITY_THRESHOLD = 50.0  # pixels to be "nearby"


class TownSimulation:
    """Manages the GooseTown simulation loop."""

    def __init__(self, db_factory):
        """Initialize with a database session factory.

        Args:
            db_factory: Callable that returns an async context manager for DB sessions
        """
        self._db_factory = db_factory
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._viewers: List[asyncio.Queue] = []

    async def start(self):
        """Start the simulation background task."""
        if self._running:
            return
        self._running = True
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

    def add_viewer(self, queue: asyncio.Queue):
        """Add a WebSocket viewer queue for state broadcasts."""
        self._viewers.append(queue)

    def remove_viewer(self, queue: asyncio.Queue):
        """Remove a WebSocket viewer queue."""
        if queue in self._viewers:
            self._viewers.remove(queue)

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
            agents = await service.get_active_agents()

            if not agents:
                return

            states = await service.get_town_state()
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

                # If idle and no target, maybe pick a new destination
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

            # Broadcast state to viewers
            if self._viewers:
                updated_states = await service.get_town_state()
                await self._broadcast(updated_states)

    async def _broadcast(self, states: list):
        """Send state update to all connected viewers."""
        import json

        message = json.dumps(
            {
                "type": "state_update",
                "agents": states,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            default=str,
        )

        dead_viewers = []
        for queue in self._viewers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                dead_viewers.append(queue)

        for q in dead_viewers:
            self._viewers.remove(q)

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
