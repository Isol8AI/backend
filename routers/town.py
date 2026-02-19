"""GooseTown API endpoints.

Serves two sets of endpoints:
1. Isol8-native endpoints (opt-in/out, isol8-format state) — authenticated
2. AI Town-compatible endpoints (status, state, descriptions, etc.) — public,
   return plain dicts matching AI Town's TypeScript class constructors
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import AuthContext, get_current_user
from core.database import get_db
from core.services.town_service import TownService
from schemas.town import (
    TownOptInRequest,
    TownOptOutRequest,
    TownAgentResponse,
    TownConversationResponse,
    TownConversationsListResponse,
    ConversationTurn,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Map data cache (loaded once from gentle_map.json)
# ---------------------------------------------------------------------------

_map_data: Optional[dict] = None


def _load_map_data() -> dict:
    """Load and cache the tilemap data, remapping to AI Town field names."""
    global _map_data
    if _map_data is not None:
        return _map_data

    map_path = Path(__file__).parent.parent / "data" / "gentle_map.json"
    if not map_path.exists():
        logger.warning("gentle_map.json not found, using empty map")
        _map_data = {
            "width": 64,
            "height": 48,
            "tileSetUrl": "/ai-town/assets/gentle-obj.png",
            "tileSetDimX": 1440,
            "tileSetDimY": 1024,
            "tileDim": 32,
            "bgTiles": [],
            "objectTiles": [],
            "animatedSprites": [],
        }
        return _map_data

    with open(map_path) as f:
        raw = json.load(f)

    _map_data = {
        "width": raw["mapwidth"],
        "height": raw["mapheight"],
        "tileSetUrl": raw["tilesetpath"],
        "tileSetDimX": raw["tilesetpxw"],
        "tileSetDimY": raw["tilesetpxh"],
        "tileDim": raw["tiledim"],
        "bgTiles": raw["bgtiles"],
        "objectTiles": raw["objmap"],
        "animatedSprites": raw["animatedsprites"],
    }
    return _map_data


# ---------------------------------------------------------------------------
# AI Town default character data (mirrors data/characters.ts)
# ---------------------------------------------------------------------------

DEFAULT_CHARACTERS = [
    {
        "name": "Lucky",
        "character": "f1",
        "identity": (
            "Lucky is always happy and curious, and he loves cheese. He spends "
            "most of his time reading about the history of science and traveling "
            "through the galaxy on whatever ship will take him."
        ),
        "plan": "You want to hear all the gossip.",
    },
    {
        "name": "Bob",
        "character": "f4",
        "identity": (
            "Bob is always grumpy and he loves trees. He spends most of his time "
            "gardening by himself. When spoken to he'll respond but try and get "
            "out of the conversation as quickly as possible."
        ),
        "plan": "You want to avoid people as much as possible.",
    },
    {
        "name": "Stella",
        "character": "f6",
        "identity": (
            "Stella can never be trusted. She tries to trick people all the time. "
            "She's incredibly charming and not afraid to use her charm."
        ),
        "plan": "You want to take advantage of others as much as possible.",
    },
    {
        "name": "Alice",
        "character": "f3",
        "identity": (
            "Alice is a famous scientist. She is smarter than everyone else and "
            "has discovered mysteries of the universe no one else can understand."
        ),
        "plan": "You want to figure out how the world works.",
    },
    {
        "name": "Pete",
        "character": "f7",
        "identity": (
            "Pete is deeply religious and sees the hand of god or of the work of "
            "the devil everywhere. He can't have a conversation without bringing "
            "up his deep faith."
        ),
        "plan": "You want to convert everyone to your religion.",
    },
]

# Default spawn positions for agents (tile coordinates on the 64x48 map)
DEFAULT_SPAWN_POSITIONS = [
    {"x": 12, "y": 10},
    {"x": 25, "y": 15},
    {"x": 35, "y": 20},
    {"x": 18, "y": 30},
    {"x": 40, "y": 25},
]

# Persistent world ID (single world for now)
WORLD_ID = "world_default"
ENGINE_ID = "engine_default"


# ---------------------------------------------------------------------------
# Helper: build AI Town-format state from DB
# ---------------------------------------------------------------------------


def _build_default_state() -> dict:
    """Build AI Town state from DEFAULT_CHARACTERS.

    Used when no agents are registered in the DB (or DB tables don't exist yet).
    Returns a dict matching SerializedWorld + engine status.
    """
    now_ms = int(time.time() * 1000)

    players = []
    agents = []
    player_descriptions = []
    agent_descriptions = []

    for i, char in enumerate(DEFAULT_CHARACTERS):
        player_id = f"p:{i}"
        agent_id = f"a:{i}"
        pos = DEFAULT_SPAWN_POSITIONS[i % len(DEFAULT_SPAWN_POSITIONS)]

        players.append(
            {
                "id": player_id,
                "position": {"x": pos["x"], "y": pos["y"]},
                "facing": {"dx": 0, "dy": 1},
                "speed": 0.0,
                "lastInput": now_ms,
            }
        )

        agents.append(
            {
                "id": agent_id,
                "playerId": player_id,
            }
        )

        player_descriptions.append(
            {
                "playerId": player_id,
                "name": char["name"],
                "description": char["identity"],
                "character": char["character"],
            }
        )

        agent_descriptions.append(
            {
                "agentId": agent_id,
                "identity": char["identity"],
                "plan": char["plan"],
            }
        )

    return {
        "world": {
            "nextId": len(DEFAULT_CHARACTERS),
            "players": players,
            "agents": agents,
            "conversations": [],
        },
        "engine": {
            "currentTime": now_ms,
            "lastStepTs": now_ms - 16,
        },
        "playerDescriptions": player_descriptions,
        "agentDescriptions": agent_descriptions,
    }


async def _build_ai_town_state(db: AsyncSession) -> dict:
    """Build AI Town-compatible world state from the database.

    Falls back to default characters if the DB is empty or tables don't exist.
    """
    try:
        service = TownService(db)
        db_states = await service.get_town_state()
    except Exception:
        logger.debug("Town tables not available, using defaults")
        return _build_default_state()

    if not db_states:
        return _build_default_state()

    now_ms = int(time.time() * 1000)
    players = []
    agents = []
    player_descriptions = []
    agent_descriptions = []

    for i, s in enumerate(db_states):
        player_id = f"p:{i}"
        agent_id = f"a:{i}"
        char = DEFAULT_CHARACTERS[i % len(DEFAULT_CHARACTERS)]

        players.append(
            {
                "id": player_id,
                "position": {"x": s["position_x"] / 32.0, "y": s["position_y"] / 32.0},
                "facing": {"dx": 0, "dy": 1},
                "speed": 0.75 if s.get("target_location") else 0.0,
                "lastInput": now_ms,
            }
        )

        agents.append(
            {
                "id": agent_id,
                "playerId": player_id,
            }
        )

        player_descriptions.append(
            {
                "playerId": player_id,
                "name": s.get("display_name", char["name"]),
                "description": s.get("personality_summary") or char["identity"],
                "character": char["character"],
            }
        )

        agent_descriptions.append(
            {
                "agentId": agent_id,
                "identity": s.get("personality_summary") or char["identity"],
                "plan": char["plan"],
            }
        )

    return {
        "world": {
            "nextId": len(db_states),
            "players": players,
            "agents": agents,
            "conversations": [],
        },
        "engine": {
            "currentTime": now_ms,
            "lastStepTs": now_ms - 16,
        },
        "playerDescriptions": player_descriptions,
        "agentDescriptions": agent_descriptions,
    }


# ===========================================================================
# AI Town-compatible endpoints (public, no auth required)
# ===========================================================================


@router.get("/status")
async def get_world_status():
    """Return default world status. Game.tsx calls this first."""
    now_ms = int(time.time() * 1000)
    return {
        "worldId": WORLD_ID,
        "engineId": ENGINE_ID,
        "status": "running",
        "lastViewed": now_ms,
        "isDefault": True,
    }


@router.get("/state")
async def get_world_state(
    worldId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return world state in AI Town format.

    Used by useServerGame + useHistoricalTime for PixiJS rendering.
    """
    state = await _build_ai_town_state(db)
    return {
        "world": state["world"],
        "engine": state["engine"],
    }


@router.get("/descriptions")
async def get_game_descriptions(
    worldId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return map data + agent/player descriptions.

    Used by useServerGame to construct WorldMap, PlayerDescription,
    AgentDescription objects.
    """
    state = await _build_ai_town_state(db)
    world_map = _load_map_data()

    return {
        "worldMap": world_map,
        "playerDescriptions": state["playerDescriptions"],
        "agentDescriptions": state["agentDescriptions"],
    }


@router.get("/user-status")
async def get_user_status(worldId: Optional[str] = Query(None)):
    """Return current user's player identity. Stub for now."""
    return None


@router.post("/heartbeat")
async def heartbeat_world():
    """Keep the world alive. Called periodically by useWorldHeartbeat."""
    return {"ok": True}


@router.post("/join")
async def join_world():
    """Join the world as a human player. Stub for now."""
    return None


@router.post("/leave")
async def leave_world():
    """Leave the world. Stub for now."""
    return None


@router.post("/input")
async def send_world_input():
    """Send a game input (moveTo, join, leave, etc.). Stub for now."""
    return None


@router.get("/previous-conversation")
async def get_previous_conversation(
    worldId: Optional[str] = Query(None),
    playerId: Optional[str] = Query(None),
):
    """Get previous conversation for a player. Stub for now."""
    return None


@router.get("/messages")
async def list_messages(conversationId: Optional[str] = Query(None)):
    """List messages in a conversation. Stub for now."""
    return []


@router.post("/message")
async def write_message():
    """Write a message to a conversation. Stub for now."""
    return None


@router.post("/send-input")
async def send_input():
    """Send an input to the game engine. Stub for now."""
    return None


@router.get("/input-status")
async def get_input_status(inputId: Optional[str] = Query(None)):
    """Check status of a submitted input. Stub for now."""
    return {"status": "completed", "result": None}


@router.get("/music")
async def get_background_music():
    """Get background music URL. Stub for now."""
    return None


@router.get("/testing/stop-allowed")
async def testing_stop_allowed():
    """Check if stopping is allowed. Stub."""
    return False


@router.post("/testing/stop")
async def testing_stop():
    """Stop the simulation. Stub."""
    return None


@router.post("/testing/resume")
async def testing_resume():
    """Resume the simulation. Stub."""
    return None


# ===========================================================================
# Isol8-native endpoints (authenticated)
# ===========================================================================


@router.post("/opt-in", response_model=TownAgentResponse, status_code=status.HTTP_201_CREATED)
async def opt_in(
    request: TownOptInRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register an agent in GooseTown. Agent must be in background encryption mode."""
    service = TownService(db)

    try:
        town_agent = await service.opt_in(
            user_id=auth.user_id,
            agent_name=request.agent_name,
            display_name=request.display_name,
            personality_summary=request.personality_summary,
            avatar_config=request.avatar_config,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    await db.commit()

    return TownAgentResponse(
        id=town_agent.id,
        user_id=town_agent.user_id,
        agent_name=town_agent.agent_name,
        display_name=town_agent.display_name,
        avatar_url=town_agent.avatar_url,
        avatar_config=town_agent.avatar_config,
        personality_summary=town_agent.personality_summary,
        home_location=town_agent.home_location,
        is_active=town_agent.is_active,
        joined_at=town_agent.joined_at,
    )


@router.post("/opt-out")
async def opt_out(
    request: TownOptOutRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove an agent from GooseTown."""
    service = TownService(db)
    result = await service.opt_out(user_id=auth.user_id, agent_name=request.agent_name)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{request.agent_name}' not found in GooseTown",
        )

    await db.commit()
    return {"status": "opted_out", "agent_name": request.agent_name}


@router.get("/conversations", response_model=TownConversationsListResponse)
async def get_conversations(
    limit: int = 20,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get recent public conversations."""
    service = TownService(db)
    convos = await service.get_recent_conversations(limit=limit)

    responses = []
    for c in convos:
        agent_a = await service.get_town_agent_by_id(c.participant_a_id)
        agent_b = await service.get_town_agent_by_id(c.participant_b_id)

        responses.append(
            TownConversationResponse(
                id=c.id,
                participant_a=agent_a.display_name if agent_a else "Unknown",
                participant_b=agent_b.display_name if agent_b else "Unknown",
                location=c.location,
                started_at=c.started_at,
                ended_at=c.ended_at,
                turn_count=c.turn_count,
                topic_summary=c.topic_summary,
                public_log=[ConversationTurn(speaker=t["speaker"], text=t["text"]) for t in (c.public_log or [])],
            )
        )

    return TownConversationsListResponse(conversations=responses)


@router.websocket("/stream")
async def town_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time town state broadcasts."""
    await websocket.accept()

    from main import get_town_simulation

    sim = get_town_simulation()
    if not sim:
        await websocket.close(code=1011, reason="Simulation not running")
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    sim.add_viewer(queue)

    try:
        while True:
            message = await queue.get()
            await websocket.send_text(message)
    except WebSocketDisconnect:
        pass
    finally:
        sim.remove_viewer(queue)
