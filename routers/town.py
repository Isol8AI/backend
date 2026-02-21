"""GooseTown API endpoints.

Serves two sets of endpoints:
1. Isol8-native endpoints (opt-in/out, isol8-format state) — authenticated
2. AI Town-compatible endpoints (status, state, descriptions, etc.) — public,
   return plain dicts matching AI Town's TypeScript class constructors
"""

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import AuthContext, get_current_user, get_optional_user
from core.database import get_db
from core.services.management_api_client import ManagementApiClient, ManagementApiClientError
from core.services.town_service import TownService
from core.town_constants import (
    AGENT_CHARACTERS,
    AVAILABLE_CHARACTERS,
    DEFAULT_CHARACTERS,
    DEFAULT_SPAWN_POSITIONS,
    TOWN_LOCATIONS,
    WALK_SPEED_DISPLAY,
)
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


# Persistent world ID (single world for now)
WORLD_ID = "world_default"
ENGINE_ID = "engine_default"

MAX_HUMAN_PLAYERS = 8

# ---------------------------------------------------------------------------
# In-memory game state for human players & inputs
# ---------------------------------------------------------------------------

# user_id -> {player_id, position, facing, character, name}
_human_players: dict[str, dict] = {}
_next_input_id: int = 0
_completed_inputs: dict[str, dict] = {}  # input_id -> {kind, value} or {kind, message}

# ---------------------------------------------------------------------------
# Real-time state push via API Gateway WebSocket
# ---------------------------------------------------------------------------
# Town viewers subscribe via the shared API Gateway WebSocket (same one used
# for chat).  When game state changes (join/leave/move), we push the combined
# state to every viewer in a single message — giving the frontend the same
# atomic update guarantee that Convex subscriptions provide.
# ---------------------------------------------------------------------------

_town_viewer_connections: set[str] = set()

# Lazy singleton — created on first push.  Returns None in local dev where
# WS_MANAGEMENT_API_URL is not set (no-op push).
_town_mgmt_api: Optional[ManagementApiClient] = None
_town_mgmt_api_failed: bool = False


def _get_town_mgmt_api() -> Optional[ManagementApiClient]:
    global _town_mgmt_api, _town_mgmt_api_failed
    if _town_mgmt_api is not None:
        return _town_mgmt_api
    if _town_mgmt_api_failed:
        return None
    try:
        _town_mgmt_api = ManagementApiClient()
        return _town_mgmt_api
    except ManagementApiClientError:
        # No WS_MANAGEMENT_API_URL — local dev, push is a no-op
        _town_mgmt_api_failed = True
        logger.info("ManagementApiClient unavailable — town WS push disabled")
        return None


async def _build_ws_message_async() -> dict:
    """Build the combined state payload for WebSocket clients (async, reads DB)."""
    from core.database import async_session_factory

    async with async_session_factory() as db:
        state = await _build_ai_town_state(db)

    world_map = _load_map_data()
    return {
        "type": "town_state",
        "worldState": {"world": state["world"], "engine": state["engine"]},
        "gameDescriptions": {
            "worldMap": world_map,
            "playerDescriptions": state["playerDescriptions"],
            "agentDescriptions": state["agentDescriptions"],
        },
    }


def _build_ws_message() -> dict:
    """Build the combined state payload for WebSocket clients (sync fallback).

    Uses default state. Prefer _build_ws_message_async() for DB-backed state.
    """
    state = _build_default_state()
    world_map = _load_map_data()
    return {
        "type": "town_state",
        "worldState": {"world": state["world"], "engine": state["engine"]},
        "gameDescriptions": {
            "worldMap": world_map,
            "playerDescriptions": state["playerDescriptions"],
            "agentDescriptions": state["agentDescriptions"],
        },
    }


def _notify_state_changed():
    """Push updated state to all connected town viewers via API Gateway.

    Called after any mutation that changes game state (join, leave, moveTo).
    Also called by TownSimulation after each tick.
    Dead connections (GoneException) are silently removed.
    """
    if not _town_viewer_connections:
        return
    mgmt = _get_town_mgmt_api()
    if mgmt is None:
        return

    # Try async DB-backed state; fall back to defaults on failure
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # Schedule async message build as a fire-and-forget task
        loop.create_task(_async_notify_viewers())
        return
    except RuntimeError:
        pass

    # No event loop — sync fallback
    message = _build_ws_message()
    _push_to_viewers(message)


async def _async_notify_viewers():
    """Async helper: build state from DB and push to all viewers."""
    mgmt = _get_town_mgmt_api()
    if mgmt is None:
        return
    try:
        message = await _build_ws_message_async()
    except Exception:
        message = _build_ws_message()
    _push_to_viewers(message)


def _push_to_viewers(message: dict):
    """Send a message dict to all connected town viewers."""
    mgmt = _get_town_mgmt_api()
    if mgmt is None:
        return
    dead: list[str] = []
    for conn_id in list(_town_viewer_connections):
        try:
            if not mgmt.send_message(conn_id, message):
                dead.append(conn_id)
        except ManagementApiClientError:
            dead.append(conn_id)
    for conn_id in dead:
        _town_viewer_connections.discard(conn_id)


def add_town_viewer(connection_id: str):
    """Register a WebSocket connection as a town viewer and push initial state."""
    import asyncio

    _town_viewer_connections.add(connection_id)
    mgmt = _get_town_mgmt_api()
    if mgmt is None:
        return

    async def _send_initial():
        try:
            msg = await _build_ws_message_async()
        except Exception:
            msg = _build_ws_message()
        try:
            mgmt.send_message(connection_id, msg)
        except ManagementApiClientError:
            _town_viewer_connections.discard(connection_id)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_initial())
    except RuntimeError:
        # No event loop — sync fallback
        try:
            mgmt.send_message(connection_id, _build_ws_message())
        except ManagementApiClientError:
            _town_viewer_connections.discard(connection_id)


def remove_town_viewer(connection_id: str):
    """Unregister a WebSocket connection from town updates."""
    _town_viewer_connections.discard(connection_id)


def _allocate_input(result: dict) -> str:
    """Create a completed input and return its ID."""
    global _next_input_id
    input_id = f"o:{_next_input_id}"
    _next_input_id += 1
    _completed_inputs[input_id] = result
    return input_id


def _next_player_id() -> str:
    """Allocate a player ID for a human player."""
    base = len(DEFAULT_CHARACTERS)
    existing_ids = {hp["player_id"] for hp in _human_players.values()}
    for n in range(base, base + MAX_HUMAN_PLAYERS + 1):
        pid = f"p:{n}"
        if pid not in existing_ids:
            return pid
    return f"p:{base + len(_human_players)}"


def _pick_spawn_position() -> dict:
    """Pick a random spawn position that isn't occupied."""
    occupied = {(hp["position"]["x"], hp["position"]["y"]) for hp in _human_players.values()}
    for _ in range(20):
        x = random.randint(5, 25)
        y = random.randint(3, 15)
        if (x, y) not in occupied:
            return {"x": x, "y": y}
    return {"x": 12, "y": 8}


def _pick_character() -> str:
    """Pick a random character sprite not already used by an AI agent."""
    human_chars = {hp["character"] for hp in _human_players.values()}
    available = [c for c in AVAILABLE_CHARACTERS if c not in AGENT_CHARACTERS and c not in human_chars]
    if not available:
        available = [c for c in AVAILABLE_CHARACTERS if c not in human_chars]
    if not available:
        available = AVAILABLE_CHARACTERS
    return random.choice(available)


# ---------------------------------------------------------------------------
# Helper: build AI Town-format state from DB
# ---------------------------------------------------------------------------


def _build_default_state() -> dict:
    """Build AI Town state from DEFAULT_CHARACTERS + human players.

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
        spawn = char.get("spawn", DEFAULT_SPAWN_POSITIONS[i % len(DEFAULT_SPAWN_POSITIONS)])

        players.append(
            {
                "id": player_id,
                "position": {"x": spawn["x"], "y": spawn["y"]},
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

    next_id = len(DEFAULT_CHARACTERS)

    # Include human players
    for user_id, hp in _human_players.items():
        players.append(
            {
                "id": hp["player_id"],
                "human": user_id,
                "position": hp["position"],
                "facing": hp["facing"],
                "speed": 0.0,
                "lastInput": now_ms,
            }
        )
        player_descriptions.append(
            {
                "playerId": hp["player_id"],
                "name": hp.get("name", "Human"),
                "description": f"{hp.get('name', 'Human')} is a human player",
                "character": hp["character"],
            }
        )
        next_id += 1

    return {
        "world": {
            "nextId": next_id,
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

    _char_by_name = {c["agent_name"]: c for c in DEFAULT_CHARACTERS}

    for i, s in enumerate(db_states):
        player_id = f"p:{i}"
        agent_id = f"a:{i}"
        char = _char_by_name.get(s.get("agent_name"), DEFAULT_CHARACTERS[i % len(DEFAULT_CHARACTERS)])

        # Compute facing direction from current position toward target
        target_loc = s.get("target_location")
        facing = {"dx": 0, "dy": 1}  # default: face down
        speed = 0.0
        if target_loc:
            target = TOWN_LOCATIONS.get(target_loc)
            if target:
                dx = target["x"] - s["position_x"]
                dy = target["y"] - s["position_y"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.01:
                    facing = {"dx": dx / dist, "dy": dy / dist}
                speed = WALK_SPEED_DISPLAY

        players.append(
            {
                "id": player_id,
                "position": {"x": s["position_x"], "y": s["position_y"]},
                "facing": facing,
                "speed": speed,
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
                "plan": char.get("plan", ""),
            }
        )

    # Append human players
    next_id = len(db_states)
    for user_id, hp in _human_players.items():
        players.append(
            {
                "id": hp["player_id"],
                "human": user_id,
                "position": hp["position"],
                "facing": hp["facing"],
                "speed": 0.0,
                "lastInput": now_ms,
            }
        )
        player_descriptions.append(
            {
                "playerId": hp["player_id"],
                "name": hp.get("name", "Human"),
                "description": f"{hp.get('name', 'Human')} is a human player",
                "character": hp["character"],
            }
        )
        next_id += 1

    return {
        "world": {
            "nextId": next_id,
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
async def get_user_status(
    worldId: Optional[str] = Query(None),
    auth: AuthContext | None = Depends(get_optional_user),
):
    """Return the current user's token identifier.

    Returns the Clerk user_id if authenticated (used by the frontend to match
    the human player via `player.human === humanTokenIdentifier`).
    Returns null if not signed in.
    """
    if auth is None:
        return None
    return auth.user_id


@router.post("/heartbeat")
async def heartbeat_world():
    """Keep the world alive. Called periodically by useWorldHeartbeat."""
    return {"ok": True}


@router.post("/join")
async def join_world(
    auth: AuthContext = Depends(get_current_user),
):
    """Join the world as a human player.

    Creates a player entity with a random sprite and spawn position.
    Returns an input ID that the frontend polls via /input-status.
    """
    user_id = auth.user_id

    if user_id in _human_players:
        return _allocate_input({"kind": "error", "message": "You are already in this game!"})

    if len(_human_players) >= MAX_HUMAN_PLAYERS:
        return _allocate_input({"kind": "error", "message": f"Only {MAX_HUMAN_PLAYERS} human players allowed at once."})

    player_id = _next_player_id()
    character = _pick_character()
    position = _pick_spawn_position()

    _human_players[user_id] = {
        "player_id": player_id,
        "position": position,
        "facing": {"dx": 0, "dy": 1},
        "character": character,
        "name": "Me",
    }

    input_id = _allocate_input({"kind": "ok", "value": player_id})
    _notify_state_changed()
    return input_id


@router.post("/leave")
async def leave_world(
    auth: AuthContext = Depends(get_current_user),
):
    """Leave the world. Removes the human player."""
    user_id = auth.user_id

    if user_id not in _human_players:
        return _allocate_input({"kind": "ok", "value": None})

    del _human_players[user_id]
    input_id = _allocate_input({"kind": "ok", "value": None})
    _notify_state_changed()
    return input_id


class SendWorldInputRequest(BaseModel):
    engineId: Optional[str] = None
    name: str
    args: dict


@router.post("/input")
async def send_world_input(
    request: SendWorldInputRequest = Body(...),
    auth: AuthContext = Depends(get_current_user),
):
    """Handle game inputs (moveTo, join, leave, etc.).

    Returns an input ID that the frontend polls via /input-status.
    """
    user_id = auth.user_id
    name = request.name
    args = request.args

    if name == "moveTo":
        player_id = args.get("playerId")
        destination = args.get("destination")

        if not player_id or not destination:
            return _allocate_input({"kind": "error", "message": "Missing playerId or destination"})

        # Find the human player that owns this player_id
        hp = _human_players.get(user_id)
        if not hp or hp["player_id"] != player_id:
            return _allocate_input({"kind": "error", "message": "Player not found"})

        # Update position immediately (no pathfinding yet — Task 5)
        hp["position"] = {"x": destination["x"], "y": destination["y"]}
        input_id = _allocate_input({"kind": "ok", "value": None})
        _notify_state_changed()
        return input_id

    if name == "join":
        # Handled by /join endpoint, but also accept via /input
        character = args.get("character", _pick_character())
        if user_id in _human_players:
            return _allocate_input({"kind": "error", "message": "You are already in this game!"})
        if len(_human_players) >= MAX_HUMAN_PLAYERS:
            return _allocate_input({"kind": "error", "message": f"Only {MAX_HUMAN_PLAYERS} human players allowed."})

        player_id = _next_player_id()
        position = _pick_spawn_position()
        _human_players[user_id] = {
            "player_id": player_id,
            "position": position,
            "facing": {"dx": 0, "dy": 1},
            "character": character,
            "name": args.get("name", "Me"),
        }
        input_id = _allocate_input({"kind": "ok", "value": player_id})
        _notify_state_changed()
        return input_id

    if name == "leave":
        if user_id in _human_players:
            del _human_players[user_id]
        input_id = _allocate_input({"kind": "ok", "value": None})
        _notify_state_changed()
        return input_id

    # Unknown input type — succeed silently for forward compatibility
    return _allocate_input({"kind": "ok", "value": None})


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
async def send_input(
    request: SendWorldInputRequest = Body(...),
    auth: AuthContext = Depends(get_current_user),
):
    """Send an input to the game engine (alias for /input)."""
    return await send_world_input(request=request, auth=auth)


@router.get("/input-status")
async def get_input_status(inputId: Optional[str] = Query(None)):
    """Check status of a submitted input.

    Returns the completed result if the input has been processed.
    The frontend polls this via watchQuery until it gets a non-null result.
    """
    if not inputId:
        return None

    result = _completed_inputs.get(inputId)
    if result is None:
        # Input not found or not yet processed
        return None

    return result


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
