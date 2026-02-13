"""GooseTown API endpoints."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import AuthContext, get_current_user
from core.database import get_db
from core.services.town_service import TownService
from schemas.town import (
    TownOptInRequest,
    TownOptOutRequest,
    TownAgentResponse,
    TownAgentStateResponse,
    TownStateResponse,
    TownConversationResponse,
    TownConversationsListResponse,
    ConversationTurn,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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


@router.get("/state", response_model=TownStateResponse)
async def get_state(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current town state (all active agents, positions, activities)."""
    service = TownService(db)
    states = await service.get_town_state()

    return TownStateResponse(
        agents=[
            TownAgentStateResponse(
                agent_id=s["agent_id"],
                display_name=s["display_name"],
                current_location=s["current_location"],
                current_activity=s["current_activity"],
                target_location=s["target_location"],
                position_x=s["position_x"],
                position_y=s["position_y"],
                mood=s["mood"],
                energy=s["energy"],
                status_message=s["status_message"],
            )
            for s in states
        ],
        timestamp=datetime.now(timezone.utc),
    )


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
