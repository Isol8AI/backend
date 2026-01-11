import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.auth import AuthContext, get_current_user
from core.config import AVAILABLE_MODELS
from core.database import get_db, get_session_factory
from core.llm import llm_service
from models.message import Message, MessageRole
from models.session import Session
from models.user import User

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str | None = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: str
    name: str

    class Config:
        from_attributes = True


class ModelOut(BaseModel):
    id: str
    name: str


@router.get("/models", response_model=list[ModelOut])
async def get_available_models() -> list[ModelOut]:
    """Get list of available LLM models."""
    return AVAILABLE_MODELS


@router.get("/sessions", response_model=list[SessionOut])
async def get_sessions(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SessionOut]:
    """Get all chat sessions for the current user in the current context.

    Sessions are scoped to the current context:
    - Personal mode: Only sessions with org_id=None
    - Org mode: Only sessions with matching org_id
    """
    query = select(Session).filter(Session.user_id == auth.user_id)

    if auth.is_org_context:
        query = query.filter(Session.org_id == auth.org_id)
    else:
        query = query.filter(Session.org_id.is_(None))

    result = await db.execute(query.order_by(Session.created_at.desc()))
    return result.scalars().all()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def get_session_messages(
    session_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    """Get all messages for a specific session.

    Session must belong to user AND match current context (personal vs org).
    """
    query = select(Session).filter(
        Session.id == session_id,
        Session.user_id == auth.user_id,
    )

    if auth.is_org_context:
        query = query.filter(Session.org_id == auth.org_id)
    else:
        query = query.filter(Session.org_id.is_(None))

    session_result = await db.execute(query)
    session = session_result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .filter(Message.session_id == session_id)
        .order_by(asc(Message.timestamp))
    )
    return result.scalars().all()


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> StreamingResponse:
    """Stream chat response using Server-Sent Events."""
    # Verify user exists
    result = await db.execute(select(User).filter(User.id == auth.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please sync first.")

    # Get or create session
    session_id = request.session_id
    if not session_id:
        # Create session with org_id from current context
        new_session = Session(
            user_id=auth.user_id,
            org_id=auth.org_id,  # None for personal, org_id for org context
            name=request.message[:30]
        )
        db.add(new_session)
        await db.flush()
        session_id = str(new_session.id)

    # Fetch conversation history for context
    history_result = await db.execute(
        select(Message)
        .filter(Message.session_id == session_id)
        .order_by(asc(Message.timestamp))
    )
    history_messages = history_result.scalars().all()

    # Build history with model attribution for assistant messages
    history = []
    for msg in history_messages:
        if msg.role == MessageRole.USER:
            history.append({"role": "user", "content": msg.content})
        else:
            model_name = msg.model_used or "unknown"
            attributed_content = f"[Response from {model_name}]\n{msg.content}"
            history.append({"role": "assistant", "content": attributed_content})

    # Save user message
    user_message = Message(
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
        model_used=request.model or "default"
    )
    db.add(user_message)
    await db.commit()

    async def generate():
        full_response = ""

        # Send session_id first
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        async for chunk in llm_service.generate_response_stream(
            current_message=request.message,
            history=history,
            model=request.model
        ):
            full_response += chunk
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        # Save assistant response after streaming completes
        async with session_factory() as save_db:
            assistant_message = Message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                model_used=request.model or "default"
            )
            save_db.add(assistant_message)
            await save_db.commit()

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )