from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import asc
from pydantic import BaseModel
from typing import List
import json
from core.database import get_db, get_session_factory
from core.auth import get_current_user
from core.llm import llm_service
from core.config import AVAILABLE_MODELS
from models.user import User
from models.session import Session
from models.message import Message, MessageRole

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


@router.get("/models", response_model=List[ModelOut])
async def get_available_models():
    """Get list of available LLM models."""
    return AVAILABLE_MODELS


@router.get("/sessions", response_model=List[SessionOut])
async def get_sessions(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all chat sessions for the current user."""
    user_id = current_user["sub"]
    result = await db.execute(
        select(Session).filter(Session.user_id == user_id).order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()
    return sessions


@router.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all messages for a specific session."""
    user_id = current_user["sub"]

    # Verify session belongs to user
    session_result = await db.execute(
        select(Session).filter(Session.id == session_id, Session.user_id == user_id)
    )
    session = session_result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .filter(Message.session_id == session_id)
        .order_by(asc(Message.timestamp))
    )
    messages = result.scalars().all()
    return messages


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session_factory = Depends(get_session_factory)
):
    """Stream chat response using Server-Sent Events."""
    user_id = current_user["sub"]

    # Verify user exists
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please sync first.")

    # Get or create session
    session_id = request.session_id
    if not session_id:
        new_session = Session(user_id=user_id, name=request.message[:30])
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