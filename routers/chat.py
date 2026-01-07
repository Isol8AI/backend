from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from core.database import get_db
from core.auth import get_current_user
from core.security import encryption_service
from core.llm import llm_service
from models.user import User
from models.user import User
from models.session import Session
from models.message import Message, MessageRole

router = APIRouter()

class ChatRequest(BaseModel):
    message: str # Plaintext from frontend (we encrypt it here)
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    user_id = current_user["sub"]
    
    # 1. Get User's Key (Decrypted)
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please sync first.")
        
    user_key = encryption_service.decrypt_user_key(user.encrypted_key)

    # 2. Get or Create Session
    session_id = request.session_id
    if not session_id:
        # Create new session if none provided
        new_session = Session(user_id=user_id, name=request.message[:30])
        db.add(new_session)
        await db.flush()
        session_id = str(new_session.id)
    
    # 3. Encrypt & Save User Message
    encrypted_user_msg = encryption_service.encrypt_message(request.message, user_key)
    user_message_Entry = Message(
        session_id=session_id,
        role=MessageRole.USER,
        content=encrypted_user_msg,
        model="hf-dedicated"
    )
    db.add(user_message_Entry)
    
    # 4. Call LLM (Send plaintext prompt)
    # In a real RAG app, you'd fetch history here, decrypt it, and append to prompt.
    llm_response_text = await llm_service.generate_response(request.message)

    # 5. Encrypt & Save AI Response
    encrypted_ai_msg = encryption_service.encrypt_message(llm_response_text, user_key)
    ai_message_entry = Message(
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=encrypted_ai_msg,
        model="hf-dedicated"
    )
    db.add(ai_message_entry)
    
    await db.commit()

    return ChatResponse(response=llm_response_text, session_id=session_id)
