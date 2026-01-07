from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from core.database import get_db
from core.auth import get_current_user
from core.security import encryption_service
from models.user import User

router = APIRouter()

@router.post("/sync")
async def sync_user(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ensures the logged-in user exists in the database.
    If not, generates a new unique encryption key for them.
    """
    user_id = current_user["sub"] # Clerk User ID
    
    # Check if user already exists
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        # Create new user with a fresh encryption key
        raw_key = encryption_service.generate_user_key()
        encrypted_key = encryption_service.encrypt_user_key_for_storage(raw_key)
        
        new_user = User(id=user_id, encrypted_key=encrypted_key)
        db.add(new_user)
        try:
            await db.commit()
            await db.refresh(new_user)
            return {"status": "created", "user_id": user_id, "message": "User initialized with secure key."}
        except Exception as e:
            await db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
            
    return {"status": "exists", "user_id": user_id, "message": "User already exists."}
