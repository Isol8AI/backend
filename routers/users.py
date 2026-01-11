from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from core.database import get_db
from core.auth import get_current_user, AuthContext
from models.user import User

router = APIRouter()

@router.post("/sync")
async def sync_user(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Ensures the logged-in user exists in the database."""
    user_id = auth.user_id

    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()

    if not user:
        new_user = User(id=user_id)
        db.add(new_user)
        try:
            await db.commit()
            return {"status": "created", "user_id": user_id}
        except Exception as e:
            await db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {"status": "exists", "user_id": user_id}
