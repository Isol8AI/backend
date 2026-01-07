from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.security import encryption_service
from core.auth import get_current_user
from routers import users, chat

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Secure Chat Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "encryption_active": encryption_service.master_cipher is not None}

@app.get("/protected")
async def protected_route(user: dict = Depends(get_current_user)):
    return {"message": "You are authenticated", "user": user}
