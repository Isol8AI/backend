import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from core.config import settings
from core.auth import get_current_user
from core.database import get_db
from core.enclave import shutdown_enclave
from routers import users, chat, organizations, context, webhooks, debug_encryption, memories

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting application...")
    yield
    # Shutdown
    logger.info("Shutting down application...")
    await shutdown_enclave()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS configuration from environment
cors_origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Public routes
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(organizations.router, prefix="/api/v1", tags=["organizations"])
app.include_router(context.router, prefix="/api/v1", tags=["context"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])

# Memory routes
app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])

# Debug routes - DEVELOPMENT ONLY
app.include_router(debug_encryption.router, prefix="/api/v1", tags=["debug"])


@app.get("/")
async def root():
    return {"message": "Welcome to Freebird Chat API"}


@app.get("/health")
async def health_check(db = Depends(get_db)):
    """Health check that validates database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected"}


@app.get("/protected")
async def protected_route(auth = Depends(get_current_user)):
    return {"message": "You are authenticated", "user_id": auth.user_id}
