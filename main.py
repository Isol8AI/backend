# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv

load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from core.config import settings
from core.auth import get_current_user
from core.database import get_db, close_memory_pool
from core.enclave import startup_enclave, shutdown_enclave
from routers import users, chat, organizations, context, webhooks, debug_encryption, websocket_chat

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting application...")
    await startup_enclave()

    yield

    # Shutdown
    logger.info("Shutting down application...")

    await close_memory_pool()
    await shutdown_enclave()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
# Required because API Gateway HTTP_PROXY integration passes OPTIONS requests to backend.
# API Gateway adds CORS headers but doesn't intercept preflight - backend must return 2xx.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Public routes
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(organizations.router, prefix="/api/v1", tags=["organizations"])
app.include_router(context.router, prefix="/api/v1", tags=["context"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])

# Debug routes - DEVELOPMENT ONLY
app.include_router(debug_encryption.router, prefix="/api/v1", tags=["debug"])

# WebSocket routes
app.include_router(websocket_chat.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Welcome to Isol8 Chat API"}


@app.get("/health")
async def health_check(db=Depends(get_db)):
    """
    Health check that validates database connectivity.

    Returns:
        HTTP 200 with {"status": "healthy"} when all checks pass
        HTTP 503 with {"status": "unhealthy"} when any check fails

    ALB health checks require proper HTTP status codes:
    - 200-299: healthy, route traffic to this instance
    - 503: unhealthy, stop routing traffic to this instance
    """
    from fastapi.responses import JSONResponse

    try:
        await db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "database": "disconnected"})


@app.get("/protected")
async def protected_route(auth=Depends(get_current_user)):
    return {"message": "You are authenticated", "user_id": auth.user_id}
