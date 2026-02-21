# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sqlalchemy import text

from core.config import settings
from core.auth import get_current_user
from core.database import get_db, close_memory_pool
from core.enclave import startup_enclave, shutdown_enclave
from core.services.town_simulation import TownSimulation
from routers import (
    users,
    chat,
    organizations,
    context,
    webhooks,
    debug_encryption,
    websocket_chat,
    agents,
    town,
    billing,
)

logger = logging.getLogger(__name__)

_town_simulation: Optional[TownSimulation] = None


def get_town_simulation() -> Optional[TownSimulation]:
    """Get the global TownSimulation instance."""
    return _town_simulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _town_simulation

    # Startup
    logger.info("Starting application...")
    await startup_enclave()

    # Start GooseTown simulation
    from core.database import get_session_factory
    from routers.town import _notify_state_changed

    _town_simulation = TownSimulation(db_factory=get_session_factory(), notify_fn=_notify_state_changed)
    await _town_simulation.start()

    yield

    # Shutdown
    logger.info("Shutting down application...")
    if _town_simulation:
        await _town_simulation.stop()

    await close_memory_pool()
    await shutdown_enclave()


openapi_tags = [
    {
        "name": "users",
        "description": "User registration, sync, and encryption key management.",
    },
    {
        "name": "chat",
        "description": "Chat sessions, messages, model listing, and SSE streaming.",
    },
    {
        "name": "organizations",
        "description": "Organization encryption setup, key distribution, and membership.",
    },
    {
        "name": "agents",
        "description": "OpenClaw agent CRUD, encrypted agent messaging, and state management.",
    },
    {
        "name": "context",
        "description": "Context and fact storage for conversation memory.",
    },
    {
        "name": "webhooks",
        "description": "Clerk webhook handlers for user and organization sync events.",
    },
    {
        "name": "websocket",
        "description": "WebSocket connect, disconnect, and message endpoints (API Gateway integration).",
    },
    {
        "name": "debug",
        "description": "Debug and diagnostic endpoints for encryption testing. Disabled in production.",
    },
    {
        "name": "town",
        "description": "GooseTown AI agent simulation endpoints.",
    },
    {
        "name": "billing",
        "description": "Billing, usage tracking, and subscription management.",
    },
    {
        "name": "health",
        "description": "Root and health check endpoints for verifying API availability.",
    },
]

app = FastAPI(
    title="Isol8 Chat API",
    description=(
        "Zero-trust encrypted chat platform powered by AWS Nitro Enclaves. "
        "All messages are end-to-end encrypted: the server never sees plaintext. "
        "Supports personal and organization-level encryption with streaming LLM inference."
    ),
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=None if settings.ENVIRONMENT == "prod" else "/docs",
    redoc_url=None if settings.ENVIRONMENT == "prod" else "/redoc",
    servers=[
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "https://api-dev.isol8.co", "description": "Development"},
        {"url": "https://api-staging.isol8.co", "description": "Staging"},
        {"url": "https://api.isol8.co", "description": "Production"},
    ],
    openapi_tags=openapi_tags,
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


def custom_openapi():
    """Override OpenAPI schema generation to inject BearerAuth security scheme."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )
    schema["components"] = schema.get("components", {})
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# Public routes
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(organizations.router, prefix="/api/v1", tags=["organizations"])
app.include_router(context.router, prefix="/api/v1", tags=["context"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])

# Debug routes - DEVELOPMENT ONLY
app.include_router(debug_encryption.router, prefix="/api/v1", tags=["debug"])

# WebSocket routes (API Gateway WebSocket -> HTTP POST)
app.include_router(websocket_chat.router, prefix="/api/v1/ws")

# Agent routes (OpenClaw integration)
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])

# Billing routes
app.include_router(billing.router, prefix="/api/v1/billing", tags=["billing"])

# GooseTown routes
app.include_router(town.router, prefix="/api/v1/town", tags=["town"])


@app.get(
    "/",
    summary="API root",
    description="Returns a welcome message. Useful for verifying the API is reachable.",
    operation_id="root",
    tags=["health"],
)
async def root():
    return {"message": "Welcome to Isol8 Chat API"}


@app.get(
    "/health",
    summary="Health check",
    description=(
        "Validates database connectivity. Returns HTTP 200 when healthy, HTTP 503 when unhealthy. "
        "Used by ALB health checks to determine whether to route traffic to this instance."
    ),
    operation_id="health_check",
    tags=["health"],
    responses={
        503: {"description": "Database connection failed"},
    },
)
async def health_check(db=Depends(get_db)):
    from fastapi.responses import JSONResponse

    try:
        await db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "database": "disconnected"})


@app.get(
    "/protected",
    summary="Test authentication",
    description="Returns the authenticated user's ID. Useful for verifying that Clerk JWT authentication is working.",
    operation_id="protected_route",
    tags=["health"],
    responses={
        401: {"description": "Missing or invalid Clerk JWT token"},
    },
)
async def protected_route(auth=Depends(get_current_user)):
    return {"message": "You are authenticated", "user_id": auth.user_id}
