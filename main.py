# Load environment variables FIRST, before any other imports
# This ensures OpenMemory SDK sees the OM_* env vars
from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from core.config import settings
from core.auth import get_current_user
from core.database import get_db, close_memory_pool
from core.enclave import shutdown_enclave
from routers import users, chat, organizations, context, webhooks, debug_encryption, memories

logger = logging.getLogger(__name__)

# Background task handles
_decay_task = None
_reflection_task = None


async def memory_decay_loop():
    """Background task that periodically applies memory decay."""
    # Import here to avoid circular imports
    try:
        import sys
        from pathlib import Path

        memory_path = Path(__file__).parent.parent / "memory" / "packages" / "openmemory-py" / "src"
        if str(memory_path) not in sys.path:
            sys.path.insert(0, str(memory_path))
        from openmemory.memory.decay import apply_decay
    except ImportError as e:
        logger.warning(f"Memory decay not available: {e}")
        return

    # Run decay every 5 minutes
    interval = 5 * 60
    logger.info(f"Memory decay loop started (interval: {interval}s)")

    while True:
        try:
            await asyncio.sleep(interval)
            await apply_decay()
        except asyncio.CancelledError:
            logger.info("Memory decay loop cancelled")
            break
        except Exception as e:
            logger.error(f"Memory decay error: {e}")
            # Continue running despite errors


async def memory_reflection_loop():
    """Background task that periodically consolidates similar memories."""
    # Import here to avoid circular imports
    try:
        import sys
        from pathlib import Path

        memory_path = Path(__file__).parent.parent / "memory" / "packages" / "openmemory-py" / "src"
        if str(memory_path) not in sys.path:
            sys.path.insert(0, str(memory_path))
        from openmemory.memory.reflect import run_reflection
    except ImportError as e:
        logger.warning(f"Memory reflection not available: {e}")
        return

    # Run reflection every 10 minutes (consolidates similar memories)
    interval = 10 * 60
    logger.info(f"Memory reflection loop started (interval: {interval}s)")

    while True:
        try:
            await asyncio.sleep(interval)
            # Use embedding-based clustering (works with encrypted content)
            result = await run_reflection(use_embedding_clustering=True)
            if result.get("created", 0) > 0:
                logger.info(f"Memory reflection: consolidated {result['created']} clusters")
        except asyncio.CancelledError:
            logger.info("Memory reflection loop cancelled")
            break
        except Exception as e:
            logger.error(f"Memory reflection error: {e}")
            # Continue running despite errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _decay_task, _reflection_task

    # Startup
    logger.info("Starting application...")

    # Start memory decay background task
    _decay_task = asyncio.create_task(memory_decay_loop())
    logger.info("Memory decay background task started")

    # Start memory reflection/consolidation background task
    _reflection_task = asyncio.create_task(memory_reflection_loop())
    logger.info("Memory reflection background task started")

    yield

    # Shutdown
    logger.info("Shutting down application...")

    # Cancel background tasks
    for task, name in [(_decay_task, "decay"), (_reflection_task, "reflection")]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Memory {name} task cancelled")

    await close_memory_pool()
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
