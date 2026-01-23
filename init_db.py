import asyncio
import os
import sys
import re
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from core.database import engine
from models import Base

# Check if OpenMemory SDK is available and configured for PostgreSQL
try:
    from openmemory.core.db import _init_pg_pool, is_pg as openmemory_is_pg
    OPENMEMORY_AVAILABLE = True
except ImportError:
    OPENMEMORY_AVAILABLE = False
    openmemory_is_pg = False


def get_schema_from_env() -> str:
    """
    Get the schema name from DATABASE_URL or ENVIRONMENT variable.
    Defaults to 'public' if not set.
    """
    # First try to parse from DATABASE_URL search_path option
    db_url = os.getenv("DATABASE_URL", "")
    match = re.search(r"search_path[=%]3D(\w+)", db_url)
    if match:
        return match.group(1)

    # Fall back to ENVIRONMENT variable
    env = os.getenv("ENVIRONMENT", "").lower()
    if env in ("dev", "staging", "prod"):
        return env

    # Default to public
    return "public"


async def init_models(reset: bool = False):
    schema = get_schema_from_env()
    print(f"Using database schema: {schema}")

    retries = 5
    while retries > 0:
        try:
            async with engine.begin() as conn:
                if reset:
                    print(f"Dropping schema '{schema}' and recreating...")
                    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

                # Create schema if it doesn't exist
                print(f"Creating schema '{schema}' if not exists...")
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                await conn.execute(text(f"GRANT ALL ON SCHEMA {schema} TO postgres"))

                # Set search_path for this connection
                await conn.execute(text(f"SET search_path TO {schema}"))

                # Enable pgvector extension (in public schema, shared across all schemas)
                print("Enabling pgvector extension...")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # Create SQLAlchemy tables in the schema
                print(f"Creating SQLAlchemy tables in '{schema}' schema...")
                await conn.run_sync(Base.metadata.create_all)

            # Initialize OpenMemory SDK tables (if available and using PostgreSQL)
            if OPENMEMORY_AVAILABLE and openmemory_is_pg:
                print("Initializing OpenMemory SDK tables...")
                try:
                    await _init_pg_pool()
                    print("OpenMemory SDK tables created successfully.")
                except Exception as e:
                    print(f"Warning: OpenMemory SDK initialization failed: {e}")
                    print("Memory features may not work until manually fixed.")
            elif OPENMEMORY_AVAILABLE:
                print("OpenMemory SDK is using SQLite, skipping PostgreSQL table creation.")
            else:
                print("OpenMemory SDK not available, skipping memory table creation.")

            print(f"Database initialization complete for schema '{schema}'.")
            return
        except OperationalError as e:
            print(f"Database not ready yet ({e}), retrying in 2 seconds...")
            retries -= 1
            await asyncio.sleep(2)

    print("Could not connect to database after retries.")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    asyncio.run(init_models(reset=reset))
