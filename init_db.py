import asyncio
import sys
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from core.database import engine
from models import Base


async def init_models(reset: bool = False):
    retries = 5
    while retries > 0:
        try:
            async with engine.begin() as conn:
                if reset:
                    print("Dropping all tables...")
                    # Drop schema and recreate to ensure clean slate
                    await conn.execute(text("DROP SCHEMA public CASCADE"))
                    await conn.execute(text("CREATE SCHEMA public"))
                    await conn.execute(text("GRANT ALL ON SCHEMA public TO postgres"))
                    await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

                # Enable pgvector extension
                print("Enabling pgvector extension...")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # Create SQLAlchemy tables
                print("Creating SQLAlchemy tables...")
                await conn.run_sync(Base.metadata.create_all)

                # Note: OpenMemory tables (openmemory_memories, openmemory_vectors, etc.)
                # are created automatically by the OpenMemory SDK when it initializes.
                # See memory/packages/openmemory-py/src/openmemory/core/db.py:_init_pg_schema()
                print("OpenMemory tables will be created on first SDK use...")

            print("Database initialization complete.")
            return
        except OperationalError:
            print("Database not ready yet, retrying in 2 seconds...")
            retries -= 1
            await asyncio.sleep(2)

    print("Could not connect to database after retries.")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    asyncio.run(init_models(reset=reset))
