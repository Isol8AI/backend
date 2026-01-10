import asyncio
import sys
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
                    await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
            print("Database tables created.")
            return
        except OperationalError:
            print("Database not ready yet, retrying in 2 seconds...")
            retries -= 1
            await asyncio.sleep(2)

    print("Could not connect to database after retries.")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    asyncio.run(init_models(reset=reset))
