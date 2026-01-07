import asyncio
import time
from sqlalchemy.exc import OperationalError
from core.database import engine
from models import Base

async def init_models():
    retries = 5
    while retries > 0:
        try:
            async with engine.begin() as conn:
                # await conn.run_sync(Base.metadata.drop_all) # Uncomment to reset
                await conn.run_sync(Base.metadata.create_all)
            print("Database tables created.")
            return
        except OperationalError:
            print("Database not ready yet, retrying in 2 seconds...")
            retries -= 1
            await asyncio.sleep(2)
    
    print("Could not connect to database after retries.")

if __name__ == "__main__":
    asyncio.run(init_models())
