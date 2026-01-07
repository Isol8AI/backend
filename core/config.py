from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Secure Chat Platform"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "changethis_in_production_to_a_long_random_string")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Master Key for DB Encryption (Must be 32 bytes URL-safe base64)
    # Generate with: cryptography.fernet.Fernet.generate_key()
    MASTER_KEY: str = os.getenv("MASTER_KEY", "change_this_to_a_valid_fernet_key_in_prod=")

    # Clerk Auth
    CLERK_ISSUER: str = os.getenv("CLERK_ISSUER", "https://your-clerk-domain.clerk.accounts.dev")
    # Audience is often empty or specific in Clerk, usually the generic claim checks are enough
    CLERK_AUDIENCE: Optional[str] = None

    # Database
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/securechat")

    # External services
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    LLM_ENDPOINT_URL: Optional[str] = os.getenv("LLM_ENDPOINT_URL")

    class Config:
        env_file = ".env"

settings = Settings()
