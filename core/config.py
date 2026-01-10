from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Freebird Chat"
    API_V1_STR: str = "/api/v1"

    # Clerk Auth
    CLERK_ISSUER: str = os.getenv("CLERK_ISSUER", "https://your-clerk-domain.clerk.accounts.dev")
    CLERK_AUDIENCE: Optional[str] = None

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/securechat")

    # HuggingFace Configuration
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    # Using HuggingFace Inference Providers router (OpenAI-compatible)
    HF_API_URL: str = os.getenv("HF_API_URL", "https://router.huggingface.co/v1")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Available models for the frontend selector
# Models confirmed to work via HuggingFace Inference Providers (serverless)
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B"},
    {"id": "google/gemma-2-2b-it", "name": "Gemma 2 2B"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "name": "DeepSeek R1 32B"},
]
