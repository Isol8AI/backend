import os
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Isol8 Chat"
    API_V1_STR: str = "/api/v1"

    # Environment mode
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Clerk Auth
    CLERK_ISSUER: str = os.getenv("CLERK_ISSUER", "https://your-clerk-domain.clerk.accounts.dev")
    CLERK_AUDIENCE: Optional[str] = None
    CLERK_WEBHOOK_SECRET: Optional[str] = os.getenv("CLERK_WEBHOOK_SECRET")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/securechat")

    # HuggingFace Configuration
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    # Using HuggingFace Inference Providers router (OpenAI-compatible)
    HF_API_URL: str = os.getenv("HF_API_URL", "https://router.huggingface.co/v1")

    # CORS Configuration
    # Comma-separated list of allowed origins (e.g., "https://dev.isol8.co,https://localhost:3000")
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

    # Enclave Configuration
    # In production, this would be the Nitro enclave endpoint
    ENCLAVE_MODE: str = os.getenv("ENCLAVE_MODE", "mock")  # "mock" or "nitro"
    ENCLAVE_INFERENCE_TIMEOUT: float = float(os.getenv("ENCLAVE_INFERENCE_TIMEOUT", "120.0"))

    @field_validator("CLERK_ISSUER")
    @classmethod
    def validate_clerk_issuer(cls, v: str) -> str:
        if "your-clerk-domain" in v:
            raise ValueError("CLERK_ISSUER not configured. Set the CLERK_ISSUER environment variable.")
        return v

    @field_validator("HUGGINGFACE_TOKEN")
    @classmethod
    def validate_hf_token(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            import warnings

            warnings.warn("HUGGINGFACE_TOKEN not set. LLM features will not work.", UserWarning)
        return v

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
