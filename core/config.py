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

    # CORS Configuration (comma-separated origins)
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS as comma-separated list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # Enclave Configuration
    # In production, this would be the Nitro enclave endpoint
    ENCLAVE_MODE: str = os.getenv("ENCLAVE_MODE", "mock")  # "mock" or "nitro"
    ENCLAVE_INFERENCE_TIMEOUT: float = float(os.getenv("ENCLAVE_INFERENCE_TIMEOUT", "120.0"))

    # Memory extraction model (smaller, faster model for fact extraction)
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "Qwen/Qwen2.5-7B-Instruct")

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
# Query available models: curl "https://huggingface.co/api/models?inference_provider=all&pipeline_tag=text-generation"
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B"},
    {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "name": "DeepSeek R1 32B"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B"},
]
