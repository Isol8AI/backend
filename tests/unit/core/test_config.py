"""Unit tests for configuration module."""
import pytest
from core.config import settings, AVAILABLE_MODELS


class TestSettings:
    """Tests for Settings configuration."""

    def test_project_name_default(self):
        """Project name has default value."""
        assert settings.PROJECT_NAME == "Freebird Chat"

    def test_api_v1_str_default(self):
        """API version string has default value."""
        assert settings.API_V1_STR == "/api/v1"

    def test_clerk_issuer_exists(self):
        """CLERK_ISSUER setting exists."""
        assert hasattr(settings, "CLERK_ISSUER")
        assert settings.CLERK_ISSUER is not None

    def test_clerk_audience_nullable(self):
        """CLERK_AUDIENCE can be None."""
        # This is optional, so it can be None
        assert hasattr(settings, "CLERK_AUDIENCE")

    def test_database_url_exists(self):
        """DATABASE_URL setting exists."""
        assert hasattr(settings, "DATABASE_URL")
        assert settings.DATABASE_URL is not None

    def test_huggingface_token_exists(self):
        """HUGGINGFACE_TOKEN setting exists."""
        assert hasattr(settings, "HUGGINGFACE_TOKEN")

    def test_hf_api_url_default(self):
        """HF_API_URL has default value."""
        assert hasattr(settings, "HF_API_URL")
        assert "huggingface" in settings.HF_API_URL.lower()


class TestAvailableModels:
    """Tests for AVAILABLE_MODELS configuration."""

    def test_available_models_is_list(self):
        """AVAILABLE_MODELS is a list."""
        assert isinstance(AVAILABLE_MODELS, list)

    def test_available_models_not_empty(self):
        """AVAILABLE_MODELS is not empty."""
        assert len(AVAILABLE_MODELS) > 0

    def test_models_have_required_fields(self):
        """Each model has id and name fields."""
        for model in AVAILABLE_MODELS:
            assert "id" in model, f"Model missing 'id': {model}"
            assert "name" in model, f"Model missing 'name': {model}"

    def test_models_id_not_empty(self):
        """Model IDs are not empty strings."""
        for model in AVAILABLE_MODELS:
            assert model["id"], f"Model has empty id: {model}"
            assert len(model["id"]) > 0

    def test_models_name_not_empty(self):
        """Model names are not empty strings."""
        for model in AVAILABLE_MODELS:
            assert model["name"], f"Model has empty name: {model}"
            assert len(model["name"]) > 0

    def test_qwen_model_available(self):
        """Qwen model is in available models."""
        model_ids = [m["id"] for m in AVAILABLE_MODELS]
        assert any("qwen" in id.lower() for id in model_ids)

    def test_llama_model_available(self):
        """Llama model is in available models."""
        model_ids = [m["id"] for m in AVAILABLE_MODELS]
        assert any("llama" in id.lower() for id in model_ids)
