"""Unit tests for configuration module."""

from core.config import AVAILABLE_MODELS, settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_project_name_default(self):
        """Project name has expected default value."""
        assert settings.PROJECT_NAME == "Freebird Chat"

    def test_api_v1_str_default(self):
        """API version string has expected default value."""
        assert settings.API_V1_STR == "/api/v1"

    def test_required_settings_exist(self):
        """Required settings are configured."""
        assert settings.CLERK_ISSUER is not None
        assert settings.DATABASE_URL is not None
        assert hasattr(settings, "HUGGINGFACE_TOKEN")
        assert hasattr(settings, "CLERK_AUDIENCE")

    def test_hf_api_url_default(self):
        """HF_API_URL points to HuggingFace."""
        assert "huggingface" in settings.HF_API_URL.lower()


class TestAvailableModels:
    """Tests for AVAILABLE_MODELS configuration."""

    def test_available_models_not_empty(self):
        """AVAILABLE_MODELS contains at least one model."""
        assert isinstance(AVAILABLE_MODELS, list)
        assert len(AVAILABLE_MODELS) > 0

    def test_models_have_required_fields(self):
        """Each model has non-empty id and name fields."""
        for model in AVAILABLE_MODELS:
            assert model.get("id"), f"Model missing or empty 'id': {model}"
            assert model.get("name"), f"Model missing or empty 'name': {model}"

    def test_expected_models_available(self):
        """Expected model families are represented."""
        model_ids = [m["id"].lower() for m in AVAILABLE_MODELS]
        assert any("qwen" in mid for mid in model_ids), "Qwen model expected"
        assert any("llama" in mid for mid in model_ids), "Llama model expected"
