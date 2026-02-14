"""Unit tests for configuration module."""

from core.config import FALLBACK_MODELS, settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_project_name_default(self):
        """Project name has expected default value."""
        assert settings.PROJECT_NAME == "Isol8 Chat"

    def test_api_v1_str_default(self):
        """API version string has expected default value."""
        assert settings.API_V1_STR == "/api/v1"

    def test_required_settings_exist(self):
        """Required settings are configured."""
        assert settings.CLERK_ISSUER is not None
        assert settings.DATABASE_URL is not None
        assert hasattr(settings, "AWS_REGION")
        assert hasattr(settings, "CLERK_AUDIENCE")

    def test_aws_region_default(self):
        """AWS_REGION has a default value."""
        assert settings.AWS_REGION is not None
        assert len(settings.AWS_REGION) > 0


class TestAvailableModels:
    """Tests for FALLBACK_MODELS configuration."""

    def test_available_models_not_empty(self):
        """FALLBACK_MODELS contains at least one model."""
        assert isinstance(FALLBACK_MODELS, list)
        assert len(FALLBACK_MODELS) > 0

    def test_models_have_required_fields(self):
        """Each model has non-empty id and name fields."""
        for model in FALLBACK_MODELS:
            assert model.get("id"), f"Model missing or empty 'id': {model}"
            assert model.get("name"), f"Model missing or empty 'name': {model}"

    def test_expected_models_available(self):
        """Expected model families are represented (AWS Bedrock models)."""
        model_ids = [m["id"].lower() for m in FALLBACK_MODELS]
        assert any("llama" in mid for mid in model_ids), "Llama model expected"
        assert any("claude" in mid or "anthropic" in mid for mid in model_ids), "Claude model expected"


class TestBillingConfig:
    """Test billing-related configuration."""

    def test_stripe_settings_have_defaults(self):
        """Stripe settings should have safe defaults for local dev."""
        from core.config import settings
        assert hasattr(settings, "STRIPE_SECRET_KEY")
        assert hasattr(settings, "STRIPE_WEBHOOK_SECRET")
        assert hasattr(settings, "STRIPE_METER_ID")
        assert hasattr(settings, "BILLING_MARKUP")
        assert settings.BILLING_MARKUP == 1.4

    def test_billing_plan_budgets_defined(self):
        """Plan budget constants should be defined."""
        from core.config import PLAN_BUDGETS, FREE_TIER_LIMIT
        assert "free" in PLAN_BUDGETS
        assert "starter" in PLAN_BUDGETS
        assert "pro" in PLAN_BUDGETS
        assert "usage_only" in PLAN_BUDGETS
        assert FREE_TIER_LIMIT == 2_000_000
