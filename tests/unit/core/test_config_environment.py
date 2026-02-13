"""Unit tests for ENVIRONMENT setting in configuration module."""

import os
from unittest.mock import patch

from pydantic_settings import BaseSettings


class TestEnvironmentSetting:
    """Tests for Settings.ENVIRONMENT configuration."""

    def test_environment_defaults_to_non_prod(self):
        """ENVIRONMENT defaults to a value that is NOT 'prod'.

        This ensures Swagger UI is visible by default in local/dev environments.
        """
        from core.config import settings

        assert settings.ENVIRONMENT != "prod", (
            f"ENVIRONMENT should not default to 'prod', got '{settings.ENVIRONMENT}'"
        )

    def test_environment_reflects_env_var(self):
        """When ENVIRONMENT env var is set, settings picks it up."""
        # We need to construct a fresh Settings instance since the module-level
        # singleton was created at import time (before we set the env var).
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            from core.config import Settings

            fresh = Settings()
            assert fresh.ENVIRONMENT == "staging"

    def test_environment_can_be_set_to_prod(self):
        """ENVIRONMENT can be explicitly set to 'prod'."""
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
            from core.config import Settings

            fresh = Settings()
            assert fresh.ENVIRONMENT == "prod"

    def test_environment_empty_string_default(self):
        """ENVIRONMENT defaults to empty string when env var is unset."""
        # Remove ENVIRONMENT from env if present, then build fresh Settings.
        env_overrides = {"ENVIRONMENT": ""}
        with patch.dict(os.environ, env_overrides):
            # Also ensure the key is truly empty by explicitly removing it
            os.environ.pop("ENVIRONMENT", None)
            from core.config import Settings

            fresh = Settings()
            assert fresh.ENVIRONMENT == ""
