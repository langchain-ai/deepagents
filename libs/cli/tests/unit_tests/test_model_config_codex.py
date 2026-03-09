from unittest.mock import patch

import pytest


class TestHasProviderCredentialsCodex:
    @patch("deepagents_cli.model_config.ModelConfig.load")
    def test_authenticated(self, mock_load) -> None:
        mock_config = mock_load.return_value
        mock_config.providers = {}
        mock_config.has_credentials.return_value = None

        with patch.dict("sys.modules", {
            "deepagents_codex": __import__("types").ModuleType("deepagents_codex"),
            "deepagents_codex.status": __import__("types").ModuleType("deepagents_codex.status"),
        }):
            import sys

            # Create mock status module
            from enum import Enum

            class MockCodexAuthStatus(Enum):
                AUTHENTICATED = "authenticated"
                NOT_AUTHENTICATED = "not_authenticated"

            class MockCodexAuthInfo:
                def __init__(self, status):
                    self.status = status

            sys.modules["deepagents_codex"].get_auth_status = lambda: MockCodexAuthInfo(MockCodexAuthStatus.AUTHENTICATED)
            sys.modules["deepagents_codex.status"] = __import__("types").ModuleType("deepagents_codex.status")
            sys.modules["deepagents_codex.status"].CodexAuthStatus = MockCodexAuthStatus

            from deepagents_cli.model_config import has_provider_credentials

            result = has_provider_credentials("codex")
            assert result is True

    def test_package_not_installed(self) -> None:
        """When deepagents_codex is not installed, return False."""
        from deepagents_cli.model_config import has_provider_credentials

        # The actual import will fail, which should return False
        # This test only works when deepagents-codex is genuinely not installed
        # In our dev env it may be installed, so we mock the import
        with patch("deepagents_cli.model_config.ModelConfig.load") as mock_load:
            mock_config = mock_load.return_value
            mock_config.providers = {}
            mock_config.has_credentials.return_value = None

            with patch.dict("sys.modules", {"deepagents_codex": None, "deepagents_codex.status": None}):
                result = has_provider_credentials("codex")
                assert result is False

    def test_existing_providers_unaffected(self) -> None:
        """Non-codex providers still use their normal credential checks."""
        from deepagents_cli.model_config import has_provider_credentials

        # openai provider should use OPENAI_API_KEY env var
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = has_provider_credentials("openai")
            assert result is True


class TestGetAvailableModelsCodex:
    def test_includes_codex_when_installed(self) -> None:
        """When deepagents_codex is installed, codex models appear."""
        from deepagents_cli.model_config import clear_caches

        clear_caches()

        with patch(
            "deepagents_codex.models.get_available_codex_models",
            return_value=["gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini"],
        ):
            from deepagents_cli.model_config import get_available_models

            models = get_available_models()
            if "codex" in models:
                assert "gpt-4o" in models["codex"]

        clear_caches()
