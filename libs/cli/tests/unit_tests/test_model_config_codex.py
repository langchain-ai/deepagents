import types
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("deepagents_codex")


class TestHasProviderCredentialsCodex:
    @patch("deepagents_cli.model_config.ModelConfig.load")
    def test_authenticated(self, mock_load: MagicMock) -> None:
        mock_config = mock_load.return_value
        mock_config.providers = {}
        mock_config.has_credentials.return_value = None

        from enum import Enum

        class _Status(Enum):
            AUTHENTICATED = "authenticated"
            NOT_AUTHENTICATED = "not_authenticated"

        from dataclasses import dataclass

        @dataclass
        class _Info:
            status: _Status

        codex_mod = types.ModuleType("deepagents_codex")
        status_mod = types.ModuleType("deepagents_codex.status")
        codex_mod.get_auth_status = lambda: _Info(  # type: ignore[attr-defined]
            _Status.AUTHENTICATED,
        )
        status_mod.CodexAuthStatus = _Status  # type: ignore[attr-defined]

        with patch.dict(
            "sys.modules",
            {
                "deepagents_codex": codex_mod,
                "deepagents_codex.status": status_mod,
            },
        ):
            from deepagents_cli.model_config import has_provider_credentials

            result = has_provider_credentials("codex")
            assert result is True

    def test_package_not_installed(self) -> None:
        """When deepagents_codex is not installed, return False."""
        from deepagents_cli.model_config import has_provider_credentials

        with (
            patch("deepagents_cli.model_config.ModelConfig.load") as mock_load,
            patch.dict(
                "sys.modules",
                {
                    "deepagents_codex": None,
                    "deepagents_codex.status": None,
                },
            ),
        ):
            mock_config = mock_load.return_value
            mock_config.providers = {}
            mock_config.has_credentials.return_value = None

            result = has_provider_credentials("codex")
            assert result is False

    def test_existing_providers_unaffected(self) -> None:
        """Non-codex providers still use their normal credential checks."""
        from deepagents_cli.model_config import has_provider_credentials

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
            return_value=["gpt-5.4", "gpt-5.3-codex"],
        ):
            from deepagents_cli.model_config import get_available_models

            models = get_available_models()
            if "codex" in models:
                assert "gpt-5.4" in models["codex"]

        clear_caches()
