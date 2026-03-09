from unittest.mock import MagicMock, patch

import pytest

from deepagents_cli.model_config import ModelConfigError


class TestCreateModelCodex:
    @patch("deepagents_cli.config.detect_provider", return_value=None)
    def test_create_codex_model(self, mock_detect) -> None:
        with patch(
            "deepagents_codex.chat_models.ChatCodexOAuth"
        ) as MockChatCodex:
            mock_instance = MagicMock()
            mock_instance._model_provider = "codex"
            mock_instance.profile = {"max_input_tokens": 128000}
            MockChatCodex.return_value = mock_instance

            from deepagents_cli.config import create_model

            result = create_model("codex:gpt-4o")
            MockChatCodex.assert_called_once()
            assert result.provider == "codex"

    def test_create_codex_missing_package(self) -> None:
        with patch.dict(
            "sys.modules", {"deepagents_codex": None, "deepagents_codex.chat_models": None}
        ):
            from deepagents_cli.config import create_model

            with pytest.raises(ModelConfigError, match="deepagents-codex"):
                create_model("codex:gpt-4o")
