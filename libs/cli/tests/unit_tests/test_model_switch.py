"""Tests for model switching functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.config import settings
from deepagents_cli.widgets.messages import AppMessage


class TestModelSwitchNoOp:
    """Tests for no-op when switching to the same model."""

    @pytest.mark.asyncio
    async def test_no_message_when_switching_to_same_model(self) -> None:
        """Switching to the already-active model should not print 'Switched to'.

        This is a regression test for the bug where selecting the same model
        from the model selector would print "Switched to X" even though no
        actual switch occurred.
        """
        app = DeepAgentsApp()
        # Replace method with mock to track calls (hence ignore)
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()  # Enable hot-swap path

        # Set current model
        settings.model_name = "claude-opus-4-5"
        settings.model_provider = "anthropic"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=True),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            # Attempt to switch to the same model
            await app._switch_model("anthropic:claude-opus-4-5")

        # Should show "Already using" message, not "Switched to"
        # Type checker doesn't track that _mount_message was replaced with mock
        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "Already using" in captured_messages[0]
        assert "Switched to" not in captured_messages[0]
