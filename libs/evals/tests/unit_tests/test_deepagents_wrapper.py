"""Unit tests for `DeepAgentsWrapper` initialization guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_harbor.deepagents_wrapper import DeepAgentsWrapper

if TYPE_CHECKING:
    from pathlib import Path


class TestModelNameRequired:
    """The wrapper rejects empty/whitespace `model_name` at construction."""

    def test_empty_string_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name="")

    def test_whitespace_only_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name="   ")


class TestOpenRouterPrefix:
    """`openrouter_provider` requires an `openrouter:` prefixed model."""

    def test_mismatched_prefix_raises(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError, match="openrouter_provider requires an openrouter: model prefix"
        ):
            DeepAgentsWrapper(
                logs_dir=tmp_path,
                model_name="claude-sonnet-4-6",
                openrouter_provider="MiniMax",
            )


class TestHappyPathConstruction:
    """Wrapper construction succeeds with a valid model name and stashes state."""

    def test_constructs_with_valid_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub the credential so `init_chat_model` doesn't reject us.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        wrapper = DeepAgentsWrapper(logs_dir=tmp_path, model_name="claude-sonnet-4-6")

        assert wrapper._model_name == "claude-sonnet-4-6"
        assert wrapper._model is not None
        assert wrapper._temperature == 0.0
