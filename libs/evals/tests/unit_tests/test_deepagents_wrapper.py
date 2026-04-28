"""Unit tests for `DeepAgentsWrapper` initialization guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_harbor.deepagents_wrapper import (
    DeepAgentsWrapper,
    _accepts_temperature,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestAcceptsTemperature:
    """`_accepts_temperature` recognizes reasoning-only model prefixes."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "o1-mini",
            "o1-preview",
            "o3-mini",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "openai:o3-mini",
            "openai:gpt-5-codex",
        ],
    )
    def test_reasoning_models_reject_temperature(self, model_name: str) -> None:
        assert _accepts_temperature(model_name) is False

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-sonnet-4-6",
            "anthropic:claude-opus-4-1",
            "gpt-4o-mini",
            "openai:gpt-4o",
            "gemini-2.5-flash",
            "openrouter:zai/glm-4.6",
        ],
    )
    def test_chat_models_accept_temperature(self, model_name: str) -> None:
        assert _accepts_temperature(model_name) is True


class TestModelNameRequired:
    """The wrapper rejects empty/whitespace `model_name` at construction."""

    def test_empty_string_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name="")

    def test_whitespace_only_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name="   ")

    def test_none_raises(self, tmp_path: Path) -> None:
        # Type system disallows this, but runtime guard must hold.
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name=None)  # type: ignore[arg-type]


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
