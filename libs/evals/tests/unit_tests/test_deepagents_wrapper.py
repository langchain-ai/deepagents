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
