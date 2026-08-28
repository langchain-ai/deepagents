"""Tests for chat input cursor-style preference loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code._env_vars import CURSOR_STYLE
from deepagents_code.app import DeepAgentsApp, _load_cursor_style_preference

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_cursor_style_defaults_to_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent config preserves the existing block cursor."""
    monkeypatch.delenv(CURSOR_STYLE, raising=False)
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
        tmp_path / "config.toml",
    )

    assert _load_cursor_style_preference() == "block"
