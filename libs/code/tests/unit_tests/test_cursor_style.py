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


def test_cursor_style_loads_underline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The underline preference is read from the `[ui]` table."""
    monkeypatch.delenv(CURSOR_STYLE, raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[ui]\ncursor_style = "underline"\n', encoding="utf-8")
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)

    assert _load_cursor_style_preference() == "underline"


async def test_app_applies_underline_cursor_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The startup preference reaches the mounted chat text area."""
    monkeypatch.delenv(CURSOR_STYLE, raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[ui]\ncursor_style = "underline"\n', encoding="utf-8")
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        assert app._chat_input is not None
        assert app._chat_input._text_area is not None
        assert app._chat_input._text_area.has_class("cursor-underline")
