"""Tests for terminal `OSC 9;4` progress preference loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.app import (
    _load_terminal_progress_preference,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestLoadTerminalProgressPreference:
    """_load_terminal_progress_preference reads config.toml correctly."""

    def test_default_true_when_no_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
            tmp_path / "config.toml",
        )
        assert _load_terminal_progress_preference() is True

    def test_returns_saved_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[ui]\nterminal_progress = false\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_terminal_progress_preference() is False

    def test_default_true_when_key_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `[ui]` table without `terminal_progress` defaults to `True`."""
        config = tmp_path / "config.toml"
        config.write_text("[ui]\ncursor_blink = false\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_terminal_progress_preference() is True

    def test_default_true_on_corrupt_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("this is not = valid = toml\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_terminal_progress_preference() is True

    def test_defaults_true_on_non_bool_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[ui]\nterminal_progress = "nope"\n', encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_terminal_progress_preference() is True

    def test_defaults_true_when_ui_not_a_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text('ui = "not a table"\n', encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_terminal_progress_preference() is True
