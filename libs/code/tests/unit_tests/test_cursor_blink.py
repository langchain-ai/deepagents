"""Tests for cursor-blink preference loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_code._env_vars import CURSOR_BLINK
from deepagents_code.app import (
    _load_cursor_blink_preference,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadCursorBlinkPreference:
    """_load_cursor_blink_preference resolves env then config.toml."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop a developer's exported override so config tests stay honest."""
        monkeypatch.delenv(CURSOR_BLINK, raising=False)

    def test_default_true_when_no_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
            tmp_path / "config.toml",
        )
        assert _load_cursor_blink_preference() is True

    def test_returns_saved_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[ui]\ncursor_blink = false\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_cursor_blink_preference() is False

    def test_default_true_on_corrupt_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("this is not = valid = toml\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_cursor_blink_preference() is True

    def test_defaults_true_on_non_bool_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[ui]\ncursor_blink = "nope"\n', encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_cursor_blink_preference() is True

    def test_defaults_true_when_ui_not_a_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text('ui = "not a table"\n', encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_cursor_blink_preference() is True

    def test_env_var_wins_over_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env var overrides an enabling `config.toml` value."""
        config = tmp_path / "config.toml"
        config.write_text("[ui]\ncursor_blink = true\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv(CURSOR_BLINK, "0")
        assert _load_cursor_blink_preference() is False

    def test_unrecognized_env_falls_through_to_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-boolean env token is ignored in favor of `config.toml`."""
        config = tmp_path / "config.toml"
        config.write_text("[ui]\ncursor_blink = false\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv(CURSOR_BLINK, "maybe")
        assert _load_cursor_blink_preference() is False

    def test_empty_env_var_opts_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly empty env value disables blinking (`empty_env_is_false`)."""
        config = tmp_path / "config.toml"
        config.write_text("[ui]\ncursor_blink = true\n", encoding="utf-8")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv(CURSOR_BLINK, "")
        assert _load_cursor_blink_preference() is False
