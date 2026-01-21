"""Tests for theme system."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from deepagents_cli.themes import (
    THEMES,
    get_textual_theme,
    is_dark_mode,
    set_theme,
)
from deepagents_cli.themes.detection import detect_dark_mode


class TestThemes:
    """Tests for theme mapping."""

    def test_default_theme_exists(self) -> None:
        """Test that default theme is available."""
        assert "default" in THEMES

    def test_tokyo_night_theme_exists(self) -> None:
        """Test that tokyo-night theme is available."""
        assert "tokyo-night" in THEMES

    def test_catppuccin_theme_exists(self) -> None:
        """Test that catppuccin theme is available."""
        assert "catppuccin" in THEMES

    def test_langchain_theme_exists(self) -> None:
        """Test that langchain theme is available."""
        assert "langchain" in THEMES

    def test_themes_map_to_textual_themes(self) -> None:
        """Test that all themes map to valid Textual theme names."""
        for name, textual_theme in THEMES.items():
            assert isinstance(textual_theme, str)
            assert len(textual_theme) > 0


class TestDarkModeDetection:
    """Tests for dark mode detection."""

    def test_defaults_to_dark(self) -> None:
        """Test that dark mode is the default."""
        with mock.patch.dict(os.environ, {"DEEPAGENTS_COLOR_MODE": ""}):
            assert detect_dark_mode() is True

    def test_explicit_dark_override(self) -> None:
        """Test DEEPAGENTS_COLOR_MODE=dark override."""
        with mock.patch.dict(os.environ, {"DEEPAGENTS_COLOR_MODE": "dark"}):
            assert detect_dark_mode() is True

    def test_explicit_light_override(self) -> None:
        """Test DEEPAGENTS_COLOR_MODE=light override."""
        with mock.patch.dict(os.environ, {"DEEPAGENTS_COLOR_MODE": "light"}):
            assert detect_dark_mode() is False

    def test_no_env_var_defaults_dark(self) -> None:
        """Test that missing env var defaults to dark."""
        env = {k: v for k, v in os.environ.items() if k != "DEEPAGENTS_COLOR_MODE"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert detect_dark_mode() is True


class TestThemeAPI:
    """Tests for the theme module API."""

    def test_set_theme_valid(self) -> None:
        """Test setting a valid theme."""
        set_theme("default")
        assert get_textual_theme() == "textual-dark"

    def test_set_theme_tokyo_night(self) -> None:
        """Test setting tokyo-night theme."""
        set_theme("tokyo-night")
        assert get_textual_theme() == "tokyo-night"

    def test_set_theme_catppuccin(self) -> None:
        """Test setting catppuccin theme."""
        set_theme("catppuccin")
        assert get_textual_theme() == "catppuccin-mocha"

    def test_set_theme_langchain(self) -> None:
        """Test setting langchain theme."""
        set_theme("langchain")
        assert get_textual_theme() == "langchain"

    def test_set_theme_invalid(self) -> None:
        """Test setting an invalid theme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown theme"):
            set_theme("nonexistent-theme")

    def test_is_dark_mode(self) -> None:
        """Test is_dark_mode returns boolean."""
        result = is_dark_mode()
        assert isinstance(result, bool)
