"""Tests for theme system."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from deepagents_cli.themes import (
    BUILTIN_THEMES,
    Theme,
    ThemeColors,
    get_theme,
    is_dark_mode,
    set_theme,
    theme,
)
from deepagents_cli.themes.detection import detect_dark_mode


class TestThemeColors:
    """Tests for ThemeColors dataclass."""

    def test_default_values(self) -> None:
        """Test that ThemeColors has sensible defaults."""
        colors = ThemeColors()
        assert colors.text.startswith("#")
        assert colors.primary.startswith("#")
        assert colors.success.startswith("#")
        assert colors.error.startswith("#")

    def test_custom_values(self) -> None:
        """Test creating ThemeColors with custom values."""
        colors = ThemeColors(
            text="#ffffff",
            primary="#0000ff",
        )
        assert colors.text == "#ffffff"
        assert colors.primary == "#0000ff"


class TestTheme:
    """Tests for Theme dataclass."""

    def test_dark_mode_colors(self) -> None:
        """Test getting dark mode colors."""
        dark_colors = ThemeColors(text="#ffffff")
        light_colors = ThemeColors(text="#000000")
        t = Theme(name="test", dark=dark_colors, light=light_colors)

        assert t.colors(dark_mode=True).text == "#ffffff"
        assert t.colors(dark_mode=False).text == "#000000"

    def test_no_light_variant_falls_back_to_dark(self) -> None:
        """Test that themes without light variant use dark for both modes."""
        dark_colors = ThemeColors(text="#ffffff")
        t = Theme(name="test", dark=dark_colors, light=None)

        assert t.colors(dark_mode=True).text == "#ffffff"
        assert t.colors(dark_mode=False).text == "#ffffff"


class TestBuiltinThemes:
    """Tests for built-in themes."""

    def test_default_theme_exists(self) -> None:
        """Test that default theme is available."""
        assert "default" in BUILTIN_THEMES

    def test_all_themes_have_required_colors(self) -> None:
        """Test that all built-in themes have required color attributes."""
        required_attrs = ["text", "primary", "success", "error", "warning"]

        for name, t in BUILTIN_THEMES.items():
            colors = t.colors(dark_mode=True)
            for attr in required_attrs:
                assert hasattr(colors, attr), f"Theme '{name}' missing '{attr}'"
                assert getattr(colors, attr).startswith("#"), (
                    f"Theme '{name}' color '{attr}' is not a hex color"
                )

    def test_default_theme_has_light_variant(self) -> None:
        """Test that default theme has a light mode variant."""
        default = BUILTIN_THEMES["default"]
        assert default.light is not None


class TestDarkModeDetection:
    """Tests for dark mode detection."""

    def test_explicit_dark_override(self) -> None:
        """Test DEEPAGENTS_COLOR_MODE=dark override."""
        with mock.patch.dict(os.environ, {"DEEPAGENTS_COLOR_MODE": "dark"}):
            assert detect_dark_mode() is True

    def test_explicit_light_override(self) -> None:
        """Test DEEPAGENTS_COLOR_MODE=light override."""
        with mock.patch.dict(os.environ, {"DEEPAGENTS_COLOR_MODE": "light"}):
            assert detect_dark_mode() is False

    def test_colorfgbg_dark(self) -> None:
        """Test COLORFGBG detection for dark background."""
        env = {"COLORFGBG": "15;0", "DEEPAGENTS_COLOR_MODE": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            # bg=0 is black, should be dark mode
            assert detect_dark_mode() is True

    def test_colorfgbg_light(self) -> None:
        """Test COLORFGBG detection for light background."""
        env = {"COLORFGBG": "0;15", "DEEPAGENTS_COLOR_MODE": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            # bg=15 is white, should be light mode
            assert detect_dark_mode() is False


class TestThemeAPI:
    """Tests for the theme module API."""

    def test_set_theme_valid(self) -> None:
        """Test setting a valid theme."""
        set_theme("default")
        colors = get_theme()
        assert colors.text.startswith("#")

    def test_set_theme_invalid(self) -> None:
        """Test setting an invalid theme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown theme"):
            set_theme("nonexistent-theme")

    def test_theme_proxy_attribute_access(self) -> None:
        """Test that theme proxy allows attribute access."""
        set_theme("default")
        # Should be able to access colors via the proxy
        assert theme.primary.startswith("#")
        assert theme.text.startswith("#")

    def test_theme_proxy_invalid_attribute(self) -> None:
        """Test that theme proxy raises AttributeError for invalid attributes."""
        with pytest.raises(AttributeError, match="no color"):
            _ = theme.nonexistent_color
