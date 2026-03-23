"""Tests for deepagents_cli.theme module."""

from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli import theme
from deepagents_cli.theme import (
    DARK_COLORS,
    DEFAULT_THEME,
    LIGHT_COLORS,
    ThemeColors,
    ThemeEntry,
    get_css_variable_defaults,
)

# ---------------------------------------------------------------------------
# ThemeColors validation
# ---------------------------------------------------------------------------


class TestThemeColorsValidation:
    """Hex color validation in ThemeColors.__post_init__."""

    def _make_kwargs(self, **overrides: str) -> dict[str, str]:
        """Return valid ThemeColors kwargs with optional overrides."""
        base = {f.name: "#AABBCC" for f in fields(ThemeColors)}
        base.update(overrides)
        return base

    def test_valid_hex_colors_accepted(self) -> None:
        tc = ThemeColors(**self._make_kwargs())
        assert tc.primary == "#AABBCC"

    def test_valid_lowercase_hex_accepted(self) -> None:
        tc = ThemeColors(**self._make_kwargs(primary="#aabbcc"))
        assert tc.primary == "#aabbcc"

    def test_valid_mixed_case_hex_accepted(self) -> None:
        tc = ThemeColors(**self._make_kwargs(primary="#AaBb99"))
        assert tc.primary == "#AaBb99"

    @pytest.mark.parametrize(
        "bad_value",
        [
            "#FFF",  # 3-char shorthand
            "#GGGGGG",  # invalid hex chars
            "red",  # named color
            "",  # empty
            "rgb(1,2,3)",  # CSS function
            "#7AA2F7FF",  # 8-char RGBA
            "7AA2F7",  # missing hash
            "#7AA2F",  # 5 hex chars
        ],
    )
    def test_invalid_hex_raises(self, bad_value: str) -> None:
        with pytest.raises(ValueError, match="7-char hex color"):
            ThemeColors(**self._make_kwargs(primary=bad_value))

    def test_validation_checks_every_field(self) -> None:
        """Ensure the last field is also validated, not just the first."""
        last_field = fields(ThemeColors)[-1].name
        with pytest.raises(ValueError, match=last_field):
            ThemeColors(**self._make_kwargs(**{last_field: "bad"}))

    def test_frozen_immutability(self) -> None:
        tc = ThemeColors(**self._make_kwargs())
        with pytest.raises(AttributeError):
            tc.primary = "#000000"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pre-built color sets
# ---------------------------------------------------------------------------


class TestColorSets:
    """DARK_COLORS and LIGHT_COLORS are valid ThemeColors instances."""

    def test_dark_colors_is_theme_colors(self) -> None:
        assert isinstance(DARK_COLORS, ThemeColors)

    def test_light_colors_is_theme_colors(self) -> None:
        assert isinstance(LIGHT_COLORS, ThemeColors)

    def test_dark_and_light_differ(self) -> None:
        assert DARK_COLORS.primary != LIGHT_COLORS.primary
        assert DARK_COLORS.background != LIGHT_COLORS.background


# ---------------------------------------------------------------------------
# ThemeEntry.REGISTRY
# ---------------------------------------------------------------------------


EXPECTED_REGISTRY_KEYS = frozenset(
    {"langchain", "langchain-light", "textual-dark", "textual-light", "textual-ansi"}
)


class TestThemeEntryRegistry:
    """ThemeEntry.REGISTRY contents and immutability."""

    def test_registry_contains_expected_keys(self) -> None:
        assert set(ThemeEntry.REGISTRY.keys()) == EXPECTED_REGISTRY_KEYS

    def test_registry_is_read_only(self) -> None:
        assert isinstance(ThemeEntry.REGISTRY, MappingProxyType)
        with pytest.raises(TypeError):
            ThemeEntry.REGISTRY["bad"] = None  # type: ignore[index]

    def test_default_theme_in_registry(self) -> None:
        assert DEFAULT_THEME in ThemeEntry.REGISTRY

    @pytest.mark.parametrize(
        ("name", "dark", "custom"),
        [
            ("langchain", True, True),
            ("langchain-light", False, True),
            ("textual-dark", True, False),
            ("textual-light", False, False),
            ("textual-ansi", True, False),
        ],
    )
    def test_entry_flags(self, name: str, dark: bool, custom: bool) -> None:
        entry = ThemeEntry.REGISTRY[name]
        assert entry.dark is dark
        assert entry.custom is custom

    def test_every_entry_has_non_empty_label(self) -> None:
        for name, entry in ThemeEntry.REGISTRY.items():
            assert entry.label.strip(), f"Entry '{name}' has empty label"

    def test_every_entry_has_valid_colors(self) -> None:
        for name, entry in ThemeEntry.REGISTRY.items():
            assert isinstance(entry.colors, ThemeColors), (
                f"Entry '{name}' has invalid colors"
            )


# ---------------------------------------------------------------------------
# get_css_variable_defaults
# ---------------------------------------------------------------------------


EXPECTED_CSS_KEYS = frozenset(
    {
        "muted",
        "tool-border",
        "tool-border-hover",
        "mode-bash",
        "mode-command",
        "diff-add-fg",
        "diff-add-bg",
        "diff-remove-fg",
        "diff-remove-bg",
        "error-bg",
    }
)


class TestGetCssVariableDefaults:
    """get_css_variable_defaults() return values."""

    def test_returns_expected_keys(self) -> None:
        result = get_css_variable_defaults(dark=True)
        assert set(result.keys()) == EXPECTED_CSS_KEYS

    def test_dark_mode_uses_dark_colors(self) -> None:
        result = get_css_variable_defaults(dark=True)
        assert result["muted"] == DARK_COLORS.muted

    def test_light_mode_uses_light_colors(self) -> None:
        result = get_css_variable_defaults(dark=False)
        assert result["muted"] == LIGHT_COLORS.muted

    def test_explicit_colors_take_precedence(self) -> None:
        result = get_css_variable_defaults(dark=True, colors=LIGHT_COLORS)
        assert result["muted"] == LIGHT_COLORS.muted

    def test_all_values_are_hex_colors(self) -> None:
        import re

        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for key, val in get_css_variable_defaults(dark=True).items():
            assert hex_re.match(val), f"CSS var '{key}' has non-hex value: {val!r}"


# ---------------------------------------------------------------------------
# Semantic module-level constants
# ---------------------------------------------------------------------------


class TestSemanticConstants:
    """Module-level constants (PRIMARY, MUTED, etc.) are valid hex colors."""

    @pytest.mark.parametrize(
        "name",
        [
            "PRIMARY",
            "PRIMARY_DEV",
            "SUCCESS",
            "WARNING",
            "MUTED",
            "MODE_BASH",
            "MODE_COMMAND",
            "DIFF_ADD_FG",
            "DIFF_ADD_BG",
            "DIFF_REMOVE_FG",
            "DIFF_REMOVE_BG",
            "DIFF_CONTEXT",
            "TOOL_BORDER",
            "TOOL_HEADER",
            "FILE_PYTHON",
            "FILE_CONFIG",
            "FILE_DIR",
            "SPINNER",
        ],
    )
    def test_constant_is_valid_hex(self, name: str) -> None:
        import re

        val = getattr(theme, name)
        assert re.match(r"^#[0-9A-Fa-f]{6}$", val), (
            f"theme.{name} = {val!r} is not a valid hex color"
        )


# ---------------------------------------------------------------------------
# _load_theme_preference / save_theme_preference
# ---------------------------------------------------------------------------


class TestLoadThemePreference:
    """_load_theme_preference reads config.toml correctly."""

    def test_returns_default_when_no_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli.app import _load_theme_preference

        monkeypatch.setattr("deepagents_cli.app.theme.DEFAULT_THEME", "langchain")
        missing = tmp_path / "nonexistent" / "config.toml"
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", missing)
        assert _load_theme_preference() == "langchain"

    def test_returns_saved_theme(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli.app import _load_theme_preference

        config = tmp_path / "config.toml"
        config.write_text('[ui]\ntheme = "langchain-light"\n')
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_theme_preference() == "langchain-light"

    def test_returns_default_for_unknown_theme(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli.app import _load_theme_preference

        config = tmp_path / "config.toml"
        config.write_text('[ui]\ntheme = "nonexistent-theme"\n')
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_theme_preference() == DEFAULT_THEME

    def test_returns_default_for_corrupt_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli.app import _load_theme_preference

        config = tmp_path / "config.toml"
        config.write_text("this is not valid toml [[[")
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_theme_preference() == DEFAULT_THEME

    def test_returns_default_when_ui_section_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli.app import _load_theme_preference

        config = tmp_path / "config.toml"
        config.write_text('[model]\nname = "gpt-4"\n')
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_theme_preference() == DEFAULT_THEME


class TestSaveThemePreference:
    """save_theme_preference writes config.toml correctly."""

    def test_creates_config_from_scratch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tomllib

        from deepagents_cli.app import save_theme_preference

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        assert save_theme_preference("langchain-light") is True
        data = tomllib.loads(config.read_text())
        assert data["ui"]["theme"] == "langchain-light"

    def test_preserves_existing_config_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tomllib

        from deepagents_cli.app import save_theme_preference

        config = tmp_path / "config.toml"
        config.write_text('[model]\nname = "gpt-4"\n')
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        assert save_theme_preference("langchain") is True
        data = tomllib.loads(config.read_text())
        assert data["model"]["name"] == "gpt-4"
        assert data["ui"]["theme"] == "langchain"

    def test_rejects_unknown_theme(self) -> None:
        from deepagents_cli.app import save_theme_preference

        assert save_theme_preference("nonexistent-theme") is False

    def test_returns_false_on_write_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli.app import save_theme_preference

        # Point to a directory that doesn't exist and can't be created
        config = tmp_path / "readonly" / "config.toml"
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)
        # Make parent read-only so mkdir fails
        (tmp_path / "readonly").mkdir()
        (tmp_path / "readonly").chmod(0o444)
        result = save_theme_preference("langchain")
        # Restore permissions for cleanup
        (tmp_path / "readonly").chmod(0o755)
        assert result is False
