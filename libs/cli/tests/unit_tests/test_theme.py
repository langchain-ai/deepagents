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
    _BUILTIN_NAMES,
    DARK_COLORS,
    DEFAULT_THEME,
    LIGHT_COLORS,
    ThemeColors,
    ThemeEntry,
    _build_registry,
    _builtin_themes,
    _load_user_themes,
    get_css_variable_defaults,
    get_theme_colors,
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


class TestThemeEntryRegistry:
    """ThemeEntry.REGISTRY contents and immutability."""

    def test_registry_contains_builtin_keys(self) -> None:
        assert set(ThemeEntry.REGISTRY.keys()) >= _BUILTIN_NAMES

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
            ("textual-ansi", False, False),
            # Community themes
            ("dracula", True, False),
            ("monokai", True, False),
            ("nord", True, False),
            ("tokyo-night", True, False),
            ("gruvbox", True, False),
            ("catppuccin-mocha", True, False),
            ("solarized-dark", True, False),
            ("solarized-light", False, False),
            ("catppuccin-latte", False, False),
            ("atom-one-dark", True, False),
            ("atom-one-light", False, False),
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
        "mode-bash",
        "mode-command",
    }
)


class TestGetCssVariableDefaults:
    """get_css_variable_defaults() return values."""

    def test_returns_expected_keys(self) -> None:
        result = get_css_variable_defaults(dark=True)
        assert set(result.keys()) == EXPECTED_CSS_KEYS

    def test_dark_mode_uses_dark_colors(self) -> None:
        result = get_css_variable_defaults(dark=True)
        assert result["mode-bash"] == DARK_COLORS.mode_bash

    def test_light_mode_uses_light_colors(self) -> None:
        result = get_css_variable_defaults(dark=False)
        assert result["mode-bash"] == LIGHT_COLORS.mode_bash

    def test_explicit_colors_take_precedence(self) -> None:
        result = get_css_variable_defaults(dark=True, colors=LIGHT_COLORS)
        assert result["mode-bash"] == LIGHT_COLORS.mode_bash

    def test_all_values_are_hex_colors(self) -> None:
        import re

        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for key, val in get_css_variable_defaults(dark=True).items():
            assert hex_re.match(val), f"CSS var '{key}' has non-hex value: {val!r}"


# ---------------------------------------------------------------------------
# Semantic module-level constants
# ---------------------------------------------------------------------------


_ANSI_COLOR_NAMES = frozenset(
    {
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "bright_black",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
    }
)
"""Standard Rich ANSI color names (base 16)."""


class TestSemanticConstants:
    """Module-level constants (PRIMARY, MUTED, etc.) are ANSI color names."""

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
    def test_constant_is_valid_ansi_name(self, name: str) -> None:
        val = getattr(theme, name)
        assert isinstance(val, str), f"theme.{name} = {val!r} is not a string"
        assert val in _ANSI_COLOR_NAMES, (
            f"theme.{name} = {val!r} is not a valid ANSI color name"
        )


# ---------------------------------------------------------------------------
# get_theme_colors
# ---------------------------------------------------------------------------


class TestGetThemeColors:
    """get_theme_colors() returns the correct ThemeColors."""

    def test_none_returns_dark_colors(self) -> None:
        assert get_theme_colors(None) is DARK_COLORS

    def test_no_args_returns_dark_colors(self) -> None:
        assert get_theme_colors() is DARK_COLORS

    def test_custom_dark_theme_returns_stored_colors(self) -> None:
        class FakeApp:
            theme = "langchain"

        assert get_theme_colors(FakeApp()) is DARK_COLORS

    def test_custom_light_theme_returns_stored_colors(self) -> None:
        class FakeApp:
            theme = "langchain-light"

        assert get_theme_colors(FakeApp()) is LIGHT_COLORS

    def test_builtin_theme_resolves_dynamically(self) -> None:
        """Built-in Textual themes derive colors from current_theme."""

        class CurrentTheme:
            dark = True
            primary = "#BD93F9"
            secondary = "#6272A4"
            accent = "#FF79C6"
            panel = "#313442"
            success = "#50FA7B"
            warning = "#FFB86C"
            error = "#FF5555"
            foreground = "#F8F8F2"
            background = "#282A36"
            surface = "#2B2E3B"

        class FakeApp:
            theme = "dracula"
            current_theme = CurrentTheme()

        colors = get_theme_colors(FakeApp())
        assert colors.primary == "#BD93F9"
        assert colors.background == "#282A36"
        # App-specific fields fall back to base
        assert colors.primary_dev == DARK_COLORS.primary_dev
        assert colors.muted == DARK_COLORS.muted

    def test_builtin_theme_ansi_falls_back(self) -> None:
        """ANSI theme has non-hex values; falls back to base palette."""

        class CurrentTheme:
            dark = False
            primary = "ansi_blue"
            secondary = "ansi_cyan"
            accent = "ansi_bright_blue"
            panel = "ansi_default"
            success = "ansi_green"
            warning = "ansi_yellow"
            error = "ansi_red"
            foreground = "ansi_default"
            background = "ansi_default"
            surface = "ansi_default"

        class FakeApp:
            theme = "textual-ansi"
            current_theme = CurrentTheme()

        colors = get_theme_colors(FakeApp())
        # Non-hex values fall back to light base (ansi theme is dark=False)
        assert colors.primary == LIGHT_COLORS.primary
        assert colors.background == LIGHT_COLORS.background

    def test_unknown_theme_dark_fallback(self) -> None:
        class CurrentTheme:
            dark = True
            primary = "#112233"
            secondary = "#445566"
            accent = "#778899"
            panel = "#AABBCC"
            success = "#AABBCC"
            warning = "#AABBCC"
            error = "#AABBCC"
            foreground = "#AABBCC"
            background = "#AABBCC"
            surface = "#AABBCC"

        class FakeApp:
            theme = "nonexistent"
            current_theme = CurrentTheme()

        colors = get_theme_colors(FakeApp())
        assert colors.primary == "#112233"

    def test_widget_with_app_property(self) -> None:
        """Simulates a mounted widget whose .app resolves to an App."""

        class FakeApp:
            theme = "langchain-light"

        class FakeWidget:
            @property
            def app(self) -> FakeApp:
                return FakeApp()

        assert get_theme_colors(FakeWidget()) is LIGHT_COLORS


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


# ---------------------------------------------------------------------------
# ThemeColors.merged
# ---------------------------------------------------------------------------


class TestThemeColorsMerged:
    """ThemeColors.merged() creates a new instance from base + overrides."""

    def test_no_overrides_returns_copy_of_base(self) -> None:
        result = ThemeColors.merged(DARK_COLORS, {})
        assert result == DARK_COLORS

    def test_single_override_applied(self) -> None:
        result = ThemeColors.merged(DARK_COLORS, {"primary": "#112233"})
        assert result.primary == "#112233"
        # Other fields unchanged
        assert result.accent == DARK_COLORS.accent

    def test_multiple_overrides(self) -> None:
        result = ThemeColors.merged(
            LIGHT_COLORS, {"primary": "#AAAAAA", "error": "#BBBBBB"}
        )
        assert result.primary == "#AAAAAA"
        assert result.error == "#BBBBBB"
        assert result.success == LIGHT_COLORS.success

    def test_unknown_keys_ignored(self) -> None:
        result = ThemeColors.merged(DARK_COLORS, {"not_a_field": "#123456"})
        assert result == DARK_COLORS

    def test_invalid_hex_raises(self) -> None:
        with pytest.raises(ValueError, match="7-char hex color"):
            ThemeColors.merged(DARK_COLORS, {"primary": "bad"})

    def test_returns_new_instance(self) -> None:
        result = ThemeColors.merged(DARK_COLORS, {"primary": "#000000"})
        assert result is not DARK_COLORS


# ---------------------------------------------------------------------------
# _load_user_themes
# ---------------------------------------------------------------------------


def _write_config(path: Path, content: str) -> None:
    """Write TOML content to a config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLoadUserThemes:
    """_load_user_themes reads [themes.*] from config.toml."""

    def test_no_config_file(self, tmp_path: Path) -> None:
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=tmp_path / "missing.toml")
        assert builtins == {}

    def test_no_themes_section(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(config, '[ui]\ntheme = "langchain"\n')
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert builtins == {}

    def test_valid_user_theme_loaded(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.my-dark]
label = "My Dark"
dark = true
primary = "#FF0000"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "my-dark" in builtins
        entry = builtins["my-dark"]
        assert entry.label == "My Dark"
        assert entry.dark is True
        assert entry.custom is True
        assert entry.colors.primary == "#FF0000"
        # Unspecified fields fall back to DARK_COLORS
        assert entry.colors.muted == DARK_COLORS.muted

    def test_light_user_theme_inherits_light_base(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.my-light]
label = "My Light"
dark = false
primary = "#0000FF"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        entry = builtins["my-light"]
        assert entry.dark is False
        assert entry.colors.primary == "#0000FF"
        assert entry.colors.muted == LIGHT_COLORS.muted

    def test_missing_label_skipped(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.bad]
dark = true
primary = "#FF0000"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "bad" not in builtins

    def test_missing_dark_skipped(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.bad]
label = "Bad Theme"
primary = "#FF0000"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "bad" not in builtins

    def test_invalid_hex_skipped(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.bad-hex]
label = "Bad Hex"
dark = true
primary = "not-a-color"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "bad-hex" not in builtins

    def test_builtin_name_shadowing_skipped(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.langchain]
label = "Fake LangChain"
dark = true
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "langchain" not in builtins

    def test_multiple_user_themes(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.alpha]
label = "Alpha"
dark = true
primary = "#111111"

[themes.beta]
label = "Beta"
dark = false
primary = "#222222"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "alpha" in builtins
        assert "beta" in builtins
        assert builtins["alpha"].colors.primary == "#111111"
        assert builtins["beta"].colors.primary == "#222222"

    def test_corrupt_toml_does_not_crash(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(config, "this is [[[not valid toml")
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert builtins == {}

    def test_non_table_themes_entry_skipped(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(config, 'themes = "not a table"\n')
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert builtins == {}


# ---------------------------------------------------------------------------
# _build_registry with user themes
# ---------------------------------------------------------------------------


class TestBuildRegistryWithUserThemes:
    """_build_registry() incorporates user themes from config."""

    def test_user_theme_in_registry(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.custom-dark]
label = "Custom Dark"
dark = true
primary = "#ABCDEF"
""",
        )
        registry = _build_registry(config_path=config)
        assert isinstance(registry, MappingProxyType)
        assert "custom-dark" in registry
        assert set(registry.keys()) >= _BUILTIN_NAMES
        assert registry["custom-dark"].colors.primary == "#ABCDEF"

    def test_no_config_still_has_builtins(self, tmp_path: Path) -> None:
        registry = _build_registry(config_path=tmp_path / "missing.toml")
        assert set(registry.keys()) == _BUILTIN_NAMES


# ---------------------------------------------------------------------------
# _BUILTIN_NAMES consistency
# ---------------------------------------------------------------------------


class TestBuiltinNamesConsistency:
    """_BUILTIN_NAMES stays in sync with _builtin_themes()."""

    def test_builtin_names_matches_builtin_themes(self) -> None:
        assert frozenset(_builtin_themes()) == _BUILTIN_NAMES


# ---------------------------------------------------------------------------
# Additional edge-case coverage
# ---------------------------------------------------------------------------


class TestLoadUserThemesEdgeCases:
    """Extra edge cases for _load_user_themes."""

    def test_whitespace_only_label_skipped(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.blank]
label = "   "
dark = true
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "blank" not in builtins

    def test_valid_theme_loads_despite_sibling_invalid(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.good]
label = "Good"
dark = true
primary = "#AABBCC"

[themes.bad]
dark = true
primary = "#FF0000"

[themes.also-good]
label = "Also Good"
dark = false
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "good" in builtins
        assert "also-good" in builtins
        assert "bad" not in builtins

    def test_non_string_color_value_uses_base_fallback(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.int-color]
label = "Int Color"
dark = true
primary = 123456
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "int-color" in builtins
        assert builtins["int-color"].colors.primary == DARK_COLORS.primary

    def test_unknown_color_field_ignored(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(
            config,
            """
[themes.typo]
label = "Typo Theme"
dark = true
primay = "#FF0000"
""",
        )
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert "typo" in builtins
        # Misspelled field ignored; primary stays at base
        assert builtins["typo"].colors.primary == DARK_COLORS.primary


# ---------------------------------------------------------------------------
# ThemeEntry.__post_init__ validation
# ---------------------------------------------------------------------------


class TestThemeEntryPostInit:
    """ThemeEntry validates label in __post_init__."""

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ThemeEntry(label="", dark=True, colors=DARK_COLORS)

    def test_whitespace_only_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ThemeEntry(label="   ", dark=True, colors=DARK_COLORS)

    def test_valid_label_accepted(self) -> None:
        entry = ThemeEntry(label="My Theme", dark=True, colors=DARK_COLORS)
        assert entry.label == "My Theme"


# ---------------------------------------------------------------------------
# save_theme_preference overwrite round-trip
# ---------------------------------------------------------------------------


class TestSaveThemePreferenceOverwrite:
    """save_theme_preference correctly overwrites an existing theme value."""

    def test_overwrite_existing_theme(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tomllib

        from deepagents_cli.app import save_theme_preference

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_cli.model_config.DEFAULT_CONFIG_PATH", config)

        # Save initial theme
        assert save_theme_preference("langchain") is True
        data = tomllib.loads(config.read_text())
        assert data["ui"]["theme"] == "langchain"

        # Overwrite with a different theme
        assert save_theme_preference("langchain-light") is True
        data = tomllib.loads(config.read_text())
        assert data["ui"]["theme"] == "langchain-light"
        # Old value should be replaced, not duplicated
        assert data["ui"]["theme"] == "langchain-light"


# ---------------------------------------------------------------------------
# ThemeSelectorScreen
# ---------------------------------------------------------------------------


def _register_lc_theme(app: object) -> None:
    """Register the LangChain theme on a test app so ThemeSelectorScreen works."""
    from textual.theme import Theme as TextualTheme

    c = DARK_COLORS
    app.register_theme(  # type: ignore[attr-defined]
        TextualTheme(
            name="langchain",
            primary=c.primary,
            secondary=c.secondary,
            accent=c.accent,
            foreground=c.foreground,
            background=c.background,
            surface=c.surface,
            panel=c.panel,
            warning=c.warning,
            error=c.error,
            success=c.success,
            dark=True,
        )
    )
    app.theme = "langchain"  # type: ignore[attr-defined]


class TestThemeSelectorScreen:
    """ThemeSelectorScreen widget tests."""

    async def test_compose_shows_all_registry_themes(self) -> None:
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()
            option_list = screen.query_one("#theme-options", OptionList)
            assert option_list.option_count == len(theme.ThemeEntry.REGISTRY)

    async def test_current_theme_highlighted(self) -> None:
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()
            option_list = screen.query_one("#theme-options", OptionList)
            assert option_list.highlighted is not None
            highlighted = option_list.get_option_at_index(option_list.highlighted)
            assert highlighted.id == "langchain"

    async def test_escape_restores_original_theme(self) -> None:
        from textual.app import App

        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen

        results: list[str | None] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)

            def on_result(result: str | None) -> None:
                results.append(result)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen, on_result)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.theme == "langchain"
            assert results == [None]

    async def test_enter_selects_theme(self) -> None:
        from textual.app import App

        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen

        results: list[str | None] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)

            def on_result(result: str | None) -> None:
                results.append(result)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen, on_result)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert len(results) == 1
            assert results[0] is not None
            assert results[0] in theme.ThemeEntry.REGISTRY
