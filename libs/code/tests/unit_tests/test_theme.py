"""Tests for deepagents_code.theme module."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any, cast

import pytest

from deepagents_code._env_vars import THEME

if TYPE_CHECKING:
    from pathlib import Path

    from _typeshed import SupportsRead

from deepagents_code import theme
from deepagents_code.theme import (
    DARK_COLORS,
    DEFAULT_THEME,
    LIGHT_COLORS,
    ThemeColors,
    ThemeEntry,
    _load_user_themes,
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


# ---------------------------------------------------------------------------
# Pre-built color sets
# ---------------------------------------------------------------------------


class TestColorSets:
    """DARK_COLORS and LIGHT_COLORS are valid ThemeColors instances."""


# ---------------------------------------------------------------------------
# Theme registry
# ---------------------------------------------------------------------------


class TestThemeEntryRegistry:
    """Theme registry contents and immutability."""


# ---------------------------------------------------------------------------
# get_css_variable_defaults
# ---------------------------------------------------------------------------


EXPECTED_CSS_KEYS = frozenset(
    {
        "mode-bash",
        "mode-command",
        "mode-incognito",
        "skill",
        "skill-hover",
        "tool",
        "tool-hover",
    }
)


class TestGetCssVariableDefaults:
    """get_css_variable_defaults() return values."""


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


# ---------------------------------------------------------------------------
# get_theme_colors
# ---------------------------------------------------------------------------


class TestGetThemeColors:
    """get_theme_colors() returns the correct ThemeColors."""


# ---------------------------------------------------------------------------
# _load_theme_preference / save_theme_preference
# ---------------------------------------------------------------------------


class TestResolveThemeName:
    """Direct unit tests for the shared theme-name resolver."""


class TestLoadTerminalDefault:
    """Direct unit tests for `_load_terminal_default`."""

    def test_returns_none_and_logs_on_corrupt_toml(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Corrupt config returns `None` and logs a warning so the user can debug."""
        from deepagents_code.app import _load_terminal_default

        config = tmp_path / "config.toml"
        config.write_text("this is not valid toml [[[")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        with caplog.at_level("WARNING", logger="deepagents_code.app"):
            assert _load_terminal_default() is None
        assert any(
            "terminal theme default" in record.getMessage() for record in caplog.records
        )


class TestLoadThemePreference:
    """_load_theme_preference reads config.toml correctly."""

    def test_returns_default_for_corrupt_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code.app import _load_theme_preference

        config = tmp_path / "config.toml"
        config.write_text("this is not valid toml [[[")
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        assert _load_theme_preference() == DEFAULT_THEME


class TestTerminalThemeMapping:
    """_load_theme_preference respects [ui.terminal_themes] mapping."""

    def test_terminal_mapping_migrates_legacy_textual_ansi(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy `textual-ansi` is migrated to `ansi-light` via the mapping.

        Mirrors the migration in the saved-theme branch (pre-Textual 8.2.5).
        """
        from deepagents_code.app import _load_theme_preference

        config = tmp_path / "config.toml"
        config.write_text('[ui.terminal_themes]\n"Apple_Terminal" = "textual-ansi"\n')
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        monkeypatch.delenv(THEME, raising=False)

        assert _load_theme_preference() == "ansi-light"


class TestSaveThemePreference:
    """save_theme_preference writes config.toml correctly."""

    def test_returns_false_on_write_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code.app import save_theme_preference

        # Point to a directory that doesn't exist and can't be created
        config = tmp_path / "readonly" / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
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


# ---------------------------------------------------------------------------
# _load_user_themes
# ---------------------------------------------------------------------------


def _write_config(path: Path, content: str) -> None:
    """Write TOML content to a config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLoadUserThemes:
    """_load_user_themes reads [themes.*] from config.toml."""

    def test_corrupt_toml_does_not_crash(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        _write_config(config, "this is [[[not valid toml")
        builtins: dict[str, ThemeEntry] = {}
        _load_user_themes(builtins, config_path=config)
        assert builtins == {}


# ---------------------------------------------------------------------------
# _build_registry with user themes
# ---------------------------------------------------------------------------


class TestBuildRegistryWithUserThemes:
    """_build_registry() incorporates user themes from config."""


# ---------------------------------------------------------------------------
# _builtin_names() consistency
# ---------------------------------------------------------------------------


class TestBuiltinNamesConsistency:
    """_builtin_names() stays in sync with _builtin_themes()."""


# ---------------------------------------------------------------------------
# Additional edge-case coverage
# ---------------------------------------------------------------------------


class TestLoadUserThemesEdgeCases:
    """Extra edge cases for _load_user_themes."""


# ---------------------------------------------------------------------------
# ThemeEntry.__post_init__ validation
# ---------------------------------------------------------------------------


class TestThemeEntryPostInit:
    """ThemeEntry validates label in __post_init__."""


# ---------------------------------------------------------------------------
# save_theme_preference overwrite round-trip
# ---------------------------------------------------------------------------


class TestSaveThemePreferenceOverwrite:
    """save_theme_preference correctly overwrites an existing theme value."""


class TestSaveTerminalThemeMapping:
    """save_terminal_theme_mapping writes [ui.terminal_themes] correctly."""

    def test_repairs_non_dict_terminal_themes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A scalar `terminal_themes` value is replaced with a fresh table.

        We can't merge into a malformed value, so the user's mistake is
        overwritten — the saved-by-the-CLI invariant trumps preserving it.
        The discarded value is logged so it remains recoverable.
        """
        import tomllib

        from deepagents_code.app import save_terminal_theme_mapping

        config = tmp_path / "config.toml"
        config.write_text('[ui]\nterminal_themes = "junk"\n')
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        with caplog.at_level("WARNING", logger="deepagents_code.app"):
            assert save_terminal_theme_mapping("Apple_Terminal", "langchain") is True
        data = tomllib.loads(config.read_text())
        assert isinstance(data["ui"]["terminal_themes"], dict)
        assert data["ui"]["terminal_themes"]["Apple_Terminal"] == "langchain"
        assert any(
            "junk" in record.getMessage() and "replacing" in record.getMessage()
            for record in caplog.records
        )

    def test_concurrent_global_and_terminal_writes_preserve_both_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overlapping config saves keep both read-modify-write mutations."""
        import contextlib
        import threading
        import tomllib

        from deepagents_code.app import (
            save_terminal_theme_mapping,
            save_theme_preference,
        )

        config = tmp_path / "config.toml"
        config.write_text('[model]\nname = "existing"\n')
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)

        original_load = tomllib.load
        barrier = threading.Barrier(2)

        def slow_load(fp: SupportsRead[bytes]) -> dict[str, object]:
            data = original_load(fp)
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=0.2)
            return data

        monkeypatch.setattr(tomllib, "load", slow_load)

        results: list[bool] = []

        def save_global() -> None:
            results.append(save_theme_preference("langchain-light"))

        def save_terminal() -> None:
            results.append(save_terminal_theme_mapping("Apple_Terminal", "langchain"))

        threads = [
            threading.Thread(target=save_global),
            threading.Thread(target=save_terminal),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        data = tomllib.loads(config.read_text())
        assert results == [True, True]
        assert data["ui"]["theme"] == "langchain-light"
        assert data["ui"]["terminal_themes"]["Apple_Terminal"] == "langchain"

    def test_returns_false_on_atomic_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clean up the temp file and return `False` on mid-write failure.

        If `tomli_w.dump` raises, the temp file is unlinked. A stray
        `.tmp` file in `~/.deepagents/` after a crash would be hard to
        diagnose.
        """
        from deepagents_code.app import save_terminal_theme_mapping

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)

        def _boom(*_args: object, **_kwargs: object) -> None:
            msg = "simulated dump failure"
            raise OSError(msg)

        monkeypatch.setattr("tomli_w.dump", _boom)
        assert save_terminal_theme_mapping("Apple_Terminal", "langchain") is False
        # Temp file (`*.tmp` sibling of the target) must not be left behind.
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# ThemeSelectorScreen
# ---------------------------------------------------------------------------


def _register_lc_theme(app: object) -> None:
    """Register the LangChain theme on a test app so ThemeSelectorScreen works."""
    app_any = cast("Any", app)
    from textual.theme import Theme as TextualTheme

    c = DARK_COLORS
    app.register_theme(  # ty: ignore
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
    app_any.theme = "langchain"


def _register_lc_light_theme(app: object) -> None:
    """Register the light LangChain theme for preview tests."""
    from textual.theme import Theme as TextualTheme

    c = LIGHT_COLORS
    app.register_theme(  # ty: ignore
        TextualTheme(
            name="langchain-light",
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
            dark=False,
        )
    )


class TestThemeSelectorScreen:
    """ThemeSelectorScreen widget tests."""

    async def test_escape_restores_original_theme(self) -> None:
        from textual.app import App

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

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

    async def test_escape_syncs_terminal_background_after_restore(self) -> None:
        from textual.app import App
        from textual.theme import Theme as TextualTheme

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        class SyncApp(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.synced: list[str] = []

            def sync_terminal_background(self) -> None:
                self.synced.append(self.theme)

        app = SyncApp()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            c = theme.LIGHT_COLORS
            app.register_theme(
                TextualTheme(
                    name="langchain-light",
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
                    dark=False,
                )
            )
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            assert app.theme == "langchain"
            assert app.synced[-1] == "langchain"

    async def test_preview_failure_syncs_terminal_background_after_rollback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from textual.app import App
        from textual.theme import Theme as TextualTheme

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        class SyncApp(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.synced: list[str] = []

            def sync_terminal_background(self) -> None:
                self.synced.append(self.theme)
                if self.theme != "langchain":
                    msg = "preview sync failed"
                    raise RuntimeError(msg)

        app = SyncApp()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            c = theme.LIGHT_COLORS
            app.register_theme(
                TextualTheme(
                    name="langchain-light",
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
                    dark=False,
                )
            )
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            with caplog.at_level(
                logging.WARNING, logger="deepagents_code.tui.widgets.theme_selector"
            ):
                await pilot.press("down")
                await pilot.pause()

            target = next(key for key in theme.get_registry() if key != "langchain")
            assert app.synced[-2:] == [target, "langchain"]
            assert app.theme == "langchain"
            assert "Failed to preview theme" in caplog.text

    async def test_t_writes_terminal_mapping_for_current_term_program(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`t` persists the highlighted theme to `[ui.terminal_themes]` only.

        Leaves the picker open so the user can confirm and keep browsing.
        `[ui].theme` is intentionally untouched — pressing `t` is "save for
        this terminal", not "save as my global default". The shared
        shared `model_config._config_write_lock` prevents racing the parent's
        Enter writer.
        """
        import tomllib

        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        results: list[str | None] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)

            def on_result(result: str | None) -> None:
                results.append(result)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen, on_result)
            await pilot.pause()

            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Picker stays open after `t`.
            assert results == []
            assert app.screen is screen

        data = tomllib.loads(config.read_text())
        assert data["ui"]["terminal_themes"]["Apple_Terminal"] == "langchain"
        # `[ui].theme` must NOT be written by the `t` action — otherwise we'd
        # race the parent's save-theme-preference path.
        assert "theme" not in data.get("ui", {})

    async def test_t_persists_moved_cursor_not_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`t` saves the *highlighted* theme, not the originally-current one."""
        import tomllib

        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            registry_keys = list(theme.get_registry())
            target_key = next(key for key in registry_keys if key != "langchain")
            target_index = registry_keys.index(target_key)

            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = target_index
            await pilot.pause()
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

        data = tomllib.loads(config.read_text())
        assert data["ui"]["terminal_themes"]["Apple_Terminal"] == target_key

    async def test_t_notifies_on_save_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warn the user when `save_terminal_theme_mapping` returns `False`.

        Surfaces a warning toast pointing at the log location instead of a
        silent no-op.
        """
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.app import _ConfigWriteResult
        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        # Force the writer to report failure without actually raising —
        # exercises the `if ok:` else-branch in `_persist`.
        monkeypatch.setattr(
            "deepagents_code.app._save_terminal_theme_mapping_result",
            lambda *_args, **_kwargs: _ConfigWriteResult(
                False, "Could not save terminal mapping.", "error"
            ),
        )

        notifications: list[tuple[str, str]] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)

            def _capture(
                message: str, *, severity: str = "information", **_kwargs: object
            ) -> None:
                notifications.append((severity, message))

            monkeypatch.setattr(app, "notify", _capture)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            target_key = "langchain-light"
            target_index = list(theme.get_registry()).index(target_key)
            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = target_index
            await pilot.pause()

            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert any(
            sev == "error" and "terminal mapping" in msg for sev, msg in notifications
        )
        assert app.theme == "langchain"

    async def test_t_notifies_on_save_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surface the exception class name in a toast on raise.

        A generic 'check logs' message hides the failure mode; the class
        name (e.g., `OSError`) tells the user where to look.
        """
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        def _boom(*_args: object, **_kwargs: object) -> None:
            msg = "simulated"
            raise OSError(msg)

        monkeypatch.setattr(
            "deepagents_code.app._save_terminal_theme_mapping_result", _boom
        )

        notifications: list[tuple[str, str]] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)

            def _capture(
                message: str, *, severity: str = "information", **_kwargs: object
            ) -> None:
                notifications.append((severity, message))

            monkeypatch.setattr(app, "notify", _capture)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            target_key = "langchain-light"
            target_index = list(theme.get_registry()).index(target_key)
            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = target_index
            await pilot.pause()

            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert any(sev == "error" and "OSError" in msg for sev, msg in notifications)
        assert app.theme == "langchain"

    async def test_escape_keeps_theme_set_for_terminal_this_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Esc keeps a theme set with `t` instead of reverting to the original.

        Pressing `t` is a deliberate choice, so Esc should preserve it rather
        than rolling back to the theme that was active when the picker opened.
        """
        from textual.app import App
        from textual.theme import Theme as TextualTheme
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        results: list[str | None] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            c = theme.LIGHT_COLORS
            app.register_theme(
                TextualTheme(
                    name="langchain-light",
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
                    dark=False,
                )
            )

            def on_result(result: str | None) -> None:
                results.append(result)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen, on_result)
            await pilot.pause()

            target_key = "langchain-light"
            target_index = list(theme.get_registry()).index(target_key)

            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = target_index
            await pilot.pause()
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

        assert app.theme == target_key
        assert results == [None]

    async def test_escape_restores_original_when_no_terminal_default_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a `t` press, Esc still reverts the previewed theme."""
        from textual.app import App

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert app.theme == "langchain"

    async def test_escape_keeps_last_terminal_default_after_multiple_t(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The last successful `t` wins; Esc keeps it over earlier saves.

        Pressing `t` on a second theme supersedes the first, and a later
        preview (without `t`) does not override the saved default — Esc
        restores the most recently saved per-terminal theme.
        """
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)
            # `textual-light` is a Textual builtin, registered on every App and
            # present in the deepagents registry — usable as a second target.
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            registry = list(theme.get_registry())
            option_list = screen.query_one("#theme-options", OptionList)

            # First `t`: save langchain-light.
            option_list.highlighted = registry.index("langchain-light")
            await pilot.pause()
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Second `t`: save textual-light, superseding the first.
            option_list.highlighted = registry.index("textual-light")
            await pilot.pause()
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Preview the original (no `t`) — this must not win on Esc.
            option_list.highlighted = registry.index("langchain")
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

        assert app.theme == "textual-light"

    async def test_escape_keeps_terminal_default_over_later_preview(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Esc restores the `t`-saved theme even after previewing another.

        After `t` saves a default, browsing to a different theme live-previews
        it; Esc must roll back to the saved default, not the last preview.
        """
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            registry = list(theme.get_registry())
            option_list = screen.query_one("#theme-options", OptionList)

            option_list.highlighted = registry.index("langchain-light")
            await pilot.pause()
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Preview a different, registered theme without pressing `t`.
            option_list.highlighted = registry.index("textual-light")
            await pilot.pause()
            assert app.theme == "textual-light"

            await pilot.press("escape")
            await pilot.pause()

        assert app.theme == "langchain-light"

    async def test_escape_restores_original_when_t_with_term_program_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`t` no-ops without `TERM_PROGRAM`, so Esc reverts the preview.

        The early return in `action_set_for_terminal` never arms the
        keep-on-Esc behavior, so Esc restores the original theme and nothing
        is persisted.
        """
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            registry = list(theme.get_registry())
            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = registry.index("langchain-light")
            await pilot.pause()
            assert app.theme == "langchain-light"

            await pilot.press("t")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert app.theme == "langchain"
        assert not config.exists()

    async def test_escape_dismisses_when_kept_theme_unregistered_midsession(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Esc never traps the user even if the kept theme can't be applied.

        If the saved per-terminal theme is unregistered after `t` (so
        reapplying it raises `InvalidThemeError`), `action_cancel` logs and
        still dismisses rather than leaving the modal stuck open.
        """
        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        results: list[str | None] = []

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)

            def on_result(result: str | None) -> None:
                results.append(result)

            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen, on_result)
            await pilot.pause()

            registry = list(theme.get_registry())
            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = registry.index("langchain-light")
            await pilot.pause()
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Move off the saved theme, then pull it out from under Esc.
            option_list.highlighted = registry.index("langchain")
            await pilot.pause()
            app.unregister_theme("langchain-light")

            await pilot.press("escape")
            await pilot.pause()

        # The modal dismissed (no exception bubbled, user not trapped).
        assert results == [None]

    async def test_escape_keeps_theme_when_save_still_in_flight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Esc keeps the `t` theme even if the config write hasn't returned.

        Regression test for the race where the per-terminal save is slow
        (disk/lock contention): pressing `t` then Esc before the write
        completes must still keep the chosen theme, because the choice is
        recorded synchronously rather than only after the worker resolves.
        """
        import asyncio
        import threading

        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.app import _ConfigWriteResult
        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        started = threading.Event()
        release = threading.Event()

        def _slow_save(*_args: object, **_kwargs: object) -> _ConfigWriteResult:
            started.set()
            release.wait(timeout=1)
            return _ConfigWriteResult(True)

        monkeypatch.setattr(
            "deepagents_code.app._save_terminal_theme_mapping_result", _slow_save
        )

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = list(theme.get_registry()).index(
                "langchain-light"
            )
            await pilot.pause()

            await pilot.press("t")
            # Wait until the worker is blocked inside the (still-unfinished) save.
            assert await asyncio.to_thread(started.wait, 1)

            # Cancel while the write is in flight — must keep, not revert.
            await pilot.press("escape")
            await pilot.pause()
            assert app.theme == "langchain-light"

            # Let the still-blocked save finish. The screen is already
            # dismissed, so suppress the badge rerender (its option list is
            # gone) to mirror real mid-write teardown before draining workers.
            screen._is_mounted = False
            release.set()
            await app.workers.wait_for_complete()

        assert app.theme == "langchain-light"

    @pytest.mark.parametrize("failure_mode", ["status", "exception"])
    async def test_escape_restores_theme_when_in_flight_save_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
    ) -> None:
        """Esc reverts once a terminal-default save kept by cancel fails."""
        import asyncio
        import threading

        from textual.app import App
        from textual.widgets import OptionList

        from deepagents_code.app import _ConfigWriteResult
        from deepagents_code.tui.widgets.theme_selector import ThemeSelectorScreen

        config = tmp_path / "config.toml"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        started = threading.Event()
        release = threading.Event()

        def _slow_failing_save(*_args: object, **_kwargs: object) -> _ConfigWriteResult:
            started.set()
            release.wait(timeout=1)
            if failure_mode == "exception":
                msg = "simulated"
                raise OSError(msg)
            return _ConfigWriteResult(
                False, "Could not save terminal mapping.", "error"
            )

        monkeypatch.setattr(
            "deepagents_code.app._save_terminal_theme_mapping_result",
            _slow_failing_save,
        )

        app = App()
        async with app.run_test() as pilot:
            _register_lc_theme(app)
            _register_lc_light_theme(app)
            screen = ThemeSelectorScreen(current_theme="langchain")
            app.push_screen(screen)
            await pilot.pause()

            option_list = screen.query_one("#theme-options", OptionList)
            option_list.highlighted = list(theme.get_registry()).index(
                "langchain-light"
            )
            await pilot.pause()

            await pilot.press("t")
            assert await asyncio.to_thread(started.wait, 1)

            await pilot.press("escape")
            await pilot.pause()
            assert app.theme == "langchain-light"

            release.set()
            await app.workers.wait_for_complete()

        assert app.theme == "langchain"
