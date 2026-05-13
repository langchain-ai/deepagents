"""Interactive theme selector screen for /theme command."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

logger = logging.getLogger(__name__)


@runtime_checkable
class _TerminalBackgroundSyncApp(Protocol):
    """App protocol for terminal background sync after theme preview changes."""

    def sync_terminal_background(self) -> None:
        """Sync the terminal background to the active theme."""


class ThemeSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for theme selection with live preview.

    Displays available themes in an `OptionList`. Navigating the option list
    applies a live preview by swapping the app theme. Returns the selected
    theme name on Enter, or `None` on Esc (restoring the original theme).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "cursor_down", "Next", show=False, priority=True),
        Binding("shift+tab", "cursor_up", "Previous", show=False, priority=True),
        Binding("n", "toggle_names", "Names", show=False),
        Binding("t", "set_for_terminal", "Set for terminal", show=False),
    ]
    """Key bindings for the selector.

    Esc dismisses and restores the original theme. Arrow keys and Enter are
    handled natively by the embedded `OptionList`; Tab / Shift+Tab are bound
    here to advance the option list cursor for consistency with other
    selector screens (where Tab cycles focus across multiple widgets).
    `action_toggle_names` toggles between human-readable labels and canonical
    registry keys, which are accepted by the theme config. The terminal-default
    action saves the highlighted theme for the current terminal and updates
    the `(default)` badge in place without closing the picker.
    """

    CSS = """
    ThemeSelectorScreen {
        align: center middle;
        background: transparent;
    }

    ThemeSelectorScreen > Vertical {
        width: 50;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ThemeSelectorScreen .theme-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ThemeSelectorScreen OptionList {
        height: auto;
        max-height: 16;
        background: $background;
    }

    ThemeSelectorScreen .theme-selector-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """
    """Styling for the centered modal shell, title, option list, and help footer."""

    def __init__(self, current_theme: str, terminal_default: str | None = None) -> None:
        """Initialize the ThemeSelectorScreen.

        Args:
            current_theme: The currently active theme name (to highlight).
            terminal_default: The theme saved in `[ui.terminal_themes]` for
                the current `TERM_PROGRAM`, if any. Badged with `(default)`
                in the option list.
        """
        super().__init__()
        self._current_theme = current_theme
        self._original_theme = current_theme
        self._terminal_default = terminal_default
        self._show_keys = False

    def _sync_terminal_background(self) -> None:
        """Ask the app to sync terminal background after preview changes."""
        if isinstance(self.app, _TerminalBackgroundSyncApp):
            self.app.sync_terminal_background()

    def _format_option(self, name: str, entry: theme.ThemeEntry) -> str:
        """Render the option text for a theme entry.

        Args:
            name: Registry key.
            entry: Registry entry.

        Returns:
            Either the human label or the registry key, with `(current)`
                and/or `(default)` suffixes — combined as
                `(current, default)` when both apply to the same theme.
        """
        text = name if self._show_keys else entry.label
        suffixes: list[str] = []
        if name == self._current_theme:
            suffixes.append("current")
        if name == self._terminal_default:
            suffixes.append("default")
        if suffixes:
            text = f"{text} ({', '.join(suffixes)})"
        return text

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the theme selector UI.
        """
        glyphs = get_glyphs()
        options: list[Option] = []
        highlight_index = 0

        for i, (name, entry) in enumerate(theme.get_registry().items()):
            options.append(Option(self._format_option(name, entry), id=name))
            if name == self._current_theme:
                highlight_index = i

        with Vertical():
            yield Static("Select Theme", classes="theme-selector-title")
            option_list = OptionList(*options, id="theme-options")
            option_list.highlighted = highlight_index
            yield option_list
            nav_line = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab switch"
                f" {glyphs.bullet} Enter select"
                f" {glyphs.bullet} Esc cancel"
            )
            action_line = f"N labels/keys  {glyphs.bullet}  T set for this terminal"
            yield Static(f"{nav_line}\n{action_line}", classes="theme-selector-help")

    def on_mount(self) -> None:
        """Apply ASCII border if needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """Live-preview the highlighted theme.

        Args:
            event: The option highlighted event.
        """
        name = event.option.id
        if name is not None and name in theme.get_registry():
            try:
                self.app.theme = name
                self._sync_terminal_background()
                # refresh_css only repaints the active (modal) screen's layout;
                # force the screen beneath us to repaint so the user sees the
                # preview through the transparent scrim.
                stack = self.app.screen_stack
                if len(stack) > 1:
                    stack[-2].refresh(layout=True)
            except Exception:
                logger.warning("Failed to preview theme '%s'", name, exc_info=True)
                try:
                    self.app.theme = self._original_theme
                    self._sync_terminal_background()
                except Exception:
                    logger.warning(
                        "Failed to restore original theme '%s'",
                        self._original_theme,
                        exc_info=True,
                    )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Commit the selected theme.

        Args:
            event: The option selected event.
        """
        name = event.option.id
        if name is not None and name in theme.get_registry():
            self.dismiss(name)
        else:
            logger.warning("Selected theme '%s' is no longer available", name)
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Restore the original theme and dismiss."""
        self.app.theme = self._original_theme
        self._sync_terminal_background()
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move the option list cursor down (Tab)."""
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option list cursor up (Shift+Tab)."""
        self.query_one(OptionList).action_cursor_up()

    def action_set_for_terminal(self) -> None:
        """Persist the highlighted theme as the default for `TERM_PROGRAM`.

        Writes `[ui.terminal_themes][TERM_PROGRAM] = name` and updates the
        `(default)` badge in the option list without closing the picker, so
        the user can confirm the change and keep browsing. `[ui].theme` is
        intentionally not touched because this action saves only the current
        terminal default. Config writes are serialized in `app.py`, so
        overlapping global-theme and per-terminal-theme saves cannot clobber
        each other's keys.

        No-ops with a warning toast if `TERM_PROGRAM` is unset, or silently
        if the option list has no highlighted entry / the highlighted id
        isn't a registered theme.
        """
        term_program = os.environ.get("TERM_PROGRAM", "").strip()
        if not term_program:
            self.app.notify(
                "TERM_PROGRAM is unset; can't set a per-terminal default. "
                "Set the [ui].theme directly with Enter.",
                severity="warning",
                markup=False,
                timeout=6,
            )
            return

        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            logger.warning("action_set_for_terminal invoked with no highlighted option")
            return
        option = option_list.get_option_at_index(option_list.highlighted)
        name = option.id
        if name is None or name not in theme.get_registry():
            logger.warning(
                "action_set_for_terminal got unregistered option id '%s'", name
            )
            return

        async def _persist() -> None:
            try:
                from deepagents_cli.app import _save_terminal_theme_mapping_result

                status = await asyncio.to_thread(
                    _save_terminal_theme_mapping_result, term_program, name
                )
            except Exception as exc:
                logger.exception("Failed to persist terminal theme mapping")
                self.app.notify(
                    f"Could not save terminal mapping ({type(exc).__name__}).",
                    severity="error",
                    markup=False,
                    timeout=6,
                )
                return
            if not status.ok:
                self.app.notify(
                    status.message or "Could not save terminal mapping.",
                    severity=status.severity,
                    markup=False,
                    timeout=6,
                )
                return
            if status.message is not None:
                self.app.notify(
                    status.message,
                    severity=status.severity,
                    markup=False,
                    timeout=6,
                )
            # Update the badge in place if the screen is still mounted.
            # The user may have dismissed the picker (Esc/Enter) while the
            # write was in flight; `is_mounted` guards the widget tree.
            if self.is_mounted:
                self._terminal_default = name
                self._rerender_options()
            self.app.notify(
                f"Set '{name}' as the default for {term_program}.",
                severity="information",
                markup=False,
                timeout=4,
            )

        # Anchor the worker on the app, not this screen — if the user
        # dismisses the picker mid-flight, the screen tears down its own
        # workers but the write should still complete and toast.
        self.app.run_worker(_persist(), exclusive=False)

    def action_toggle_names(self) -> None:
        """Toggle between human labels and registry keys in the option list.

        Useful for copying the canonical key into `[ui.terminal_themes]` or
        `[ui].theme` without leaving the picker.
        """
        self._show_keys = not self._show_keys
        self._rerender_options()

    def _rerender_options(self) -> None:
        """Rebuild the option list, preserving the cursor position.

        Used when the badge text or label/key mode changes — Textual's
        `OptionList` doesn't expose a way to mutate a rendered prompt, so
        we recreate the options.
        """
        option_list = self.query_one(OptionList)
        cursor = option_list.highlighted
        registry = theme.get_registry()
        new_options = [
            Option(self._format_option(name, entry), id=name)
            for name, entry in registry.items()
        ]
        option_list.clear_options()
        option_list.add_options(new_options)
        if cursor is not None:
            option_list.highlighted = cursor
