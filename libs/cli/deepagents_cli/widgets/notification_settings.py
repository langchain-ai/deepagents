"""Notification settings screen for /notifications command."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import VerticalGroup
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

logger = logging.getLogger(__name__)

# Warning keys and their user-facing labels.
# Checked = warning is shown at startup (not suppressed). Unchecked = suppressed.
WARNING_TOGGLES: list[tuple[str, str]] = [
    ("ripgrep", "Warn when ripgrep is not installed"),
    ("tavily", "Warn when TAVILY_API_KEY is not set (web search)"),
]


class NotificationSettingsScreen(ModalScreen[None]):
    """Modal dialog for managing startup warning preferences.

    Each checkbox maps to a key in `[warnings].suppress` in
    `~/.deepagents/config.toml`. Toggling a checkbox immediately
    persists the change.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False),
        Binding("up", "app.focus_previous", "Previous", show=False, priority=True),
        Binding("down", "app.focus_next", "Next", show=False, priority=True),
        Binding("tab", "app.focus_next", "Next", show=False, priority=True),
        Binding(
            "shift+tab",
            "app.focus_previous",
            "Previous",
            show=False,
            priority=True,
        ),
    ]

    CSS = """
    NotificationSettingsScreen {
        align: center middle;
        background: transparent;
    }

    NotificationSettingsScreen > VerticalGroup {
        width: 65;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    NotificationSettingsScreen .ns-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    NotificationSettingsScreen .ns-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }

    NotificationSettingsScreen Checkbox {
        margin: 0;
        border: none;
        &:focus {
            border: none;
        }
    }
    """

    def __init__(self, suppressed: set[str]) -> None:
        """Initialize the notification settings screen.

        Args:
            suppressed: Set of currently suppressed warning keys.
        """
        super().__init__()
        self._suppressed = suppressed

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the notification settings UI.
        """
        glyphs = get_glyphs()
        with VerticalGroup():
            yield Static("Notification Settings", classes="ns-title")
            for key, label in WARNING_TOGGLES:
                yield Checkbox(
                    label,
                    value=key not in self._suppressed,
                    id=f"ns-{key}",
                )
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab navigate"
                f" {glyphs.bullet} Space toggle"
                f" {glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="ns-help")

    def on_mount(self) -> None:
        """Apply ASCII border if needed."""
        if is_ascii_mode():
            container = self.query_one(VerticalGroup)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Persist warning suppression toggle to config.toml on change."""
        event.stop()
        checkbox_id = event.checkbox.id
        if not checkbox_id or not checkbox_id.startswith("ns-"):
            return
        key = checkbox_id.removeprefix("ns-")
        enabled = event.value

        async def _persist() -> None:
            from deepagents_cli.model_config import (
                suppress_warning,
                unsuppress_warning,
            )

            try:
                if enabled:
                    ok = await asyncio.to_thread(unsuppress_warning, key)
                else:
                    ok = await asyncio.to_thread(suppress_warning, key)
            except Exception:
                logger.warning(
                    "Failed to persist notification setting for %r",
                    key,
                    exc_info=True,
                )
                ok = False
            if not ok:
                self.app.notify(
                    "Could not save notification preference. "
                    "Check file permissions for ~/.deepagents/config.toml.",
                    severity="warning",
                    timeout=6,
                    markup=False,
                )

        self.call_later(_persist)

    def action_cancel(self) -> None:
        """Close the screen."""
        self.dismiss(None)
