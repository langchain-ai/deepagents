r"""Read-only in-app Debug Console modal.

Toggled with `Ctrl+\` (or the hidden `/debug` command), this overlay shows a
point-in-time session/runtime snapshot plus a live tail of recent
`deepagents_code.*` log records sourced from the in-memory ring buffer in
`_debug_buffer`. It never mutates session state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import RichLog, Static

from deepagents_code._debug_buffer import get_log_buffer
from deepagents_code.clipboard import copy_text_to_clipboard

if TYPE_CHECKING:
    from collections.abc import Sequence

    from textual.app import ComposeResult

DEBUG_TOGGLE_KEY = "ctrl+backslash"
r"""Textual key name for the `Ctrl+\` chord that toggles the console."""

_REFRESH_INTERVAL = 0.5
"""Seconds between log-tail refresh ticks."""


class DebugConsoleScreen(ModalScreen[None]):
    """Modal showing a session snapshot and a live tail of recent log records."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False, priority=True),
        Binding(DEBUG_TOGGLE_KEY, "close", "Close", show=False, priority=True),
        Binding("ctrl+l", "clear_view", "Clear view", show=False, priority=True),
        Binding("c", "copy", "Copy", show=False, priority=True),
    ]
    """Close, clear-the-view, and copy bindings (all `priority` for the modal)."""

    CSS = """
    DebugConsoleScreen {
        align: center middle;
    }

    DebugConsoleScreen > Vertical {
        width: 100;
        max-width: 95%;
        height: 85%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    DebugConsoleScreen .debug-console-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    DebugConsoleScreen .debug-console-snapshot {
        margin-bottom: 1;
    }

    DebugConsoleScreen .debug-console-log {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
        border: solid $primary-lighten-2;
    }

    DebugConsoleScreen .debug-console-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, snapshot: Sequence[tuple[str, str]]) -> None:
        """Initialize with a captured *snapshot* of session/runtime fields.

        Args:
            snapshot: Ordered `(label, value)` pairs rendered in the header.
        """
        super().__init__()
        self._snapshot = list(snapshot)
        # Absolute index of the next unrendered log line (incremental writes).
        self._rendered_upto = 0
        # Absolute index floor for Copy: lines retained since the last clear.
        self._view_floor = 0
        # One-shot guard so the "buffer unavailable" notice is written only once.
        self._missing_notice_shown = False

    def compose(self) -> ComposeResult:
        """Lay out the title, snapshot, log tail, and key-hint footer.

        Yields:
            The child widgets composing the console.
        """
        with Vertical():
            yield Static("Debug Console", classes="debug-console-title")
            yield Static(self._render_snapshot(), classes="debug-console-snapshot")
            yield RichLog(
                highlight=False,
                markup=False,
                wrap=True,
                auto_scroll=True,
                id="debug-log",
                classes="debug-console-log",
            )
            yield Static(self._render_help(), classes="debug-console-help")

    def on_mount(self) -> None:
        """Start the refresh timer and render the current buffer contents."""
        self.set_interval(_REFRESH_INTERVAL, self._poll_logs)
        self._poll_logs()

    def _render_snapshot(self) -> Content:
        """Build the right-aligned `label: value` snapshot block.

        Returns:
            The formatted snapshot block.
        """
        if not self._snapshot:
            return Content.styled("(no session data)", "dim italic")
        width = max(len(label) for label, _ in self._snapshot)
        lines = [
            Content.assemble((f"{label:>{width}}  ", "bold"), value)
            for label, value in self._snapshot
        ]
        return Content("\n").join(lines)

    @staticmethod
    def _render_help() -> Content:
        """Build the footer key-hint line.

        Returns:
            The formatted key-hint line.
        """
        return Content.styled(
            "Esc close \u00b7 Ctrl+L clear view \u00b7 c copy", "dim italic"
        )

    def _poll_logs(self) -> None:
        """Append any log lines emitted since the last tick to the log view."""
        log = self.query_one("#debug-log", RichLog)
        buffer = get_log_buffer()
        if buffer is None:
            if not self._missing_notice_shown:
                log.write("(log buffer unavailable)")
                self._missing_notice_shown = True
            return
        lines, total = buffer.snapshot_since(self._rendered_upto)
        for line in lines:
            log.write(line)
        self._rendered_upto = total

    def action_clear_view(self) -> None:
        """Clear the on-screen log view; the in-memory buffer keeps accruing."""
        self.query_one("#debug-log", RichLog).clear()
        buffer = get_log_buffer()
        if buffer is not None:
            total = buffer.total_emitted
            self._rendered_upto = total
            self._view_floor = total

    def action_copy(self) -> None:
        """Copy the log lines retained since the last clear to the clipboard."""
        buffer = get_log_buffer()
        lines = buffer.snapshot_since(self._view_floor)[0] if buffer is not None else []
        text = "\n".join(lines)
        if not text:
            self.app.notify(
                "No log lines to copy", severity="information", timeout=2, markup=False
            )
            return
        success, error = copy_text_to_clipboard(self.app, text)
        if success:
            self.app.notify(
                "Debug log copied", severity="information", timeout=2, markup=False
            )
            return
        suffix = f": {error}" if error else ""
        self.app.notify(
            f"Failed to copy debug log{suffix}",
            severity="warning",
            timeout=3,
            markup=False,
        )

    def action_close(self) -> None:
        """Close the debug console."""
        self.dismiss(None)
