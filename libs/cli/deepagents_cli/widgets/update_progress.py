"""Progress modal for CLI self-update installs."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Log, Static

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.timer import Timer

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode
from deepagents_cli.widgets.loading import Spinner


class UpdateProgressScreen(ModalScreen[None]):
    """Modal that shows self-update progress and a bounded log tail."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("d", "toggle_details", "Details", show=False),
        Binding("c", "copy_log_path", "Copy log path", show=False),
        Binding("escape", "cancel", "Close", show=False),
    ]

    CSS = """
    UpdateProgressScreen {
        align: center middle;
    }

    UpdateProgressScreen > Vertical {
        width: 78;
        max-width: 92%;
        height: auto;
        max-height: 85%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    UpdateProgressScreen .up-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    UpdateProgressScreen .up-status {
        color: $text;
    }

    UpdateProgressScreen .up-status-row {
        height: auto;
        margin-bottom: 1;
    }

    UpdateProgressScreen .up-spinner {
        width: auto;
        color: $primary;
        margin-right: 1;
    }

    UpdateProgressScreen .up-details,
    UpdateProgressScreen .up-log,
    UpdateProgressScreen Log.up-tail,
    UpdateProgressScreen .up-help {
        color: $text-muted;
    }

    UpdateProgressScreen .up-details {
        margin-top: 1;
        margin-bottom: 1;
    }

    UpdateProgressScreen Log.up-tail {
        height: 10;
        min-height: 10;
        max-height: 10;
        background: $surface-lighten-1;
        border: solid $surface-lighten-1;
        overflow-x: hidden;
    }

    UpdateProgressScreen .up-log {
        margin-top: 1;
        margin-bottom: 1;
    }

    UpdateProgressScreen .up-help {
        height: 1;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        latest: str,
        command: str,
        log_path: Path,
        tail_limit: int = 30,
    ) -> None:
        """Initialize the progress modal.

        Args:
            latest: Version being installed.
            command: Upgrade command being run.
            log_path: Persisted log file path.
            tail_limit: Maximum output lines to keep in memory.
        """
        super().__init__()
        self._latest = latest
        self._command = command
        self._log_path = log_path
        self._tail_limit = tail_limit
        self._tail: deque[str] = deque(maxlen=tail_limit)
        self._details_visible = False
        self._done = False
        self._status = f"Installing v{latest}..."
        self._status_widget: Static | None = None
        self._spinner = Spinner()
        self._spinner_widget: Static | None = None
        self._spinner_timer: Timer | None = None
        self._command_widget: Static | None = None
        self._log_path_widget: Static | None = None
        self._tail_widget: Log | None = None
        self._help_widget: Static | None = None

    def compose(self) -> ComposeResult:
        """Compose the modal.

        Yields:
            Static widgets for status, command, output tail, log path, and help.
        """
        with Vertical():
            yield Static("Updating Deep Agents CLI", classes="up-title")
            with Horizontal(classes="up-status-row"):
                self._spinner_widget = Static(
                    self._spinner.current_frame(),
                    classes="up-spinner",
                    markup=False,
                )
                yield self._spinner_widget
                self._status_widget = Static(
                    self._status, classes="up-status", markup=False
                )
                yield self._status_widget
            self._command_widget = Static(
                f"Running command: {self._command}",
                classes="up-details",
                markup=False,
            )
            self._command_widget.display = False
            yield self._command_widget
            self._tail_widget = Log(
                highlight=False,
                max_lines=self._tail_limit,
                auto_scroll=True,
                classes="up-tail",
            )
            self._tail_widget.display = False
            yield self._tail_widget
            self._log_path_widget = Static(
                f"Log: {self._log_path}",
                classes="up-log",
                markup=False,
            )
            self._log_path_widget.display = False
            yield self._log_path_widget
            self._help_widget = Static(self._help_text(), classes="up-help")
            yield self._help_widget

    def on_mount(self) -> None:
        """Apply ASCII border when configured."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.primary)
        self._spinner_timer = self.set_interval(0.1, self._update_spinner)

    def append_line(self, line: str) -> None:
        """Append a command output line to the in-memory tail."""
        self._tail.append(line)
        if self._tail_widget is not None:
            self._tail_widget.write_line(line)

    def mark_success(self) -> None:
        """Render the completed-success state."""
        self._done = True
        self._status = f"Update complete. Restart the CLI to use v{self._latest}."
        self._stop_spinner_timer()
        self._refresh_status()

    def mark_failure(self, command: str) -> None:
        """Render the completed-failure state.

        Args:
            command: Manual command users can run to retry.
        """
        self._done = True
        self._status = f"Update failed. Try manually: {command}"
        self._details_visible = True
        self._stop_spinner_timer()
        self._refresh_status()
        self._apply_details_visibility()

    def action_toggle_details(self) -> None:
        """Show or hide the live log tail."""
        self._details_visible = not self._details_visible
        self._apply_details_visibility()

    def action_cancel(self) -> None:
        """Close the modal only after the update command has finished."""
        if self._done:
            self.dismiss(None)

    def action_copy_log_path(self) -> None:
        """Copy the persisted log path when details are visible."""
        if not self._details_visible:
            return
        self.app.copy_to_clipboard(str(self._log_path))
        self.app.notify(
            "Copied log path.",
            severity="information",
            timeout=3,
            markup=False,
        )

    def _refresh_status(self) -> None:
        if self._status_widget is not None:
            self._status_widget.update(self._status)
        if self._spinner_widget is not None:
            if self._done:
                self._spinner_widget.update(get_glyphs().checkmark)
                self._spinner_widget.display = True
            else:
                self._spinner_widget.display = True
        if self._help_widget is not None:
            self._help_widget.update(self._help_text())

    def _apply_details_visibility(self) -> None:
        if self._tail_widget is None:
            return
        if self._command_widget is not None:
            self._command_widget.display = self._details_visible
        if self._log_path_widget is not None:
            self._log_path_widget.display = self._details_visible
        self._tail_widget.display = self._details_visible
        if self._help_widget is not None:
            self._help_widget.update(self._help_text())

    def _stop_spinner_timer(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _help_text(self) -> str:
        glyphs = get_glyphs()
        details = "Hide details" if self._details_visible else "Show details"
        close = "Esc close" if self._done else "Esc close when complete"
        parts = [f"d {details}", close]
        if self._details_visible:
            parts.insert(1, "c copy log path")
        return f" {glyphs.bullet} ".join(parts)

    def _update_spinner(self) -> None:
        """Advance the glyph spinner while the update is running."""
        if self._done or self._spinner_widget is None:
            return
        self._spinner_widget.update(self._spinner.next_frame())
