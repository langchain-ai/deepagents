"""Loading widget with animated spinner for agent activity."""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, ClassVar

from textual.containers import Horizontal
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class BrailleSpinner:
    """Animated braille spinner."""

    FRAMES: ClassVar[tuple[str, ...]] = (
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    )

    def __init__(self) -> None:
        """Initialize spinner."""
        self._position = 0

    def next_frame(self) -> str:
        """Get next animation frame."""
        frame = self.FRAMES[self._position]
        self._position = (self._position + 1) % len(self.FRAMES)
        return frame

    def current_frame(self) -> str:
        """Get current frame without advancing."""
        return self.FRAMES[self._position]


class LoadingWidget(Static):
    """Animated loading indicator with status text and elapsed time.

    Displays: ⠋ Thinking... (3s, esc to interrupt)
    """

    DEFAULT_CSS = """
    LoadingWidget {
        height: auto;
        padding: 0 1;
    }

    LoadingWidget .loading-container {
        height: auto;
        width: 100%;
    }

    LoadingWidget .loading-spinner {
        width: auto;
        color: $warning;
    }

    LoadingWidget .loading-status {
        width: auto;
        color: $warning;
    }

    LoadingWidget .loading-hint {
        width: auto;
        color: $text-muted;
        margin-left: 1;
    }
    """

    def __init__(self, status: str = "Thinking") -> None:
        """Initialize loading widget.

        Args:
            status: Initial status text to display
        """
        super().__init__()
        self._status = status
        self._spinner = BrailleSpinner()
        self._start_time: float | None = None
        self._spinner_widget: Static | None = None
        self._status_widget: Static | None = None
        self._hint_widget: Static | None = None
        self._paused = False
        self._paused_elapsed: int = 0

    def compose(self) -> ComposeResult:
        """Compose the loading widget layout."""
        with Horizontal(classes="loading-container"):
            self._spinner_widget = Static(self._spinner.current_frame(), classes="loading-spinner")
            yield self._spinner_widget

            self._status_widget = Static(f" {self._status}... ", classes="loading-status")
            yield self._status_widget

            self._hint_widget = Static("(0s, esc to interrupt)", classes="loading-hint")
            yield self._hint_widget

    def on_mount(self) -> None:
        """Start animation on mount."""
        self._start_time = time()
        self.set_interval(0.1, self._update_animation)

    def _update_animation(self) -> None:
        """Update spinner and elapsed time."""
        if self._paused:
            return

        if self._spinner_widget:
            frame = self._spinner.next_frame()
            self._spinner_widget.update(f"[#FFD800]{frame}[/]")

        if self._hint_widget and self._start_time is not None:
            elapsed = int(time() - self._start_time)
            self._hint_widget.update(f"({elapsed}s, esc to interrupt)")

    def set_status(self, status: str) -> None:
        """Update the status text.

        Args:
            status: New status text
        """
        self._status = status
        if self._status_widget:
            self._status_widget.update(f" {self._status}... ")

    def pause(self, status: str = "Awaiting decision") -> None:
        """Pause the animation and update status.

        Args:
            status: Status to show while paused
        """
        self._paused = True
        if self._start_time is not None:
            self._paused_elapsed = int(time() - self._start_time)
        self._status = status
        if self._status_widget:
            self._status_widget.update(f" {status}... ")
        if self._hint_widget:
            self._hint_widget.update(f"(paused at {self._paused_elapsed}s)")
        if self._spinner_widget:
            self._spinner_widget.update("[dim]⏸[/dim]")

    def resume(self) -> None:
        """Resume the animation."""
        self._paused = False
        self._status = "Thinking"
        if self._status_widget:
            self._status_widget.update(f" {self._status}... ")

    def stop(self) -> None:
        """Stop the animation (widget will be removed by caller)."""
