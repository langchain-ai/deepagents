"""Custom header widget for deepagents-cli."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.widgets import Static

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from textual.app import ComposeResult


class DeepAgentsHeader(Horizontal):
    """Custom header showing thread ID, title, and localized time.

    Displays:
    - Left: Thread ID
    - Center: Application title
    - Right: Localized time in HH:MM:SS format
    """

    DEFAULT_CSS = """
    DeepAgentsHeader {
        height: 2;
        dock: top;
        background: $surface;
        padding: 0 1;
        border-bottom: solid $primary;
    }

    DeepAgentsHeader .header-thread {
        width: auto;
        color: $text-muted;
    }

    DeepAgentsHeader .header-title {
        width: 1fr;
        text-align: center;
        text-style: bold;
    }

    DeepAgentsHeader .header-time {
        width: auto;
        text-align: right;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        thread_id: str,
        title: str = "Deep Agents",
    ) -> None:
        """Initialize the header.

        Args:
            thread_id: The thread identifier to display on the left.
            title: The title to display in the center.
        """
        super().__init__()
        self._thread_id = thread_id
        self._title = title

    def compose(self) -> ComposeResult:
        """Compose the header layout.

        Yields:
            Static widgets for thread ID, title, and time.
        """
        yield Static(
            f"Thread: {self._thread_id}",
            id="header-thread",
            classes="header-thread",
        )
        yield Static(self._title, id="header-title", classes="header-title")
        yield Static("", id="header-time", classes="header-time")

    def on_mount(self) -> None:
        """Start the time update timer when mounted."""
        # Update time immediately
        self._update_time()
        # Set up a recurring timer to update time every second
        self.set_interval(1.0, self._update_time)

    def _update_time(self) -> None:
        """Update the time display with current localized time."""
        try:
            time_widget = self.query_one("#header-time", Static)
            current_time = datetime.now().strftime("%H:%M:%S")
            time_widget.update(current_time)
        except Exception as e:  # noqa: BLE001
            # Silently fail if widget is not available
            logger.debug("Failed to update header time: %s", e)

    def update_thread_id(self, thread_id: str) -> None:
        """Update the thread ID displayed in the header.

        Args:
            thread_id: The new thread identifier to display.
        """
        self._thread_id = thread_id
        try:
            thread_widget = self.query_one("#header-thread", Static)
            thread_widget.update(f"Thread: {self._thread_id}")
        except Exception as e:  # noqa: BLE001
            # Silently fail if widget is not available (e.g., during initialization)
            logger.debug("Failed to update header thread ID: %s", e)
