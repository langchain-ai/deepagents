"""Feedback choice screen for /feedback command."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING, Any

from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class FeedbackOption(Static):
    """Clickable feedback option widget."""

    class Clicked(Message):
        """Message sent when a feedback option is clicked."""

        def __init__(self, url: str) -> None:
            """Initialize the Clicked message.

            Args:
                url: The URL to open when clicked.
            """
            super().__init__()
            self.url = url

    def __init__(
        self,
        label: str,
        url: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the FeedbackOption.

        Args:
            label: The display label for this option.
            url: The URL to open when this option is clicked.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(label, **kwargs)
        self.label = label
        self.url = url

    def compose(self) -> ComposeResult:
        """Compose the widget content.

        Yields:
            The Static widget with the label.
        """
        yield Static(self.label)

    def on_click(self) -> None:
        """Handle click on this option by posting a Clicked message."""
        self.post_message(self.Clicked(self.url))


class FeedbackChoiceScreen(ModalScreen[None]):
    """Screen for choosing feedback type (bug report or feature request)."""

    BUG_URL = (
        "https://github.com/langchain-ai/deepagents/issues/new?template=bug_report.yml"
    )
    FEATURE_URL = "https://github.com/langchain-ai/deepagents/issues/new?template=feature_request.yml"

    def compose(self) -> ComposeResult:
        """Compose the feedback choice screen.

        Yields:
            The Static title and feedback option widgets.
        """
        yield Static("What type of feedback would you like to submit?", id="title")
        yield FeedbackOption("🐛 Bug Report", self.BUG_URL, id="bug-option")
        yield FeedbackOption(
            "✨ Feature Request", self.FEATURE_URL, id="feature-option"
        )

    def on_feedback_option_clicked(self, event: FeedbackOption.Clicked) -> None:
        """Handle option click - open URL and close screen.

        Args:
            event: The Clicked event containing the URL to open.
        """
        webbrowser.open(event.url)
        self.app.pop_screen()
