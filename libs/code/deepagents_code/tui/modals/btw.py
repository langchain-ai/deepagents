"""Ephemeral side-question prompt and answer."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, LoadingIndicator, Markdown, Static

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from textual.app import ComposeResult


logger = logging.getLogger(__name__)


class BtwScreen(ModalScreen[None]):
    """Ask independently of the main run and discard the answer on dismissal."""

    CSS_PATH = "btw.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Dismiss", show=False),
    ]

    def __init__(
        self, answer: Callable[[str], Awaitable[str]], question: str = ""
    ) -> None:
        """Capture the answer callback and optional question."""
        super().__init__()
        self._answer = answer
        self._question = question

    def compose(self) -> ComposeResult:
        """Build the side-question dialog.

        Yields:
            The prompt, answer, and dismissal hint.
        """
        with Vertical(id="btw-dialog"):
            yield Static("/btw", id="btw-title")
            yield Static(
                "A side question. No tools. Not added to the conversation.",
                id="btw-subtitle",
            )
            yield Input(
                placeholder="Ask anything about this conversation",
                id="btw-input",
                max_length=16_000,
            )
            yield Static(self._question, id="btw-question", markup=False)
            yield LoadingIndicator(id="btw-loading")
            with VerticalScroll(id="btw-scroll"):
                yield Markdown("", id="btw-answer", open_links=False)
                yield Static("", id="btw-error", markup=False)
            yield Static("Esc dismiss · Up/Down scroll", id="btw-help")

    def on_mount(self) -> None:
        """Start an independent worker only after the modal is mounted."""
        self.query_one("#btw-loading").display = False
        self.query_one("#btw-scroll").display = False
        if self._question:
            self._start(self._question)
        else:
            self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit the modal field without sending a chat message."""
        event.stop()
        if question := event.value.strip():
            self._start(question)

    def _start(self, question: str) -> None:
        self.query_one(Input).display = False
        self.query_one("#btw-question", Static).update(question)
        self.query_one("#btw-loading").display = True
        self._generate(question)

    @work(exclusive=True)
    async def _generate(self, question: str) -> None:
        try:
            text = await self._answer(question)
            await self.query_one(Markdown).update(text)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("Side question failed", exc_info=True)
            from deepagents_code.client.remote_client import format_agent_exception

            self.query_one("#btw-error", Static).update(format_agent_exception(exc))
        if self.is_mounted:
            self.query_one("#btw-loading").display = False
            scroll = self.query_one("#btw-scroll", VerticalScroll)
            scroll.display = True
            scroll.focus()

    def action_cancel(self) -> None:
        """Dismiss only this answer, leaving the main run untouched."""
        self.workers.cancel_node(self)
        self.dismiss(None)
