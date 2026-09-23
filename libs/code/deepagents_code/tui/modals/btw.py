"""Ephemeral side conversation with follow-up questions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar, override

from textual import work
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets._inline_prompt import (
    InlinePromptTextArea,
    newline_hint,
)
from deepagents_code.tui.widgets.loading import Spinner

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from textual.app import ComposeResult
    from textual.timer import Timer
    from textual.widget import Widget


logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000


class BtwTextArea(InlinePromptTextArea):
    """Side-question editor with the shared multiline and paste conventions."""

    class Submitted(InlinePromptTextArea.Submitted):
        """Posted when Enter submits the complete side question."""


class BtwScreen(ModalScreen[None]):
    """Chat independently of the main run and discard exchanges on dismissal."""

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
        self._spinner = Spinner()
        self._spinner_timer: Timer | None = None
        self._pending = False

    @override
    def compose(self) -> ComposeResult:
        """Build the side-question dialog.

        Yields:
            The prompt, answer, and dismissal hint.
        """
        with Vertical(id="btw-dialog"):
            yield Static("/btw", id="btw-title")
            yield Static(
                "Ask a quick question without interrupting your conversation.",
                id="btw-description",
            )
            with VerticalScroll(id="btw-scroll"):
                yield Static(id="btw-loading")
            yield BtwTextArea(
                placeholder="Ask anything about this conversation",
                id="btw-input",
            )
            yield Static(
                f" {get_glyphs().bullet} ".join(
                    ("Enter ask", newline_hint(), "Tab history/input", "Esc dismiss")
                ),
                id="btw-help",
            )

    async def on_mount(self) -> None:
        """Start an independent worker only after the modal is mounted."""
        self.query_one("#btw-loading").display = False
        self.query_one("#btw-scroll").display = False
        if self._question:
            await self._start(self._question)
        else:
            self.query_one(BtwTextArea).focus()

    async def on_btw_text_area_submitted(self, event: BtwTextArea.Submitted) -> None:
        """Submit the expanded modal text without sending a chat message."""
        event.stop()
        if self._pending:
            return
        if question := event.value.strip():
            if len(question) > _MAX_QUESTION_LENGTH:
                self.notify(
                    "Question must contain at most 16,000 characters.",
                    severity="warning",
                )
                return
            await self._start(question)

    async def _start(self, question: str) -> None:
        self._pending = True
        editor = self.query_one(BtwTextArea)
        editor.disabled = True
        editor.reset_paste_state()
        editor.clear()
        self.add_class("has-history")
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        scroll.display = True
        scroll.focus()
        answer = Markdown("", open_links=False)
        error = Static("", classes="btw-error", markup=False)
        answer.display = error.display = False
        await scroll.mount(
            Static(question, classes="btw-question", markup=False),
            answer,
            error,
            before="#btw-loading",
        )
        self._start_spinner()
        self._generate(question, answer, error)

    def _start_spinner(self) -> None:
        loading = self.query_one("#btw-loading", Static)
        loading.update(f"{self._spinner.current_frame()} Thinking...")
        loading.display = True
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)
        # Let the new exchange lay out before scroll_end schedules its scroll;
        # wrapped Markdown can change the content height on the next refresh.
        self.call_after_refresh(
            self.query_one("#btw-scroll", VerticalScroll).scroll_end, animate=False
        )

    def _tick_spinner(self) -> None:
        self.query_one("#btw-loading", Static).update(
            f"{self._spinner.next_frame()} Thinking..."
        )

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    @work(exclusive=True)
    async def _generate(self, question: str, answer: Markdown, error: Static) -> None:
        target: Widget = answer
        try:
            text = await self._answer(question)
            await answer.update(text)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("Side question failed", exc_info=True)
            from deepagents_code.client.remote_client import format_agent_exception

            target = error
            error.update(format_agent_exception(exc))
        self._stop_spinner()
        if self.is_mounted:
            target.display = True
            self.query_one("#btw-loading").display = False
            self._pending = False
            editor = self.query_one(BtwTextArea)
            editor.disabled = False
            editor.placeholder = "Ask a follow-up"
            editor.focus(scroll_visible=False)
            self.call_after_refresh(self._reveal_answer, target)

    def _reveal_answer(self, target: Widget) -> None:
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        if (
            target.region.y < scroll.content_region.y
            or target.region.bottom > scroll.content_region.bottom
        ):
            target.scroll_visible(top=True, animate=False)

    def on_unmount(self) -> None:
        """Stop the loading animation when the modal closes."""
        self._stop_spinner()

    def action_cancel(self) -> None:
        """Dismiss the side conversation, leaving the main run untouched."""
        self.workers.cancel_node(self)
        self.dismiss(None)
