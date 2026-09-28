"""Side conversation with follow-ups and app-owned history."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, ClassVar, override

from textual import work
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.worker import WorkerCancelled

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets._inline_prompt import (
    InlinePromptTextArea,
    newline_hint,
)
from deepagents_code.tui.widgets.loading import Spinner
from deepagents_code.tui.widgets.messages import AssistantMessage, UserMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from textual.app import ComposeResult
    from textual.geometry import Size
    from textual.timer import Timer
    from textual.widget import Widget


logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000


class BtwTextArea(InlinePromptTextArea):
    """Side-question editor with the shared multiline and paste conventions."""

    class Submitted(InlinePromptTextArea.Submitted):
        """Posted when Enter submits the complete side question."""


class BtwScreen(ModalScreen[None]):
    """Chat independently of the main run with restorable side exchanges."""

    CSS_PATH = "btw.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Dismiss", show=False),
        Binding("ctrl+x", "clear", "Clear", show=False, priority=True),
    ]

    def __init__(
        self,
        answer: Callable[[str], Awaitable[str]],
        question: str = "",
        *,
        history: Sequence[tuple[str, str]] = (),
        stream_answer: Callable[[str, Callable[[str], Awaitable[None]]], Awaitable[str]]
        | None = None,
        on_clear: Callable[[], None] | None = None,
    ) -> None:
        """Capture the answer callback and optional question.

        Args:
            answer: Generate a complete side answer.
            question: Initial question, or empty to start with the editor.
            history: Completed exchanges to restore when reopening the conversation.
            stream_answer: Optional generator that delivers fragments to its receiver.
            on_clear: Discard the caller's side-conversation history on reset.
        """
        super().__init__()
        self._answer = answer
        self._stream_answer = stream_answer
        self._on_clear = on_clear
        self._question = question
        self._history = tuple(history)
        self._spinner = Spinner()
        self._spinner_timer: Timer | None = None
        self._pending = False
        # Content may grow before the deferred scroll runs; retain follow intent
        # through that layout so stream completion cannot mistake it for a scroll up.
        self._follow_pending = False

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
            yield Static(self._help_text(), id="btw-help")

    def _help_text(self) -> str:
        hints = ["Enter submit", newline_hint()]
        if self.has_class("has-history"):
            hints.extend(("Tab history/input", "Ctrl+X clear"))
        hints.append("Esc hide")
        return f" {get_glyphs().bullet} ".join(hints)

    async def on_mount(self) -> None:
        """Start an independent worker only after the modal is mounted."""
        self.query_one("#btw-loading").display = False
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        scroll.display = False
        self.watch(
            scroll, "virtual_size", partial(self._follow_rendered_content, scroll)
        )
        await self._restore_history()
        if self._question:
            await self._start(self._question)
        else:
            self.query_one(BtwTextArea).focus(scroll_visible=False)

    def _follow_rendered_content(
        self, scroll: VerticalScroll, old_size: Size, new_size: Size
    ) -> None:
        old_bottom = max(
            0,
            old_size.height
            - scroll.container_size.height
            + scroll.scrollbar_size_horizontal,
        )
        if new_size.height > old_size.height and scroll.scroll_y >= old_bottom:
            self._follow_answer()

    async def _restore_history(self) -> None:
        if not self._history:
            return
        self.add_class("has-history")
        self.query_one("#btw-help", Static).update(self._help_text())
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        scroll.display = True
        for question, text in self._history:
            answer = AssistantMessage()
            await scroll.mount(
                UserMessage(question, classes="btw-question", detect_mode=False),
                answer,
                before="#btw-loading",
            )
            await answer.set_content(text)
        self.query_one(BtwTextArea).placeholder = "Ask a follow-up"
        self.call_after_refresh(scroll.scroll_end, animate=False)

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
        self.query_one("#btw-help", Static).update(self._help_text())
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        scroll.display = True
        scroll.focus()
        answer = AssistantMessage()
        error = Static("", classes="btw-error", markup=False)
        answer.display = error.display = False
        await scroll.mount(
            UserMessage(question, classes="btw-question", detect_mode=False),
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

    def _follow_answer(self) -> None:
        self._follow_pending = True
        self.call_after_refresh(self._scroll_to_latest)

    def _scroll_to_latest(self) -> None:
        self.query_one("#btw-scroll", VerticalScroll).scroll_end(
            animate=False, immediate=True
        )
        self._follow_pending = False

    async def _append_answer(self, answer: AssistantMessage, text: str) -> None:
        if not text:
            return
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        follow = self._follow_pending or scroll.is_vertical_scroll_end
        answer.display = True
        self._stop_spinner()
        self.query_one("#btw-loading").display = False
        await answer.append_content(text)
        if follow:
            self._follow_answer()

    async def _complete_answer(self, answer: AssistantMessage, text: str) -> None:
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        follow = answer.display and (
            self._follow_pending or scroll.is_vertical_scroll_end
        )
        await answer.set_content(text)
        if follow:
            self._follow_answer()

    @work(exclusive=True)
    async def _generate(
        self, question: str, answer: AssistantMessage, error: Static
    ) -> None:
        target: Widget = answer
        try:
            text = (
                await self._stream_answer(
                    question, partial(self._append_answer, answer)
                )
                if self._stream_answer is not None
                else await self._answer(question)
            )
            await self._complete_answer(answer, text)
        except asyncio.CancelledError:
            await answer.stop_stream()
            return
        except Exception as exc:
            await answer.stop_stream()
            logger.debug("Side question failed", exc_info=True)
            from deepagents_code.client.remote_client import format_agent_exception

            target = error
            error.update(format_agent_exception(exc))
        self._finish_answer(target, reveal=target is error or not answer.display)

    def _finish_answer(self, target: Widget, *, reveal: bool) -> None:
        self._stop_spinner()
        if self.is_mounted:
            target.display = True
            self.query_one("#btw-loading").display = False
            self._pending = False
            editor = self.query_one(BtwTextArea)
            editor.disabled = False
            editor.placeholder = "Ask a follow-up"
            editor.focus(scroll_visible=False)
            if reveal:
                self.call_after_refresh(self._reveal_answer, target)

    def _reveal_answer(self, target: Widget) -> None:
        if not target.is_mounted:
            return
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        if (
            target.region.y < scroll.content_region.y
            or target.region.bottom > scroll.content_region.bottom
        ):
            target.scroll_visible(top=True, animate=False)

    async def action_clear(self) -> None:
        """Cancel the side answer and start fresh without closing the modal."""
        self._stop_spinner()
        # Finish cancellation before removing widgets or resetting history so a
        # late answer cannot repopulate the new conversation.
        for worker in self.workers.cancel_node(self):
            with suppress(WorkerCancelled):
                await worker.wait()
        if self._on_clear is not None:
            self._on_clear()
        await self._clear_transcript()
        editor = self.query_one(BtwTextArea)
        editor.reset_paste_state()
        editor.load_text("")
        editor.disabled = False
        editor.placeholder = "Ask anything about this conversation"
        editor.focus()

    async def _clear_transcript(self) -> None:
        self._pending = self._follow_pending = False
        scroll = self.query_one("#btw-scroll", VerticalScroll)
        await scroll.remove_children(
            [child for child in scroll.children if child.id != "btw-loading"]
        )
        scroll.display = False
        self.query_one("#btw-loading").display = False
        self.remove_class("has-history")
        self.query_one("#btw-help", Static).update(self._help_text())

    def on_unmount(self) -> None:
        """Stop the loading animation when the modal closes."""
        self._stop_spinner()

    def action_cancel(self) -> None:
        """Dismiss the side conversation, leaving the main run untouched."""
        self.workers.cancel_node(self)
        self.dismiss(None)
