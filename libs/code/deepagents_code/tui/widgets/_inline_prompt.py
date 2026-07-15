"""Shared primitives for inline prompts."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from textual.binding import Binding, BindingType
from textual.content import Content
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.widget import Widget

from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.tui.widgets._paste_textarea import CollapsingPasteTextArea

ResultT = TypeVar("ResultT")

_UNSET: Any = object()


class InlinePromptCompletion(Generic[ResultT]):
    """Resolve an inline prompt result at most once.

    `set_future` and `resolve` may be called in either order: a result
    recorded before the future is wired is delivered as soon as the future
    arrives, so a late `set_future` never strands an awaiter.
    """

    def __init__(self) -> None:
        """Initialize an unresolved completion."""
        self._future: asyncio.Future[ResultT] | None = None
        self._resolved = False
        self._result: ResultT | Any = _UNSET

    @property
    def resolved(self) -> bool:
        """Whether a terminal result has been recorded."""
        return self._resolved

    def set_future(self, future: asyncio.Future[ResultT]) -> None:
        """Set the future to resolve with the terminal result.

        Delivers an already-recorded result immediately, so callers may wire
        the future either before or after `resolve`.

        Args:
            future: Future owned by the application request path.
        """
        self._future = future
        if self._resolved and self._result is not _UNSET and not future.done():
            future.set_result(self._result)

    def resolve(self, result: ResultT) -> bool:
        """Record the first terminal result and resolve the future if set.

        The result is retained, so a future wired later via `set_future` still
        receives it.

        Args:
            result: Terminal prompt result.

        Returns:
            `True` when this is the first terminal result, otherwise `False`.
        """
        if self._resolved:
            return False
        self._resolved = True
        self._result = result
        if self._future is not None and not self._future.done():
            self._future.set_result(result)
        return True


class InlinePromptTextArea(CollapsingPasteTextArea):
    """Soft-wrapping text input shared by inline prompts.

    Matches the primary chat input's paste handling: a multi-line paste stays
    grouped instead of submitting on the first embedded newline, and a large
    paste collapses into a compact `[Pasted text #N]` placeholder that expands
    back to the full text via `submitted_value`.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(
            "shift+enter,alt+enter,ctrl+enter,ctrl+j",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+backspace,alt+backspace",
            "delete_word_left",
            "Delete left to start of word",
            show=False,
        ),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter to submit text.

        Subclasses should re-declare a nested `Submitted` so Textual derives a
        distinct handler name (e.g. `on_goal_review_text_area_submitted`).
        Without it, a host mounting more than one inline prompt cannot tell
        their submissions apart, and the base handler name goes unhandled.
        """

        def __init__(self, text_area: InlinePromptTextArea, value: str) -> None:
            """Initialize a text submission message.

            Args:
                text_area: Input that emitted the submission.
                value: Complete input text at submission time, with any
                    collapsed-paste placeholders expanded to their full content.
            """
            super().__init__()
            self.text_area = text_area
            self.value = value

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an inline prompt text area."""
        classes = kwargs.pop("classes", None)
        prompt_classes = (
            "inline-prompt-input"
            if classes is None
            else f"inline-prompt-input {classes}".strip()
        )
        super().__init__(classes=prompt_classes, **kwargs)
        self.show_line_numbers = False
        self.soft_wrap = True

    async def _on_key(self, event: events.Key) -> None:
        now = time.monotonic()

        # Drive the shared paste-burst state machine so a paste replayed as rapid
        # key events (no bracketed paste) stays grouped and can be collapsed.
        if await self._absorb_key_into_burst(event, now):
            event.prevent_default()
            event.stop()
            return

        if self._maybe_start_burst(event, now):
            event.prevent_default()
            event.stop()
            return

        if self._track_burst_run(event, now):
            event.prevent_default()
            event.stop()
            return

        if event.key == "backspace" and self._delete_placeholder_token(backwards=True):
            event.prevent_default()
            event.stop()
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            # Keep a paste's embedded newlines from submitting mid-stream.
            if self._consume_enter_as_burst_newline(now):
                return
            self.post_message(self.Submitted(self, self.submitted_value))
            return

        await super()._on_key(event)


class InlinePromptOption(Static):
    """Render a selectable inline-prompt option with a cursor."""

    def __init__(
        self,
        text: str,
        index: int,
        *,
        selected: bool = False,
        selected_class: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an option.

        Args:
            text: Option label.
            index: Position in its owning prompt's option list.
            selected: Whether to render the option selected initially.
            selected_class: CSS class applied while the option is highlighted.
            **kwargs: Additional `Static` arguments.
        """
        self.option_index = index
        self._cursor_visible = selected
        self._highlighted = selected
        self._text = text
        self._selected_class = selected_class
        super().__init__(self._render(), **kwargs)
        self._sync_selected_class()

    @property
    def selected(self) -> bool:
        """Whether the selection cursor is currently shown on this option."""
        return self._cursor_visible

    def select(self) -> None:
        """Mark this option as selected."""
        self.set_state(cursor=True, highlighted=True)

    def deselect(self) -> None:
        """Mark this option as deselected."""
        self.set_state(cursor=False, highlighted=False)

    def set_state(self, *, cursor: bool, highlighted: bool) -> None:
        """Update cursor visibility and visual highlighting independently.

        Args:
            cursor: Whether to render the selection cursor.
            highlighted: Whether to apply the selected CSS class.
        """
        self._cursor_visible = cursor
        self._highlighted = highlighted
        self.update(self._render())
        self._sync_selected_class()

    def _render(self) -> Content:
        glyphs = get_glyphs()
        prefix = f"{glyphs.cursor} " if self._cursor_visible else "  "
        return Content.from_markup("$prefix$text", prefix=prefix, text=self._text)

    def _sync_selected_class(self) -> None:
        if self._selected_class is None:
            return
        self.set_class(self._highlighted, self._selected_class)


def apply_inline_prompt_border(widget: Widget) -> None:
    """Flag the prompt shell for the ASCII border variant when required.

    The border type and color live in CSS (`.inline-prompt`); adding the
    `-ascii` class lets the stylesheet swap to an ASCII-safe border on terminals
    that cannot render box-drawing characters. Keeping the border in CSS (rather
    than an inline style) lets focus-driven rules like `:focus-within` restyle it.

    Args:
        widget: Mounted prompt shell receiving the border style.
    """
    if is_ascii_mode():
        widget.add_class("-ascii")


def stop_inline_prompt_blur(event: events.Blur) -> None:
    """Keep blur from being interpreted as prompt dismissal.

    Args:
        event: Textual blur event emitted by an inline prompt.
    """
    event.stop()
