"""Shared primitives for inline prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from textual.binding import Binding, BindingType
from textual.content import Content
from textual.message import Message
from textual.widgets import Static, TextArea

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.widget import Widget

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode

ResultT = TypeVar("ResultT")


class InlinePromptCompletion(Generic[ResultT]):
    """Resolve an inline prompt result at most once."""

    def __init__(self) -> None:
        """Initialize an unresolved completion."""
        self._future: asyncio.Future[ResultT] | None = None
        self._resolved = False

    @property
    def resolved(self) -> bool:
        """Whether a terminal result has been accepted."""
        return self._resolved

    def set_future(self, future: asyncio.Future[ResultT]) -> None:
        """Set the future to resolve with the terminal result.

        Args:
            future: Future owned by the application request path.
        """
        self._future = future

    def resolve(self, result: ResultT) -> bool:
        """Resolve the configured future unless a result was already accepted.

        Args:
            result: Terminal prompt result.

        Returns:
            `True` when this is the first terminal result, otherwise `False`.
        """
        if self._resolved:
            return False
        self._resolved = True
        if self._future is not None and not self._future.done():
            self._future.set_result(result)
        return True


class InlinePromptTextArea(TextArea):
    """Soft-wrapping text input shared by inline prompts."""

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
        """Posted when the user presses Enter to submit text."""

        def __init__(self, text_area: InlinePromptTextArea, value: str) -> None:
            """Initialize a text submission message.

            Args:
                text_area: Input that emitted the submission.
                value: Complete input text at submission time.
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

    def action_insert_newline(self) -> None:
        """Insert a newline at the cursor."""
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self, self.text))


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
        self.selected = selected
        self._cursor_visible = selected
        self._highlighted = selected
        self._text = text
        self._selected_class = selected_class
        super().__init__(self._render(), **kwargs)
        self._sync_selected_class()

    def toggle(self) -> None:
        """Toggle the selected state."""
        self.set_state(cursor=not self.selected, highlighted=not self.selected)

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
        self.selected = cursor
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
    """Use the ASCII border variant when the active terminal requires it.

    Args:
        widget: Mounted prompt shell receiving the border style.
    """
    if is_ascii_mode():
        colors = theme.get_theme_colors(widget)
        widget.styles.border = ("ascii", colors.success)


def stop_inline_prompt_blur(event: events.Blur) -> None:
    """Keep blur from being interpreted as prompt dismissal.

    Args:
        event: Textual blur event emitted by an inline prompt.
    """
    event.stop()
