"""Shared primitives for inline prompts."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from textual.content import Content
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from pathlib import Path

    from textual import events
    from textual.widget import Widget

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.tui.widgets._paste_textarea import CollapsingPasteTextArea

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")

_UNSET: Any = object()

MEDIA_UNSUPPORTED_TOAST_PREFIX = "Only text is supported here"
"""Leading clause of the toast shown when media is dropped on an inline prompt.

Public so tests can assert on the toast without duplicating the whole message,
which names the discarded files (see `_media_unsupported_toast`).
"""


def _media_unsupported_toast(paths: list[Path]) -> str:
    """Build the toast for a rejected media drop.

    Names each discarded file rather than only the media ones, because the whole
    payload is swallowed — a mixed drop otherwise loses its non-media paths with
    nothing to explain where they went. Paths are identified by name, falling
    back to the full path when two drops share a basename, so the listing always
    distinguishes every file it counts.

    Args:
        paths: All resolved paths in the rejected payload. Never empty — the
            caller only builds a toast once it has found media.

    Returns:
        Toast text naming the files that were not inserted.
    """
    unique = list(dict.fromkeys(paths))
    name_counts = Counter(path.name for path in unique)
    labels = [
        path.name if name_counts[path.name] == 1 else str(path) for path in unique
    ]
    noun = "file" if len(labels) == 1 else "files"
    listing = ", ".join(labels)
    return f"{MEDIA_UNSUPPORTED_TOAST_PREFIX}; {noun} not inserted: {listing}."


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

        # Some terminals (e.g. VSCode built-in) send a literal backslash followed
        # by enter for shift+enter; treat that pair as a newline before the enter
        # below would otherwise submit.
        if self._consume_backslash_enter_newline(event, now):
            return

        self._track_backslash_pending(event, now)

        # Modifier+Enter (and Ctrl+J) insert a newline rather than submitting.
        if self._consume_modifier_newline(event):
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

    async def _on_paste(self, event: events.Paste) -> None:
        """Reject a dragged media file, else defer to shared paste handling."""
        # Flush first, matching the base handler: a rejection returns early, and
        # leaving a pending burst behind would let its timer insert the buffered
        # keystrokes after the paste was already refused.
        if self._paste_burst_buffer:
            await self._flush_paste_burst()

        if await self._reject_dropped_media(event.text):
            event.prevent_default()
            event.stop()
            return

        # Don't call super() here — Textual dispatches a message to *every*
        # class in the MRO that defines `_on_paste`, in order, so the base
        # handlers already run after this one returns and super() would invoke
        # them a second time. The `prevent_default()` above is what stops that
        # walk on the rejection path.

    async def _dispatch_burst_payload(self, payload: str) -> None:
        """Reject a media file replayed as a key burst, else defer to the base."""
        if await self._reject_dropped_media(payload):
            return
        await super()._dispatch_burst_payload(payload)

    async def _reject_dropped_media(self, text: str) -> bool:
        """Toast and swallow a dropped payload containing an image or video.

        Free-text prompts accept only text, so a dragged media file is rejected
        here instead of inserting its path. Detection requires the payload to
        resolve to files that exist on disk in the shape a terminal emits for a
        drop, so free-form prose is unaffected (see `dropped_payload_paths`).

        The whole payload is swallowed when any path is media, so a mixed drop
        does not half-insert; the toast names each discarded file to make that
        visible.

        Args:
            text: Raw pasted/dropped text payload.

        Returns:
            `True` when a media payload was detected and swallowed.
        """
        from deepagents_code.input import (
            dropped_payload_paths,
            looks_like_dropped_payload,
        )
        from deepagents_code.media_utils import is_media_path

        # Screen with the pure string guard before hopping to a thread: the
        # thread hop is an `await`, and `_dispatch_burst_payload` runs from the
        # burst flush timer's own task rather than the widget message queue, so
        # yielding there lets a concurrent keystroke land ahead of the buffered
        # payload. Ordinary typing and pasting never pays that cost now.
        if not looks_like_dropped_payload(text):
            return False

        try:
            paths = await asyncio.to_thread(dropped_payload_paths, text)
        except Exception:
            # The parser guards its own filesystem probes, but
            # `_resolve_with_unicode_space_variants` calls `expanduser()` and
            # `Path.cwd()` unguarded, so a deleted working directory or an
            # unresolvable home still surfaces here. Log at warning (not debug)
            # so it survives without DEEPAGENTS_CODE_DEBUG, since falling
            # through re-inserts the path this method exists to reject. The
            # message never includes the payload, though an OSError traceback
            # may name a path.
            logger.warning(
                "Media-payload detection failed; treating paste as text",
                exc_info=True,
            )
            return False
        if not any(is_media_path(path) for path in paths):
            return False
        if not self.is_mounted:
            # The prompt was resolved while the probe was in flight; `self.app`
            # still resolves via the active-app ContextVar, so notifying here
            # would toast about a field the user can no longer see.
            return True
        self.app.notify(
            _media_unsupported_toast(paths),
            severity="warning",
            timeout=5,
            markup=False,
        )
        return True


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


def newline_hint() -> str:
    """Return the newline-shortcut hint fragment (e.g. 'Ctrl+J newline')."""
    from deepagents_code.config import newline_shortcut

    return f"{newline_shortcut()} newline"


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
