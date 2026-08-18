"""Message widgets."""

from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.events import Click
from textual.geometry import Offset
from textual.message import Message
from textual.message_pump import NoActiveAppError
from textual.reactive import var
from textual.selection import Selection
from textual.style import Style as TStyle
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code._ask_user_types import (
    ASK_USER_ANSWERED_SUMMARY,
    ASK_USER_FAILED_SUMMARY,
    AskUserRowSummary,
)
from deepagents_code.config import (
    MODE_DISPLAY_GLYPHS,
    detect_mode_prefix,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_code.diff_utils import (
    DiffStats,
    count_diff_change_lines,
    is_truncation_marker,
    split_diff_lines,
)
from deepagents_code.file_ops import (
    DiffOutcome,
    display_caveat,
    is_sensitive_file_path,
)
from deepagents_code.formatting import format_duration
from deepagents_code.input import EMAIL_PREFIX_PATTERN, INPUT_HIGHLIGHT_PATTERN
from deepagents_code.tool_display import (
    EXECUTE_HEADER_MAX_LENGTH,
    JS_EVAL_HEADER_MAX_LENGTH,
    format_tool_display,
)
from deepagents_code.tui.widgets._js_eval_display import (
    JsEvalBlock,
    JsEvalError,
    JsEvalResult,
    JsEvalStdout,
    parse_js_eval_blocks,
)
from deepagents_code.tui.widgets._links import (
    event_targets_link,
    open_checked_url_async,
    open_style_link,
)
from deepagents_code.tui.widgets.diff import (
    compose_diff_lines,
    format_diff_stats,
    highlight_source_prefixes,
)
from deepagents_code.unicode_security import render_with_unicode_markers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from rich.console import (
        Console as RichConsole,
        ConsoleOptions,
        RenderResult,
    )
    from textual.app import ComposeResult
    from textual.css.types import PointerShape
    from textual.events import MouseMove
    from textual.timer import Timer
    from textual.widget import Widget
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownStream

    from deepagents_code.input import MediaTracker
    from deepagents_code.theme import ThemeColors
    from deepagents_code.tui.widgets.message_store import MessageData

    type _SummaryCall = tuple[str, Mapping[str, Any]]
    """One tool call as the summary code sees it: `(raw tool name, parsed args)`."""

    type _SummaryCacheKey = tuple[tuple[str, str | None], ...]
    """Opaque identity of a summary line's inputs — compare only for equality."""

    type _LiveSummaryKey = tuple[_SummaryCacheKey, _SummaryCacheKey]
    """The `(completed, pending)` key pair behind a cached live summary line."""

logger = logging.getLogger(__name__)


def _mode_color(mode: str | None, widget_or_app: object | None = None) -> str:
    """Return the hex color string for a mode, falling back to primary.

    Args:
        mode: Mode name (e.g. `'shell'`, `'command'`) or `None`.
        widget_or_app: Textual widget or `App` for theme-aware lookup.

    Returns:
        Color string from the active theme's `ThemeColors`.
    """
    colors = theme.get_theme_colors(widget_or_app)
    if not mode:
        return colors.primary
    if mode == "shell_incognito":
        return colors.mode_incognito
    if mode == "shell":
        return colors.mode_bash
    if mode == "command":
        return colors.mode_command
    logger.warning("Missing color for mode '%s'; falling back to primary.", mode)
    return colors.primary


def _event_targets_rendered_text(event: MouseMove) -> bool:
    """Return whether the hovered cell was rendered from widget content.

    Textual tags every content segment's style with a selection `offset`
    (`Content.to_strip` via `Style.rich_style_with_offset`) and reads the same
    key back in `Compositor.get_widget_and_offset_at` to map a screen cell to a
    text position. Alignment padding and cells past the end of a line carry no
    such meta, so the key doubles as a rendered-text hit test that agrees
    exactly with what Textual can resolve to an offset.

    `offset` and both of its producers are private Textual API. Re-verify these
    names on every Textual bump: if the key moves, every cell reads as blank
    and the text pointer silently stops appearing.

    This tracks *rendered* text, not *selectable* text — the meta is attached
    unconditionally, so a widget with `ALLOW_SELECT = False` still reports
    `True` here.

    Args:
        event: The Textual mouse-move event to inspect.

    Returns:
        `True` when the hovered cell holds content-rendered text.
    """
    return "offset" in event.style.meta


def _pointer_shape_for(event: MouseMove) -> PointerShape:
    """Return the pointer shape to show for the cell under the mouse.

    Textual applies a widget's pointer across its whole rectangle, so a message
    declaring `pointer: text` in CSS shows an I-beam over the blank space beside
    its short lines. Resolving the shape per cell instead keeps the I-beam on
    text the reader can actually select.

    The result is only accurate as of the last mouse movement: content that
    grows or scrolls under a stationary pointer leaves the previous shape in
    place until the mouse moves again.

    Args:
        event: The Textual mouse-move event to inspect.

    Returns:
        `'pointer'` over links, `'text'` over rendered text, else `'default'`.
    """
    if event_targets_link(event):
        return "pointer"
    if _event_targets_rendered_text(event):
        return "text"
    return "default"


@dataclass(frozen=True, slots=True)
class FormattedOutput:
    """Result of formatting tool output for display."""

    content: Content
    """Styled `Content` for the formatted output."""

    truncation: str | None = None
    """Description of truncated content (e.g., "10 more lines"), or None if no
    truncation occurred."""


# Maximum number of tool arguments to display inline
_MAX_INLINE_ARGS = 3

# Truncation limits for display
_MAX_TODO_CONTENT_LEN = 70
_DEFAULT_TODO_WRAP_WIDTH = 80
_TODO_WRAP_GUARD_COLUMNS = 4
_MAX_WEB_CONTENT_LEN = 100

# User message display truncation — when content exceeds this many characters,
# only the head and tail are rendered with an elision marker in between.
# This keeps very large pastes from flooding the conversation scrollback.
_USER_MSG_MAX_DISPLAY_CHARS = 10_000
_USER_MSG_TRUNCATE_HEAD_CHARS = 2_500
_USER_MSG_TRUNCATE_TAIL_CHARS = 2_500

# Tools that have their key info already in the header (no need for args line)
_TOOLS_WITH_HEADER_INFO: set[str] = {
    # Filesystem tools
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "execute",  # sandbox shell
    "js_eval",  # JS interpreter
    # Web tools
    "web_search",
    "fetch_url",
    "ask_user",
    # Agent tools
    "task",
    "write_todos",
}


# Tools whose key info (file path / search pattern) is already in the header, so
# their output body is collapsed entirely by default — an expand affordance
# replaces the inline preview. `read_file` echoes the file; grep/glob echo the
# matches for a pattern the header already names.
_COLLAPSE_OUTPUT_BY_DEFAULT: set[str] = {
    "read_file",
    "grep",
    "glob",
}


_TOOL_SUPERSEDED_BY_DIFF = "edit_file"
"""The one tool whose successful row is replaced by the `DiffMessage` after it.

The row self-hides via `mark_superseded_by_diff`, and only ever behind a diff
that can actually stand in for it — the adapter requires a non-empty body and a
`shown` outcome. `DiffOutcome` explains why the other outcomes cannot.

Ask `ToolCallMessage.can_be_superseded` rather than comparing against this
constant — the adapter checks a tool name from a different source, and the two
must not drift.
"""


# Tools whose collapsed body is always the formatter's compact preview, no
# matter how short the raw output is, and whose expandability is therefore
# decided by the formatter rather than by the raw size thresholds. `write_todos`
# renders a per-item summary; `ask_user` renders a one-line summary so a
# two-line transcript still keeps its answers behind an expand click.
_ALWAYS_PREVIEW_TOOLS: frozenset[str] = frozenset({"write_todos", "ask_user"})


# An `ask_user` row whose recorded output is exactly one of these holds only a
# fallback summary — no `ToolMessage` transcript ever arrived — so there is
# nothing for an expand click to reveal. Recognized by value rather than by the
# `_deferred_success_settled` flag so the suppression also holds for a row rebuilt
# from the message store, where that flag is not persisted (a rehydrated row is
# always already terminal). A real transcript always begins `Q: `, so it can never
# collide with these.
_ASK_USER_ROW_SUMMARIES: frozenset[str] = frozenset(
    {ASK_USER_ANSWERED_SUMMARY, ASK_USER_FAILED_SUMMARY}
)


# Long-running tools whose completed status row reports how long they ran
# ("Took <duration>") when a run was timed, instead of being hidden. `execute`
# shells and `task` subagent dispatches can both run for a while, so the elapsed
# time is useful.
_TIMED_SUCCESS_TOOLS: set[str] = {
    "execute",
    "task",
}


# CSS classes applied to a `ToolCallMessage` to tint the whole row by terminal
# outcome (see its `DEFAULT_CSS`). Running/pending states carry none of these.
_STATUS_CLASSES: frozenset[str] = frozenset(
    {"-status-success", "-status-error", "-status-rejected", "-status-skipped"}
)


_SUCCESS_EXIT_RE = re.compile(r"\n?\[Command succeeded with exit code 0\]\s*$")
"""Strip the SDK's `[Command succeeded with exit code 0]` trailer from tool output."""


_READ_FILE_GUTTER_RE = re.compile(r"^ *(\d+(?:\.\d+)?)(?:  |\t)(.*)$")
"""Match a `read_file` gutter row into (marker, source).

The marker is a bare `N` or `N.M` (the latter a wrapped-line continuation) —
both sides of the dot required, so a stray `.5` head is not a gutter. The
separator is exactly two spaces (current format) or a single tab (legacy
`cat -n`). Only the separator is consumed and leading padding is spaces-only, so
source indentation — including leading tabs — after the gutter stays put. Kept in
sync with the separator emitted by deepagents' `format_content_with_line_numbers`
(the authoritative producer). See `ToolCallMessage._compact_line_gutter`.
"""


def _strip_success_exit_line(text: str) -> str:
    """Remove the `[Command succeeded with exit code 0]` trailer.

    Non-zero exit codes are left intact (they come through `set_error`).

    Args:
        text: Raw tool output string.

    Returns:
        Text with the success exit-code trailer removed, if present.
    """
    return _SUCCESS_EXIT_RE.sub("", text)


# Visual width of the prompt prefix (glyph + trailing space, e.g. "> ", "$ ").
# Glyphs are single characters, so the prefix is always two columns wide.
_PROMPT_PREFIX_WIDTH = 2


def _strip_prompt_prefix(
    result: tuple[str, str] | None,
    selection: Selection,
) -> tuple[str, str] | None:
    """Drop the leading prompt prefix glyph from a selected range.

    The prefix is only rendered on the first row, so it is stripped only when
    the selection begins there. This keeps triple-click / select-all copies to
    the message body instead of the decorative `"> "` (or mode glyph) prefix.

    Args:
        result: The `(text, ending)` tuple returned by `Static.get_selection`.
        selection: The active selection geometry.

    Returns:
        The selection with the prefix removed from row 0, or `result` unchanged.
    """
    if result is None:
        return None
    text, ending = result
    start = selection.start
    if start is not None and start.y != 0:
        return result
    start_x = 0 if start is None else start.x
    prefix_chars = max(0, _PROMPT_PREFIX_WIDTH - start_x)
    return text[prefix_chars:], ending


def _select_prompt_body(widget: Static) -> None:
    """Select the user message body without its decorative prompt glyph.

    Args:
        widget: User message widget whose body should be selected.
    """
    widget.screen.selections = {  # ty: ignore[invalid-assignment]  # Textual reactive descriptor assignment updates selection watchers; `set_reactive` would skip them.
        widget: Selection(Offset(_PROMPT_PREFIX_WIDTH, 0), None),
    }


@dataclass(frozen=True, slots=True)
class _UserMessageFull:
    """A user message short enough to render verbatim."""

    text: str
    """The original body, unmodified."""


@dataclass(frozen=True, slots=True)
class _UserMessageCollapsed:
    """A user message elided to head + tail for transcript display.

    Each variant names only the fields that mean something for it, so states
    like "not collapsed but 5 hidden lines" are unrepresentable and consumers
    dispatch by `isinstance` rather than reading an overloaded flag.
    """

    head: str
    """Leading slice kept verbatim, rendered above the elision marker."""

    tail: str
    """Trailing slice kept verbatim, rendered below the elision marker."""

    hidden_lines: int
    """Newlines in the elided middle."""

    hidden_chars: int
    """Characters in the elided middle.

    Reported in place of `hidden_lines` for single-line bodies (base64 blobs,
    minified JSON), where "+0 lines" would imply nothing was hidden.
    """

    @property
    def text(self) -> str:
        """Single-string collapsed form, for callers that cannot emit spans.

        Returns:
            Head and tail joined by an elision marker.
        """
        ellipsis = get_glyphs().ellipsis
        return (
            f"{self.head}\n{ellipsis} +{self.hidden_lines} lines {ellipsis}\n"
            f"{self.tail}"
        )


_UserMessageDisplay = _UserMessageFull | _UserMessageCollapsed
"""Either form a user-message body can take in the transcript."""


def _will_collapse(text: str) -> bool:
    """Return whether `text` exceeds the transcript display threshold.

    Single source of truth for the threshold, shared by `_collapse_user_message`
    and `UserMessage.will_truncate` so the render decision and the expand
    affordance can never disagree.

    Args:
        text: Candidate body text.

    Returns:
        `True` when the body is long enough to collapse.
    """
    return len(text) > _USER_MSG_MAX_DISPLAY_CHARS


def _collapse_user_message(text: str) -> _UserMessageDisplay:
    """Collapse a very long user message for transcript display.

    Keeps the first and last portions and elides the middle. This mirrors
    Claude Code's `UserPromptMessage` head+tail truncation for rendering
    performance.

    Args:
        text: Full message content.

    Returns:
        `_UserMessageCollapsed` when the body exceeds the display threshold,
        otherwise `_UserMessageFull` carrying the original text.
    """
    if not _will_collapse(text):
        return _UserMessageFull(text=text)
    hidden_start = _USER_MSG_TRUNCATE_HEAD_CHARS
    hidden_end = len(text) - _USER_MSG_TRUNCATE_TAIL_CHARS
    return _UserMessageCollapsed(
        head=text[:_USER_MSG_TRUNCATE_HEAD_CHARS],
        tail=text[-_USER_MSG_TRUNCATE_TAIL_CHARS:],
        # Counted over a range rather than a slice so a multi-megabyte paste
        # does not allocate a copy of its own middle on every render.
        hidden_lines=text.count("\n", hidden_start, hidden_end),
        hidden_chars=hidden_end - hidden_start,
    )


def _truncate_for_display(text: str) -> str:
    """Truncate very long user message text for display in the conversation.

    Thin string-returning wrapper around `_collapse_user_message` for
    `QueuedUserMessage.render` and tests, which only need the joined text.
    `UserMessage.render` consumes the head/tail fields directly so it can
    interleave the clickable affordance between them.

    Args:
        text: Full message content.

    Returns:
        Truncated text with an elision marker, or the original text when
        it does not exceed the display threshold.
    """
    return _collapse_user_message(text).text


class UserMessage(Static):
    """Widget displaying a user message.

    Very long messages are collapsed in the transcript by default (head+tail
    elision) to protect scrollback performance. The full text remains on the
    widget for copy/select, and the collapsed form is reversible via click or
    Ctrl+O.
    """

    class ExpansionChanged(Message):
        """Posted when the collapsed-body expansion state changes."""

        def __init__(self, widget: UserMessage, expanded: bool) -> None:
            """Initialize an expansion-state message.

            Args:
                widget: The user message whose expansion state changed.
                expanded: Whether the full body is now shown.
            """
            super().__init__()
            self.widget = widget
            self.expanded = expanded

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $primary;
        /* The expand affordance carries `@click` meta, which Textual styles as
           a link (underline, and bold on an accent block when hovered).
           Neutralize both so the hint renders as plain inherited-colour dim
           italic, matching every other "click or Ctrl+O" hint in this module.
           Bold in particular has to go: it cancels dim in most terminals. */
        link-color: $text;
        link-style: not underline;
        link-color-hover: $text;
        link-background-hover: transparent;
        link-style-hover: not bold not underline;
    }

    UserMessage.-cancelled {
        opacity: 0.6;
    }
    """
    """`-cancelled` dims a prompt whose turn was interrupted by the user."""

    _expanded: var[bool] = var(False)

    def __init__(
        self,
        content: str,
        *,
        media_snapshot: MediaTracker | None = None,
        detect_mode: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a user message.

        Args:
            content: The message content
            media_snapshot: Optional media tracker state captured at submission.
            detect_mode: When `True` (default), a leading mode trigger (`/`,
                `!`, `!!`) is rendered with its shell/command glyph, border, and
                highlight. Set to `False` for text submitted as literal agent
                input (e.g. via `-m`/`--message`), which never triggers a
                shell/command mode, so a leading slash (like a file path) must
                render as a plain user message rather than a slash command.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content
        self._media_snapshot = media_snapshot
        self._detect_mode = detect_mode
        self._deferred_expanded = False
        # Last expansion value published to the message store. Deduping against
        # it keeps the reactive's initialization watcher and the deferred
        # restore from re-emitting a value the store already holds.
        self._published_expanded = False

    @staticmethod
    def will_truncate(content: str) -> bool:
        """Return whether `content` would collapse in the transcript.

        Prefer the `has_expandable_body` property when a widget is in hand: it
        applies the same mode-prefix stripping as `render()`, so it cannot
        disagree with what is actually on screen. This static form is for
        callers that only have the raw string and know no prefix applies.

        Args:
            content: Candidate user-message body (mode prefix already stripped
                when applicable — matching what `render()` collapses).

        Returns:
            `True` when the body exceeds the display character threshold.
        """
        return _will_collapse(content)

    @property
    def raw_text(self) -> str:
        """The original, untruncated message text as the user submitted it.

        Named `raw_text` rather than `content` to avoid shadowing Textual's
        read/write `Static.content` property (backed by a mangled attribute);
        overriding it getter-only would make `self.content = ...` raise.
        """
        return self._content

    @property
    def media_snapshot(self) -> MediaTracker | None:
        """Media tracker state captured when the message was submitted."""
        return self._media_snapshot

    def _body_for_display(self) -> str:
        """Return the message body after stripping a detected mode trigger.

        Unlike `_prefix_and_body`, this does not look up theme colors, so it is
        safe to call before mount (e.g. `has_expandable_body` / Ctrl+O routing).

        Returns:
            Body text used for collapse decisions (`has_expandable_body`).
        """
        content = self._content
        mode_match = detect_mode_prefix(content) if self._detect_mode else None
        if mode_match:
            prefix_text, _mode = mode_match
            return content[len(prefix_text) :]
        return content

    @property
    def has_expandable_body(self) -> bool:
        """Whether this message is long enough to collapse/expand in display."""
        return self.will_truncate(self._body_for_display())

    def set_cancelled(self) -> None:
        """Dim the message to mark its turn as interrupted by the user."""
        self.add_class("-cancelled")

    def toggle_expanded(self) -> None:
        """Toggle between collapsed and full-body transcript display."""
        if not self.has_expandable_body:
            return
        self._expanded = not self._expanded

    def action_toggle_expand(self) -> None:
        """Textual `@click` target for the expand/collapse affordance."""
        self.toggle_expanded()

    def watch__expanded(self, expanded: bool) -> None:
        """Relayout and publish user-driven expansion for virtualization."""
        self.refresh(layout=True)
        # Publish only genuine changes: dedupe against the store's known value
        # to drop the reactive's initialization watcher and the deferred
        # restore, and require `is_attached` so `_expanded` set in pre-mount
        # test setup does not `post_message` on a detached widget.
        if self.is_attached and expanded != self._published_expanded:
            self._published_expanded = expanded
            self.post_message(self.ExpansionChanged(self, expanded))

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return selected text, preferring the full content over the render.

        `render()` truncates long messages, so for a full-message selection
        (select-all / select-to-end, where `selection.end` is `None`) the text
        is extracted from the untruncated content so copy yields the complete
        original.  A partial selection is extracted from the base (on-screen)
        render so its offsets stay aligned with what the user highlighted.

        Args:
            selection: The active selection geometry.

        Returns:
            The `(text, ending)` selection with the prefix removed, or `None`.
        """
        if selection.end is not None:
            return _strip_prompt_prefix(super().get_selection(selection), selection)
        text = str(self._build_full_render())
        return _strip_prompt_prefix((selection.extract(text), "\n"), selection)

    def _prefix_and_body(self) -> tuple[tuple[str, str], str]:
        """Compute the styled mode prefix and the body with its trigger stripped.

        Returns:
            A `(prefix, body)` pair where `prefix` is a `(text, style)` tuple and
            `body` is the content with any mode-trigger prefix removed.
        """
        colors = theme.get_theme_colors(self)
        content = self._content
        mode_match = detect_mode_prefix(content) if self._detect_mode else None
        if mode_match:
            prefix_text, mode = mode_match
            glyph = MODE_DISPLAY_GLYPHS.get(mode, prefix_text[0])
            return (
                (f"{glyph} ", f"bold {_mode_color(mode, self)}"),
                content[len(prefix_text) :],
            )
        return ("> ", f"bold {colors.primary}"), content

    def _build_full_render(self) -> Content:
        """Build a Content from the full content without display truncation.

        Omits the expand/collapse hint spans in both the collapsed and expanded
        states, so select-all yields only the original message body. Drag
        selections do not route here (see `get_selection`) and can still pick up
        hint text along with the body.

        Returns:
            Content with the mode prefix glyph and the full message body.
        """
        prefix, body = self._prefix_and_body()
        return Content.assemble(prefix, body)

    def text_select_all(self) -> None:
        """Select the message body without the prompt prefix glyph."""
        _select_prompt_body(self)

    def on_mount(self) -> None:
        """Add mode/ASCII CSS classes and restore deferred expansion state."""
        mode_match = detect_mode_prefix(self._content) if self._detect_mode else None
        if mode_match:
            _prefix, mode = mode_match
            self.add_class(f"-mode-{mode.replace('_', '-')}")
        if is_ascii_mode():
            self.add_class("-ascii")
        # The store already holds the restored state, so record it as published
        # first; the assignment below then dedupes instead of re-emitting it.
        self._published_expanded = self._deferred_expanded
        if self._deferred_expanded:
            self._expanded = True
            self._deferred_expanded = False

    def _append_highlighted_body(
        self,
        parts: list[str | tuple[str, str] | Content],
        content: str,
        *,
        colors: ThemeColors,
    ) -> None:
        """Append body text to `parts`, highlighting @mentions and /commands.

        Args:
            parts: Accumulator for `Content.assemble`.
            content: Body text to highlight and append.
            colors: Active theme colors.
        """
        last_end = 0
        for match in INPUT_HIGHLIGHT_PATTERN.finditer(content):
            start, end = match.span()
            token = match.group()

            # Skip @mentions that look like email addresses
            if token.startswith("@") and start > 0:
                char_before = content[start - 1]
                if EMAIL_PREFIX_PATTERN.match(char_before):
                    continue

            # Add text before the match (unstyled)
            if start > last_end:
                parts.append(content[last_end:start])

            # The regex only matches tokens starting with / or @
            if token.startswith("/") and start == 0:
                # A leading `/command` is only highlighted when mode detection
                # is on; otherwise it is literal text (e.g. a file path passed
                # via `-m`) and must render plain so the token is not dropped.
                if self._detect_mode:
                    parts.append((token, f"bold {colors.warning}"))
                else:
                    parts.append(token)
            elif token.startswith("@"):
                # @file mention
                parts.append((token, f"bold {colors.primary}"))
            last_end = end

        # Add remaining text after last match
        if last_end < len(content):
            parts.append(content[last_end:])

    @staticmethod
    def _hint_style() -> TStyle:
        """Style for the expand/collapse affordance.

        Returns:
            Dim italic style carrying the `@click` hit-target meta.
        """
        # `@click` meta is what Textual uses for Markdown links; body text stays
        # free of meta so regular clicks select/copy without toggling. The
        # link-* rules in `DEFAULT_CSS` keep Textual's automatic link styling
        # from overriding the dim italic.
        return TStyle(dim=True, italic=True) + TStyle.from_meta(
            {"@click": "toggle_expand"}
        )

    @classmethod
    def _collapse_hint_content(cls, collapsed: _UserMessageCollapsed) -> Content:
        """Build the clickable "show full message" affordance.

        Args:
            collapsed: The collapse result describing the elided middle.

        Returns:
            Dim hit-target Content wired to `action_toggle_expand`.
        """
        ellipsis = get_glyphs().ellipsis
        # A single-line paste (base64 blob, minified JSON) hides no newlines, so
        # "+0 lines" would read as "nothing is hidden" exactly when the most is.
        if collapsed.hidden_lines:
            amount = f"+{collapsed.hidden_lines:,} lines"
        else:
            amount = f"+{collapsed.hidden_chars:,} characters"
        return Content.styled(
            f"{ellipsis} {amount} · click or Ctrl+O to show full message",
            cls._hint_style(),
        )

    @classmethod
    def _expand_hint_content(cls) -> Content:
        """Build the clickable "collapse" affordance shown when expanded.

        Returns:
            Dim hit-target Content wired to `action_toggle_expand`.
        """
        return Content.styled("click or Ctrl+O to collapse", cls._hint_style())

    def render(self) -> Content:
        """Render the styled user message.

        Returns:
            Styled Content with mode prefix and highlighted mentions. Long
            messages are collapsed by default with a clickable expand
            affordance; when expanded they show the full body plus a collapse
            hint. Select-all still uses `_build_full_render` (no hints).
        """
        colors = theme.get_theme_colors(self)

        # Use mode-specific prefix indicator when content starts with a
        # mode trigger character (e.g. "!" for shell, "/" for commands).
        # The display glyph may differ from the trigger (e.g. "$" for shell).
        prefix, body = self._prefix_and_body()
        parts: list[str | tuple[str, str] | Content] = [prefix]
        collapse = _collapse_user_message(body)

        if isinstance(collapse, _UserMessageFull):
            self._append_highlighted_body(parts, body, colors=colors)
            return Content.assemble(*parts)

        if self._expanded:
            self._append_highlighted_body(parts, body, colors=colors)
            parts.extend(("\n", self._expand_hint_content()))
            return Content.assemble(*parts)

        # Collapsed: head + clickable elision line + tail. The middle marker is
        # the affordance (not a second trailing line) so the collapse stays
        # one glanceable region instead of an invisible middle ellipsis. Head
        # and tail come from the collapse result rather than being re-sliced
        # here, so the reported amount always describes the gap on screen.
        self._append_highlighted_body(parts, collapse.head, colors=colors)
        parts.extend(("\n", self._collapse_hint_content(collapse), "\n"))
        self._append_highlighted_body(parts, collapse.tail, colors=colors)
        return Content.assemble(*parts)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Match the pointer shape to the cell under the mouse."""
        self.styles.pointer = _pointer_shape_for(event)

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the message."""
        self.styles.pointer = "default"


class QueuedUserMessage(Static):
    """Widget displaying a queued (pending) user message in grey.

    This is an ephemeral widget that gets removed when the message is dequeued.
    """

    DEFAULT_CSS = """
    QueuedUserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $panel;
        opacity: 0.6;
    }
    """
    """Dimmed border + reduced opacity to distinguish queued messages from sent ones."""

    def __init__(
        self, content: str, *, detect_mode: bool = True, **kwargs: Any
    ) -> None:
        """Initialize a queued user message.

        Args:
            content: The message content
            detect_mode: When `True` (default), a leading mode trigger (`/`,
                `!`, `!!`) is rendered with its shell/command glyph. Set to
                `False` for text queued as literal agent input (e.g. via
                `-m`/`--message`), so a leading slash (like a file path) renders
                as a plain user message rather than a slash command.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content
        self._detect_mode = detect_mode

    def on_mount(self) -> None:
        """Add ASCII border class when in ASCII mode."""
        if is_ascii_mode():
            self.add_class("-ascii")

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return selected text, preferring the full content over the render.

        See `UserMessage.get_selection`: full-message selections extract from
        the untruncated content, partial selections defer to the on-screen
        render so offsets stay aligned.

        Args:
            selection: The active selection geometry.

        Returns:
            The `(text, ending)` selection with the prefix removed, or `None`.
        """
        if selection.end is not None:
            return _strip_prompt_prefix(super().get_selection(selection), selection)
        text = str(self._build_full_render())
        return _strip_prompt_prefix((selection.extract(text), "\n"), selection)

    def _prefix_and_body(self) -> tuple[tuple[str, str], str]:
        """Compute the muted mode prefix and body with its trigger stripped.

        Returns:
            A `(prefix, body)` pair where `prefix` is a `(text, style)` tuple.
        """
        colors = theme.get_theme_colors(self)
        content = self._content
        mode_match = detect_mode_prefix(content) if self._detect_mode else None
        if mode_match:
            prefix_text, mode = mode_match
            glyph = MODE_DISPLAY_GLYPHS.get(mode, prefix_text[0])
            return (f"{glyph} ", f"bold {colors.muted}"), content[len(prefix_text) :]
        return ("> ", f"bold {colors.muted}"), content

    def _build_full_render(self) -> Content:
        """Build a Content from the full content without display truncation.

        Returns:
            Content with the mode prefix glyph and the full message body.
        """
        prefix, body = self._prefix_and_body()
        return Content.assemble(prefix, body)

    def text_select_all(self) -> None:
        """Select the message body without the prompt prefix glyph."""
        _select_prompt_body(self)

    def render(self) -> Content:
        """Render the queued user message (greyed out).

        Returns:
            Styled Content with dimmed prefix and body.
        """
        colors = theme.get_theme_colors(self)
        prefix, content = self._prefix_and_body()
        content = _truncate_for_display(content)
        return Content.assemble(prefix, (content, colors.muted))

    def on_mouse_move(self, event: MouseMove) -> None:
        """Match the pointer shape to the cell under the mouse."""
        self.styles.pointer = _pointer_shape_for(event)

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the message."""
        self.styles.pointer = "default"


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by `---` markers.

    Args:
        text: Raw `SKILL.md` content.

    Returns:
        Body text with frontmatter removed and leading whitespace stripped.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return text
    # Find closing --- (skip the opening line)
    end = stripped.find("\n---", 3)
    if end == -1:
        return text
    # Skip past the closing --- and its trailing newline
    after = end + 4  # len("\n---")
    return stripped[after:].lstrip("\n")


class _SkillToggle(Static):
    """Clickable header/hint area for toggling skill body expansion.

    Referenced by name in `SkillMessage._on_toggle_click`'s `@on(Click)`
    CSS selector — rename with care.
    """


class SkillMessage(Vertical):
    """Widget displaying a skill invocation with collapsible body.

    Shows skill name, source badge, description, and user args as a compact
    header. The full SKILL.md body (frontmatter stripped) is hidden behind a
    preview/expand toggle (click or Ctrl+O).  The expanded view renders
    markdown via Rich's `Markdown` inside a single `Static` widget.

    Visibility is driven by a CSS class (`-expanded`) toggled via a Textual
    reactive `var`. Click handlers are scoped to the header and hint widgets
    (`_SkillToggle`) so clicks on the rendered markdown body do not trigger
    expansion toggles (preserving text selection, for instance).
    """

    DEFAULT_CSS = """
    SkillMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $skill;
    }

    SkillMessage .skill-header {
        height: auto;
    }

    SkillMessage .skill-description {
        color: $text-muted;
        margin-left: 3;
    }

    SkillMessage .skill-args {
        margin-left: 3;
        margin-top: 0;
    }

    SkillMessage #skill-md {
        margin-left: 3;
        margin-top: 0;
        padding: 0;
        display: none;
    }

    SkillMessage .skill-hint {
        margin-left: 3;
        color: $text-muted;
    }

    SkillMessage.-expanded #skill-md {
        display: block;
    }

    SkillMessage:hover {
        border-left: wide $skill-hover;
    }
    """

    _PREVIEW_LINES = 4
    _PREVIEW_CHARS = 300

    _expanded: var[bool] = var(False, toggle_class="-expanded")

    def __init__(
        self,
        skill_name: str,
        description: str = "",
        source: str = "",
        body: str = "",
        args: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize a skill message.

        Args:
            skill_name: Skill identifier.
            description: Short description of the skill.
            source: Origin label (e.g., `'built-in'`, `'user'`).
            body: Full SKILL.md content (frontmatter included).
            args: User-provided arguments.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._skill_name = skill_name
        self._description = description
        self._source = source
        self._body = body
        self._stripped_body = _strip_frontmatter(body)
        self._args = args
        self._md_widget: Static | None = None
        self._hint_widget: _SkillToggle | None = None
        self._deferred_expanded: bool = False
        self._md_rendered: bool = False

    def compose(self) -> ComposeResult:
        """Compose the skill message layout.

        Yields:
            Widgets for header, description, args, and collapsible body.
        """
        colors = theme.get_theme_colors()
        source_tag = f" [{self._source}]" if self._source else ""
        yield _SkillToggle(
            Content.styled(
                f"/ skill:{self._skill_name}{source_tag}",
                f"bold {colors.skill}",
            ),
            classes="skill-header",
        )
        if self._description:
            yield _SkillToggle(
                Content.styled(self._description, "dim"),
                classes="skill-description",
            )
        if self._args:
            yield Static(
                Content.assemble(
                    ("User request: ", "bold"),
                    self._args,
                ),
                classes="skill-args",
            )
        yield Static("", id="skill-md")
        yield _SkillToggle("", classes="skill-hint", id="skill-hint")

    def on_mount(self) -> None:
        """Cache widget references, render initial state.

        Ordering matters: widget refs must be cached before `_prepare_body`
        or `_deferred_expanded` assignment, because either may set
        `_expanded` which fires `watch__expanded` synchronously.
        """
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border_left = ("ascii", colors.skill)

        self._md_widget = self.query_one("#skill-md", Static)
        self._hint_widget = self.query_one("#skill-hint", _SkillToggle)

        body = self._stripped_body.strip()
        if body:
            self._prepare_body(body)

        if self._deferred_expanded:
            self._expanded = self._deferred_expanded
            self._deferred_expanded = False

    def _prepare_body(self, body: str) -> None:
        """Set initial hint text. Full body render is deferred to first expand.

        Args:
            body: Stripped markdown body text.
        """
        lines = body.split("\n")
        total_lines = len(lines)
        needs_truncation = (
            total_lines > self._PREVIEW_LINES or len(body) > self._PREVIEW_CHARS
        )

        if needs_truncation:
            remaining = total_lines - self._PREVIEW_LINES
            ellipsis = get_glyphs().ellipsis
            if self._hint_widget:
                self._hint_widget.update(
                    Content.styled(
                        f"{ellipsis} {remaining} more lines"
                        " — click or Ctrl+O to expand",
                        "dim italic",
                    )
                )
        else:
            # Short body — show fully rendered, no preview needed.
            self._ensure_md_rendered(body)
            self._expanded = True

    def _ensure_md_rendered(self, body: str) -> None:
        """Render markdown into the Static widget on first call, then no-op.

        Args:
            body: Stripped markdown body text.
        """
        if self._md_rendered or not self._md_widget:
            return
        try:
            from rich.markdown import Markdown as RichMarkdown

            self._md_widget.update(RichMarkdown(body))
        except Exception:
            logger.warning(
                "Failed to render skill body as markdown; falling back to plain text",
                exc_info=True,
            )
            self._md_widget.update(body)
        self._md_rendered = True

    def toggle_body(self) -> None:
        """Toggle between preview and full body display."""
        if not self._stripped_body.strip():
            return
        self._expanded = not self._expanded

    def watch__expanded(self, expanded: bool) -> None:
        """Lazy-render markdown on first expand; update hint text."""
        body = self._stripped_body.strip()
        if not body:
            return

        if expanded:
            self._ensure_md_rendered(body)

        if not self._hint_widget:
            return

        lines = body.split("\n")
        total_lines = len(lines)
        needs_truncation = (
            total_lines > self._PREVIEW_LINES or len(body) > self._PREVIEW_CHARS
        )

        if not needs_truncation:
            # Short body — always fully visible, no hint needed.
            self._hint_widget.display = False
            return

        if expanded:
            self._hint_widget.update(
                Content.styled("click or Ctrl+O to collapse", "dim italic")
            )
        else:
            remaining = total_lines - self._PREVIEW_LINES
            ellipsis = get_glyphs().ellipsis
            self._hint_widget.update(
                Content.styled(
                    f"{ellipsis} {remaining} more lines — click or Ctrl+O to expand",
                    "dim italic",
                )
            )

    @on(Click, "_SkillToggle")
    def _on_toggle_click(self, event: Click) -> None:
        """Toggle expansion when header or hint is clicked."""
        event.stop()
        if self._stripped_body.strip():
            self.toggle_body()


class AssistantMessage(Vertical):
    """Widget displaying an assistant message with markdown support.

    Uses MarkdownStream for smoother streaming instead of re-rendering
    the full content on each update. Once a stream finishes, the message
    is re-rendered from the complete source via `Markdown.update()` to
    work around Textualize/textual#6518: `MarkdownFence._update_from_block`
    refreshes the visible `Label` but leaves `_highlighted_code` pinned to
    the first chunk, so any later recompose (click, focus change, theme
    update) re-yields the stale value and wrapped fenced-code bodies vanish.
    A full re-parse rebuilds every fence with correct internal state.

    Streamed tokens are coalesced in `_pending_append` and flushed to the
    `MarkdownStream` on a throttled timer (`_STREAM_FLUSH_INTERVAL`). Writing
    every token immediately forced a markdown re-parse per chunk on the UI
    event loop, which starved keyboard input while the model streamed.
    Batching the writes keeps the event loop free so typing stays responsive.
    """

    _STREAM_FLUSH_INTERVAL: ClassVar[float] = 0.1
    """Seconds between coalesced flushes of streamed text to the markdown widget."""

    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    AssistantMessage Markdown {
        padding: 0;
        margin: 0;
    }

    /* Markdown blocks carry a bottom margin for inter-block spacing; drop it
       on the final block so the message has no trailing blank row. */
    AssistantMessage Markdown > *:last-child {
        margin-bottom: 0;
    }
    """

    def __init__(
        self, content: str = "", *, local_only: bool = False, **kwargs: Any
    ) -> None:
        """Initialize an assistant message.

        Args:
            content: Initial markdown content
            local_only: `True` when the content came from the client rather
                than the agent — currently only non-incognito `!` shell
                output, which borrows this widget for its markdown rendering
                and streaming. Callers that ask "did the agent do anything in
                this thread" must not count such a message.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._local_only = local_only
        self._content_parts: list[str] = [content] if content else []
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None
        self._pending_append = ""
        self._flush_timer: Timer | None = None

    @property
    def _content(self) -> str:
        """Full message text, materialized from streamed chunks on access."""
        if len(self._content_parts) > 1:
            self._content_parts = ["".join(self._content_parts)]
        return self._content_parts[0] if self._content_parts else ""

    @_content.setter
    def _content(self, value: str) -> None:
        self._content_parts = [value] if value else []

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Compose the assistant message layout.

        Yields:
            Markdown widget for rendering assistant content.
        """
        from textual.widgets import Markdown

        yield Markdown("", id="assistant-content", open_links=False)

    def on_mount(self) -> None:
        """Store reference to markdown widget."""
        from textual.widgets import Markdown

        self._markdown = self.query_one("#assistant-content", Markdown)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Match the pointer shape to the cell under the mouse."""
        self.styles.pointer = _pointer_shape_for(event)

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the message."""
        self.styles.pointer = "default"

    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open Markdown links with the same toast feedback as style links."""
        event.stop()
        await open_checked_url_async(event.href, app=self.app, notify_on_success=True)

    def _get_markdown(self) -> Markdown:
        """Get the markdown widget, querying if not cached.

        Returns:
            The Markdown widget for this message.
        """
        if self._markdown is None:
            from textual.widgets import Markdown

            self._markdown = self.query_one("#assistant-content", Markdown)
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        """Ensure the markdown stream is initialized.

        Returns:
            The MarkdownStream instance for streaming content.
        """
        if self._stream is None:
            from textual.widgets import Markdown

            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    async def append_content(self, text: str) -> None:
        """Append streamed content, coalescing writes onto a throttled timer.

        Tokens are buffered in `_pending_append` and written to the
        `MarkdownStream` at most once per `_STREAM_FLUSH_INTERVAL` so the UI
        event loop stays free to process keypresses while the model streams.

        Args:
            text: Text to append
        """
        if not text:
            return
        self._content_parts.append(text)
        self._pending_append += text
        if self._flush_timer is None:
            self._flush_timer = self.set_interval(
                self._STREAM_FLUSH_INTERVAL, self._flush_pending_append
            )

    async def _flush_pending_append(self) -> None:
        """Write any buffered streamed text to the markdown stream.

        Runs from a Textual timer callback, where an unhandled exception
        escalates to `App._handle_exception` and tears down the whole REPL.
        On a transient write failure the buffer is restored (re-prepended
        ahead of any text that arrived in the meantime) so the next tick
        retries instead of silently dropping the fragment.
        """
        if not self._pending_append:
            return
        pending = self._pending_append
        self._pending_append = ""
        try:
            stream = self._ensure_stream()
            await stream.write(pending)
        except Exception:  # a render hiccup must not crash the app
            self._pending_append = pending + self._pending_append
            logger.exception("Failed to flush streamed markdown fragment")

    def _stop_flush_timer(self) -> None:
        """Cancel the coalescing flush timer if it is running."""
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

    async def write_initial_content(self) -> None:
        """Write initial content if provided at construction time."""
        if self._content:
            await self._get_markdown().update(self._content)

    async def stop_stream(self) -> None:
        """Stop the streaming and finalize the content."""
        self._stop_flush_timer()
        await self._flush_pending_append()
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
            await self._get_markdown().update(self._content)

    async def set_content(self, content: str) -> None:
        """Set the full message content.

        Cancels any active stream and renders the new content with a
        single `Markdown.update()` (avoiding a redundant intermediate
        update of the in-flight content).

        Args:
            content: The markdown content to display
        """
        self._stop_flush_timer()
        self._pending_append = ""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        self._content = content
        if self._markdown:
            await self._markdown.update(content)


_ToolStatus = Literal["pending", "running", "success", "error", "rejected", "skipped"]
"""The full set of lifecycle states a tool call can hold.

Kept as a closed `Literal` so `ty` flags typos at the assignment sites and so
the grouping predicates (`is_success`/`is_failed`/`is_pending`) partition a
known universe.
"""

_TOOL_AWAITING_APPROVAL_ACCESSORY_CLASS = "-tool-awaiting-approval-accessory"
"""Marker class hiding a tool's accessories while an approval prompt replaces it.

Deliberately distinct from `_TOOL_GROUP_COLLAPSED_ACCESSORY_CLASS`: a footer can
be hidden for more than one reason at once, and releasing one reason must not
un-hide a footer still hidden by another. Merging reasons into a single class
would make
`ToolGroupSummary._release_collapsible` reveal a footer whose tool is still
hidden behind an approval prompt.

Applied with `set_class` rather than by assigning `display`. An inline `display`
permanently outranks the CSS cascade, so assigning it here would strand the
footer against the user's `/timestamps` preference forever. Styled in
`app.tcss`, which relies on rule order to win the specificity tie against that
preference's own class.
"""

_TOOL_SUPERSEDED_ACCESSORY_CLASS = "-tool-superseded-accessory"
"""Hides a row's decorations when its diff has taken the row's place.

A third, independent hide reason. See `_TOOL_AWAITING_APPROVAL_ACCESSORY_CLASS`
for why each reason gets its own class (they must release independently) and
why it is applied with `set_class` rather than `display`.
"""


class ToolCallMessage(Vertical):
    """Widget displaying a tool call with collapsible output.

    Tool outputs are shown as a 3-line preview by default.
    Press Ctrl+O to expand/collapse the full output.
    Shows an animated "Running..." indicator while the tool is executing.
    """

    DEFAULT_CSS = """
    ToolCallMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $tool;
    }

    ToolCallMessage .tool-header {
        height: auto;
        color: $tool;
        text-style: bold;
    }

    ToolCallMessage .tool-task-desc {
        color: $text-muted;
        margin-left: 3;
        text-style: italic;
    }

    ToolCallMessage .tool-args {
        color: $text-muted;
        margin-left: 3;
    }

    ToolCallMessage .tool-status {
        margin-left: 3;
    }

    ToolCallMessage .tool-status.pending {
        color: $warning;
    }

    ToolCallMessage .tool-status.success {
        color: $success;
    }

    ToolCallMessage .tool-status.error {
        color: $error;
    }

    ToolCallMessage .tool-status.rejected {
        color: $warning;
    }

    ToolCallMessage .tool-reject-reason {
        margin-left: 3;
        margin-top: 0;
        height: auto;
        color: $text-muted;
    }

    ToolCallMessage .tool-output-row {
        layout: horizontal;
        height: auto;
        width: 1fr;
    }

    /* Fixed gutter holds the output glyph so soft-wrapped content lines stay
       aligned to a single hanging indent instead of falling under the glyph. */
    ToolCallMessage .tool-output-gutter {
        width: 2;
        height: 1;
        color: $text-muted;
    }

    ToolCallMessage .tool-output {
        margin-left: 0;
        margin-top: 0;
        padding: 0;
        height: auto;
        width: 1fr;
    }

    ToolCallMessage .tool-output-preview {
        margin-left: 0;
        margin-top: 0;
        width: 1fr;
    }

    ToolCallMessage .tool-output-hint {
        margin-left: 0;
        color: $text-muted;
    }

    /* Terminal outcome tints the row: green success, red error, amber
       rejected/skipped. A faint background keeps text readable across
       light/dark/ansi themes while the border carries the primary signal. */
    ToolCallMessage.-status-success {
        border-left: wide $success;
        background: $success 8%;
    }

    ToolCallMessage.-status-error {
        border-left: wide $error;
        background: $error 10%;
    }

    ToolCallMessage.-status-rejected,
    ToolCallMessage.-status-skipped {
        border-left: wide $warning;
        background: $warning 8%;
    }

    ToolCallMessage:hover {
        border-left: wide $tool-hover;
    }
    """
    """Left border tracks tool lifecycle; hover brightens for interactivity."""

    _PREVIEW_LINES = 6
    """Maximum number of lines to show in preview mode."""

    _PREVIEW_CHARS = 400
    """Maximum number of characters to show in preview mode."""

    _JS_EVAL_INLINE_RESULT_MAX = 80
    """Maximum single-line `js_eval` result length rendered inline.

    Inline rendering uses `result: value` rather than a standalone labeled block.
    """

    _TASK_DESC_MAX_LENGTH = 120
    """Maximum `task` description length shown before it is truncated.

    A longer description collapses to at most this many characters (trailing
    whitespace trimmed) with a trailing ellipsis and becomes expandable via
    click or Ctrl+O.
    """

    _RUNNING_TIMER_THRESHOLD_SECS = 10
    """Seconds a tool must run before the elapsed-time counter appears.

    Short tool calls finish well under this threshold, so the timer would only
    flicker on briefly; suppressing it until the tool is genuinely slow keeps
    the "Running..." row quiet for the common case.
    """

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a tool call message.

        Args:
            tool_name: Name of the tool being called
            args: Tool arguments (optional)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._args = args or {}
        self._status: _ToolStatus = "pending"  # Waiting for approval or auto-approve
        self._output: str = ""
        self._expanded: bool = False
        self._args_expanded: bool = False
        self._task_desc_expanded: bool = False
        # User-provided reason attached to a HITL reject decision (if any).
        self._reject_reason: str | None = None
        # Widget references (set in on_mount)
        self._status_widget: Static | None = None
        self._header_widget: Static | None = None
        self._task_desc_widget: Static | None = None
        self._task_desc_hint_widget: Static | None = None
        self._args_widget: Static | None = None
        self._args_hint_widget: Static | None = None
        self._preview_widget: Static | None = None
        self._preview_row: Horizontal | None = None
        self._hint_widget: Static | None = None
        self._full_widget: Static | None = None
        self._full_row: Horizontal | None = None
        self._reject_reason_widget: Static | None = None
        # Animation state
        self._spinner_position = 0
        self._start_time: float | None = None
        self._duration: float | None = None
        self._animation_timer: Timer | None = None
        # Terminal success this row earned but has not rendered. See
        # `defer_success`; `_deferred_success_settled` separates "still awaiting
        # the richer result" from "already fell back to the summary".
        self._deferred_success_output: str | None = None
        self._deferred_success_settled: bool = False
        # One-shot guard so `_format_ask_user_output` reports unusable `questions`
        # args once per widget rather than on every re-render.
        self._ask_user_args_warned: bool = False
        # Deferred state for hydration (set by MessageData.to_widget)
        self._deferred_status: str | None = None
        self._deferred_output: str | None = None
        self._deferred_duration: float | None = None
        self._deferred_expanded: bool = False
        self._deferred_reject_reason: str | None = None
        # Whether the widget is currently hidden because an approval prompt
        # is rendering the same content (see `set_awaiting_approval`).
        self._awaiting_approval: bool = False
        # Transcript decorations that must follow approval visibility without
        # losing their independent user-controlled visibility state.
        self._visibility_accessories: list[Widget] = []
        self._diff_superseded: bool = False
        self._self_hidden: bool = False
        self._has_display_caveat: bool = False

    def compose(self) -> ComposeResult:
        """Compose the tool call message layout.

        Yields:
            Widgets for header, arguments, status, and output display.
        """
        tool_label = format_tool_display(self._tool_name, self._args)
        yield Static(tool_label, markup=False, classes="tool-header", id="tool-header")
        # Task: dedicated description line (dim, truncated). A long description
        # collapses to a truncated preview that expands on click or Ctrl+O.
        if self._tool_name == "task":
            if self._task_description():
                yield Static(
                    self._task_desc_content(),
                    classes="tool-task-desc",
                    id="task-desc",
                )
                yield Static("", classes="tool-output-hint", id="task-desc-hint")
        # Only show args for tools where header doesn't capture the key info
        elif self._tool_name not in _TOOLS_WITH_HEADER_INFO:
            args = self._filtered_args()
            if args:
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in list(args.items())[:_MAX_INLINE_ARGS]
                )
                if len(args) > _MAX_INLINE_ARGS:
                    args_str += ", ..."
                yield Static(
                    Content.from_markup("[dim]($args)[/dim]", args=args_str),
                    classes="tool-args",
                )
        # Collapsed argument detail for tools whose args are too noisy inline.
        # Mounted for every tool but only populated when `has_expandable_args` is True.
        yield Static("", classes="tool-args", id="args-full")
        yield Static("", classes="tool-output-hint", id="args-hint")
        # Status - shows running animation while pending, then final status
        yield Static("", classes="tool-status", id="status")
        # Optional HITL reject reason (only shown when user rejected with a message)
        yield Static("", classes="tool-reject-reason", id="reject-reason")
        # Output area - hidden initially, shown when output is set. The glyph
        # lives in a fixed-width gutter so wrapped content aligns to a single
        # hanging indent rather than wrapping back under the glyph.
        output_prefix = get_glyphs().output_prefix
        yield Horizontal(
            Static(output_prefix, classes="tool-output-gutter"),
            Static("", classes="tool-output-preview", id="output-preview"),
            classes="tool-output-row",
            id="output-preview-row",
        )
        yield Horizontal(
            Static(output_prefix, classes="tool-output-gutter"),
            Static("", classes="tool-output", id="output-full"),
            classes="tool-output-row",
            id="output-full-row",
        )
        yield Static("", classes="tool-output-hint", id="output-hint")

    def on_mount(self) -> None:
        """Cache widget references and hide all status/output areas initially."""
        if is_ascii_mode():
            self.add_class("-ascii")

        self._status_widget = self.query_one("#status", Static)
        self._header_widget = self.query_one("#tool-header", Static)
        try:
            self._task_desc_widget = self.query_one("#task-desc", Static)
            self._task_desc_hint_widget = self.query_one("#task-desc-hint", Static)
        except NoMatches:
            # Only mounted for `task` calls that carry a description.
            self._task_desc_widget = None
            self._task_desc_hint_widget = None
        self._args_widget = self.query_one("#args-full", Static)
        self._args_hint_widget = self.query_one("#args-hint", Static)
        self._preview_widget = self.query_one("#output-preview", Static)
        self._preview_row = self.query_one("#output-preview-row", Horizontal)
        self._hint_widget = self.query_one("#output-hint", Static)
        self._full_widget = self.query_one("#output-full", Static)
        self._full_row = self.query_one("#output-full-row", Horizontal)
        self._reject_reason_widget = self.query_one("#reject-reason", Static)
        # Hide everything initially - status only shown when running or on error/reject
        self._status_widget.display = False
        self._args_widget.display = False
        self._args_hint_widget.display = False
        self._preview_row.display = False
        self._hint_widget.display = False
        self._full_row.display = False
        self._reject_reason_widget.display = False
        self._update_args_display()
        self._update_task_desc_display()

        # Restore deferred state if this widget was hydrated from data
        self._restore_deferred_state()
        # `to_widget` sets `_diff_superseded` before mount, but not every
        # `_restore_deferred_state` branch applies visibility. Applied here so
        # hiding does not depend on which branch a tool takes.
        self._apply_own_visibility()

    def _restore_deferred_state(self) -> None:
        """Restore state from deferred values (used when hydrating from data)."""
        if self._deferred_status is None:
            return

        status = self._deferred_status
        output = self._deferred_output or ""
        duration = self._deferred_duration
        self._expanded = self._deferred_expanded
        if self._deferred_reject_reason:
            self._reject_reason = self._deferred_reject_reason

        # Clear deferred values
        self._deferred_status = None
        self._deferred_output = None
        self._deferred_duration = None
        self._deferred_expanded = False
        self._deferred_reject_reason = None

        # Restore based on status (don't restart animations for running tools)
        colors = theme.get_theme_colors(self)
        match status:
            case "success":
                self._status = "success"
                self._output = output
                self._duration = duration
                self._apply_status_class("success")
                if self._tool_name in _TIMED_SUCCESS_TOOLS and duration is not None:
                    self._show_timed_success_status(duration)
                else:
                    self._show_success_status()
                self._update_output_display()
            case "error":
                self._status = "error"
                self._output = output
                self._apply_status_class("error")
                if self._status_widget:
                    self._status_widget.add_class("error")
                    error_icon = get_glyphs().error
                    self._status_widget.update(
                        Content.styled(f"{error_icon} Error", colors.error)
                    )
                    self._status_widget.display = True
                self._update_output_display()
            case "rejected":
                self._status = "rejected"
                self._apply_status_class("rejected")
                if self._status_widget:
                    self._status_widget.add_class("rejected")
                    error_icon = get_glyphs().error
                    self._status_widget.update(
                        Content.styled(f"{error_icon} Rejected", colors.warning)
                    )
                    self._status_widget.display = True
                self._update_reject_reason_display()
            case "skipped":
                self._status = "skipped"
                self._apply_status_class("skipped")
                if self._status_widget:
                    self._status_widget.add_class("rejected")
                    self._status_widget.update(Content.styled("- Skipped", "dim"))
                    self._status_widget.display = True
            case "running":
                # For running tools, show static "Running..." without animation
                # (animations shouldn't be restored for archived tools)
                self._status = "running"
                if self._status_widget:
                    self._status_widget.add_class("pending")
                    frame = get_glyphs().spinner_frames[0]
                    self._status_widget.update(
                        Content.styled(f"{frame} Running...", colors.warning)
                    )
                    self._status_widget.display = True
            case _:
                # pending or unknown - leave as default
                pass

    def set_running(self) -> None:
        """Mark the tool as running (approved and executing).

        Call this when approval is granted to start the running animation.
        """
        if self._status == "running":
            return  # Already running

        self._status = "running"
        self._duration = None
        self._start_time = time()
        if self._status_widget:
            self._status_widget.add_class("pending")
            self._status_widget.display = True
        self._update_running_animation()
        self._animation_timer = self.set_interval(0.1, self._update_running_animation)

    def _update_running_animation(self) -> None:
        """Update the running spinner animation."""
        if self._status != "running" or self._status_widget is None:
            return

        spinner_frames = get_glyphs().spinner_frames
        frame = spinner_frames[self._spinner_position]
        self._spinner_position = (self._spinner_position + 1) % len(spinner_frames)

        elapsed = ""
        if self._start_time is not None:
            elapsed_secs = int(time() - self._start_time)
            if elapsed_secs >= self._RUNNING_TIMER_THRESHOLD_SECS:
                elapsed = f" ({format_duration(elapsed_secs)})"

        text = f"{frame} Running...{elapsed}"
        self._status_widget.update(
            Content.styled(text, theme.get_theme_colors(self).warning)
        )

    def pause_running(self) -> None:
        """Pause the running spinner while the tool awaits a user decision.

        Reverts the row to its pending appearance (status hidden) and stops the
        animation so a tool blocked on HITL approval or `ask_user` input does
        not misleadingly display "Running...". Resume with `set_running`, which
        restarts the elapsed timer from the moment execution actually begins.
        """
        if self._status != "running":
            return
        self._stop_animation()
        self._status = "pending"
        self._start_time = None
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.display = False

    def _stop_animation(self) -> None:
        """Stop the running animation."""
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

    def _apply_status_class(self, status: str) -> None:
        """Tint the whole row to match a terminal outcome.

        Swaps the `-status-*` CSS class so the row border and background
        reflect success/error/rejected/skipped. Running and pending states keep
        the default `$tool` accent, so they clear any prior status class.

        Args:
            status: Terminal status name (`success`, `error`, `rejected`,
                `skipped`); any other value clears the tint.
        """
        for name in _STATUS_CLASSES:
            self.remove_class(name)
        class_name = f"-status-{status}"
        if class_name in _STATUS_CLASSES:
            self.add_class(class_name)

    def defer_success(self, output: AskUserRowSummary) -> None:
        """Record a terminal success this row earned but has not yet rendered.

        An answered `ask_user` deliberately stays in `_current_tool_messages` so
        the streamed `ToolMessage` can settle it with the full Q&A transcript.
        That leaves the row non-terminal in the meantime, and every teardown
        sweep treats a still-tracked row as a failure — so without this the row
        renders as rejected or as an agent error, and its `tool.result` reports
        `tool_status="error"`, for a question the user answered normally.

        Args:
            output: Summary to settle with if the `ToolMessage` never arrives.
                Narrowed to `AskUserRowSummary` because `_format_ask_user_output`
                recognizes exactly those values as "no transcript behind this row"
                and suppresses the expand affordance for them. Passing the
                transcript here would strand it unreadable on the row.
        """
        self._deferred_success_output = output
        self._deferred_success_settled = False

    @property
    def deferred_success_output(self) -> str | None:
        """Terminal output for a row that earned a success it did not render.

        Set while the row awaits its richer result and deliberately kept after a
        fallback settle, because a settled row can still be tracked in
        `_current_tool_messages` and swept again later (`textual_adapter`'s
        `finally` backstop). `_dispatch_terminal_tool_result_hooks` reads this as
        the "this row already succeeded" flag, so clearing it on settle would make
        that later sweep report a fabricated failure.
        """
        return self._deferred_success_output

    @property
    def is_awaiting_deferred_result(self) -> bool:
        """Whether this row still expects a richer result to replace its summary.

        Distinct from `deferred_success_output`, which stays set after a fallback
        settle. Callers that must not act on an already-settled row — recovering
        an interrupted turn's `tool_calls`, or imposing a terminal failure — ask
        this instead.
        """
        return self._deferred_success_output is not None and (
            not self._deferred_success_settled
        )

    def clear_deferred_success(self) -> None:
        """Drop the deferred outcome once an authoritative result supersedes it.

        Called when the streamed `ToolMessage` settles the row, so its real
        status wins — including an error, which `set_error` would otherwise
        redirect back to the deferred success.
        """
        self._deferred_success_output = None
        self._deferred_success_settled = False

    def settle_deferred_success(self) -> bool:
        """Settle this row with its deferred success, if it is awaiting one.

        Idempotent: a row that already fell back returns False rather than
        re-rendering, so callers need no `is_awaiting_deferred_result` guard of
        their own. Records that the fallback fired but keeps the output — see
        `deferred_success_output` for why a later sweep still needs to read it.

        Returns:
            True if the row was settled. False if it had no deferred outcome, has
                already settled, or is rejected/skipped so `set_success` would
                ignore it — in each case the caller should record its own terminal
                state.
        """
        output = self._deferred_success_output
        if output is None or self._deferred_success_settled:
            # Mirrors `is_awaiting_deferred_result`, spelled out so the type
            # checker can narrow `output` to `str`.
            return False
        if self._status in {"rejected", "skipped"}:
            return False
        # Before `set_success`, which re-renders synchronously. Nothing in that
        # render path reads this flag today (`_format_ask_user_output` derives
        # "no transcript" from the output value instead, so the suppression also
        # survives rehydration), but ordering the flag first keeps the object
        # consistent for anything the render does reach.
        self._deferred_success_settled = True
        self.set_success(output)
        return True

    def set_success(self, result: str = "") -> None:
        """Mark the tool call as successful.

        For long-running tools (`execute`, `task`) that actually ran (a start
        time was recorded via `set_running`), the elapsed run time is shown via
        `_show_timed_success_status`; every other case routes through
        `_show_success_status`.

        Args:
            result: Tool output/result to display
        """
        if self._status in {"rejected", "skipped"}:
            # A rejected tool (or one skipped due to a sibling rejection) never
            # legitimately becomes successful. A resumed turn can still stream a
            # synthetic ToolMessage for such a tool (see the reasoned-reject path
            # in `textual_adapter`); ignore it so the row keeps its terminal
            # rejected/skipped state instead of flipping.
            return
        elapsed = time() - self._start_time if self._start_time is not None else None
        self._stop_animation()
        self._status = "success"
        # This call owns `_output`, so any caveat a previous completion put
        # there is gone. Clearing here keeps the flag from outliving the
        # sentence it describes and leaving the row unfoldable for no visible
        # reason; `set_success_with_caveat` re-sets it after delegating here.
        self._has_display_caveat = False
        self._duration = (
            elapsed
            if self._tool_name in _TIMED_SUCCESS_TOOLS and elapsed is not None
            else None
        )
        # Strip redundant command success trailers — the UI already conveys
        # success. `ask_user` output is a user-authored Q&A transcript, though,
        # so text that resembles a command trailer must remain verbatim.
        self._output = (
            result
            if self._tool_name == "ask_user"
            else _strip_success_exit_line(result)
        )
        self._apply_status_class("success")
        if self._duration is not None:
            self._show_timed_success_status(self._duration)
        else:
            self._show_success_status()
        self._update_output_display()

    def _show_timed_success_status(self, duration: float) -> None:
        """Render the preserved duration for a completed timed tool call.

        Args:
            duration: Elapsed tool run time in seconds.
        """
        if self._status_widget is None:
            return
        self._status_widget.remove_class("pending")
        self._status_widget.update(
            Content.styled(f"Took {format_duration(duration)}", "dim")
        )
        self._status_widget.display = True

    def _show_success_status(self) -> None:
        """Render the status marker for a completed successful call.

        When the call produces visible output it speaks for itself and the
        status stays hidden; otherwise show a "Success!" marker so a completed
        call isn't left without any outcome indicator. A row already marked as
        replaced by a mounted diff hides entirely.
        """
        if self._status_widget is None:
            return
        self._status_widget.remove_class("pending")
        if self._superseded_by_diff:
            self._apply_own_visibility()
            return
        if self._format_output(self._output, is_preview=False).content.plain.strip():
            self._status_widget.remove_class("success")
            self._status_widget.display = False
            return
        glyph = get_glyphs().checkmark
        colors = theme.get_theme_colors(self)
        self._status_widget.add_class("success")
        self._status_widget.update(Content.styled(f"{glyph} Success!", colors.success))
        self._status_widget.display = True

    @staticmethod
    def can_be_superseded(tool_name: str | None) -> bool:
        """Return whether a diff may stand in for this tool's row.

        The public form of the `_TOOL_SUPERSEDED_BY_DIFF` check, so the adapter
        does not reach for a private constant to ask the same question from a
        different name source. `mark_superseded_by_diff` still enforces it — this
        only lets a caller avoid tripping the warning.

        Args:
            tool_name: Raw name of the tool that produced the row.

        Returns:
            Whether a mounted `DiffMessage` may hide the row.
        """
        return tool_name == _TOOL_SUPERSEDED_BY_DIFF

    def mark_superseded_by_diff(self) -> None:
        """Hide a successful file-tool row after its diff has mounted.

        Rejects any tool other than `_TOOL_SUPERSEDED_BY_DIFF`, leaving the row
        visible and logging at warning — so this is not safe to call
        speculatively. That guard is load-bearing rather than defensive:
        `MessageStore.to_widget` routes a stored flag through this method
        precisely to inherit it, so rehydration cannot hide a row the live path
        would have left visible.
        """
        if self._tool_name != _TOOL_SUPERSEDED_BY_DIFF:
            # A broken invariant, not a routine skip: the caller decided this row
            # was superseded from a *different* name source (the adapter gates on
            # `record.tool_name`), so a divergence leaves an empty-bodied diff
            # rendering "no changes" beside a row that stayed visible.
            logger.warning(
                "mark_superseded_by_diff called on %r; only %r may be superseded",
                self._tool_name,
                _TOOL_SUPERSEDED_BY_DIFF,
            )
            return
        self._diff_superseded = True
        self._apply_own_visibility()

    def set_error(self, error: str) -> None:
        """Mark the tool call as failed.

        Args:
            error: Error message
        """
        if self._status in {"rejected", "skipped"}:
            # A rejected/skipped tool never legitimately errors. A resumed turn
            # can stream a synthetic error ToolMessage for a reasoned-reject tool
            # (see `textual_adapter`); ignore it so the row keeps its rejected
            # state rather than flipping to "Error" (which also left the stale
            # `rejected` CSS class alongside `error`).
            return
        if self.settle_deferred_success():
            # A teardown sweep imposing a generic failure on a row that already
            # succeeded (an answered `ask_user` awaiting its transcript). The
            # authoritative `ToolMessage` calls `clear_deferred_success` first, so
            # a *real* tool error still lands below. `settle_deferred_success` is
            # idempotent, so the redirect fires once: a row that already fell back
            # keeps no immunity against a later genuine error.
            #
            # INFO, not DEBUG: turning a failure into a success is the single
            # highest-stakes decision on this path, and the always-on debug ring
            # buffer that backs the in-app console only captures INFO and above.
            logger.info(
                "Suppressed error on tool row with a deferred success: %s", error
            )
            return
        self._stop_animation()
        self._status = "error"
        self._apply_status_class("error")
        # Not a no-op: `_superseded_by_diff` is gated on success, so this is what
        # reveals a row that was hidden behind a diff before its status flipped.
        self._apply_own_visibility()
        # For shell commands, prepend the full command so users can see what failed
        command = self._args.get("command") if self._tool_name == "execute" else None
        if command and isinstance(command, str) and command.strip():
            self._output = f"$ {command}\n\n{error}"
        else:
            self._output = error
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("error")
            error_icon = get_glyphs().error
            colors = theme.get_theme_colors(self)
            self._status_widget.update(
                Content.styled(f"{error_icon} Error", colors.error)
            )
            self._status_widget.display = True
        # Always show full error - errors should be visible
        self._expanded = True
        self._update_output_display()

    def set_rejected(self, *, reason: str | None = None) -> None:
        """Mark the tool call as rejected by user.

        Args:
            reason: Optional free-text reason supplied via the HITL reject
                widget; rendered as a dim line beneath the status.
        """
        if self.settle_deferred_success():
            # A turn-cancel sweep rejecting every tracked row; an answered
            # `ask_user` among them still succeeded, so it keeps its own outcome.
            # (Interrupt rejections leave these rows tracked instead — see
            # `_pop_rows_not_awaiting_deferred_result`.) INFO for the same reason
            # as the redirect in `set_error`.
            logger.info(
                "Suppressed rejection on tool row with a deferred success: %s", reason
            )
            return
        self._stop_animation()
        self._status = "rejected"
        self._apply_status_class("rejected")
        if reason and reason.strip():
            self._reject_reason = reason.strip()
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")
            error_icon = get_glyphs().error
            text = f"{error_icon} Rejected"
            colors = theme.get_theme_colors(self)
            self._status_widget.update(Content.styled(text, colors.warning))
            self._status_widget.display = True
        self._update_reject_reason_display()

    def _update_reject_reason_display(self) -> None:
        """Render the rejection reason line if a reason is set."""
        if self._reject_reason_widget is None:
            return
        if self._reject_reason:
            self._reject_reason_widget.update(
                Content.from_markup(
                    "[dim italic]Reason: $reason[/dim italic]",
                    reason=self._reject_reason,
                )
            )
            self._reject_reason_widget.display = True
        else:
            self._reject_reason_widget.display = False

    def set_skipped(self) -> None:
        """Mark the tool call as skipped (due to another rejection)."""
        self._stop_animation()
        self._status = "skipped"
        self._apply_status_class("skipped")
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")  # Use same styling as rejected
            self._status_widget.update(Content.styled("- Skipped", "dim"))
            self._status_widget.display = True

    def set_awaiting_approval(self) -> None:
        """Hide the tool call while an approval prompt mirrors its content.

        Used to avoid showing the same shell command in both the streamed tool
        call header and the HITL approval dialog at the same time. The widget
        is restored via `clear_awaiting_approval` once the user decides.
        """
        self._awaiting_approval = True
        self._apply_own_visibility()

    def clear_awaiting_approval(self) -> None:
        """Restore the tool call after `set_awaiting_approval`.

        No-op if `set_awaiting_approval` was not previously called, so the
        method is safe to call unconditionally from a `finally` block.
        """
        if not self._awaiting_approval:
            return
        self._awaiting_approval = False
        self._apply_own_visibility()

    def _register_visibility_accessories(self, *accessories: Widget) -> None:
        """Link transcript decorations whose visibility follows this tool.

        Idempotent: `Widget` uses identity equality, so re-registering the same
        accessory (a regroup folding an already-folded tool) cannot double-add.
        """
        for accessory in accessories:
            if accessory not in self._visibility_accessories:
                self._visibility_accessories.append(accessory)
        self._sync_own_hide_accessories()

    @property
    def has_own_hide_reason(self) -> bool:
        """Whether this row hides itself, regardless of any group collapse.

        Group code must consult this before revealing a row: the two mechanisms
        are independent, so an unconditional `display = True` would reveal a row
        that is hiding for its own reasons.
        """
        return self._awaiting_approval or self._superseded_by_diff

    def _apply_own_visibility(self) -> None:
        """Apply self-hide reasons without disturbing group visibility.

        Only touches `display` when a self-hide reason applies or is being
        released — a row with no self-hide history is left exactly as the group
        set it. Releasing is the narrower guarantee: it restores `display` to
        `True` unconditionally, so a row that was *both* group-collapsed and
        self-hidden would reveal itself into a collapsed group. Not reachable
        today (the one supersedable tool is group-excluded, and rows awaiting
        approval are evicted rather than folded), but a new hide reason that can
        coexist with a group must consult the group's state here.

        The other direction is `has_own_hide_reason`, which group code checks
        before revealing — and which this reads, so the set of hide reasons is
        defined in exactly one place and a new one cannot be honoured by the
        group checks while being ignored here.
        """
        if self.has_own_hide_reason:
            self.display = False
            self._self_hidden = True
        elif self._self_hidden:
            self.display = True
            self._self_hidden = False
        self._sync_own_hide_accessories()

    def _sync_own_hide_accessories(self) -> None:
        """Mirror this row's hide reasons onto linked decorations.

        Tests each reason individually rather than reading `has_own_hide_reason`:
        each carries its own class so the reasons release independently, which the
        aggregate cannot express. A new hide reason needs a class and an entry
        here as well as a term in `has_own_hide_reason`.
        """
        for accessory in self._visibility_accessories:
            accessory.set_class(
                self._awaiting_approval,
                _TOOL_AWAITING_APPROVAL_ACCESSORY_CLASS,
            )
            accessory.set_class(
                self._superseded_by_diff,
                _TOOL_SUPERSEDED_ACCESSORY_CLASS,
            )

    def toggle_output(self) -> None:
        """Toggle expansion of the tool's preview/full output."""
        if not self._output:
            return
        # No-op in both directions when nothing is hidden: the collapsed and
        # expanded forms are identical, so toggling only flickers the hint.
        # This also covers force-expanded errors (see `set_error`).
        if not self._has_expandable_output():
            return
        self._expanded = not self._expanded
        self._update_output_display()

    def toggle_args(self) -> None:
        """Toggle display of collapsed tool arguments."""
        if not self.has_expandable_args:
            return
        self._args_expanded = not self._args_expanded
        self._update_args_display()

    def toggle_task_desc(self) -> None:
        """Toggle between the truncated and full `task` description."""
        if not self.has_expandable_task_desc:
            return
        self._task_desc_expanded = not self._task_desc_expanded
        self._update_task_desc_display()

    def on_click(self, event: Click) -> None:
        """Toggle output/argument/description expansion.

        A click on the header/args region (the truncated command or code line
        and its hint) toggles the collapsible args/code block directly, so an
        `execute` command or `js_eval` program can be expanded even when the
        output below it is *also* expandable. A `task` row routes clicks on its
        description region to the description toggle for the same reason.
        Otherwise prefer toggling output, falling through to the args/code block
        only when the output can't expand — `js_eval` commonly has a short,
        unexpandable result sitting below a multi-line, collapsible code block,
        and the old "output wins whenever it exists" rule left that code block
        stuck.
        """
        event.stop()  # Prevent click from bubbling up and scrolling
        if self.has_expandable_task_desc and self._click_targets_task_desc_region(
            event.widget
        ):
            self.toggle_task_desc()
        elif self.has_expandable_args and self._click_targets_args_region(event.widget):
            self.toggle_args()
        elif self._output and self.has_expandable_output:
            self.toggle_output()
        elif self.has_expandable_args:
            self.toggle_args()
        elif self.has_expandable_task_desc:
            self.toggle_task_desc()

    def _click_targets_args_region(self, widget: object) -> bool:
        """Whether a click landed on the header/args block (not the output).

        Walks up from the clicked widget to `self`, matching the cached
        header, collapsed-args, and args-hint widgets. The walk is bounded so a
        mock or detached node (which never reaches `self` via `.parent`) returns
        `False` instead of looping, preserving the generic "prefer output"
        routing for those cases.

        Returns:
            `True` if the click landed on the header/args region.
        """
        targets = tuple(
            target
            for target in (
                self._header_widget,
                self._args_widget,
                self._args_hint_widget,
            )
            if target is not None
        )
        if not targets:
            # A click can only arrive post-mount, where these refs are always
            # cached, so an empty tuple means a regression nulled them out. Log
            # it rather than silently routing every click to output (mirrors
            # `_update_args_display`).
            logger.debug("_click_targets_args_region: header/args refs not cached")
            return False
        # The header/args/hint widgets are direct children of `self` (a click on
        # rendered text reports a descendant, so the real match depth is 0-1).
        # 8 is generous headroom that also bounds the walk for a detached or mock
        # node, whose `.parent` chain never reaches `self`.
        node = widget
        for _ in range(8):
            if node is None or node is self:
                return False
            if any(node is target for target in targets):
                return True
            node = getattr(node, "parent", None)
        return False

    def _click_targets_task_desc_region(self, widget: object) -> bool:
        """Whether a click landed on the `task` header/description block.

        Mirrors `_click_targets_args_region` but matches the cached header,
        description, and description-hint widgets so a `task` row expands its
        description when clicked, even when its output below is also expandable.

        Returns:
            `True` if the click landed on the header/description region.
        """
        targets = tuple(
            target
            for target in (
                self._header_widget,
                self._task_desc_widget,
                self._task_desc_hint_widget,
            )
            if target is not None
        )
        if not targets:
            logger.debug("_click_targets_task_desc_region: header/desc refs not cached")
            return False
        node = widget
        for _ in range(8):
            if node is None or node is self:
                return False
            if any(node is target for target in targets):
                return True
            node = getattr(node, "parent", None)
        return False

    def _format_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format tool output based on tool type for nicer display.

        Args:
            output: Raw output string
            is_preview: Whether this is for preview (truncated) display

        Returns:
            FormattedOutput with content and optional truncation info.
        """
        # Trim surrounding blank lines and trailing whitespace, but preserve the
        # command's own leading indentation on the first content line. A bare
        # `strip()` would lstrip the first line only — continuation lines keep
        # their indent — so output that indents every row (e.g. `git branch -r`,
        # which prefixes each branch with two spaces) renders with line 0 flush
        # and the rest indented beside the fixed glyph gutter.
        output = output.rstrip().lstrip("\n")
        if not output:
            return FormattedOutput(content=Content(""))

        # Tool-specific formatting using dispatch table
        formatters = {
            "write_todos": self._format_todos_output,
            "ls": self._format_ls_output,
            "read_file": self._format_file_output,
            "write_file": self._format_file_output,
            "edit_file": self._format_edit_file_output,
            "grep": self._format_search_output,
            "glob": self._format_search_output,
            "execute": self._format_shell_output,
            "js_eval": self._format_js_eval_output,
            "web_search": self._format_web_output,
            "fetch_url": self._format_web_output,
            "task": self._format_task_output,
            "ask_user": self._format_ask_user_output,
        }

        formatter = formatters.get(self._tool_name)
        if formatter:
            return formatter(output, is_preview=is_preview)

        return self._format_generic_output(output, is_preview=is_preview)

    def _format_generic_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format output using generic size-based truncation.

        Used for tools with no dedicated formatter, and by a dedicated formatter
        that cannot parse its input and so must still cap an arbitrarily long
        body rather than dumping it into the collapsed row.

        Args:
            output: Tool output. `_format_output` has stripped trailing whitespace
                and leading newlines, but deliberately preserves the first line's
                leading indentation — do not assume it is fully trimmed.
            is_preview: Whether to truncate for the collapsed row.

        Returns:
            FormattedOutput, carrying truncation info only when `is_preview` and
                the body exceeds the line or character threshold.
        """
        if is_preview:
            lines = output.split("\n")
            if len(lines) > self._PREVIEW_LINES:
                return self._format_lines_output(lines, is_preview=True)
            if len(output) > self._PREVIEW_CHARS:
                truncated = output[: self._PREVIEW_CHARS]
                truncation = f"{len(output) - self._PREVIEW_CHARS} more chars"
                return FormattedOutput(
                    content=Content(truncated), truncation=truncation
                )

        # Default: plain text (Content treats input as literal)
        return FormattedOutput(content=Content(output))

    @property
    def has_expandable_output(self) -> bool:
        """Whether collapsed output has hidden content worth a toggle.

        Public wrapper around `_has_expandable_output` so toggle routing (click
        and Ctrl+O) can tell "has output" apart from "has output that can
        actually expand/collapse". `js_eval` results are frequently short and
        unexpandable while the code block above them *is* collapsible, so the
        routing must fall through to args when output cannot toggle.
        """
        return self._has_expandable_output()

    def _is_search_no_result_output(self, output: str) -> bool:
        """Return whether search output is a terminal no-result message.

        These sentinels must match the empty-result strings the SDK emits
        (`format_grep_matches` in `deepagents.backends.utils` and
        `_format_file_paths` in `deepagents.middleware.filesystem`). If those
        change, this silently stops matching and empty searches revert to
        collapsing behind an expand affordance rather than rendering inline.
        """
        if self._tool_name == "grep":
            return output.strip() == "No matches found"
        if self._tool_name == "glob":
            return output.strip() == "No files found"
        return False

    def _has_expandable_output(self) -> bool:
        """Return whether collapsed output has hidden content to expand."""
        output = self._output.strip()
        if not output or self._is_search_no_result_output(output):
            return False

        # Tools in `_COLLAPSE_OUTPUT_BY_DEFAULT` (read_file, grep, glob) collapse
        # their body entirely by default (the header already carries the file
        # path / search pattern), so any result with something to show is
        # expandable regardless of size. The exception is a search that finds
        # nothing: grep/glob return the terminal "No matches found" / "No files
        # found" message, caught by `_is_search_no_result_output` above so it
        # renders inline (see `_update_output_display`) instead of hiding a
        # "nothing found" result behind an expand click. Beyond that, confirm the
        # formatted output is non-empty rather than trusting the raw string —
        # output that formats to blank (all whitespace, or a serialized empty
        # collection like `[]`) has nothing to reveal. Successful `edit_file`
        # similarly hides its redundant success line in the collapsed view while
        # keeping the raw output expandable. This mirrors the empty-output guard
        # in `_update_output_display`, which suppresses any body that would
        # render blank before the collapse branch is reached — the two must move
        # together if that assumption changes. Errors are excluded because
        # `set_error` force-expands every error; treating a short error as
        # always-expandable would offer a collapse that hides it entirely.
        if self._tool_name in _COLLAPSE_OUTPUT_BY_DEFAULT and self._status != "error":
            formatted = self._format_output(output, is_preview=False)
            return bool(formatted.content.plain.strip())
        if self._tool_name == "edit_file" and self._status == "success":
            return True

        # See `_ALWAYS_PREVIEW_TOOLS`: the formatter decides whether these have
        # anything left to reveal, rather than the raw size thresholds below.
        # (A formatter that cannot parse its input may delegate back to them.)
        if self._tool_name in _ALWAYS_PREVIEW_TOOLS:
            return self._format_output(output, is_preview=True).truncation is not None

        lines = output.split("\n")
        if len(lines) > self._PREVIEW_LINES or len(output) > self._PREVIEW_CHARS:
            # The outer size threshold is necessary but not sufficient: only
            # treat output as expandable if the formatter actually hides
            # content. Some formatters cap by line count alone (task and the
            # web fallback, via `_format_task_output` / `_format_lines_output`),
            # so a long single line crosses the char threshold yet renders in
            # full with nothing hidden.
            return self._format_output(output, is_preview=True).truncation is not None

        return False

    def _format_todos_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format write_todos output as a checklist.

        Returns:
            FormattedOutput with checklist content and optional truncation info.
        """
        items = self._parse_todo_items(output)
        if items is None:
            return FormattedOutput(content=Content(output))

        if not items:
            return FormattedOutput(content=Content.styled("No todos", "dim"))

        lines: list[Content] = []
        max_items = 4 if is_preview else len(items)

        # Build stats header
        stats = self._build_todo_stats(items)
        if stats:
            lines.extend([stats, Content("")])

        # Format each item
        lines.extend(
            self._format_single_todo(item, is_preview=is_preview)
            for item in items[:max_items]
        )

        truncation = None
        if is_preview:
            hidden_items = len(items) - max_items
            if hidden_items > 0:
                truncation = f"{hidden_items} more"
            elif any(
                len(self._todo_text(item)) > _MAX_TODO_CONTENT_LEN
                for item in items[:max_items]
            ):
                truncation = "full todo text"

        return FormattedOutput(content=Content("\n").join(lines), truncation=truncation)

    @staticmethod
    def _todo_text(item: dict | str) -> str:
        """Return display text for a todo item.

        Args:
            item: Todo item dictionary or plain string.

        Returns:
            Todo content text.
        """
        if isinstance(item, dict):
            return str(item.get("content", str(item)))
        return str(item)

    def _parse_todo_items(self, output: str) -> list | None:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """Parse todo items from output.

        Returns:
            List of todo items, or None if parsing fails.
        """
        list_match = re.search(r"\[(\{.*\})\]", output.replace("\n", " "), re.DOTALL)
        if list_match:
            try:
                return ast.literal_eval("[" + list_match.group(1) + "]")
            except (ValueError, SyntaxError):
                return None
        try:
            items = ast.literal_eval(output)
            return items if isinstance(items, list) else None
        except (ValueError, SyntaxError):
            return None

    def _build_todo_stats(self, items: list) -> Content:
        """Build stats content for todo list.

        Returns:
            Styled `Content` showing active, pending, and completed counts.
        """
        colors = theme.get_theme_colors(self)
        completed = sum(
            1 for i in items if isinstance(i, dict) and i.get("status") == "completed"
        )
        active = sum(
            1 for i in items if isinstance(i, dict) and i.get("status") == "in_progress"
        )
        pending = len(items) - completed - active

        parts: list[Content] = []
        if active:
            parts.append(Content.styled(f"{active} active", colors.warning))
        if pending:
            parts.append(Content.styled(f"{pending} pending", "dim"))
        if completed:
            parts.append(Content.styled(f"{completed} done", colors.success))
        return Content.styled(" | ", "dim").join(parts) if parts else Content("")

    def _todo_content_width(self, indent_width: int) -> int:
        """Return the todo content wrap width for the current widget size.

        Args:
            indent_width: Display width before todo content starts.

        Returns:
            Width available for todo content wrapping.
        """
        display_width = 0
        for widget in (self._full_widget, self._preview_widget, self):
            if widget and widget.is_mounted and widget.size.width > 0:
                display_width = widget.size.width
                break

        if not display_width:
            try:
                display_width = self.app.size.width
            except NoActiveAppError:
                display_width = _DEFAULT_TODO_WRAP_WIDTH

        # The content widgets measured above live inside the gutter row, so
        # their width already excludes the output glyph column; the guard
        # columns absorb the gutter offset for the self/app fallback width.
        available = display_width - indent_width - _TODO_WRAP_GUARD_COLUMNS
        return max(20, available)

    def _format_todo_line(
        self,
        prefix: Content,
        text: str,
        *,
        is_preview: bool,
        text_style: str | None = None,
    ) -> Content:
        """Format a todo row, wrapping expanded content under the text column.

        Args:
            prefix: Styled status prefix before todo content.
            text: Todo text to render.
            is_preview: Whether the compact preview is being rendered.
            text_style: Optional style for todo content.

        Returns:
            Styled `Content` for one todo row.
        """
        if is_preview and len(text) > _MAX_TODO_CONTENT_LEN:
            text = text[: _MAX_TODO_CONTENT_LEN - 3] + "..."

        if is_preview:
            content = Content.styled(text, text_style) if text_style else Content(text)
            return Content.assemble(prefix, content)

        indent = " " * len(prefix.plain)
        wrapped = textwrap.wrap(
            text,
            width=self._todo_content_width(len(prefix.plain)),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        parts: list[Content] = [prefix]
        for index, line in enumerate(wrapped):
            if index:
                parts.append(Content("\n" + indent))
            content = Content.styled(line, text_style) if text_style else Content(line)
            parts.append(content)
        return Content.assemble(*parts)

    def _format_single_todo(self, item: dict | str, *, is_preview: bool) -> Content:
        """Format a single todo item.

        Args:
            item: Todo item dictionary or plain string.
            is_preview: Whether the compact preview is being rendered.

        Returns:
            Styled `Content` with checkbox and status styling.
        """
        colors = theme.get_theme_colors(self)
        if isinstance(item, dict):
            text = self._todo_text(item)
            status = item.get("status", "pending")
        else:
            text = self._todo_text(item)
            status = "pending"

        glyphs = get_glyphs()
        if status == "completed":
            return self._format_todo_line(
                Content.styled(f"{glyphs.checkmark} done   ", colors.success),
                text,
                is_preview=is_preview,
                text_style="dim",
            )
        if status == "in_progress":
            return self._format_todo_line(
                Content.styled(f"{glyphs.circle_filled} active ", colors.warning),
                text,
                is_preview=is_preview,
            )
        return self._format_todo_line(
            Content.styled(f"{glyphs.circle_empty} todo   ", "dim"),
            text,
            is_preview=is_preview,
        )

    def _format_ls_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format ls output as a clean directory listing.

        Returns:
            FormattedOutput with directory listing and optional truncation info.
        """
        # Try to parse as a Python list (common format)
        try:
            items = ast.literal_eval(output)
            if isinstance(items, list):
                lines: list[Content] = []
                max_items = 5 if is_preview else len(items)
                for item in items[:max_items]:
                    path = Path(str(item))
                    name = path.name
                    if path.suffix in {".py", ".pyx"}:
                        lines.append(Content.styled(name, theme.FILE_PYTHON))
                    elif path.suffix in {".json", ".yaml", ".yml", ".toml"}:
                        lines.append(Content.styled(name, theme.FILE_CONFIG))
                    elif not path.suffix:
                        lines.append(Content.styled(f"{name}/", theme.FILE_DIR))
                    else:
                        lines.append(Content(name))

                truncation = None
                if is_preview and len(items) > max_items:
                    truncation = f"{len(items) - max_items} more"

                return FormattedOutput(
                    content=Content("\n").join(lines), truncation=truncation
                )
        except (ValueError, SyntaxError):
            pass

        # Fallback: plain text
        return FormattedOutput(content=Content(output))

    @staticmethod
    def _compact_line_gutter(output: str) -> str:
        r"""Tighten `read_file`'s line-number gutter for display.

        `read_file` prefixes each row with a right-justified line marker — `N`,
        or `N.M` for a wrapped-line continuation — then two spaces, then the
        original source content. (Output from deepagents versions predating the
        gutter disambiguation in #4561 used the older `cat -n` gutter — a wide
        right-justified number and a tab — which may still surface from cached or
        persisted transcripts.) The model needs the raw gutter for edits, but the
        TUI re-justifies markers to the widest marker actually present, then two
        spaces, mirroring how grep/glob results sit flush left. Source
        indentation after the gutter is preserved untouched.

        The gutter shape is `_READ_FILE_GUTTER_RE`. Lines that don't match a
        gutter shape (e.g. test fixtures or non-numbered output) are passed
        through unchanged.

        Returns:
            The output with compacted gutters, or the original string if no
                line-numbered content was found.
        """
        lines = output.split("\n")
        parsed: list[tuple[str, str] | None] = []
        width = 0
        for line in lines:
            match = _READ_FILE_GUTTER_RE.match(line)
            if match:
                marker, source = match.groups()
                parsed.append((marker, source))
                width = max(width, len(marker))
            else:
                parsed.append(None)

        if width == 0:
            return output

        compacted: list[str] = []
        for line, row in zip(lines, parsed, strict=True):
            if row is None:
                compacted.append(line)
            else:
                marker, source = row
                compacted.append(f"{marker:>{width}}  {source}")
        return "\n".join(compacted)

    def _format_edit_file_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Render edit_file output, hiding success only in the preview.

        On success the collapsed status glyph and the diff already convey the
        outcome, so the "Successfully replaced ..." line is hidden by default.
        The full rendering still shows the raw tool output so clicking the row
        can recover the original message. Errors still render in both modes.

        Returns:
            Empty preview on success, otherwise the file formatter.
        """
        if self._status == "success" and is_preview:
            return FormattedOutput(content=Content(""))
        return self._format_file_output(output, is_preview=is_preview)

    def _format_file_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format file read/write output.

        Preview mode caps both line count and total characters so that files
        with very long lines (minified HTML/JS/CSS) don't wrap and overflow
        the widget.

        Returns:
            FormattedOutput with file content and optional truncation info.
        """
        output = self._compact_line_gutter(output)
        lines = output.split("\n")
        # Files conventionally end in "\n"; the trailing empty element isn't a
        # real line and would inflate truncation counts.
        had_trailing_newline = bool(lines) and not lines[-1]
        if had_trailing_newline:
            lines = lines[:-1]
        max_lines = 4 if is_preview else len(lines)
        char_budget = self._PREVIEW_CHARS if is_preview else None

        shown, chars_used, char_truncated = self._truncate_to_budget(
            lines, max_lines=max_lines, char_budget=char_budget
        )
        parts = [Content(line) for line in shown]
        content = Content("\n").join(parts) if parts else Content("")

        truncation = self._build_truncation_hint(
            output=output,
            lines=lines,
            parts_count=len(parts),
            chars_used=chars_used,
            char_truncated=char_truncated,
            had_trailing_newline=had_trailing_newline,
            is_preview=is_preview,
        )

        return FormattedOutput(content=content, truncation=truncation)

    @staticmethod
    def _truncate_to_budget(
        lines: list[str], *, max_lines: int, char_budget: int | None
    ) -> tuple[list[str], int, bool]:
        """Apply line- and character-count caps to a list of display lines.

        Shared by the file, shell, and search formatters so preview truncation
        stays identical across tool outputs. When `char_budget` is `None` (the
        expanded, non-preview view) only the line cap applies.

        Args:
            lines: Candidate display lines, already cleaned by the caller.
            max_lines: Maximum number of lines to emit.
            char_budget: Maximum characters to emit across all lines, counting
                the newline separators between them, or `None` for no cap.

        Returns:
            The lines to show, the characters consumed (including separators),
            and whether the character budget forced truncation.
        """
        shown: list[str] = []
        chars_used = 0
        char_truncated = False
        for line in lines[:max_lines]:
            display_line = line
            if char_budget is not None:
                separator_cost = 1 if shown else 0
                remaining = char_budget - chars_used - separator_cost
                if remaining <= 0:
                    char_truncated = True
                    break
                if len(line) > remaining:
                    display_line = line[:remaining]
                    char_truncated = True
                chars_used += separator_cost + len(display_line)
            shown.append(display_line)
            if char_truncated:
                break
        return shown, chars_used, char_truncated

    @staticmethod
    def _build_truncation_hint(
        *,
        output: str,
        lines: list[str],
        parts_count: int,
        chars_used: int,
        char_truncated: bool,
        had_trailing_newline: bool,
        is_preview: bool,
        line_unit: Literal["files", "lines"] = "lines",
    ) -> str | None:
        """Compose the truncation hint, preferring line counts over char counts.

        When both the line cap and the char cap were hit, hidden-line count is
        the more useful signal for the user — char counts dominate the hint
        for big files where what they really want to know is "how many more
        lines am I missing?". `line_unit` names the hidden-row noun ("lines"
        for text output, "files" for glob path lists).

        Returns:
            Hint string for the UI, or `None` if nothing was truncated.
        """
        if not is_preview:
            return None
        hidden_lines = len(lines) - parts_count
        if hidden_lines > 0:
            return f"{hidden_lines} more {line_unit}"
        if char_truncated:
            effective_output_len = len(output) - (1 if had_trailing_newline else 0)
            hidden_chars = effective_output_len - chars_used
            return f"{hidden_chars} more chars"
        return None

    def _format_search_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format grep/glob search output.

        Returns:
            FormattedOutput with search results and optional truncation info.
        """
        # Try to parse as a Python list (glob returns list of paths). The
        # except is scoped to detection only — formatting runs outside it so a
        # bug in `_format_search_lines` can't silently reroute to the fallback.
        try:
            items = ast.literal_eval(output.strip())
        except (ValueError, SyntaxError):
            items = None

        if isinstance(items, list):
            paths: list[str] = []
            for item in items:
                path = Path(str(item))
                try:
                    display = str(path.relative_to(Path.cwd()))
                except ValueError:
                    display = path.name
                paths.append(display)
            return self._format_search_lines(
                paths, is_preview=is_preview, line_unit="files"
            )

        # Fallback: line-based output (grep results)
        lines = [
            raw_line.strip() for raw_line in output.split("\n") if raw_line.strip()
        ]
        return self._format_search_lines(
            lines, is_preview=is_preview, line_unit="lines"
        )

    def _format_search_lines(
        self,
        lines: list[str],
        *,
        is_preview: bool,
        line_unit: Literal["files", "lines"],
    ) -> FormattedOutput:
        """Format search result rows with line and character preview caps.

        `line_unit` names the hidden-row noun for the hint — "files" for glob
        path lists, "lines" for grep matches.

        Returns:
            FormattedOutput with search rows and optional truncation info.
        """
        # Search rows are denser than file/shell output, so the preview shows
        # one extra row (5) before truncating.
        max_lines = 5 if is_preview else len(lines)
        char_budget = self._PREVIEW_CHARS if is_preview else None

        shown, chars_used, char_truncated = self._truncate_to_budget(
            lines, max_lines=max_lines, char_budget=char_budget
        )
        parts = [Content(line) for line in shown]
        content = Content("\n").join(parts) if parts else Content("")

        # The cleaned `lines` carry no trailing-newline element, so the joined
        # length is the full preview-able content length.
        truncation = self._build_truncation_hint(
            output="\n".join(lines),
            lines=lines,
            parts_count=len(parts),
            chars_used=chars_used,
            char_truncated=char_truncated,
            had_trailing_newline=False,
            is_preview=is_preview,
            line_unit=line_unit,
        )

        return FormattedOutput(content=content, truncation=truncation)

    def _format_shell_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format shell command output.

        Returns:
            FormattedOutput with shell output and optional truncation info.
        """
        lines = output.split("\n")
        had_trailing_newline = bool(lines) and not lines[-1]
        if had_trailing_newline:
            lines = lines[:-1]
        max_lines = 4 if is_preview else len(lines)
        char_budget = self._PREVIEW_CHARS if is_preview else None

        shown, chars_used, char_truncated = self._truncate_to_budget(
            lines, max_lines=max_lines, char_budget=char_budget
        )
        # Dim the leading `$ command` echo; only the first row can carry it.
        parts = [
            Content.styled(line, "dim")
            if index == 0 and line.startswith("$ ")
            else Content(line)
            for index, line in enumerate(shown)
        ]
        content = Content("\n").join(parts) if parts else Content("")

        truncation = self._build_truncation_hint(
            output=output,
            lines=lines,
            parts_count=len(parts),
            chars_used=chars_used,
            char_truncated=char_truncated,
            had_trailing_newline=had_trailing_newline,
            is_preview=is_preview,
        )

        return FormattedOutput(content=content, truncation=truncation)

    def _format_js_eval_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format `js_eval` (JS interpreter) output.

        Unwraps the REPL's `<stdout>` / `<result>` / `<error>` envelope into
        labeled, styled sections instead of dumping the raw XML-escaped blob.

        Returns:
            FormattedOutput with the formatted REPL output and optional
            truncation info.
        """
        blocks = parse_js_eval_blocks(output)
        if blocks is None:
            # Unexpected shape — fall back to plain line rendering.
            return self._format_lines_output(output.split("\n"), is_preview=is_preview)

        colors = theme.get_theme_colors(self)

        # Common case: a single short scalar result with no stdout. Rendering a
        # standalone "result" header above a one-word value reads as a
        # misplaced badge, so collapse it to an inline `result: value` line.
        if len(blocks) == 1:
            block = blocks[0]
            if (
                isinstance(block, JsEvalResult)
                and not block.kind
                and "\n" not in block.body
                and len(block.body) <= self._JS_EVAL_INLINE_RESULT_MAX
            ):
                content = Content.assemble(
                    Content.styled("result: ", colors.success),
                    Content(block.body),
                )
                return FormattedOutput(content=content)
        lines: list[Content] = []
        total_lines = 0
        max_lines = self._PREVIEW_LINES if is_preview else None
        # Char budget mirrors the other formatters so a single very long body
        # line (e.g. a 10k-char result) is clipped instead of flooding the
        # collapsed preview. `None` outside preview means no char cap.
        remaining_chars = self._PREVIEW_CHARS if is_preview else None
        # Chars hidden when a single over-budget body line is clipped. Only
        # meaningful for the hint when no whole lines were dropped (line counts
        # take precedence below, matching `_build_truncation_hint`).
        clipped_chars = 0

        def add_section(label: Content, body: str) -> None:
            nonlocal total_lines, remaining_chars, clipped_chars
            if max_lines is not None and total_lines >= max_lines:
                return
            if remaining_chars is not None and remaining_chars <= 0:
                return
            lines.append(label)
            total_lines += 1
            body_lines = body.split("\n") if body else [""]
            for body_line in body_lines:
                if max_lines is not None and total_lines >= max_lines:
                    break
                if remaining_chars is not None:
                    if remaining_chars <= 0:
                        break
                    if len(body_line) > remaining_chars:
                        # Clip the over-budget line and stop adding more.
                        lines.append(Content(f"  {body_line[:remaining_chars]}"))
                        total_lines += 1
                        clipped_chars = len(body_line) - remaining_chars
                        remaining_chars = 0
                        break
                    remaining_chars -= len(body_line)
                lines.append(Content(f"  {body_line}"))
                total_lines += 1

        for block in blocks:
            if isinstance(block, JsEvalStdout):
                add_section(Content.styled("stdout", "dim"), block.body)
            elif isinstance(block, JsEvalError):
                header = f"error ({block.error_type})" if block.error_type else "error"
                add_section(Content.styled(header, colors.error), block.body)
            else:  # JsEvalResult
                label = "result (handle)" if block.kind else "result"
                add_section(Content.styled(label, colors.success), block.body)

        content = Content("\n").join(lines) if lines else Content("")
        truncation = self._build_js_eval_truncation_hint(
            blocks=blocks,
            shown_lines=total_lines,
            clipped_chars=clipped_chars,
            is_preview=is_preview,
        )
        return FormattedOutput(content=content, truncation=truncation)

    @staticmethod
    def _build_js_eval_truncation_hint(
        *,
        blocks: list[JsEvalBlock],
        shown_lines: int,
        clipped_chars: int,
        is_preview: bool,
    ) -> str | None:
        """Quantify how much `js_eval` preview content was hidden.

        Prefers a hidden-line count over a hidden-char count (mirroring
        `_build_truncation_hint`): when whole sections were dropped, "N more
        lines" is the more useful signal; a lone clipped body line reports the
        chars it lost.

        Args:
            blocks: The parsed blocks, used to compute the full (untruncated)
                display-line count.
            shown_lines: Display lines actually emitted into the preview.
            clipped_chars: Chars dropped from a single clipped body line, if any.
            is_preview: Whether this is preview rendering; full renders never
                truncate.

        Returns:
            A hint string for the UI, or `None` when nothing was hidden.
        """
        if not is_preview:
            return None
        # Each block renders as one label line plus its body lines; an empty
        # body still occupies one (blank) line.
        full_lines = sum(
            1 + (len(block.body.split("\n")) if block.body else 1) for block in blocks
        )
        hidden_lines = full_lines - shown_lines
        if hidden_lines > 0:
            return f"{hidden_lines} more lines"
        if clipped_chars > 0:
            return f"{clipped_chars} more chars"
        return None

    def _format_web_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format web_search/fetch_url output.

        Returns:
            FormattedOutput with web response and optional truncation info.
        """
        data = self._try_parse_web_data(output)
        if isinstance(data, dict):
            return self._format_web_dict(data, is_preview=is_preview)

        # Fallback: plain text
        return self._format_lines_output(output.split("\n"), is_preview=is_preview)

    @staticmethod
    def _try_parse_web_data(output: str) -> dict | None:
        """Try to parse web output as JSON or dict.

        Returns:
            Parsed dict if successful, None otherwise.
        """
        try:
            if output.strip().startswith("{"):
                return json.loads(output)
            return ast.literal_eval(output)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            return None

    def _format_web_dict(self, data: dict, *, is_preview: bool) -> FormattedOutput:
        """Format a parsed web response dict.

        Returns:
            FormattedOutput with web response content and optional truncation info.
        """
        # Handle web_search results
        if "results" in data:
            return self._format_web_search_results(
                data.get("results", []), is_preview=is_preview
            )

        # Handle fetch_url response
        if "markdown_content" in data:
            lines = data["markdown_content"].split("\n")
            return self._format_lines_output(lines, is_preview=is_preview)

        # Generic dict - show key fields
        parts: list[Content] = []
        max_keys = 3 if is_preview else len(data)
        for k, v in list(data.items())[:max_keys]:
            v_str = str(v)
            if is_preview and len(v_str) > _MAX_WEB_CONTENT_LEN:
                v_str = v_str[:_MAX_WEB_CONTENT_LEN] + "..."
            parts.append(Content(f"  {k}: {v_str}"))
        truncation = None
        if is_preview and len(data) > max_keys:
            truncation = f"{len(data) - max_keys} more"
        return FormattedOutput(
            content=Content("\n").join(parts) if parts else Content(""),
            truncation=truncation,
        )

    def _format_web_search_results(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, results: list, *, is_preview: bool
    ) -> FormattedOutput:
        """Format web search results.

        Returns:
            FormattedOutput with search results and optional truncation info.
        """
        if not results:
            return FormattedOutput(content=Content.styled("No results", "dim"))
        parts: list[Content] = []
        max_results = 3 if is_preview else len(results)
        for r in results[:max_results]:
            title = r.get("title", "")
            url = r.get("url", "")
            parts.extend(
                [
                    Content.styled(f"  {title}", "bold"),
                    Content.styled(f"  {url}", "dim"),
                ]
            )
        truncation = None
        if is_preview and len(results) > max_results:
            truncation = f"{len(results) - max_results} more results"
        return FormattedOutput(content=Content("\n").join(parts), truncation=truncation)

    def _format_lines_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, lines: list[str], *, is_preview: bool
    ) -> FormattedOutput:
        """Format a list of lines with optional preview truncation.

        Returns:
            FormattedOutput with lines content and optional truncation info.
        """
        max_lines = 4 if is_preview else len(lines)
        parts = [Content(line) for line in lines[:max_lines]]
        content = Content("\n").join(parts) if parts else Content("")
        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"
        return FormattedOutput(content=content, truncation=truncation)

    def _format_task_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format task (subagent) output.

        Returns:
            FormattedOutput with task output and optional truncation info.
        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        parts = [Content(line) for line in lines[:max_lines]]
        content = Content("\n").join(parts) if parts else Content("")

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

    def _ask_user_question_count(self) -> int:
        """Return the number of valid question objects in this tool call.

        The count comes from the structured tool arguments rather than parsing
        the free-form transcript. This keeps arbitrary answer text opaque while
        still supporting the collapsed `N answers` affordance.

        Returns:
            The question count, or zero unless `questions` is a non-empty list of
                dicts each carrying non-blank `question` text. Deliberately looser
                than `ask_user._validate_questions` — it accepts payloads that
                rejects, such as an unknown `type` or a `choices`/`type` mismatch —
                because it only needs to guard the fields the count reads. Of the
                three paths that populate `_args`, only the `ask_user` interrupt
                (validated in `textual_adapter` via `ask_user_adapter`) is checked;
                the streamed tool call and the persisted store
                (`message_store.to_widget`) are not, so malformed shapes do reach
                here and must degrade rather than raise.
        """
        questions = self._args.get("questions")
        if not isinstance(questions, list) or not questions:
            return 0
        if not all(
            isinstance(question, dict)
            and isinstance(question.get("question"), str)
            and bool(question["question"].strip())
            for question in questions
        ):
            return 0
        return len(questions)

    def _format_ask_user_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format an `ask_user` result for the collapsed or expanded row.

        The inline question widget is unmounted once answered, so this row is the
        only place the answers stay visible in the live session — the thread's
        own `ToolMessage` is what a reload re-renders from. Collapsed, the row
        keeps a one-line summary; expanded, it shows what was sent back.

        The summary is derived from the recorded status, never from the answer
        text (the question count only labels the expand affordance). The
        placeholders are in-band, so a user who types `(cancelled)` or
        `(error: ...)` must not have their answer read as control state. The cost
        is that a cancelled prompt resumed by a non-TUI client — which `ask_user`
        records as `status="success"` with `(cancelled)` placeholders — reads as
        answered until expanded.

        Returns:
            FormattedOutput with the status-derived summary when `is_preview`, or
                the output rendered literally when expanded. A row holding only a
                fallback summary advertises no expansion. Falls back to generic
                formatting when the structured question args are unavailable.
        """
        question_count = self._ask_user_question_count()
        if question_count == 0:
            # Route through the generic path rather than returning the body bare:
            # `ask_user` is in `_ALWAYS_PREVIEW_TOOLS`, so the size thresholds in
            # `_has_expandable_output`/`_update_output_display` no longer gate it
            # and an arbitrarily long body would otherwise fill the collapsed row
            # with no expand affordance. `_format_generic_output` reapplies them.
            if not self._ask_user_args_warned:
                # Once per widget: this runs on every re-render, and the
                # condition cannot change without a new `_args`.
                self._ask_user_args_warned = True
                logger.warning(
                    "ask_user row has no usable `questions` args (got %r); the "
                    "collapsed row will show the transcript instead of a summary",
                    self._args.get("questions"),
                )
            return self._format_generic_output(output, is_preview=is_preview)

        if output in _ASK_USER_ROW_SUMMARIES:
            # No authoritative ToolMessage arrived, so this row holds only the
            # fallback summary. There is no transcript for expansion to reveal;
            # advertising the question count would create a dead affordance.
            return FormattedOutput(content=Content.styled(output, "dim"))

        if not is_preview:
            return FormattedOutput(content=Content(output))

        if self._status == "error":
            # The transcript holds `(error: ...)` placeholders, not answers, so
            # count the questions instead of promising answers.
            summary = ASK_USER_FAILED_SUMMARY
            noun = "question" if question_count == 1 else "questions"
        else:
            summary = ASK_USER_ANSWERED_SUMMARY
            noun = "answer" if question_count == 1 else "answers"
        return FormattedOutput(
            content=Content.styled(summary, "dim"),
            truncation=f"{question_count} {noun}",
        )

    def _update_output_display(self) -> None:
        """Update the output display based on expanded state."""
        # Guard: all widgets must be initialized before updating display state
        if (
            not self._output
            or not self._preview_widget
            or not self._preview_row
            or not self._full_widget
            or not self._full_row
            or not self._hint_widget
        ):
            return

        output_stripped = self._output.strip()
        lines = output_stripped.split("\n")
        total_lines = len(lines)
        total_chars = len(output_stripped)

        # Truncate if too many lines OR too many characters
        needs_truncation = (
            total_lines > self._PREVIEW_LINES or total_chars > self._PREVIEW_CHARS
        )

        # Some output is a non-empty raw string that the formatter renders as no
        # visible content — all whitespace, or a serialized empty collection like
        # `[]`. The raw `_output` is truthy, so the early-return guard at the top
        # of this method doesn't catch it, but rendering it would show an empty
        # box with a misleading expand affordance. Treat it like empty output and
        # render nothing. (A search that found nothing is not this case: grep/glob
        # return a human-readable "No matches found" / "No files found" that
        # formats non-empty and renders inline; see the collapse branch below.)
        # This also subsumes the all-whitespace case, so the collapsed branch
        # below no longer needs its own empty guard.
        #
        # This fires for errors too, but never hides one: a real error body is
        # human-readable text that formats non-empty (and execute errors keep
        # the `$ command` echo), so it only triggers on a body that has nothing
        # to render anyway. The "error" status badge stays visible regardless.
        full = self._format_output(self._output, is_preview=False)
        if not full.content.plain.strip():
            self._preview_row.display = False
            self._full_row.display = False
            self._hint_widget.display = False
            return

        if self._expanded:
            # Show full output with formatting
            self._preview_row.display = False
            self._full_widget.update(full.content)
            self._full_row.display = True
            # Only offer a collapse affordance when collapsing would actually
            # hide something. Errors are force-expanded (see `set_error`), so a
            # short single-line error has no smaller collapsed form — showing
            # "click to collapse" there is misleading.
            if self._has_expandable_output():
                self._hint_widget.update(
                    Content.styled(
                        f"{self._output_hint_keys()} to collapse", "dim italic"
                    )
                )
                self._hint_widget.display = True
            else:
                self._hint_widget.display = False
        else:
            # Show collapsed preview
            self._full_row.display = False
            # `read_file` echoes the file the agent read, grep/glob echo the
            # matches for a pattern the header already names, and `edit_file`
            # success output repeats the status/diff — so the body is noise by
            # default. Collapse it entirely (no preview) while keeping the
            # original output expandable for when the user does want to see it.
            # A grep/glob that found nothing is excluded: its terminal "No
            # matches/files found" message is the whole result, so it renders
            # inline rather than hiding behind an expand click.
            if not self._is_search_no_result_output(self._output) and (
                self._tool_name in _COLLAPSE_OUTPUT_BY_DEFAULT
                or (self._tool_name == "edit_file" and self._status == "success")
            ):
                self._preview_row.display = False
                ellipsis = get_glyphs().ellipsis
                self._hint_widget.update(
                    Content.styled(
                        f"{ellipsis} {self._output_hint_keys()} to expand", "dim italic"
                    )
                )
                self._hint_widget.display = True
                return
            # Truncate the preview only when the output is large enough to
            # warrant it; `_ALWAYS_PREVIEW_TOOLS` use their compact preview
            # regardless of size.
            is_preview = needs_truncation or self._tool_name in _ALWAYS_PREVIEW_TOOLS
            # Pass the raw output, not `output_stripped`: `_format_output`
            # normalizes whitespace while preserving the first line's leading
            # indentation. Pre-stripping here flattens that indent on line 0 only,
            # misaligning uniformly indented output (e.g. `git branch -r`). The
            # expanded branch above already passes raw `self._output`.
            result = self._format_output(self._output, is_preview=is_preview)
            self._preview_widget.update(result.content)
            self._preview_row.display = True

            # Offer expansion only when the formatter actually hid content.
            # The raw size threshold can trip without anything being hidden, and
            # promising an expansion that reveals nothing is misleading.
            if result.truncation:
                ellipsis = get_glyphs().ellipsis
                self._hint_widget.update(
                    Content.styled(
                        f"{ellipsis} {result.truncation} — "
                        f"{self._output_hint_keys()} to expand",
                        "dim italic",
                    )
                )
                self._hint_widget.display = True
            else:
                self._hint_widget.display = False

    def _output_hint_keys(self) -> str:
        """Affordances to advertise in the output expand/collapse hint.

        Ctrl+O routes to the collapsible command/code block whenever this row
        has one (see `action_toggle_tool_output`), and to a truncated `task`
        description when the row is a `task` call, so the output hint only
        advertises Ctrl+O when Ctrl+O would actually toggle the *output*. When a
        command/code block or expandable `task` description owns Ctrl+O the
        output is reachable by clicking its own region instead.

        Returns:
            `"click"` when an expandable command/code block or `task`
                description owns Ctrl+O, otherwise `"click or Ctrl+O"`.
        """
        if self.has_expandable_args or self.has_expandable_task_desc:
            return "click"
        return "click or Ctrl+O"

    @property
    def has_output(self) -> bool:
        """Check if this tool message has output to display.

        Returns:
            True if there is output content, False otherwise.
        """
        return bool(self._output)

    @property
    def tool_name(self) -> str:
        """Public read-only accessor for the underlying tool name."""
        return self._tool_name

    @property
    def args(self) -> dict[str, Any]:
        """Public read-only accessor for the parsed tool-call arguments.

        Returns a shallow copy so a consumer (e.g. a hook payload built from
        `args`) cannot rebind the widget's top-level keys by reference. Nested
        mutable values are shared, not deep-copied, so callers must treat them as
        read-only and must not deep-mutate a returned nested value.
        """
        return dict(self._args)

    @property
    def summary_call(self) -> _SummaryCall:
        """This row as `(tool name, args)` for the group summary line.

        Unlike `args` this does not copy: the group rebuilds its cache key from
        every member on each spinner tick, so a copy per member per tick buys
        nothing the caller uses. Safe because the `Mapping` return type is
        read-only and this widget never mutates `_args` after construction — the
        dict is the caller's, though, so do not widen this to a caller that
        mutates.
        """
        return (self._tool_name, self._args)

    @property
    def is_success(self) -> bool:
        """Whether the tool completed successfully."""
        return self._status == "success"

    @property
    def _superseded_by_diff(self) -> bool:
        """Whether this row hides because its `DiffMessage` says it all.

        Gated on success so a row whose status later flips to error is revealed
        again: `_diff_superseded` stays set, this goes `False`, and `set_error`
        re-applies visibility for exactly that reason. Without the conjunct a
        failure would be hidden behind a diff of the change it did not make.
        """
        return self.is_success and self._diff_superseded

    @property
    def has_display_caveat(self) -> bool:
        """Whether this row's output leads with a caveat that must stay visible.

        A caveat is the user's only account of a change the transcript cannot
        render, and it is carried inside this row's output. A groupable tool
        (`write_file`, `delete`) is folded at mount, and the collapsed summary
        line is built from tool names and arguments, never from tool output — so
        without this the caveat is folded away and destroying a 5,000-line file
        whose contents could not be read renders as `▸ Deleted 1 file`, exactly
        like destroying an empty one.

        Read by the group code alongside `is_failed`: a caveated row is not a
        failure, but it has the same claim on staying on screen.
        """
        return self._has_display_caveat

    def set_success_with_caveat(self, caveat: str, output: str) -> bool:
        """Complete this row successfully, leading with a caveat if there is one.

        The live path's single entry point, because the flag and the prose it
        describes must not be settable apart: a flag without the sentence leaves
        a row unfoldable for no visible reason, and — the costly direction — the
        sentence without the flag lets the change's only account be folded into
        a group summary.

        Not routed through `set_error`: a display problem is not a tool failure.
        That would stamp the row "Error", overwrite the tool's success message,
        and make a completed operation count toward every failure surface, so a
        user would retry an edit that already applied. Prepended rather than
        appended so the caveat survives the collapsed output preview.

        Args:
            caveat: The caveat to lead with, or empty when there is none.
            output: The tool's own output.

        Returns:
            Whether a caveat was applied, for callers tracking whether any
            surface carried it.
        """
        self.set_success("\n\n".join(part for part in (caveat, output) if part))
        self._has_display_caveat = bool(caveat)
        return bool(caveat)

    def _mark_display_caveat(self) -> None:
        """Restore the display-caveat flag on a rehydrated row.

        For `MessageData.to_widget` only, where the output carrying the caveat
        is restored separately through `_deferred_output`. On the live path use
        `set_success_with_caveat`, which sets both together.

        Private so it is not available as a public way to set the flag alone:
        the store already restores this widget's other private state, and a
        caller outside that path wanting the flag without the sentence is the
        split `set_success_with_caveat` exists to prevent.
        """
        self._has_display_caveat = True

    @property
    def is_failed(self) -> bool:
        """Whether the tool did not succeed and should stay visible.

        Covers errored, rejected, and skipped tools. `skipped` is included so a
        reject-cascade (one tool rejected, the rest skipped) keeps the skipped
        rows visible and out of the group's success count, matching how
        `_regroup_completed_tools` treats a hydrated transcript.
        """
        return self._status in {"error", "rejected", "skipped"}

    @property
    def is_pending(self) -> bool:
        """Whether the tool has not finished (awaiting approval or running)."""
        return self._status in {"pending", "running"}

    @property
    def has_expandable_args(self) -> bool:
        """Whether the tool's args are large enough to deserve a collapsible block.

        - `ask_user`: its `questions` payload is too noisy to render inline.
        - `js_eval`: the header shows only the first code line (truncated at
            `JS_EVAL_HEADER_MAX_LENGTH`), so the full program is offered as a
            collapsible block whenever it spans more than one non-blank line *or*
            a single line is long enough to be truncated in the header.
        - `execute`: the header truncates the shell command at
            `EXECUTE_HEADER_MAX_LENGTH`, so the full command is offered as a
            collapsible block when the command, after stripping surrounding
            whitespace, is longer than `EXECUTE_HEADER_MAX_LENGTH`.
        """
        if self._tool_name == "ask_user":
            return bool(self._args)
        if self._tool_name == "js_eval":
            code = self._args.get("code")
            if isinstance(code, str) and code.strip():
                non_blank = sum(1 for line in code.splitlines() if line.strip())
                return non_blank > 1 or len(code.strip()) > JS_EVAL_HEADER_MAX_LENGTH
        if self._tool_name == "execute":
            command = self._args.get("command")
            if isinstance(command, str) and command.strip():
                return len(command.strip()) > EXECUTE_HEADER_MAX_LENGTH
        return False

    @property
    def has_expandable_task_desc(self) -> bool:
        """Whether the `task` description is long enough to be truncated.

        A `task` row renders its description on a dedicated dim line, truncated
        at `_TASK_DESC_MAX_LENGTH`. When the full description exceeds that, the
        truncated preview becomes expandable via click or Ctrl+O.
        """
        return len(self._task_description()) > self._TASK_DESC_MAX_LENGTH

    def _task_description(self) -> str:
        """Return the `task` call's description string, or empty when absent.

        A non-string `description` (schema-typed as a string) is coerced to
        `""` so downstream length/slice logic stays safe; the anomaly is logged.
        """
        if self._tool_name != "task":
            return ""
        desc = self._args.get("description", "")
        if isinstance(desc, str):
            return desc
        if desc is not None:
            logger.debug("task description is not a string: %r", type(desc))
        return ""

    def _task_desc_content(self) -> Content:
        """Render the `task` description, truncated unless expanded.

        Returns:
            Dim `Content`: the full description when expanded or when it already
            fits within `_TASK_DESC_MAX_LENGTH`; otherwise the preview truncated
            to that length (trailing whitespace trimmed) with a trailing
            ellipsis.
        """
        desc = self._task_description()
        if self._task_desc_expanded or len(desc) <= self._TASK_DESC_MAX_LENGTH:
            text = desc
        else:
            ellipsis = get_glyphs().ellipsis
            text = desc[: self._TASK_DESC_MAX_LENGTH].rstrip() + ellipsis
        return Content.styled(text, "dim")

    def _update_task_desc_display(self) -> None:
        """Update the truncated/expanded `task` description and its hint."""
        if self._task_desc_widget is None or self._task_desc_hint_widget is None:
            # Refs are legitimately None for non-`task` rows (never mounted). Log
            # only when a `task` row that carries a description is missing them,
            # so a regression that nulls them post-mount isn't a silent no-op.
            if self._task_description():
                logger.debug("_update_task_desc_display: task-desc refs not cached")
            return
        if not self._task_description():
            self._task_desc_widget.display = False
            self._task_desc_hint_widget.display = False
            return
        self._task_desc_widget.update(self._task_desc_content())
        self._task_desc_widget.display = True
        if not self.has_expandable_task_desc:
            self._task_desc_hint_widget.display = False
            return
        verb = "collapse" if self._task_desc_expanded else "expand"
        self._task_desc_hint_widget.update(
            Content.styled(f"click or Ctrl+O to {verb}", "dim italic")
        )
        self._task_desc_hint_widget.display = True

    def _format_code_detail(self) -> Content:
        """Render the `js_eval` program for the collapsible code block.

        The code is shown verbatim and left-aligned (its own indentation is the
        only indentation), as plain uncolored `Content`. Blank lines of
        top/bottom padding add breathing room between the `js_eval` header above
        and the "show/hide code" hint below.

        Returns:
            A plain `Content` renderable with a blank line of padding on
                top and bottom.
        """
        code = self._args.get("code")
        code_str = code.strip("\n") if isinstance(code, str) else str(code)
        code_str = render_with_unicode_markers(code_str)

        # Blank lines of top/bottom padding separate the block from the header
        # line above and the "show/hide code" hint below.
        return Content("\n").join((Content(""), Content(code_str), Content("")))

    def _format_command_detail(self) -> Content:
        """Render the full `execute` command for the collapsible block.

        The command is shown verbatim and left-aligned, as plain uncolored
        `Content`, mirroring `_format_code_detail`. Hidden/deceptive Unicode is
        rendered as visible markers so a truncated header can't conceal it.

        Returns:
            A plain `Content` renderable with a blank line of padding on
                top and bottom.
        """
        command = self._args.get("command")
        command_str = command.strip("\n") if isinstance(command, str) else str(command)
        command_str = render_with_unicode_markers(command_str)
        return Content("\n").join((Content(""), Content(command_str), Content("")))

    def _format_args_detail(self) -> Content:
        """Render tool arguments as an indented `Content` block.

        Renders JSON-pretty-printed args, falling back to `str(self._args)`
        (with a visible marker) when JSON serialization fails — `default=str`
        already handles most non-serializable values, so reaching the fallback
        indicates a deeper issue worth logging. `js_eval` code is handled
        separately by `_format_code_detail`.

        Returns:
            Indented `Content` containing JSON-pretty-printed arguments, or a
            marked fallback rendering on serialization failure.
        """
        try:
            text = json.dumps(self._args, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "ask_user args not JSON-serializable; using repr fallback: %r", exc
            )
            text = f"# (fallback rendering)\n{self._args!s}"
        lines = Content(text).split("\n")
        return Content("\n").join(Content.assemble("  ", line) for line in lines)

    def _update_args_display(self) -> None:
        """Update the collapsed/expanded argument display."""
        if self._args_widget is None or self._args_hint_widget is None:
            # Toggle invoked before on_mount cached the refs; log so a regression
            # that nulls them out post-mount doesn't appear as a silent no-op.
            logger.debug("_update_args_display called before widget refs are cached")
            return

        if not self.has_expandable_args:
            self._args_widget.display = False
            self._args_hint_widget.display = False
            return

        if self._tool_name == "js_eval":
            noun, detail_fn = "code", self._format_code_detail
        elif self._tool_name == "execute":
            noun, detail_fn = "command", self._format_command_detail
        else:
            noun, detail_fn = "arguments", self._format_args_detail
        if self._args_expanded:
            self._args_widget.update(detail_fn())
            self._args_widget.display = True
            self._args_hint_widget.update(
                Content.styled(f"click or Ctrl+O to hide {noun}", "dim italic")
            )
        else:
            self._args_widget.display = False
            self._args_hint_widget.update(
                Content.styled(f"click or Ctrl+O to show {noun}", "dim italic")
            )
        self._args_hint_widget.display = True

    def _filtered_args(self) -> dict[str, Any]:
        """Filter large tool args for display.

        Returns:
            Filtered args dict with only display-relevant keys for write/edit tools.
        """
        if self._tool_name not in {"write_file", "edit_file"}:
            return self._args

        filtered: dict[str, Any] = {}
        for key in ("file_path", "path", "replace_all"):
            if key in self._args:
                filtered[key] = self._args[key]
        return filtered


# Maps a tool name to the summary category it aggregates under. grep/glob share
# "search" so a mixed run folds into a single "Searched for N patterns" segment.
_TOOL_SUMMARY_CATEGORY: dict[str, str] = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "delete": "delete",
    "ls": "ls",
    "grep": "search",
    "glob": "search",
    "execute": "shell",
    "js_eval": "js",
    "web_search": "web_search",
    "fetch_url": "fetch",
    "task": "task",
    "write_todos": "todos",
}

# category -> (present verb, past verb, singular noun, plural noun).
_TOOL_SUMMARY_PHRASES: dict[str, tuple[str, str, str, str]] = {
    "read": ("Reading", "Read", "file", "files"),
    "write": ("Writing", "Wrote", "file", "files"),
    "edit": ("Editing", "Edited", "file", "files"),
    "delete": ("Deleting", "Deleted", "file", "files"),
    "ls": ("Listing", "Listed", "directory", "directories"),
    "search": ("Searching for", "Searched for", "pattern", "patterns"),
    "shell": ("Running", "Ran", "shell command", "shell commands"),
    "js": ("Running", "Ran", "JS evaluation", "JS evaluations"),
    "fetch": ("Fetching", "Fetched", "URL", "URLs"),
    "task": ("Running", "Ran", "agent", "agents"),
}

# category -> tool-arg names naming the thing the call acts on, in fallback
# order. Only categories whose summary noun is a durable object belong here:
# their counts claim "N distinct things", so repeat calls on one target must
# collapse (see `_tally_categories`).
#
# Every other category is absent on purpose. "shell", "js", "task", and "search"
# count attempts, not objects — running one command or grepping one pattern twice
# is genuinely two pieces of work. "web_search" phrases its own repeats
# ("Searched the web 2 times") and "todos" carries no count at all. "ls" is
# excluded because a listing is a snapshot, not a durable object: a group spans a
# whole step, so an intervening write can make the second listing of one
# directory show different contents.
_TOOL_SUMMARY_TARGET_ARGS: dict[str, tuple[str, ...]] = {
    "read": ("file_path", "path"),
    "write": ("file_path", "path"),
    "edit": ("file_path", "path"),
    "delete": ("file_path", "path"),
    "fetch": ("url",),
}

_PATH_TARGET_CATEGORIES = frozenset({"read", "write", "edit", "delete"})
"""Categories from `_TOOL_SUMMARY_TARGET_ARGS` whose target is a filesystem path.

Only these are normalized before comparison. Path rules are wrong for a URL:
`http://x/a//b`, `http://x/a/` and `http://x/a` would each be judged the same
target as a URL the server can answer differently. That undercounts, which this
code must never do. Any category added here needs a genuine path, and any other
target is compared exactly.
"""


def _summary_target(tool_name: str, args: Mapping[str, Any]) -> str | None:
    """Identify the object a call acts on, for repeat-call collapsing.

    Args:
        tool_name: Raw tool name for the call.
        args: The call's parsed arguments.

    Returns:
        An identity for the target, or None when the category counts attempts
        rather than objects, or the naming argument is missing or not a non-empty
        string. None means "cannot be judged a repeat", so the call is always
        counted — an argument list this code cannot read undercounts nothing.
    """
    category = _TOOL_SUMMARY_CATEGORY.get(tool_name, tool_name)
    arg_names = _TOOL_SUMMARY_TARGET_ARGS.get(category)
    if arg_names is None:
        return None
    for arg_name in arg_names:
        value = args.get(arg_name)
        if isinstance(value, str) and value:
            if category not in _PATH_TARGET_CATEGORIES:
                # Not a path, so compare it exactly — see
                # `_PATH_TARGET_CATEGORIES` for why path rules undercount a URL.
                return f"{category}:{value}"
            # The category prefix is load-bearing: `_tally_categories` shares one
            # `seen` set across categories, so identities must be namespaced or
            # reading and then editing one file would read as a repeat.
            return f"{category}:{_normalize_path_target(value)}"
    return None


def _normalize_path_target(value: str) -> str:
    r"""Normalize a path using the filesystem middleware's canonical form.

    Defers to `validate_path` rather than reimplementing it, so the identity is
    exactly the string the file tools act on and cannot drift from it. Every
    path the middleware rejects — `..` traversal, a leading `~`, a drive prefix
    like `C:/` — is returned verbatim: a call that could not have run has no
    canonical form, and canonicalizing one would fold it into a valid target's
    tally. `~` and `/~` are different reads and must count as two.

    Args:
        value: The raw path string as the tool was called with it.

    Returns:
        The normalized virtual path, or `value` when the middleware rejects it.
    """
    from deepagents.backends.utils import validate_path

    try:
        return validate_path(value)
    except ValueError:
        return value


# category -> plural noun for the operation, used to report repeat work on one
# target alongside the target count, e.g. "Edited 1 file (3 edits)".
#
# Every mutating category qualifies: each call is an event that changed the tree
# and owns a diff, so collapsing three edits of one file to a bare "Edited 1
# file" hides work the reader wants. A group spans a whole step, so one path can
# genuinely be mutated twice — `delete a.py`, `write_file a.py`, `delete a.py`
# is two real deletions. "fetch" qualifies too: a repeated fetch of one URL is a
# deliberate re-request, not pagination, and "Fetched 1 URL" for two requests
# reads like the second one vanished. Of the categories that name a target, only
# "read" is absent — a repeat read is usually pagination, where the count is
# noise, so reads collapse silently.
#
# A category here has no effect unless it is also in `_TOOL_SUMMARY_TARGET_ARGS`:
# without a target, `calls` can never exceed `targets`.
_REPEAT_COUNT_NOUNS: dict[str, str] = {
    "edit": "edits",
    "write": "writes",
    "delete": "deletions",
    "fetch": "calls",
}


class _CategoryTally(NamedTuple):
    """One category's contribution to a summary line."""

    category: str
    """The summary category being counted."""

    rep_name: str
    """First raw tool name seen for the category, for fallback phrasing."""

    targets: int
    """Distinct targets touched. Calls with no identifiable target each count
    for themselves, so this never drops below the honest minimum."""

    calls: int
    """Total calls, including repeats on one target. Equals `targets` unless
    something was touched more than once."""


def _tally_categories(
    calls: Sequence[_SummaryCall],
) -> list[_CategoryTally]:
    """Aggregate calls by category, counting distinct targets and total calls.

    Both numbers are needed because the phrasing claims nouns ("Read 2 files")
    while the work is calls: a file read twice is one file, and an edit made
    twice is one file but two edits.

    Args:
        calls: `(raw tool name, parsed args)` for each call, in call order.

    Returns:
        One tally per category, in first-appearance order.
    """
    tallies: dict[str, _CategoryTally] = {}
    # One set across all categories is safe because `_summary_target` namespaces
    # every identity by category. That is also why the first-call branch below
    # may ignore `repeat`: a repeat implies an earlier call in the same category,
    # which must already have created that category's tally.
    seen: set[str] = set()
    for tool_name, args in calls:
        category = _TOOL_SUMMARY_CATEGORY.get(tool_name, tool_name)
        target = _summary_target(tool_name, args)
        repeat = target is not None and target in seen
        if target is not None:
            seen.add(target)
        current = tallies.get(category)
        if current is None:
            tallies[category] = _CategoryTally(
                category=category, rep_name=tool_name, targets=1, calls=1
            )
            continue
        tallies[category] = current._replace(
            targets=current.targets + (0 if repeat else 1),
            calls=current.calls + 1,
        )
    return list(tallies.values())


_DIFF_HEADER_CATEGORIES = frozenset({"write", "edit", "delete"})
"""Summary categories whose past verb heads a `DiffMessage`.

The file-mutating tools — the only ones that produce a diff to head.
"""


def _diff_header_verb(tool_name: str | None) -> str:
    """Return the past-tense verb naming the change a diff shows.

    The verb text is shared with the group-summary tables, but eligibility is
    not: a newly added file tool needs an entry in `_DIFF_HEADER_CATEGORIES` as
    well, or it heads its diff with no verb at all.

    Args:
        tool_name: Raw name of the tool that produced the diff.

    Returns:
        The verb, or empty when the tool does not mutate a file or has no
        phrasing registered for its category.
    """
    category = _TOOL_SUMMARY_CATEGORY.get(tool_name or "", "")
    if category not in _DIFF_HEADER_CATEGORIES:
        return ""
    phrases = _TOOL_SUMMARY_PHRASES.get(category)
    if phrases is None:
        # A category listed as diff-eligible but never given phrasing. Losing
        # the verb beats raising inside `compose` and killing the whole diff.
        logger.warning("No summary phrasing registered for category %r", category)
        return ""
    return phrases[1]


_Tense = Literal["present", "past"]


def _summary_segment(tally: _CategoryTally, tense: _Tense) -> str:
    """Phrase one category's segment, e.g. "Read 2 files" / "Reading 2 files".

    The lead noun counts distinct targets. When a category in
    `_REPEAT_COUNT_NOUNS` repeated work on one of them, the operation count
    trails in parentheses ("Edited 1 file (3 edits)") so neither number is lost.

    Args:
        tally: The category's distinct-target and total-call counts.
        tense: Whether to phrase the segment in the present or past tense.

    Returns:
        The phrased segment for this category and tense.
    """
    category, tool_name, count = tally.category, tally.rep_name, tally.targets
    if category == "web_search":
        base = "Searching the web" if tense == "present" else "Searched the web"
        return base if count == 1 else f"{base} {count} times"
    if category == "todos":
        return "Updating todos" if tense == "present" else "Updated todos"
    phrase = _TOOL_SUMMARY_PHRASES.get(category)
    if phrase is None:
        present, past = "Running", "Ran"
        singular, plural = f"{tool_name} call", f"{tool_name} calls"
    else:
        present, past, singular, plural = phrase
    verb = present if tense == "present" else past
    noun = singular if count == 1 else plural
    segment = f"{verb} {count} {noun}"
    repeat_noun = _REPEAT_COUNT_NOUNS.get(category)
    if repeat_noun is not None and tally.calls > count:
        # `calls > count` implies at least two calls, so the noun is always
        # plural.
        segment += f" ({tally.calls} {repeat_noun})"
    return segment


def summarize_tool_group(
    calls: Sequence[_SummaryCall], *, tense: _Tense = "past"
) -> str:
    """Build a one-line summary of a run of tool calls.

    Aggregates by category in first-appearance order and lowercases the lead
    word of every segment after the first, e.g. two `read_file` calls on
    different paths plus an `execute` -> "Read 2 files, ran 1 shell command".

    Takes args, not just names, because the counts claim distinct nouns: without
    the argument naming each call's target, one file read twice is
    indistinguishable from two files read once.

    Args:
        calls: `(raw tool name, parsed args)` for each call, in call order.
        tense: Whether to phrase the summary in the present or past tense.

    Returns:
        The aggregated one-line summary string in the requested tense.
    """
    tallies = _tally_categories(calls)
    if not tallies:
        return "Running tools" if tense == "present" else "Ran tools"
    return _join_segments([_summary_segment(tally, tense) for tally in tallies])


def _join_segments(segments: list[str]) -> str:
    """Join summary segments, lowercasing the lead word of all but the first.

    Args:
        segments: Pre-phrased segments in display order.

    Returns:
        The segments joined with ", ", e.g. `["Read 2 files", "Running 1 agent"]`
        -> "Read 2 files, running 1 agent".
    """
    first, *rest = segments
    lowered = [f"{seg[0].lower()}{seg[1:]}" if seg else seg for seg in rest]
    return ", ".join([first, *lowered])


def summarize_live_tool_group(
    completed_calls: Sequence[_SummaryCall],
    pending_calls: Sequence[_SummaryCall],
) -> str:
    """Summarize an in-flight run, mixing past and present tense.

    Completed calls are phrased in the past tense so the work already done in
    the step stays visible, and the still-running calls are phrased in the
    present tense, e.g. two completed `execute` calls plus a pending `task` ->
    "Ran 2 shell commands, running 1 agent".

    Each half is tallied independently, so a file whose second read is still
    running is counted in both — collapsing across the split would drop it from
    the present-tense half and the line would stop reporting the step as
    reading.

    Args:
        completed_calls: `(raw tool name, parsed args)` for calls that finished
            successfully, in call order. Failed/rejected calls are evicted
            before this runs.
        pending_calls: `(raw tool name, parsed args)` for calls still pending or
            running, in call order.

    Returns:
        The combined one-line summary. Empty when neither half has members.
    """
    segments: list[str] = []
    if completed_calls:
        segments.append(summarize_tool_group(completed_calls, tense="past"))
    if pending_calls:
        segments.append(summarize_tool_group(pending_calls, tense="present"))
    if not segments:
        return ""
    return _join_segments(segments)


def _summary_cache_key(
    calls: Sequence[_SummaryCall],
) -> _SummaryCacheKey:
    """Build a cache key capturing everything a summary line depends on.

    The line is a function of each call's `(name, target)`, so the key is too.
    A names-only key would in fact be sufficient today — every membership
    mutation, append and eviction alike, clears the cache outright, so two
    cached states cannot share a name list while differing in targets. Deriving
    the key from the summarizer's real inputs instead of relying on that
    argument keeps it correct if grouping ever changes. Over-invalidation is the
    only cost: reordering calls within a category rebuilds identical text.

    Args:
        calls: `(raw tool name, parsed args)` for each call, in call order.

    Returns:
        A hashable key, order-sensitive to match segment ordering.
    """
    return tuple((name, _summary_target(name, args)) for name, args in calls)


_TOOL_GROUP_COLLAPSED_ACCESSORY_CLASS = "-tool-group-collapsed-accessory"
"""Marker class hiding a collapsed group's accessory widgets.

See `_TOOL_AWAITING_APPROVAL_ACCESSORY_CLASS` for why each hide reason carries its
own class and why none may be replaced by assigning `display`.
"""


def _hides_itself(widget: Widget) -> bool:
    """Whether a widget is hiding itself for reasons a group must not override.

    Returns:
        `True` when the widget tracks its own hide reasons and one applies. Only
        `ToolCallMessage` does; anything else is governed by its group alone.
    """
    return isinstance(widget, ToolCallMessage) and widget.has_own_hide_reason


class ToolGroupSummary(Static):
    """Collapsed one-line stand-in for an assistant step's tool calls.

    Tools are hidden from the moment they start; this single line shows live
    progress ("Running 1 shell command…") and flips to the fully past-tense
    line ("Ran 1 shell command") once every tool finishes. While the step is
    live, finished calls stay visible in the past tense next to the ones still
    running in the present tense (e.g. "Ran 2 shell commands, running 1 agent…")
    so the work already done in the step doesn't disappear. Failed, rejected,
    and skipped tools are evicted to standalone rows (see `_evict_unfoldable`) so
    errors stay visible. Clicking the line or pressing Ctrl+O expands the
    underlying tool rows (and their diffs).

    Two modes:

    - **live** (streaming): created empty, members added via `add_member` as
      they mount, a spinner timer animates the line and re-renders present/past
      tense, and failed tools are ejected back into view so errors stay visible.
    - **finalized** (`live=False`, used for hydration/resume): a fixed set of
      completed tools rendered straight to the past tense with no timer.

    Purely presentational — never tracked by the message store; it is re-derived
    from the mounted tool widgets on each stream boundary and on hydration.
    """

    DEFAULT_CSS = """
    ToolGroupSummary {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
        pointer: pointer;
    }

    ToolGroupSummary:hover {
        color: $text;
    }
    """

    _SPINNER_INTERVAL: ClassVar[float] = 0.1

    _collapsed: var[bool] = var(True)

    def __init__(
        self,
        tools: list[ToolCallMessage] | None = None,
        collapsible: list[Widget] | None = None,
        *,
        accessories: dict[Widget, list[Widget]] | None = None,
        live: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the summary.

        Args:
            tools: Tool widgets the summary aggregates (drives its text). May be
                empty for a live group that grows via `add_member`.
            collapsible: Every widget hidden/shown with the group, including the
                tool widgets and any interleaved diff previews.
            accessories: Decorations (e.g. timestamp footers) keyed by the
                collapsible they trail, hidden and shown with that owner via a
                marker class rather than `display`. Collapsing therefore never
                clears an accessory's own visibility class, so the
                `/timestamps` preference reasserts itself on expand. Keys must
                appear in `collapsible`: `_apply_visibility` iterates
                `collapsible`, so an accessory keyed by a non-member is never
                synced.
            live: When True, animate progress and accept new members until
                `close`. When False, render a finalized past-tense summary.
            **kwargs: Additional arguments passed to `Static`.
        """
        super().__init__("", **kwargs)
        self._tools = list(tools or [])
        self._collapsible = list(collapsible or [])
        self._accessories: dict[Widget, list[Widget]] = {}
        for owner, widgets in (accessories or {}).items():
            self._attach_accessories(owner, widgets)
        self._accepting_members = live
        self._finalized = not live
        self._spinner_pos = 0
        self._timer: Timer | None = None
        # Cached summary phrasing, rebuilt only when membership changes (not on
        # every spinner tick). None means "recompute on next render".
        self._present_text: str | None = None
        self._past_text: str | None = None
        # The (completed, pending) `_summary_cache_key` pair the cached live line
        # was built from. The line mixes finished (past tense) and running
        # (present tense) members, so it must be rebuilt whenever a member
        # finishes, not just when membership grows.
        self._present_key: _LiveSummaryKey | None = None

    def on_mount(self) -> None:
        """Apply initial visibility, render, and arm the spinner if live."""
        self._apply_visibility()
        self._render_line()
        self._sync_timer()

    def _attach_accessories(self, owner: Widget, accessories: Iterable[Widget]) -> None:
        """Record `owner`'s accessories and link them to its approval hiding.

        The single registration point for every fold path (constructor,
        `add_member`, `add_collapsible`) so the group's `_accessories` map and a
        tool's own `_visibility_accessories` cannot drift apart — a tool folded
        via one path would otherwise hide its footer on collapse but not while
        an approval prompt replaced it.
        """
        linked = [a for a in accessories if a not in self._accessories.get(owner, ())]
        if not linked:
            return
        self._accessories.setdefault(owner, []).extend(linked)
        if isinstance(owner, ToolCallMessage):
            owner._register_visibility_accessories(*linked)

    def add_member(self, tool: ToolCallMessage, *accessories: Widget) -> None:
        """Add a tool to a live group and link its accessories.

        Args:
            tool: Tool widget folded into the group.
            accessories: Decorations (e.g. the tool's timestamp footer) that
                follow the tool's visibility via a marker class. Not group
                members: they take neither `-grouped` nor a `display` flip, so
                their own visibility class survives the fold. Also linked to
                the tool's own hide reasons — approval and diff supersession
                both — via `has_own_hide_reason`.
        """
        tool.add_class("-grouped")
        self._tools.append(tool)
        self._collapsible.append(tool)
        self._attach_accessories(tool, accessories)
        self._present_text = self._past_text = self._present_key = None
        self._apply_visibility()
        in_progress = self._sync_lifecycle()
        self._render_line(in_progress=in_progress)

    def add_collapsible(self, widget: Widget, *accessories: Widget) -> None:
        """Attach a non-tool widget (e.g. a diff) and its accessories.

        Args:
            widget: Non-tool widget folded with the group.
            accessories: Decorations (e.g. the widget's timestamp footer) that
                follow the widget's visibility via a marker class. Not group
                members, so their own visibility class survives the fold. No
                approval linkage: only a `ToolCallMessage` can await approval.
        """
        widget.add_class("-grouped")
        self._collapsible.append(widget)
        self._attach_accessories(widget, accessories)
        self._apply_collapsible_visibility(widget, visible=not self._collapsed)

    def close(self) -> None:
        """Stop accepting members and finalize after every tool settles.

        A non-tool stream event can close a group before middleware-generated
        terminal results arrive. Keep the live timer running in that case so a
        later error or rejection is evicted instead of being summarized in the
        past tense as though the tool ran successfully.
        """
        self._accepting_members = False
        self._evict_unfoldable()
        in_progress = self._sync_lifecycle()
        if not self.is_attached:
            return
        if self._tools:
            self._render_line(in_progress=in_progress)
        else:
            # Every tool failed and was ejected — nothing left to summarize.
            # Release whatever is still folded first: once this summary is gone
            # nothing can expand it, so a retained widget (and its accessories)
            # would stay hidden for the rest of the session.
            self._release_all_collapsible()
            self.remove()

    def _release_collapsible(self, widget: Widget) -> None:
        """Drop a widget's group linkage and clear its collapsed accessory class.

        Only the *group's* hide reason is released. A tool's own linkage
        (`_visibility_accessories`) intentionally survives, so a revealed row
        still hides its footer for any of its own hide reasons — an approval
        prompt replacing it, or a diff superseding it. See `has_own_hide_reason`
        for the full set.
        """
        if widget in self._collapsible:
            self._collapsible.remove(widget)
        widget.remove_class("-grouped")
        for accessory in self._accessories.pop(widget, []):
            accessory.remove_class(_TOOL_GROUP_COLLAPSED_ACCESSORY_CLASS)

    def _release_all_collapsible(self) -> None:
        """Release and reveal every remaining folded widget.

        Must run before this summary is removed: a widget left folded keeps
        `-grouped`, `display = False`, and its accessories' marker class with no
        summary left to expand it, which no later toggle can undo.
        """
        for widget in list(self._collapsible):
            self._release_collapsible(widget)
            if widget.is_attached and not _hides_itself(widget):
                widget.display = True

    def reveal_pending(self) -> None:
        """Remove unfinished tool calls from the collapsed group."""
        pending = [tool for tool in self._tools if tool.is_pending]
        if not pending:
            return
        for tool in pending:
            self._tools.remove(tool)
            self._release_collapsible(tool)
            if tool.is_attached and not _hides_itself(tool):
                tool.display = True
        self._present_text = self._past_text = self._present_key = None
        in_progress = self._sync_lifecycle()
        if self._tools:
            self._render_line(in_progress=in_progress)
            return
        self._release_all_collapsible()
        if self.is_attached:
            self.remove()

    @property
    def has_attached_members(self) -> bool:
        """Whether any collapsed widget is still attached to the DOM."""
        return any(widget.is_attached for widget in self._collapsible)

    def toggle(self) -> None:
        """Toggle between collapsed and expanded."""
        self._collapsed = not self._collapsed

    def watch__collapsed(self, _collapsed: bool) -> None:
        """Re-render and re-apply member visibility when the state changes.

        Coalesced into one repaint so expanding a multi-tool group reveals every
        row at once instead of bouncing the transcript per member.
        """
        if not self.is_attached:
            self._apply_visibility()
            self._render_line()
            return
        with self.app.batch_update():
            self._apply_visibility()
            self._render_line()

    def on_click(self, event: Click) -> None:
        """Toggle the group on click."""
        event.stop()
        self.toggle()

    def _in_progress(self) -> bool:
        """Whether any member tool is still pending or running.

        Returns:
            True if at least one member tool has not finished.
        """
        return any(tool.is_pending for tool in self._tools)

    def _sync_lifecycle(self, *, in_progress: bool | None = None) -> bool:
        """Finalize only once a closed group's retained tools have settled.

        Returns:
            Whether any retained tool is still in progress.
        """
        if in_progress is None:
            in_progress = self._in_progress()
        self._finalized = not self._accepting_members and not in_progress
        self._sync_timer()
        return in_progress

    def _evict_unfoldable(self) -> None:
        """Un-fold tools the summary line cannot speak for.

        Two reasons qualify. A non-success (errored, rejected, skipped) must stay
        visible so a failure is not summarized away. So must a success whose
        output opens with a display caveat: the summary is built from tool names
        and arguments, never from tool output, so folding one hides the only
        statement that the change could not be shown — see
        `ToolCallMessage.has_display_caveat`.
        """
        failed = [t for t in self._tools if t.is_failed or t.has_display_caveat]
        if not failed:
            return
        for tool in failed:
            self._tools.remove(tool)
            self._release_collapsible(tool)
            if tool.is_attached and not _hides_itself(tool):
                tool.display = True
        self._present_text = self._past_text = self._present_key = None

    def _sync_timer(self) -> None:
        """Run the spinner timer only while live members are in progress."""
        if not self._finalized and self._in_progress():
            if self._timer is None:
                self._timer = self.set_interval(self._SPINNER_INTERVAL, self._tick)
        else:
            self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        """Advance the spinner, eject failures, and flip to past tense when done."""
        try:
            self._spinner_pos += 1
            before = len(self._tools)
            self._evict_unfoldable()
            evicted = len(self._tools) != before
            if self._collapsed:
                # Re-assert hidden state in case a member was shown externally
                # (e.g. ToolCallMessage.clear_awaiting_approval after HITL).
                self._apply_visibility()
            if not self._tools:
                self._sync_lifecycle(in_progress=False)
                # Nothing can expand this summary once it is gone, so release
                # anything still folded before removing it.
                self._release_all_collapsible()
                if self.is_attached:
                    self.remove()
                return
            in_progress = self._sync_lifecycle()
            # A bare spinner advance keeps the line height. `_render_line`
            # promotes this to a layout update if the pending summary changed.
            self._render_line(
                in_progress=in_progress, layout=evicted or not in_progress
            )
        except Exception:
            # Fires ~10x/second, so an unhandled raise would propagate out of the
            # interval callback and can crash the app repeatedly. The group is
            # purely presentational; stop animating and log rather than take the
            # transcript down.
            logger.exception("ToolGroupSummary spinner tick failed; stopping timer")
            self._stop_timer()

    def _apply_collapsible_visibility(self, widget: Widget, *, visible: bool) -> None:
        """Apply the group's visibility to a widget and its accessories.

        The owner is driven directly via `display`; accessories are driven by a
        marker class instead, so hiding them leaves their independent visibility
        class intact and it reasserts itself when the group expands. The two
        mechanisms are not interchangeable — see
        `_TOOL_AWAITING_APPROVAL_ACCESSORY_CLASS`.

        Accessories are classed even while detached: `set_class` is safe off-DOM
        and nothing revisits a skipped accessory, so guarding on `is_attached`
        here would leave a late-mounted footer stranded visible over a hidden
        row.

        Expanding the group does not reveal a widget hiding for its own reasons
        (`_hides_itself`); its accessories still follow the group, since their
        self-hide reasons carry their own independent classes.
        """
        target = visible and not _hides_itself(widget)
        if widget.is_attached and widget.display != target:
            widget.display = target
        for accessory in self._accessories.get(widget, []):
            accessory.set_class(not visible, _TOOL_GROUP_COLLAPSED_ACCESSORY_CLASS)

    def _apply_visibility(self) -> None:
        """Show or hide every folded widget, and its accessories, per collapse."""
        visible = not self._collapsed
        for widget in self._collapsible:
            self._apply_collapsible_visibility(widget, visible=visible)

    def _render_line(
        self, *, in_progress: bool | None = None, layout: bool = True
    ) -> None:
        """Refresh the summary line for the current tense and collapsed state.

        Args:
            in_progress: Pre-computed progress state to avoid re-scanning members
                on the spinner hot path; recomputed when omitted.
            layout: Whether to force a layout update. A changed summary always
                triggers layout; the spinner hot path passes False so a bare
                glyph swap doesn't relayout the whole transcript 10x/second.
        """
        if not self.is_attached:
            return
        if not self._tools:
            self.update(Content(""), layout=layout)
            return
        glyphs = get_glyphs()
        if in_progress is None:
            in_progress = self._in_progress()
        if not self._finalized and in_progress:
            # Tallied per bucket, not across both: a file whose second read is
            # still in flight must stay visible as being read.
            pending = [t.summary_call for t in self._tools if t.is_pending]
            completed = [t.summary_call for t in self._tools if not t.is_pending]
            key = (_summary_cache_key(completed), _summary_cache_key(pending))
            summary_changed = self._present_text is None or key != self._present_key
            if summary_changed:
                self._present_text = summarize_live_tool_group(completed, pending)
                self._present_key = key
            frames = glyphs.spinner_frames
            spinner = frames[self._spinner_pos % len(frames)]
            self.update(
                Content(f"{spinner} {self._present_text}{glyphs.ellipsis}"),
                layout=layout or summary_changed,
            )
        else:
            mark = (
                glyphs.disclosure_collapsed
                if self._collapsed
                else glyphs.disclosure_expanded
            )
            if self._past_text is None:
                self._past_text = summarize_tool_group(
                    [tool.summary_call for tool in self._tools], tense="past"
                )
            self.update(Content(f"{mark} {self._past_text}"), layout=layout)


class LazyToolGroupSummary(Vertical):
    """Data-backed tool summary that creates detail widgets only when expanded."""

    class ExpansionChanged(Message):
        """Posted after lazy tool details mount or unmount."""

        def __init__(self, widget: LazyToolGroupSummary, expanded: bool) -> None:
            """Initialize an expansion notification.

            Args:
                widget: Lazy group whose detail state changed.
                expanded: Whether its detail widgets are now mounted.
            """
            super().__init__()
            self.widget = widget
            self.expanded = expanded

    DEFAULT_CSS = """
    LazyToolGroupSummary {
        height: auto;
        margin: 0 0 1 0;
    }

    LazyToolGroupSummary .lazy-tool-group-header {
        height: auto;
        padding: 0 1;
        color: $text-muted;
        pointer: pointer;
    }

    LazyToolGroupSummary .lazy-tool-group-header:hover {
        color: $text;
    }

    LazyToolGroupSummary .lazy-tool-group-details {
        height: auto;
    }
    """

    def __init__(
        self,
        messages: list[MessageData],
        *,
        detail_builder: Callable[[MessageData], tuple[Widget, Widget | None]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a collapsed summary from retained message data.

        Args:
            messages: Completed tool and associated diff rows in display order.
            detail_builder: Optional factory for a detail widget and its timestamp.
            **kwargs: Additional arguments passed to `Vertical`.
        """
        super().__init__(**kwargs)
        self._message_data = list(messages)
        self._detail_builder = detail_builder
        self._detail_widgets: list[tuple[MessageData, Widget]] = []
        self._expanded = False
        self._deferred_expanded = False
        self._transitioning = False
        self._details: Vertical | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Compose the persistent header and initially empty detail container.

        Yields:
            Header and detail container widgets.
        """
        yield Static("", classes="lazy-tool-group-header")
        yield Vertical(classes="lazy-tool-group-details")

    def on_mount(self) -> None:
        """Render the summary and restore deferred expansion when requested."""
        self._details = self.query_one(".lazy-tool-group-details", Vertical)
        self._details.display = False
        self._update_header()
        if self._deferred_expanded:
            self._deferred_expanded = False
            self.toggle()

    def _update_header(self) -> None:
        """Refresh disclosure state without constructing any detail widget."""
        if not self.is_attached:
            return
        glyphs = get_glyphs()
        mark = (
            glyphs.disclosure_expanded
            if self._expanded
            else glyphs.disclosure_collapsed
        )
        calls = [
            (data.tool_name, data.tool_args or {})
            for data in self._message_data
            if data.tool_name is not None
        ]
        self.query_one(".lazy-tool-group-header", Static).update(
            Content(f"{mark} {summarize_tool_group(calls)}")
        )

    def toggle(self) -> None:
        """Schedule expansion or collapse without blocking the input handler."""
        if not self._transitioning:
            self.run_worker(self._set_expanded(not self._expanded))

    def _snapshot_detail_state(self) -> None:
        """Persist mutable child state before detail widgets leave the DOM."""
        for data, widget in self._detail_widgets:
            if isinstance(widget, ToolCallMessage):
                data.tool_expanded = widget._expanded

    async def _set_expanded(self, expanded: bool) -> None:
        """Create or discard detail widgets for one explicit toggle."""
        if expanded == self._expanded or self._transitioning:
            return
        details = self._details
        if details is None or not details.is_attached:
            return

        self._transitioning = True
        try:
            if expanded:
                try:
                    entries = []
                    for data in self._message_data:
                        widget, footer = (
                            self._detail_builder(data)
                            if self._detail_builder is not None
                            else (data.to_widget(), None)
                        )
                        entries.append((data, widget, footer))
                    nodes = [
                        node
                        for _data, widget, footer in entries
                        for node in (
                            (widget, footer) if footer is not None else (widget,)
                        )
                    ]
                    await details.mount(*nodes)
                except Exception:
                    logger.warning("Failed to expand lazy tool group", exc_info=True)
                    try:
                        await details.remove_children()
                    except Exception:
                        logger.warning(
                            "Failed to clean up lazy tool group expansion",
                            exc_info=True,
                        )
                    self.app.notify(
                        "Tool details could not be expanded. See the debug log.",
                        severity="warning",
                        timeout=6,
                        markup=False,
                    )
                    return
                self._detail_widgets = [
                    (data, widget) for data, widget, _footer in entries
                ]
                details.display = True
            else:
                self._snapshot_detail_state()
                details.display = False
                try:
                    await details.remove_children()
                except Exception:
                    logger.warning("Failed to collapse lazy tool group", exc_info=True)
                    details.display = True
                    self.app.notify(
                        "Tool details could not be collapsed. See the debug log.",
                        severity="warning",
                        timeout=6,
                        markup=False,
                    )
                    return
                self._detail_widgets = []

            self._expanded = expanded
            self._update_header()
            self.post_message(self.ExpansionChanged(self, expanded))
        finally:
            self._transitioning = False

    @on(Click, ".lazy-tool-group-header")
    def _on_header_click(self, event: Click) -> None:
        """Toggle details when the summary row is clicked."""
        event.stop()
        self.toggle()


class DiffMessage(Static):
    """Widget displaying a diff with syntax highlighting.

    Two behaviors beyond rendering, both easy to break from `compose` without
    noticing, and documented per-parameter on `__init__`:

    - Any `outcome` other than `shown` replaces the diff body with a caveat
      sentence. A body rendered under a lost pre-image would be a whole-file
      insertion that never happened, so suppression is the honest result rather
      than a degradation.
    - A path this session treats as sensitive suppresses the body *and* the
      counts. Leaking `+40 -3` for a credentials file still describes its
      contents, so the header cannot survive a redacted body.

    `renders_caveat` reports whether this widget's body states the change could
    not be shown. The adapter reads it back when deciding whether any surface
    carried the caveat — conjoined with the mount result, since this says only
    what the widget would render, not that it reached the screen.
    """

    DEFAULT_CSS = """
    DiffMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $panel;
    }

    DiffMessage .diff-header {
        margin-bottom: 1;
    }
    """
    """Deliberately carries no per-line color: the row gutters and the
    `.diff-line-*` backgrounds supply it."""

    def __init__(
        self,
        diff_content: str,
        file_path: str = "",
        *,
        tool_name: str | None = None,
        before: str = "",
        after: str = "",
        stats: DiffStats | None = None,
        outcome: DiffOutcome = "shown",
        show_caveat: bool = True,
        show_numbers: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a diff message.

        Args:
            diff_content: The unified diff content
            file_path: Path to the file being modified
            tool_name: Name of the file tool that produced the diff
            before: Source aligned to the diff's old line numbers. Pass the
                whole file, or a prefix `highlight_source_prefixes` previously
                produced — which is what `MessageData` round-trips. Stored
                trimmed, and dropped entirely for a credential path.
            after: Source aligned to the diff's new line numbers, same contract.
            stats: Authoritative `(additions, deletions)`, counted before
                truncation. Always preferred over recounting the diff body;
                `None` recounts.
            outcome: What the operation can honestly say about what it changed.
                Anything but `shown` suppresses the body and replaces it with
                that outcome's caveat. Taken as the outcome rather than as
                independent flags because they are not independent: a
                `stats`-plus-"counts are fiction" pair is representable, and
                whichever of the two a reader trusts, the other contradicts it.
            show_caveat: Whether to render the outcome's caveat. Suppresses only
                the sentence, never the body — an untrusted body stays hidden
                either way. Pass `False` only when a tool row already on screen
                carries the identical sentence, which for `edit_file` is
                guaranteed: it can never be folded into a group, so both would
                render adjacent. The caller owns that judgement because this
                widget cannot see what else is mounted.
            show_numbers: Whether file-relative line numbers may be rendered.
                Diffs whose numbers are not file-relative remain unnumbered.
                The caller owns this judgement: a live `edit_file` diff is
                computed from the full before/after file contents and has
                file-relative numbers, while a resumed-thread `edit_file`
                diff is rebuilt from `old_string`/`new_string` fragments and
                does not.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._diff_content = diff_content
        self._file_path = file_path
        self._tool_name = tool_name
        self._redacted = is_sensitive_file_path(file_path)
        if self._redacted:
            self._before = ""
            self._after = ""
        else:
            self._before, self._after = highlight_source_prefixes(
                diff_content, before, after
            )
        self._stats = stats
        self._outcome = outcome
        self._show_caveat = show_caveat
        self._show_numbers = show_numbers

    @property
    def renders_caveat(self) -> bool:
        """Whether this widget's body states the change could not be shown.

        Read by the adapter instead of inferring from the record's outcome: the
        adapter cannot otherwise know whether the mounted widget actually put
        the caveat on screen, and asserting it did is how a caveat came to be
        both suppressed and reported as delivered.
        """
        return self._outcome != "shown" and self._show_caveat

    def compose(self) -> ComposeResult:
        """Compose the diff message layout.

        Yields:
            Widgets displaying the diff header and formatted content.
        """
        parts: list[str | tuple[str, str] | Content] = []
        if verb := _diff_header_verb(self._tool_name):
            parts.append((f"{verb} ", "bold"))
        parts.append(Content.from_markup("[dim]$path[/dim]", path=self._file_path))

        # Never render the contents or line counts of credential files (e.g.
        # `.env`) — the diff would leak secrets into the terminal UI and
        # scrollback, and the counts would describe them.
        if self._redacted:
            yield Static(Content.assemble(*parts), classes="diff-header")
            if self.renders_caveat:
                # Rendered alongside the redaction notice, not instead of it.
                # The two say different things: redaction means the diff was
                # withheld deliberately, the caveat means it could not be
                # produced at all. Showing only the former tells a reader the
                # change is known and merely hidden. The caveat names the tool
                # and nothing else, so it leaks no file content.
                yield Static(
                    Content.styled(
                        display_caveat(self._outcome, self._tool_name or "operation"),
                        "dim",
                    )
                )
            yield Static(
                Content.styled("Diff hidden — file may contain credentials", "dim")
            )
        elif self._outcome != "shown":
            # The body cannot be trusted. Under `untrusted_before` the diff was
            # computed against a stand-in empty file, so a one-line edit renders
            # as a whole-file insertion; suppressing only the counts would leave
            # the body making the same false claim more loudly, so the caveat
            # replaces it outright.
            #
            # Gated on the outcome, not on `renders_caveat`: `show_caveat`
            # suppresses the sentence when a row already carries it, and must
            # never be able to bring the untrusted body back.
            #
            # The caveat is the shared one, so this widget stands on its own:
            # the tool row that also carries it can be folded into a group, or
            # never have mounted at all, and pointing at it would leave the
            # reader chasing text that is not on screen.
            yield Static(Content.assemble(*parts), classes="diff-header")
            if self.renders_caveat:
                yield Static(
                    Content.styled(
                        display_caveat(self._outcome, self._tool_name or "operation"),
                        "dim",
                    )
                )
        else:
            stats = self._stats if self._stats is not None else self._recount()
            if stats is None:
                parts.append(("  change counts unavailable", "dim"))
            elif stats.additions or stats.deletions:
                parts += ["  ", format_diff_stats(stats)]
            elif not self._diff_content:
                parts.append(("  no changes", "dim"))
            header = Content.assemble(*parts)
            if header.plain:
                yield Static(header, classes="diff-header")
            if self._diff_content:
                yield from compose_diff_lines(
                    self._diff_content,
                    max_lines=100,
                    path=self._file_path,
                    before=self._before,
                    after=self._after,
                    show_numbers=self._show_numbers,
                )

    def _recount(self) -> DiffStats | None:
        """Recount the diff body when no authoritative counts were supplied.

        Returns:
            Counts from the body, or `None` when the body was clipped. A
            truncated diff is missing lines by construction, so counting it would
            assert a number that is known to be short and indistinguishable from
            a correct one. `None` says the counts are unavailable, which is
            different from — and more honest than — silently showing none.
        """
        lines = split_diff_lines(self._diff_content)
        # Any position, not just the last line, and via the shared predicate:
        # the renderer marks a truncated row wherever it appears, and the two
        # must not disagree about whether this body is complete.
        if any(is_truncation_marker(line) for line in lines):
            return None
        return count_diff_change_lines(lines)

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border_left = ("ascii", colors.panel)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Match the pointer shape to the cell under the mouse."""
        self.styles.pointer = _pointer_shape_for(event)

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the message."""
        self.styles.pointer = "default"


class ErrorMessage(Static):
    """Widget displaying an error message."""

    DEFAULT_CSS = """
    ErrorMessage {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $error-muted;
        color: white;
        border-left: wide $error;
    }
    """
    """Tinted background + left border to visually separate errors from output."""

    def __init__(self, error: str | Content, **kwargs: Any) -> None:
        """Initialize an error message.

        Args:
            error: Plain string, or `Content` for pre-styled bodies
                (e.g. with `link`-styled spans).
            **kwargs: Additional arguments passed to parent.
        """
        self._content = error
        super().__init__(**kwargs)

    def render(self) -> Content:
        """Render with theme-aware colors.

        Returns:
            Styled error content; spans on a `Content` body are preserved.
        """
        colors = theme.get_theme_colors(self)
        return Content.assemble(
            Content.styled("Error: ", f"bold {colors.error}"),
            self._content,
        )

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border_left = ("ascii", colors.error)

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open clicked URLs."""
        if event.style.link:
            open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Match the pointer shape to the cell under the mouse."""
        self.styles.pointer = _pointer_shape_for(event)

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the message."""
        self.styles.pointer = "default"


class _RubricResultToggle(Static):
    """Clickable summary or hint for a rubric result."""


class RubricResultMessage(Vertical):
    """Compact grader result with complete, scrollable details on demand."""

    class ExpansionChanged(Message):
        """Posted when the grader-details expansion state changes."""

        def __init__(self, widget: RubricResultMessage, expanded: bool) -> None:
            """Initialize an expansion-state message.

            Args:
                widget: The rubric result whose expansion state changed.
                expanded: Whether the grader details are now expanded.
            """
            super().__init__()
            self.widget = widget
            self.expanded = expanded

    DEFAULT_CSS = """
    RubricResultMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
        border-left: wide $warning;
    }

    RubricResultMessage .rubric-result-summary {
        height: auto;
    }

    RubricResultMessage .rubric-result-details-scroll {
        display: none;
        height: auto;
        max-height: 16;
        margin: 1 0 0 2;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
    }

    RubricResultMessage .rubric-result-details {
        height: auto;
        padding: 0;
    }

    RubricResultMessage .rubric-result-hint {
        height: auto;
        margin-left: 2;
        color: $text-muted;
    }

    RubricResultMessage.-expanded .rubric-result-details-scroll {
        display: block;
    }
    """

    _expanded: var[bool] = var(False, toggle_class="-expanded")

    def __init__(
        self,
        summary: str,
        details: str,
        **kwargs: Any,
    ) -> None:
        """Initialize a grader result.

        Args:
            summary: Concise default transcript line.
            details: Complete user-facing grader explanation and criteria gaps.
            **kwargs: Additional arguments passed to `Vertical`.
        """
        super().__init__(**kwargs)
        self._summary = summary
        self._details = details
        self._hint_widget: _RubricResultToggle | None = None
        self._deferred_expanded = False
        # Last expansion value published to the message store. Deduping against it
        # keeps the reactive's initialization watcher and the deferred restore from
        # re-emitting a value the store already holds.
        self._published_expanded = False

    def compose(self) -> ComposeResult:
        """Compose the compact summary, details viewport, and expansion hint.

        Yields:
            Summary, scrollable details, and toggle hint widgets.
        """
        yield _RubricResultToggle(
            Content.styled(self._summary, "dim italic"),
            classes="rubric-result-summary",
        )
        with VerticalScroll(classes="rubric-result-details-scroll"):
            yield Static(
                Content(self._details),
                classes="rubric-result-details",
            )
        yield _RubricResultToggle("", classes="rubric-result-hint")

    def on_mount(self) -> None:
        """Initialize the expansion hint and restore deferred state."""
        self._hint_widget = self.query_one(
            ".rubric-result-hint",
            _RubricResultToggle,
        )
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border_left = ("ascii", colors.warning)
        if not self._details:
            self._hint_widget.display = False
            return
        # The store already holds the restored state, so record it as published
        # first; the assignment below then dedupes instead of re-emitting it.
        self._published_expanded = self._deferred_expanded
        if self._deferred_expanded:
            self._expanded = True
            self._deferred_expanded = False
        self._update_hint()

    def toggle_details(self) -> None:
        """Toggle the complete grader details."""
        if self._details:
            self._expanded = not self._expanded

    def watch__expanded(self, expanded: bool) -> None:
        """Refresh the hint and publish user-driven expansion for virtualization."""
        self._update_hint()
        # Publish only genuine changes: dedupe against the store's known value to
        # drop the reactive's initialization watcher and the deferred restore, and
        # require `is_attached` so `_expanded` set in pre-mount test setup does not
        # `post_message` on a detached widget (NoActiveAppError).
        if self.is_attached and expanded != self._published_expanded:
            self._published_expanded = expanded
            self.post_message(self.ExpansionChanged(self, expanded))

    def _update_hint(self) -> None:
        """Render the current expansion hint."""
        if self._hint_widget is None or not self._details:
            return
        action = "hide" if self._expanded else "show"
        self._hint_widget.update(
            Content.styled(f"click or Ctrl+O to {action} details", "dim italic")
        )

    @on(Click, "_RubricResultToggle")
    def _on_toggle_click(self, event: Click) -> None:
        """Toggle details from the summary or hint."""
        event.stop()
        self.toggle_details()


class _MutedRichMarkdown:
    """Render Rich markdown to match `AppMessage`'s muted-italic base.

    Plain `AppMessage` strings render as `dim italic` via `Content.styled`
    plus the widget's CSS. Rich's default markdown theme paints h2-h4
    magenta and table headers/borders cyan, and doesn't apply `dim` to
    paragraphs, so markdown blocks look visually distinct. This wrapper:

    - Applies a `rich.theme.Theme` while rendering that strips the stock
        colors while keeping structural emphasis (bold/underline/italic), and
    - Layers `dim` over the whole document via `rich.styled.Styled` so
        body text matches the `dim italic` baseline used elsewhere.
    """

    _THEME_OVERRIDES: ClassVar[dict[str, str]] = {
        "markdown.h1": "bold underline",
        "markdown.h2": "bold underline",
        "markdown.h3": "bold",
        "markdown.h4": "italic",
        "markdown.table.header": "bold",
        "markdown.table.border": "",
    }

    def __init__(self, markup: str) -> None:
        from rich.markdown import (
            Markdown as RichMarkdown,
            MarkdownElement,
            TableElement,
        )
        from rich.table import Table

        class _FoldingTableElement(TableElement):
            """Render long Markdown table cells by folding instead of eliding."""

            def __rich_console__(  # noqa: PLW3201  # Rich renderable protocol
                self, console: RichConsole, options: ConsoleOptions
            ) -> RenderResult:
                for renderable in super().__rich_console__(console, options):
                    if isinstance(renderable, Table):
                        for column in renderable.columns:
                            column.overflow = "fold"
                    yield renderable

        class _FoldingMarkdown(RichMarkdown):
            """Rich Markdown variant that never ellipsizes table cells."""

            elements: ClassVar[dict[str, type[MarkdownElement]]] = {
                **RichMarkdown.elements,
                "table_open": _FoldingTableElement,
            }

        self._markdown = _FoldingMarkdown(markup)
        self._markup = markup

    def __rich_console__(  # noqa: PLW3201  # Rich renderable protocol
        self, console: RichConsole, options: ConsoleOptions
    ) -> RenderResult:
        from rich.styled import Styled
        from rich.theme import Theme

        theme = Theme(self._THEME_OVERRIDES, inherit=True)
        try:
            with console.use_theme(theme):
                yield from Styled(self._markdown, "dim").__rich_console__(
                    console, options
                )
        except Exception:
            # Rich markdown or theme application blew up on malformed input.
            # Fall back to the raw source so the chat view keeps rendering.
            logger.warning(
                "Rich markdown rendering failed; falling back to plain text",
                exc_info=True,
            )
            yield from Styled(self._markup, "dim italic").__rich_console__(
                console, options
            )


# Floor for markdown layout width so a not-yet-sized widget still renders a
# readable table instead of collapsing to a single column.
_MARKDOWN_MIN_RENDER_WIDTH = 20

# One-shot flag (mutable holder to avoid a `global` statement) set once the first
# markdown style conversion fails, so a systematic breakage (e.g. a Rich/Textual
# version drift) surfaces at `warning` once instead of staying invisible at
# `debug`, without spamming a line per unconvertible span.
_markdown_style_conversion_warned = [False]


def _markdown_to_content(
    markup: str, width: int, console: RichConsole | None = None
) -> Content:
    """Render muted markdown to selectable `Content` at a fixed width.

    Textual's mouse text-selection only works over widgets whose rendered
    visual is `Content` or Rich `Text`; a raw Rich renderable (such as
    `_MutedRichMarkdown`) renders as a `RichVisual`, which carries none of the
    per-cell offset metadata selection relies on, so its text can be neither
    highlighted nor copied. Rendering the markdown to segments and rebuilding
    them as `Content` preserves the visual (tables, rules, emphasis) while
    making the text selectable.

    Args:
        markup: The markdown source to render.
        width: Target render width in cells; the markdown is laid out to fit.
        console: Console used to render segments; a default is created when
            `None`.

    Returns:
        `Content` visually equivalent to the rendered markdown, with trailing
            whitespace trimmed from each line so copies stay clean.
    """
    from rich.console import Console
    from rich.segment import Segment
    from textual.content import Span
    from textual.style import Style

    render_width = max(width, 1)
    if console is None:
        console = Console(width=render_width)
    segments = console.render(
        _MutedRichMarkdown(markup), console.options.update_width(render_width)
    )
    content_lines: list[Content] = []
    for line in Segment.split_lines(segments):
        text = "".join(segment.text for segment in line)
        stripped = text.rstrip()
        spans: list[Span] = []
        position = 0
        for segment in line:
            start = position
            position += len(segment.text)
            if start >= len(stripped):
                break
            end = min(position, len(stripped))
            if segment.style is not None and end > start:
                try:
                    style = Style.from_rich_style(segment.style)
                except Exception:  # style conversion is best-effort
                    if not _markdown_style_conversion_warned[0]:
                        _markdown_style_conversion_warned[0] = True
                        logger.warning(
                            "Failed to convert a markdown style; markdown will "
                            "render without some styling (later occurrences log "
                            "at debug)",
                            exc_info=True,
                        )
                    else:
                        logger.debug(
                            "Skipping unconvertible markdown style", exc_info=True
                        )
                else:
                    spans.append(Span(start, end, style))
        content_lines.append(Content(stripped, spans))
    while content_lines and not content_lines[-1].plain:
        content_lines.pop()
    return Content("\n").join(content_lines)


class AppMessage(Static):
    """Widget displaying an app message."""

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    AppMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(
        self,
        message: str | Content,
        *,
        markdown: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a system message.

        Args:
            message: The system message as a string or pre-styled `Content`.
            markdown: When `True`, render `message` as markdown (tables,
                headings, bold, etc.).

                Requires a string message — `Content` objects already carry
                their own structure.
            **kwargs: Additional arguments passed to parent.

        Raises:
            TypeError: If `markdown=True` is combined with a non-string
                `message`.
        """
        self._content = message
        self._is_markdown = markdown
        # Markdown is rendered lazily in `render()` so it can be laid out to the
        # widget's current width and rebuilt as selectable `Content`.
        self._markdown_cache: tuple[int, Content] | None = None
        if markdown:
            if not isinstance(message, str):
                msg = "AppMessage(markdown=True) requires a string message"
                raise TypeError(msg)
            rendered: Content = Content("")
        elif isinstance(message, Content):
            rendered = message
        else:
            rendered = Content.styled(message, "dim italic")
        super().__init__(rendered, **kwargs)

    def render(self) -> Content:
        """Render the message, laying out markdown to the current width.

        Returns:
            The message `Content`. Markdown is rendered to selectable `Content`
                sized to the widget's current render width so it can be
                highlighted and copied.
        """
        if not self._is_markdown:
            return super().render()  # ty: ignore[invalid-return-type]
        width = self._markdown_render_width()
        # The cache is keyed on width only: captured spans are late-bound styles
        # (`dim`, `bold`, ANSI colors) that Textual resolves against the active
        # theme at display time, so cached `Content` still recolors on a theme
        # switch and does not need re-rendering.
        if self._markdown_cache is None or self._markdown_cache[0] != width:
            try:
                console = self.app.console
            except NoActiveAppError:
                console = None
            content = _markdown_to_content(str(self._content), width, console)
            self._markdown_cache = (width, content)
        return self._markdown_cache[1]

    def _markdown_render_width(self) -> int:
        """Best-known content width for laying out markdown, with fallbacks.

        Returns:
            The widget's content width, falling back to the container or app
                width, and never below `_MARKDOWN_MIN_RENDER_WIDTH`.
        """
        width = self.content_size.width
        if width <= 0:
            # `content_size` is not known yet; fall back to the container (then
            # app) width and subtract this widget's own horizontal padding,
            # which those outer widths — unlike `content_size` — don't exclude.
            outer = self.container_size.width
            if outer <= 0:
                try:
                    outer = self.app.size.width
                except NoActiveAppError:
                    outer = 0
            padding = self.styles.padding
            width = outer - padding.left - padding.right
        return max(width, _MARKDOWN_MIN_RENDER_WIDTH)

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open style-embedded hyperlinks on single click."""
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Match the pointer shape to the cell under the mouse."""
        self.styles.pointer = _pointer_shape_for(event)

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the message."""
        self.styles.pointer = "default"


class SummarizationMessage(AppMessage):
    """Widget displaying a summarization completion notification."""

    DEFAULT_CSS = """
    SummarizationMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $primary;
        background: $surface;
        border-left: wide $primary;
        text-style: bold;
    }
    """

    def __init__(self, message: str | Content | None = None, **kwargs: Any) -> None:
        """Initialize a summarization notification message.

        Args:
            message: Optional message override used when rehydrating from the
                message store.

                Defaults to the standard summary notification.
            **kwargs: Additional arguments passed to parent.
        """
        self._raw_message = message
        # Pass the default text to AppMessage for _content serialization;
        # render() supplies theme-aware styling at display time.
        super().__init__(message or "✓ Conversation offloaded", **kwargs)

    def render(self) -> Content:
        """Render with theme-aware colors.

        Returns:
            Styled summarization content with theme-appropriate color.
        """
        colors = theme.get_theme_colors(self)
        if self._raw_message is None:
            return Content.styled("✓ Conversation offloaded", f"bold {colors.primary}")
        if isinstance(self._raw_message, Content):
            return self._raw_message
        return Content.styled(self._raw_message, f"bold {colors.primary}")
