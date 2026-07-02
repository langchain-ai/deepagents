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
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from textual import on
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.events import Click
from textual.geometry import Offset
from textual.message_pump import NoActiveAppError
from textual.reactive import var
from textual.selection import Selection
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import (
    MODE_DISPLAY_GLYPHS,
    detect_mode_prefix,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_code.formatting import format_duration
from deepagents_code.input import EMAIL_PREFIX_PATTERN, INPUT_HIGHLIGHT_PATTERN
from deepagents_code.tool_display import (
    JS_EVAL_HEADER_MAX_LENGTH,
    format_tool_display,
)
from deepagents_code.unicode_security import render_with_unicode_markers
from deepagents_code.widgets._js_eval_display import (
    JsEvalBlock,
    JsEvalError,
    JsEvalResult,
    JsEvalStdout,
    parse_js_eval_blocks,
)
from deepagents_code.widgets._links import (
    event_targets_link,
    open_checked_url_async,
    open_style_link,
)
from deepagents_code.widgets.diff import compose_diff_lines

if TYPE_CHECKING:
    from rich.console import (
        Console as RichConsole,
        ConsoleOptions,
        RenderResult,
    )
    from textual.app import ComposeResult
    from textual.events import MouseMove
    from textual.timer import Timer
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownStream

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


_SUCCESS_EXIT_RE = re.compile(r"\n?\[Command succeeded with exit code 0\]\s*$")
"""Strip the SDK's `[Command succeeded with exit code 0]` trailer from tool output."""


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


class UserMessage(Static):
    """Widget displaying a user message."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $primary;
        pointer: text;
    }

    UserMessage.-cancelled {
        opacity: 0.6;
    }
    """
    """`-cancelled` dims a prompt whose turn was interrupted by the user."""

    def __init__(self, content: str, **kwargs: Any) -> None:
        """Initialize a user message.

        Args:
            content: The message content
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content

    def set_cancelled(self) -> None:
        """Dim the message to mark its turn as interrupted by the user."""
        self.add_class("-cancelled")

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Exclude the prompt prefix glyph from copied text.

        Args:
            selection: The active selection geometry.

        Returns:
            The `(text, ending)` selection with the prefix removed, or `None`.
        """
        return _strip_prompt_prefix(super().get_selection(selection), selection)

    def text_select_all(self) -> None:
        """Select the message body without the prompt prefix glyph."""
        _select_prompt_body(self)

    def on_mount(self) -> None:
        """Add CSS classes for mode-specific border and ASCII border type."""
        mode_match = detect_mode_prefix(self._content)
        if mode_match:
            _prefix, mode = mode_match
            self.add_class(f"-mode-{mode.replace('_', '-')}")
        if is_ascii_mode():
            self.add_class("-ascii")

    def render(self) -> Content:
        """Render the styled user message.

        Returns:
            Styled Content with mode prefix and highlighted mentions.
        """
        colors = theme.get_theme_colors(self)
        parts: list[str | tuple[str, str]] = []
        content = self._content

        # Use mode-specific prefix indicator when content starts with a
        # mode trigger character (e.g. "!" for shell, "/" for commands).
        # The display glyph may differ from the trigger (e.g. "$" for shell).
        mode_match = detect_mode_prefix(content)
        if mode_match:
            prefix_text, mode = mode_match
            glyph = MODE_DISPLAY_GLYPHS.get(mode, prefix_text[0])
            parts.append((f"{glyph} ", f"bold {_mode_color(mode, self)}"))
            content = content[len(prefix_text) :]
        else:
            parts.append(("> ", f"bold {colors.primary}"))

        # Highlight @mentions and /commands in the content
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
                # /command at start
                parts.append((token, f"bold {colors.warning}"))
            elif token.startswith("@"):
                # @file mention
                parts.append((token, f"bold {colors.primary}"))
            last_end = end

        # Add remaining text after last match
        if last_end < len(content):
            parts.append(content[last_end:])

        return Content.assemble(*parts)


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
        pointer: text;
    }
    """
    """Dimmed border + reduced opacity to distinguish queued messages from sent ones."""

    def __init__(self, content: str, **kwargs: Any) -> None:
        """Initialize a queued user message.

        Args:
            content: The message content
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content

    def on_mount(self) -> None:
        """Add ASCII border class when in ASCII mode."""
        if is_ascii_mode():
            self.add_class("-ascii")

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Exclude the prompt prefix glyph from copied text.

        Args:
            selection: The active selection geometry.

        Returns:
            The `(text, ending)` selection with the prefix removed, or `None`.
        """
        return _strip_prompt_prefix(super().get_selection(selection), selection)

    def text_select_all(self) -> None:
        """Select the message body without the prompt prefix glyph."""
        _select_prompt_body(self)

    def render(self) -> Content:
        """Render the queued user message (greyed out).

        Returns:
            Styled Content with dimmed prefix and body.
        """
        colors = theme.get_theme_colors(self)
        content = self._content
        mode_match = detect_mode_prefix(content)
        if mode_match:
            prefix_text, mode = mode_match
            glyph = MODE_DISPLAY_GLYPHS.get(mode, prefix_text[0])
            prefix = (f"{glyph} ", f"bold {colors.muted}")
            content = content[len(prefix_text) :]
        else:
            prefix = ("> ", f"bold {colors.muted}")
        return Content.assemble(prefix, (content, colors.muted))


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
                        "dim",
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
                    "dim",
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
        pointer: text;
    }

    /* Markdown blocks carry a bottom margin for inter-block spacing; drop it
       on the final block so the message has no trailing blank row. */
    AssistantMessage Markdown > *:last-child {
        margin-bottom: 0;
    }
    """

    def __init__(self, content: str = "", **kwargs: Any) -> None:
        """Initialize an assistant message.

        Args:
            content: Initial markdown content
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
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
        """Show a pointer cursor over markdown links, text cursor elsewhere.

        The pointer is set on the inner `Markdown` widget because it carries a
        non-default (`text`) pointer in CSS, so the screen resolves its shape
        before reaching this container.
        """
        if self._markdown is not None:
            self._markdown.styles.pointer = (
                "pointer" if event_targets_link(event) else "text"
            )

    def on_leave(self) -> None:
        """Reset the markdown pointer shape when the mouse leaves the message."""
        if self._markdown is not None:
            self._markdown.styles.pointer = "text"

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
        self._status = "pending"  # Waiting for approval or auto-approve
        self._output: str = ""
        self._expanded: bool = False
        self._args_expanded: bool = False
        # User-provided reason attached to a HITL reject decision (if any).
        self._reject_reason: str | None = None
        # Widget references (set in on_mount)
        self._status_widget: Static | None = None
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
        self._animation_timer: Timer | None = None
        # Deferred state for hydration (set by MessageData.to_widget)
        self._deferred_status: str | None = None
        self._deferred_output: str | None = None
        self._deferred_expanded: bool = False
        self._deferred_reject_reason: str | None = None
        # Whether the widget is currently hidden because an approval prompt
        # is rendering the same content (see `set_awaiting_approval`).
        self._awaiting_approval: bool = False

    def compose(self) -> ComposeResult:
        """Compose the tool call message layout.

        Yields:
            Widgets for header, arguments, status, and output display.
        """
        tool_label = format_tool_display(self._tool_name, self._args)
        yield Static(tool_label, markup=False, classes="tool-header")
        # Task: dedicated description line (dim, truncated)
        if self._tool_name == "task":
            desc = self._args.get("description", "")
            if desc:
                max_len = 120
                suffix = "..." if len(desc) > max_len else ""
                truncated = desc[:max_len].rstrip() + suffix
                yield Static(
                    Content.styled(truncated, "dim"),
                    classes="tool-task-desc",
                )
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

        # Restore deferred state if this widget was hydrated from data
        self._restore_deferred_state()

    def _restore_deferred_state(self) -> None:
        """Restore state from deferred values (used when hydrating from data)."""
        if self._deferred_status is None:
            return

        status = self._deferred_status
        output = self._deferred_output or ""
        self._expanded = self._deferred_expanded
        if self._deferred_reject_reason:
            self._reject_reason = self._deferred_reject_reason

        # Clear deferred values
        self._deferred_status = None
        self._deferred_output = None
        self._deferred_expanded = False
        self._deferred_reject_reason = None

        # Restore based on status (don't restart animations for running tools)
        colors = theme.get_theme_colors(self)
        match status:
            case "success":
                self._status = "success"
                self._output = output
                self._show_success_status()
                self._update_output_display()
            case "error":
                self._status = "error"
                self._output = output
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

    def set_success(self, result: str = "") -> None:
        """Mark the tool call as successful.

        Args:
            result: Tool output/result to display
        """
        self._stop_animation()
        self._status = "success"
        # Strip redundant success trailer — the UI already conveys success
        self._output = _strip_success_exit_line(result)
        self._show_success_status()
        self._update_output_display()

    def _show_success_status(self) -> None:
        """Render the status marker for a completed successful call.

        When the call produces visible output it speaks for itself and the
        status stays hidden; otherwise show a "Success!" marker so a completed
        call (e.g. `edit_file`) isn't left without any outcome indicator.
        """
        if self._status_widget is None:
            return
        self._status_widget.remove_class("pending")
        if (
            self._tool_name != "edit_file"
            and self._format_output(
                self._output, is_preview=False
            ).content.plain.strip()
        ):
            self._status_widget.remove_class("success")
            self._status_widget.display = False
            return
        glyph = get_glyphs().checkmark
        colors = theme.get_theme_colors(self)
        self._status_widget.add_class("success")
        self._status_widget.update(Content.styled(f"{glyph} Success!", colors.success))
        self._status_widget.display = True

    def set_error(self, error: str) -> None:
        """Mark the tool call as failed.

        Args:
            error: Error message
        """
        self._stop_animation()
        self._status = "error"
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
        self._stop_animation()
        self._status = "rejected"
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
        self.display = False

    def clear_awaiting_approval(self) -> None:
        """Restore the tool call after `set_awaiting_approval`.

        No-op if `set_awaiting_approval` was not previously called, so the
        method is safe to call unconditionally from a `finally` block.
        """
        if not self._awaiting_approval:
            return
        self._awaiting_approval = False
        self.display = True

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

    def on_click(self, event: Click) -> None:
        """Toggle output/argument expansion.

        Prefer toggling output, but only when the output can actually
        expand/collapse. Otherwise fall through to the collapsible args/code
        block — `js_eval` commonly has a short, unexpandable result sitting
        below a multi-line, collapsible code block, and the old
        "output wins whenever it exists" rule left that code block stuck.
        """
        event.stop()  # Prevent click from bubbling up and scrolling
        if self._output and self.has_expandable_output:
            self.toggle_output()
        elif self.has_expandable_args:
            self.toggle_args()

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
        }

        formatter = formatters.get(self._tool_name)
        if formatter:
            return formatter(output, is_preview=is_preview)

        if is_preview:
            # Fallback for unknown tools: use generic truncation
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

    def _has_expandable_output(self) -> bool:
        """Return whether collapsed output has hidden content to expand."""
        output = self._output.strip()
        if not output:
            return False

        # Tools in `_COLLAPSE_OUTPUT_BY_DEFAULT` (read_file, grep, glob) collapse
        # their body entirely by default (the header already carries the file
        # path / search pattern), so any result with something to show is
        # expandable regardless of size. `read_file`'s formatter never drops body
        # text, but grep/glob serialize an empty match set as `[]`, which formats
        # to nothing — so confirm the formatted output is non-empty rather than
        # trusting the raw string. Successful `edit_file` similarly hides its
        # redundant success line in the collapsed view while keeping the raw
        # output expandable. This mirrors the empty-output guard in
        # `_update_output_display`, which suppresses any body that would render
        # blank before the collapse branch is reached — the two must move
        # together if that assumption changes. Errors are excluded because
        # `set_error` force-expands every error; treating a short error as
        # always-expandable would offer a collapse that hides it entirely.
        if self._tool_name in _COLLAPSE_OUTPUT_BY_DEFAULT and self._status != "error":
            formatted = self._format_output(output, is_preview=False)
            return bool(formatted.content.plain.strip())
        if self._tool_name == "edit_file" and self._status == "success":
            return True

        if self._tool_name == "write_todos":
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
            return FormattedOutput(content=Content.styled("    No todos", "dim"))

        lines: list[Content] = []
        max_items = 4 if is_preview else len(items)

        # Build stats header
        stats = self._build_todo_stats(items)
        if stats:
            lines.extend([Content.assemble("    ", stats), Content("")])

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
                Content.styled(f"    {glyphs.checkmark} done   ", colors.success),
                text,
                is_preview=is_preview,
                text_style="dim",
            )
        if status == "in_progress":
            return self._format_todo_line(
                Content.styled(f"    {glyphs.circle_filled} active ", colors.warning),
                text,
                is_preview=is_preview,
            )
        return self._format_todo_line(
            Content.styled(f"    {glyphs.circle_empty} todo   ", "dim"),
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
                        lines.append(Content.styled(f"    {name}", theme.FILE_PYTHON))
                    elif path.suffix in {".json", ".yaml", ".yml", ".toml"}:
                        lines.append(Content.styled(f"    {name}", theme.FILE_CONFIG))
                    elif not path.suffix:
                        lines.append(Content.styled(f"    {name}/", theme.FILE_DIR))
                    else:
                        lines.append(Content(f"    {name}"))

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
        r"""Tighten `read_file`'s cat -n line-number gutter for display.

        The tool emits `f"{line_num:6d}\t{line}"` — a 6-wide right-justified
        number plus a tab — so even single-digit line numbers carry five
        leading spaces and the tab pushes content to a distant tab stop. The
        model needs that raw format for edits, but the TUI renders a compact
        gutter instead: numbers right-justified to the widest number actually
        present, then two spaces, mirroring how grep/glob results sit flush
        left. Source indentation after the gutter is preserved untouched.

        Lines that don't match the cat -n shape (e.g. test fixtures or
        non-numbered output) are passed through unchanged.

        Returns:
            The output with compacted gutters, or the original string if no
                line-numbered content was found.
        """
        lines = output.split("\n")
        # Split each line on its gutter tab into (number, source). The gutter
        # tab is always the first one; any tabs in `text` are real source
        # indentation and stay put. The head must be a bare `N` or `N.M` (the
        # latter is a wrapped-line continuation marker) — both sides of the dot
        # are required, so a stray `.5` head marks a non-gutter line.
        parsed: list[tuple[str, str] | None] = []
        width = 0
        for line in lines:
            head, tab, text = line.partition("\t")
            num = head.strip()
            whole, dot, frac = num.partition(".")
            if tab and whole.isdigit() and (not dot or frac.isdigit()):
                parsed.append((num, text))
                width = max(width, len(num))
            else:
                parsed.append(None)

        if width == 0:
            return output

        return "\n".join(
            f"{row[0]:>{width}}  {row[1]}" if row else line
            for line, row in zip(lines, parsed, strict=True)
        )

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

        # Some tools serialize an empty successful result as a non-empty literal
        # (e.g. glob "[]") that formats to no visible content. The raw `_output`
        # is truthy, so the early-return guard at the top of this method doesn't
        # catch it, but rendering it would show an empty box with a misleading
        # expand affordance. Treat it like empty output and render nothing. This
        # also subsumes the all-whitespace case (formats to empty), so the
        # collapsed branch below no longer needs its own empty guard.
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
                    Content.styled("click or Ctrl+O to collapse", "dim italic")
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
            if self._tool_name in _COLLAPSE_OUTPUT_BY_DEFAULT or (
                self._tool_name == "edit_file" and self._status == "success"
            ):
                self._preview_row.display = False
                ellipsis = get_glyphs().ellipsis
                self._hint_widget.update(
                    Content.styled(f"{ellipsis} click or Ctrl+O to expand", "dim")
                )
                self._hint_widget.display = True
                return
            # Truncate the preview only when the output is large enough to
            # warrant it; `write_todos` always uses its compact per-item preview
            # regardless of size.
            is_preview = needs_truncation or self._tool_name == "write_todos"
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
                        f"{ellipsis} {result.truncation} — click or Ctrl+O to expand",
                        "dim",
                    )
                )
                self._hint_widget.display = True
            else:
                self._hint_widget.display = False

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
    def has_expandable_args(self) -> bool:
        """Whether the tool's args are large enough to deserve a collapsible block.

        - `ask_user`: its `questions` payload is too noisy to render inline.
        - `js_eval`: the header shows only the first code line (truncated at
            `JS_EVAL_HEADER_MAX_LENGTH`), so the full program is offered as a
            collapsible block whenever it spans more than one non-blank line *or*
            a single line is long enough to be truncated in the header.
        """
        if self._tool_name == "ask_user":
            return bool(self._args)
        if self._tool_name == "js_eval":
            code = self._args.get("code")
            if isinstance(code, str) and code.strip():
                non_blank = sum(1 for line in code.splitlines() if line.strip())
                return non_blank > 1 or len(code.strip()) > JS_EVAL_HEADER_MAX_LENGTH
        return False

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

        is_code = self._tool_name == "js_eval"
        noun = "code" if is_code else "arguments"
        if self._args_expanded:
            detail = (
                self._format_code_detail() if is_code else self._format_args_detail()
            )
            self._args_widget.update(detail)
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


class DiffMessage(Static):
    """Widget displaying a diff with syntax highlighting."""

    DEFAULT_CSS = """
    DiffMessage {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
        pointer: text;
    }

    DiffMessage .diff-header {
        text-style: bold;
        margin-bottom: 1;
    }

    DiffMessage .diff-add {
        color: $text-success;
        background: $success-muted;
    }

    DiffMessage .diff-remove {
        color: $text-error;
        background: $error-muted;
    }

    DiffMessage .diff-context {
        color: $text-muted;
    }

    DiffMessage .diff-hunk {
        color: $secondary;
        text-style: bold;
    }
    """
    """Diff syntax coloring per theme: additions, removals, muted context."""

    def __init__(self, diff_content: str, file_path: str = "", **kwargs: Any) -> None:
        """Initialize a diff message.

        Args:
            diff_content: The unified diff content
            file_path: Path to the file being modified
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._diff_content = diff_content
        self._file_path = file_path

    def compose(self) -> ComposeResult:
        """Compose the diff message layout.

        Yields:
            Widgets displaying the diff header and formatted content.
        """
        if self._file_path:
            yield Static(
                Content.from_markup("[bold]File: $path[/bold]", path=self._file_path),
                classes="diff-header",
            )

        # Render the diff with per-line Statics (CSS-driven backgrounds)
        yield from compose_diff_lines(self._diff_content, max_lines=100)

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.primary)


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
        pointer: text;
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
        from rich.markdown import Markdown as RichMarkdown

        self._markdown = RichMarkdown(markup)
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
        pointer: text;
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
            markdown: When `True`, render `message` as markdown via Rich's
                markdown renderer (tables, headings, bold, etc.).

                Requires a string message — `Content` objects already carry
                their own structure.
            **kwargs: Additional arguments passed to parent.

        Raises:
            TypeError: If `markdown=True` is combined with a non-string
                `message`.
        """
        self._content = message
        self._is_markdown = markdown
        if markdown:
            if not isinstance(message, str):
                msg = "AppMessage(markdown=True) requires a string message"
                raise TypeError(msg)
            rendered = _MutedRichMarkdown(message)
        elif isinstance(message, Content):
            rendered = message
        else:
            rendered = Content.styled(message, "dim italic")
        super().__init__(rendered, **kwargs)

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open style-embedded hyperlinks on single click."""
        open_style_link(event)


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
        pointer: text;
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
