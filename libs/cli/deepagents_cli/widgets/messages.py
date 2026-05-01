"""Message widgets for deepagents-cli."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, ClassVar

from textual import on
from textual.containers import Vertical
from textual.content import Content
from textual.events import Click
from textual.reactive import var
from textual.widgets import Static

from deepagents_cli import theme
from deepagents_cli.config import (
    MODE_DISPLAY_GLYPHS,
    PREFIX_TO_MODE,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_cli.formatting import format_duration
from deepagents_cli.input import EMAIL_PREFIX_PATTERN, INPUT_HIGHLIGHT_PATTERN
from deepagents_cli.tool_display import format_tool_display
from deepagents_cli.widgets._links import open_style_link
from deepagents_cli.widgets.diff import compose_diff_lines

if TYPE_CHECKING:
    from rich.console import Console as RichConsole, ConsoleOptions, RenderResult
    from textual.app import ComposeResult
    from textual.timer import Timer
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownStream

logger = logging.getLogger(__name__)


def _show_timestamp_toast(widget: Static | Vertical) -> None:
    """Show a toast with the message's creation timestamp.

    No-ops silently if the widget is not mounted or has no associated message
    data in the store.

    Args:
        widget: The message widget whose timestamp to display.
    """
    from datetime import UTC, datetime

    try:
        app = widget.app
    except Exception:  # noqa: BLE001  # Textual raises when widget has no app
        return
    if not widget.id:
        return
    store = app._message_store  # type: ignore[attr-defined]
    data = store.get_message(widget.id)
    if not data:
        return
    dt = datetime.fromtimestamp(data.timestamp, tz=UTC).astimezone()
    label = f"{dt:%b} {dt.day}, {dt.hour % 12 or 12}:{dt:%M:%S} {dt:%p}"
    app.notify(label, timeout=3)


class _TimestampClickMixin:
    """Mixin that shows a timestamp toast on click.

    Add to any message widget that should display its creation timestamp when
    clicked. Widgets needing additional click behavior (e.g. `ToolCallMessage`,
    `AppMessage`) should override `on_click` and call `_show_timestamp_toast`
    directly instead.
    """

    def on_click(self, event: Click) -> None:  # noqa: ARG002  # Textual event handler
        """Show timestamp toast on click."""
        _show_timestamp_toast(self)  # type: ignore[arg-type]


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
_MAX_WEB_CONTENT_LEN = 100

# Tools that have their key info already in the header (no need for args line)
_TOOLS_WITH_HEADER_INFO: set[str] = {
    # Filesystem tools
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",  # sandbox shell
    # Shell tools
    "shell",  # local shell
    # Web tools
    "web_search",
    "fetch_url",
    # Agent tools
    "task",
    "write_todos",
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


class UserMessage(_TimestampClickMixin, Static):
    """Widget displaying a user message."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $primary;
    }
    """

    def __init__(self, content: str, **kwargs: Any) -> None:
        """Initialize a user message.

        Args:
            content: The message content
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content

    def on_mount(self) -> None:
        """Add CSS classes for mode-specific border and ASCII border type."""
        mode = PREFIX_TO_MODE.get(self._content[:1]) if self._content else None
        if mode:
            self.add_class(f"-mode-{mode}")
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
        mode = PREFIX_TO_MODE.get(content[:1]) if content else None
        if mode:
            glyph = MODE_DISPLAY_GLYPHS.get(mode, content[0])
            parts.append((f"{glyph} ", f"bold {_mode_color(mode, self)}"))
            content = content[1:]
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

    def render(self) -> Content:
        """Render the queued user message (greyed out).

        Returns:
            Styled Content with dimmed prefix and body.
        """
        colors = theme.get_theme_colors(self)
        content = self._content
        mode = PREFIX_TO_MODE.get(content[:1]) if content else None
        if mode:
            glyph = MODE_DISPLAY_GLYPHS.get(mode, content[0])
            prefix = (f"{glyph} ", f"bold {colors.muted}")
            content = content[1:]
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
        else:
            _show_timestamp_toast(self)


class AssistantMessage(_TimestampClickMixin, Vertical):
    """Widget displaying an assistant message with markdown support.

    Uses MarkdownStream for smoother streaming instead of re-rendering
    the full content on each update.
    """

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
    """

    def __init__(self, content: str = "", **kwargs: Any) -> None:
        """Initialize an assistant message.

        Args:
            content: Initial markdown content
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Compose the assistant message layout.

        Yields:
            Markdown widget for rendering assistant content.
        """
        from textual.widgets import Markdown

        yield Markdown("", id="assistant-content")

    def on_mount(self) -> None:
        """Store reference to markdown widget."""
        from textual.widgets import Markdown

        self._markdown = self.query_one("#assistant-content", Markdown)

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
        """Append content to the message (for streaming).

        Uses MarkdownStream for smoother rendering instead of re-rendering
        the full content on each chunk.

        Args:
            text: Text to append
        """
        if not text:
            return
        self._content += text
        stream = self._ensure_stream()
        await stream.write(text)

    async def write_initial_content(self) -> None:
        """Write initial content if provided at construction time."""
        if self._content:
            stream = self._ensure_stream()
            await stream.write(self._content)

    async def stop_stream(self) -> None:
        """Stop the streaming and finalize the content."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None

    async def set_content(self, content: str) -> None:
        """Set the full message content.

        This stops any active stream and sets content directly.

        Args:
            content: The markdown content to display
        """
        await self.stop_stream()
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

    ToolCallMessage .tool-output {
        margin-left: 0;
        margin-top: 0;
        padding: 0;
        height: auto;
    }

    ToolCallMessage .tool-output-preview {
        margin-left: 0;
        margin-top: 0;
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

    # Max lines/chars to show in preview mode
    _PREVIEW_LINES = 6
    _PREVIEW_CHARS = 400

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
        # Widget references (set in on_mount)
        self._status_widget: Static | None = None
        self._preview_widget: Static | None = None
        self._hint_widget: Static | None = None
        self._full_widget: Static | None = None
        # Animation state
        self._spinner_position = 0
        self._start_time: float | None = None
        self._animation_timer: Timer | None = None
        # Deferred state for hydration (set by MessageData.to_widget)
        self._deferred_status: str | None = None
        self._deferred_output: str | None = None
        self._deferred_expanded: bool = False

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
        # Status - shows running animation while pending, then final status
        yield Static("", classes="tool-status", id="status")
        # Output area - hidden initially, shown when output is set
        yield Static("", classes="tool-output-preview", id="output-preview")
        yield Static("", classes="tool-output", id="output-full")
        yield Static("", classes="tool-output-hint", id="output-hint")

    def on_mount(self) -> None:
        """Cache widget references and hide all status/output areas initially."""
        if is_ascii_mode():
            self.add_class("-ascii")

        self._status_widget = self.query_one("#status", Static)
        self._preview_widget = self.query_one("#output-preview", Static)
        self._hint_widget = self.query_one("#output-hint", Static)
        self._full_widget = self.query_one("#output-full", Static)
        # Hide everything initially - status only shown when running or on error/reject
        self._status_widget.display = False
        self._preview_widget.display = False
        self._hint_widget.display = False
        self._full_widget.display = False

        # Restore deferred state if this widget was hydrated from data
        self._restore_deferred_state()

    def _restore_deferred_state(self) -> None:
        """Restore state from deferred values (used when hydrating from data)."""
        if self._deferred_status is None:
            return

        status = self._deferred_status
        output = self._deferred_output or ""
        self._expanded = self._deferred_expanded

        # Clear deferred values
        self._deferred_status = None
        self._deferred_output = None
        self._deferred_expanded = False

        # Restore based on status (don't restart animations for running tools)
        colors = theme.get_theme_colors(self)
        match status:
            case "success":
                self._status = "success"
                self._output = output
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
            elapsed = f" ({format_duration(elapsed_secs)})"

        text = f"{frame} Running...{elapsed}"
        self._status_widget.update(
            Content.styled(text, theme.get_theme_colors(self).warning)
        )

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
        if self._status_widget:
            self._status_widget.remove_class("pending")
            # Hide status on success - output speaks for itself
            self._status_widget.display = False
        self._update_output_display()

    def set_error(self, error: str) -> None:
        """Mark the tool call as failed.

        Args:
            error: Error message
        """
        self._stop_animation()
        self._status = "error"
        # For shell commands, prepend the full command so users can see what failed
        command = (
            self._args.get("command")
            if self._tool_name in {"shell", "bash", "execute"}
            else None
        )
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

    def set_rejected(self) -> None:
        """Mark the tool call as rejected by user."""
        self._stop_animation()
        self._status = "rejected"
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")
            error_icon = get_glyphs().error
            text = f"{error_icon} Rejected"
            colors = theme.get_theme_colors(self)
            self._status_widget.update(Content.styled(text, colors.warning))
            self._status_widget.display = True

    def set_skipped(self) -> None:
        """Mark the tool call as skipped (due to another rejection)."""
        self._stop_animation()
        self._status = "skipped"
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")  # Use same styling as rejected
            self._status_widget.update(Content.styled("- Skipped", "dim"))
            self._status_widget.display = True

    def toggle_output(self) -> None:
        """Toggle between preview and full output display."""
        if not self._output:
            return
        self._expanded = not self._expanded
        self._update_output_display()

    def on_click(self, event: Click) -> None:
        """Toggle output expansion, or show timestamp if no output."""
        event.stop()  # Prevent click from bubbling up and scrolling
        if self._output:
            self.toggle_output()
        else:
            _show_timestamp_toast(self)

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
        output = output.strip()
        if not output:
            return FormattedOutput(content=Content(""))

        # Tool-specific formatting using dispatch table
        formatters = {
            "write_todos": self._format_todos_output,
            "ls": self._format_ls_output,
            "read_file": self._format_file_output,
            "write_file": self._format_file_output,
            "edit_file": self._format_file_output,
            "grep": self._format_search_output,
            "glob": self._format_search_output,
            "shell": self._format_shell_output,
            "bash": self._format_shell_output,
            "execute": self._format_shell_output,
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

    def _prefix_output(self, content: Content) -> Content:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """Prefix output with output marker and indent continuation lines.

        Args:
            content: The styled output content to prefix and indent.

        Returns:
            `Content` with output prefix on first line and indented
                continuation.
        """
        if not content.plain:
            return Content("")
        output_prefix = get_glyphs().output_prefix
        lines = content.split("\n")
        prefixed = [Content.assemble(f"{output_prefix} ", lines[0])]
        prefixed.extend(Content.assemble("  ", line) for line in lines[1:])
        return Content("\n").join(prefixed)

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
        lines.extend(self._format_single_todo(item) for item in items[:max_items])

        truncation = None
        if is_preview and len(items) > max_items:
            truncation = f"{len(items) - max_items} more"

        return FormattedOutput(content=Content("\n").join(lines), truncation=truncation)

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

    def _format_single_todo(self, item: dict | str) -> Content:
        """Format a single todo item.

        Returns:
            Styled `Content` with checkbox and status styling.
        """
        colors = theme.get_theme_colors(self)
        if isinstance(item, dict):
            text = item.get("content", str(item))
            status = item.get("status", "pending")
        else:
            text = str(item)
            status = "pending"

        if len(text) > _MAX_TODO_CONTENT_LEN:
            text = text[: _MAX_TODO_CONTENT_LEN - 3] + "..."

        glyphs = get_glyphs()
        if status == "completed":
            return Content.assemble(
                Content.styled(f"    {glyphs.checkmark} done", colors.success),
                Content.styled(f"   {text}", "dim"),
            )
        if status == "in_progress":
            return Content.assemble(
                Content.styled(f"    {glyphs.circle_filled} active", colors.warning),
                f" {text}",
            )
        return Content.assemble(
            Content.styled(f"    {glyphs.circle_empty} todo", "dim"),
            f"   {text}",
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

    def _format_file_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format file read/write output.

        Returns:
            FormattedOutput with file content and optional truncation info.
        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        parts = [Content(line) for line in lines[:max_lines]]
        content = Content("\n").join(parts)

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

    def _format_search_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format grep/glob search output.

        Returns:
            FormattedOutput with search results and optional truncation info.
        """
        # Try to parse as a Python list (glob returns list of paths)
        try:
            items = ast.literal_eval(output.strip())
            if isinstance(items, list):
                parts: list[Content] = []
                max_items = 5 if is_preview else len(items)
                for item in items[:max_items]:
                    path = Path(str(item))
                    try:
                        rel = path.relative_to(Path.cwd())
                        display = str(rel)
                    except ValueError:
                        display = path.name
                    parts.append(Content(f"    {display}"))

                truncation = None
                if is_preview and len(items) > max_items:
                    truncation = f"{len(items) - max_items} more files"

                return FormattedOutput(
                    content=Content("\n").join(parts), truncation=truncation
                )
        except (ValueError, SyntaxError):
            pass

        # Fallback: line-based output (grep results)
        lines = output.split("\n")
        max_lines = 5 if is_preview else len(lines)

        parts = [
            Content(f"    {raw_line.strip()}")
            for raw_line in lines[:max_lines]
            if raw_line.strip()
        ]

        content = Content("\n").join(parts) if parts else Content("")
        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more"

        return FormattedOutput(content=content, truncation=truncation)

    def _format_shell_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format shell command output.

        Returns:
            FormattedOutput with shell output and optional truncation info.
        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        parts: list[Content] = []
        for i, line in enumerate(lines[:max_lines]):
            if i == 0 and line.startswith("$ "):
                parts.append(Content.styled(line, "dim"))
            else:
                parts.append(Content(line))

        content = Content("\n").join(parts) if parts else Content("")

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

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
            or not self._full_widget
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

        if self._expanded:
            # Show full output with formatting
            self._preview_widget.display = False
            result = self._format_output(self._output, is_preview=False)
            prefixed = self._prefix_output(result.content)
            self._full_widget.update(prefixed)
            self._full_widget.display = True
            # Show collapse hint underneath
            self._hint_widget.update(
                Content.styled("click or Ctrl+O to collapse", "dim italic")
            )
            self._hint_widget.display = True
        else:
            # Show preview
            self._full_widget.display = False
            if needs_truncation:
                result = self._format_output(self._output, is_preview=True)
                prefixed = self._prefix_output(result.content)
                self._preview_widget.update(prefixed)
                self._preview_widget.display = True

                # Build hint with truncation info if available
                if result.truncation:
                    ellipsis = get_glyphs().ellipsis
                    hint = Content.styled(
                        f"{ellipsis} {result.truncation} — click or Ctrl+O to expand",
                        "dim",
                    )
                else:
                    hint = Content.styled("click or Ctrl+O to expand", "dim italic")
                self._hint_widget.update(hint)
                self._hint_widget.display = True
            elif output_stripped:
                # Output fits in preview, show formatted
                result = self._format_output(output_stripped, is_preview=False)
                prefixed = self._prefix_output(result.content)
                self._preview_widget.update(prefixed)
                self._preview_widget.display = True
                self._hint_widget.display = False
            else:
                self._preview_widget.display = False
                self._hint_widget.display = False

    @property
    def has_output(self) -> bool:
        """Check if this tool message has output to display.

        Returns:
            True if there is output content, False otherwise.
        """
        return bool(self._output)

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


class DiffMessage(_TimestampClickMixin, Static):
    """Widget displaying a diff with syntax highlighting."""

    DEFAULT_CSS = """
    DiffMessage {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
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


class ErrorMessage(_TimestampClickMixin, Static):
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

    def __init__(self, error: str, **kwargs: Any) -> None:
        """Initialize an error message.

        Args:
            error: The error message
            **kwargs: Additional arguments passed to parent
        """
        # Store raw content for serialization
        self._content = error
        super().__init__(**kwargs)

    def render(self) -> Content:
        """Render with theme-aware colors.

        Returns:
            Styled error content with theme-appropriate color.
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

    def on_click(self, event: Click) -> None:
        """Open style-embedded hyperlinks on single click and show timestamp."""
        open_style_link(event)
        _show_timestamp_toast(self)


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
