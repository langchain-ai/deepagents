"""Message widgets for deepagents-cli."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Markdown, Static

from deepagents_cli.config import (
    COLORS,
    MODE_PREFIXES,
    CharsetMode,
    _detect_charset_mode,
    get_glyphs,
)
from deepagents_cli.input import EMAIL_PREFIX_PATTERN, INPUT_HIGHLIGHT_PATTERN
from deepagents_cli.tool_display import format_tool_display
from deepagents_cli.widgets._links import open_style_link
from deepagents_cli.widgets.diff import format_diff_textual

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click
    from textual.timer import Timer
    from textual.widgets._markdown import MarkdownStream

logger = logging.getLogger(__name__)

_PREFIX_TO_MODE: dict[str, str] = {v: k for k, v in MODE_PREFIXES.items()}
"""Reverse lookup: trigger character -> mode name."""


def _mode_color(mode: str | None) -> str:
    """Return the color string for a mode, falling back to primary.

    Args:
        mode: Mode name (e.g. `'bash'`, `'command'`) or `None`.

    Returns:
        Hex color string from `COLORS`.
    """
    if not mode:
        return COLORS["primary"]
    color = COLORS.get(f"mode_{mode}")
    if color is None:
        logger.warning(
            "Missing color key 'mode_%s' in COLORS; falling back to primary.", mode
        )
        return COLORS["primary"]
    return color


@dataclass(frozen=True, slots=True)
class FormattedOutput:
    """Result of formatting tool output for display.

    Attributes:
        content: The formatted output content with Rich markup.
        truncation: Description of truncated content (e.g., "10 more lines"),
            or None if no truncation occurred.
    """

    content: str
    truncation: str | None = None


# Maximum number of tool arguments to display inline
_MAX_INLINE_ARGS = 3

# Truncation limits for display
_MAX_TODO_CONTENT_LEN = 70
_MAX_WEB_CONTENT_LEN = 100
_MAX_WEB_PREVIEW_LEN = 150

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
    "http_request",
    # Agent tools
    "task",
    "write_todos",
}


class UserMessage(Static):
    """Widget displaying a user message."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0 0 0;
        background: transparent;
        border-left: wide #10b981;
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
        """Set border style based on charset mode and content prefix."""
        mode = _PREFIX_TO_MODE.get(self._content[:1]) if self._content else None
        color = _mode_color(mode)
        border_type = "ascii" if _detect_charset_mode() == CharsetMode.ASCII else "wide"
        self.styles.border_left = (border_type, color)

    def compose(self) -> ComposeResult:
        """Compose the user message layout.

        Yields:
            Static widget containing the formatted user message.
        """
        text = Text()
        content = self._content

        # Use mode-specific prefix indicator when content starts with a
        # mode trigger character (e.g. "!" for bash, "/" for commands).
        mode = _PREFIX_TO_MODE.get(content[:1]) if content else None
        if mode:
            text.append(f"{content[0]} ", style=f"bold {_mode_color(mode)}")
            content = content[1:]
        else:
            text.append("> ", style=f"bold {COLORS['primary']}")

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
                text.append(content[last_end:start])

            # The regex only matches tokens starting with / or @
            if token.startswith("/") and start == 0:
                # /command at start - yellow/gold
                text.append(token, style="bold #fbbf24")
            elif token.startswith("@"):
                # @file mention - green
                text.append(token, style="bold #10b981")
            last_end = end

        # Add remaining text after last match
        if last_end < len(content):
            text.append(content[last_end:])

        yield Static(text)


class QueuedUserMessage(Static):
    """Widget displaying a queued (pending) user message in grey.

    This is an ephemeral widget that gets removed when the message is dequeued.
    """

    DEFAULT_CSS = """
    QueuedUserMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0 0 0;
        background: transparent;
        border-left: wide #6b7280;
        opacity: 0.6;
    }
    """

    def __init__(self, content: str, **kwargs: Any) -> None:
        """Initialize a queued user message.

        Args:
            content: The message content
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._content = content

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border_left = ("ascii", "#6b7280")

    def compose(self) -> ComposeResult:
        """Compose the queued user message layout.

        Yields:
            Static widget containing the formatted queued message (greyed out).
        """
        text = Text()
        content = self._content
        mode = _PREFIX_TO_MODE.get(content[:1]) if content else None
        if mode:
            text.append(f"{content[0]} ", style=f"bold {COLORS['dim']}")
            content = content[1:]
        else:
            text.append("> ", style=f"bold {COLORS['dim']}")
        text.append(content, style="#9ca3af")
        yield Static(text)


class AssistantMessage(Vertical):
    """Widget displaying an assistant message with markdown support.

    Uses MarkdownStream for smoother streaming instead of re-rendering
    the full content on each update.
    """

    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0 0 0;
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
        yield Markdown("", id="assistant-content")

    def on_mount(self) -> None:
        """Store reference to markdown widget."""
        self._markdown = self.query_one("#assistant-content", Markdown)

    def _get_markdown(self) -> Markdown:
        """Get the markdown widget, querying if not cached.

        Returns:
            The Markdown widget for this message.
        """
        if self._markdown is None:
            self._markdown = self.query_one("#assistant-content", Markdown)
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        """Ensure the markdown stream is initialized.

        Returns:
            The MarkdownStream instance for streaming content.
        """
        if self._stream is None:
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
    Press Ctrl+E to expand/collapse the full output.
    Shows an animated "Running..." indicator while the tool is executing.
    """

    DEFAULT_CSS = """
    ToolCallMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide #3b3b3b;
    }

    ToolCallMessage .tool-header {
        height: auto;
    }

    ToolCallMessage .tool-args {
        color: #6b7280;
        margin-left: 3;
    }

    ToolCallMessage .tool-status {
        margin-left: 3;
    }

    ToolCallMessage .tool-status.pending {
        color: #f59e0b;
    }

    ToolCallMessage .tool-status.success {
        color: #10b981;
    }

    ToolCallMessage .tool-status.error {
        color: #ef4444;
    }

    ToolCallMessage .tool-status.rejected {
        color: #f59e0b;
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
        color: #6b7280;
    }

    ToolCallMessage:hover {
        border-left: wide #525252;
    }
    """

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
        tool_label = escape_markup(format_tool_display(self._tool_name, self._args))
        yield Static(
            f"[bold #f59e0b]{tool_label}[/bold #f59e0b]",
            classes="tool-header",
        )
        # Only show args for tools where header doesn't capture the key info
        if self._tool_name not in _TOOLS_WITH_HEADER_INFO:
            args = self._filtered_args()
            if args:
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in list(args.items())[:_MAX_INLINE_ARGS]
                )
                if len(args) > _MAX_INLINE_ARGS:
                    args_str += ", ..."
                yield Static(
                    f"[dim]({escape_markup(args_str)})[/dim]",
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
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border_left = ("ascii", "#3b3b3b")

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
                    self._status_widget.update("[red]✗ Error[/red]")
                    self._status_widget.display = True
                self._update_output_display()
            case "rejected":
                self._status = "rejected"
                if self._status_widget:
                    self._status_widget.add_class("rejected")
                    self._status_widget.update("[yellow]✗ Rejected[/yellow]")
                    self._status_widget.display = True
            case "skipped":
                self._status = "skipped"
                if self._status_widget:
                    self._status_widget.add_class("rejected")
                    self._status_widget.update("[dim]- Skipped[/dim]")
                    self._status_widget.display = True
            case "running":
                # For running tools, show static "Running..." without animation
                # (animations shouldn't be restored for archived tools)
                self._status = "running"
                if self._status_widget:
                    self._status_widget.add_class("pending")
                    self._status_widget.update("[yellow]⠿ Running...[/yellow]")
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
            elapsed = f" ({elapsed_secs}s)"

        self._status_widget.update(f"[yellow]{frame} Running...{elapsed}[/yellow]")

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
        self._output = result
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
            self._status_widget.update(f"[red]{error_icon} Error[/red]")
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
            self._status_widget.update(f"[yellow]{error_icon} Rejected[/yellow]")
            self._status_widget.display = True

    def set_skipped(self) -> None:
        """Mark the tool call as skipped (due to another rejection)."""
        self._stop_animation()
        self._status = "skipped"
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")  # Use same styling as rejected
            self._status_widget.update("[dim]- Skipped[/dim]")
            self._status_widget.display = True

    def toggle_output(self) -> None:
        """Toggle between preview and full output display."""
        if not self._output:
            return
        self._expanded = not self._expanded
        self._update_output_display()

    def on_click(self, event: Click) -> None:
        """Handle click to toggle output expansion."""
        event.stop()  # Prevent click from bubbling up and scrolling
        self.toggle_output()

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
            return FormattedOutput(content="")

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
            "http_request": self._format_web_output,
            "task": self._format_task_output,
        }

        formatter = formatters.get(self._tool_name)
        if formatter:
            return formatter(output, is_preview=is_preview)

        # Default: return as-is but escape markup
        return FormattedOutput(content=escape_markup(output))

    def _prefix_output(self, content: str) -> str:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """Prefix output with output marker and indent continuation lines.

        Args:
            content: The output content to prefix and indent.

        Returns:
            Content with output prefix on first line and indented continuation.
        """
        lines = content.split("\n")
        if not lines:
            return ""
        output_prefix = get_glyphs().output_prefix
        prefixed = f"{output_prefix} {lines[0]}"
        if len(lines) > 1:
            prefixed += "\n" + "\n".join(f"  {line}" for line in lines[1:])
        return prefixed

    def _format_todos_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format write_todos output as a checklist.

        Returns:
            FormattedOutput with checklist content and optional truncation info.
        """
        items = self._parse_todo_items(output)
        if items is None:
            return FormattedOutput(content=escape_markup(output))

        if not items:
            return FormattedOutput(content="    [dim]No todos[/dim]")

        lines: list[str] = []
        max_items = 4 if is_preview else len(items)

        # Build stats header
        stats_header = self._build_todo_stats(items)
        if stats_header:
            lines.extend([f"    [dim]{stats_header}[/dim]", ""])

        # Format each item
        lines.extend(self._format_single_todo(item) for item in items[:max_items])

        truncation = None
        if is_preview and len(items) > max_items:
            truncation = f"{len(items) - max_items} more"

        return FormattedOutput(content="\n".join(lines), truncation=truncation)

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

    def _build_todo_stats(self, items: list) -> str:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """Build stats string for todo list.

        Returns:
            Formatted stats string showing active, pending, and completed counts.
        """
        completed = sum(
            1 for i in items if isinstance(i, dict) and i.get("status") == "completed"
        )
        active = sum(
            1 for i in items if isinstance(i, dict) and i.get("status") == "in_progress"
        )
        pending = len(items) - completed - active

        parts = []
        if active:
            parts.append(f"[yellow]{active} active[/yellow]")
        if pending:
            parts.append(f"{pending} pending")
        if completed:
            parts.append(f"[green]{completed} done[/green]")
        return " | ".join(parts)

    def _format_single_todo(self, item: dict | str) -> str:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """Format a single todo item.

        Returns:
            Rich-formatted string with checkbox and status styling.
        """
        if isinstance(item, dict):
            content = item.get("content", str(item))
            status = item.get("status", "pending")
        else:
            content = str(item)
            status = "pending"

        if len(content) > _MAX_TODO_CONTENT_LEN:
            content = content[: _MAX_TODO_CONTENT_LEN - 3] + "..."

        glyphs = get_glyphs()
        escaped = escape_markup(content)
        if status == "completed":
            return f"    [green]{glyphs.checkmark} done[/green]   [dim]{escaped}[/dim]"
        if status == "in_progress":
            return f"    [yellow]{glyphs.circle_filled} active[/yellow] {escaped}"
        return f"    [dim]{glyphs.circle_empty} todo[/dim]   {escaped}"

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
                lines = []
                max_items = 5 if is_preview else len(items)  # Show all when expanded
                for item in items[:max_items]:
                    path = Path(str(item))
                    name = path.name
                    # Color by file type
                    if path.suffix in {".py", ".pyx"}:
                        lines.append(f"    [#3b82f6]{name}[/#3b82f6]")
                    elif path.suffix in {".md", ".txt", ".rst"}:
                        lines.append(f"    {name}")
                    elif path.suffix in {".json", ".yaml", ".yml", ".toml"}:
                        lines.append(f"    [#f59e0b]{name}[/#f59e0b]")
                    elif not path.suffix:
                        # Likely a directory or no extension
                        lines.append(f"    [#10b981]{name}/[/#10b981]")
                    else:
                        lines.append(f"    {name}")

                truncation = None
                if is_preview and len(items) > max_items:
                    truncation = f"{len(items) - max_items} more"

                return FormattedOutput(content="\n".join(lines), truncation=truncation)
        except (ValueError, SyntaxError):
            pass

        # Fallback: just escape and return
        return FormattedOutput(content=escape_markup(output))

    def _format_file_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format file read/write output.

        Returns:
            FormattedOutput with file content and optional truncation info.
        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        formatted_lines = [escape_markup(line) for line in lines[:max_lines]]
        content = "\n".join(formatted_lines)

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
                lines = []
                max_items = 5 if is_preview else len(items)  # Show all when expanded
                for item in items[:max_items]:
                    # Show just filename or relative path
                    path = Path(str(item))
                    try:
                        rel = path.relative_to(Path.cwd())
                        display = str(rel)
                    except ValueError:
                        display = path.name
                    lines.append(f"    {display}")

                truncation = None
                if is_preview and len(items) > max_items:
                    truncation = f"{len(items) - max_items} more files"

                return FormattedOutput(content="\n".join(lines), truncation=truncation)
        except (ValueError, SyntaxError):
            pass

        # Fallback: line-based output (grep results)
        lines = output.split("\n")
        max_lines = 5 if is_preview else len(lines)

        formatted_lines = [
            f"    {escape_markup(raw_line.strip())}"
            for raw_line in lines[:max_lines]
            if raw_line.strip()
        ]

        content = "\n".join(formatted_lines)
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
        max_lines = 4 if is_preview else len(lines)  # Show all when expanded

        formatted_lines = []
        for i, line in enumerate(lines[:max_lines]):
            escaped = escape_markup(line)
            # Style only the first line (the command) in dim grey
            if i == 0 and escaped.startswith("$ "):
                formatted_lines.append(f"[dim]{escaped}[/dim]")
            else:
                formatted_lines.append(escaped)

        content = "\n".join(formatted_lines)

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

    def _format_web_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """Format web_search/fetch_url/http_request output.

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

        # Handle fetch_url/http_request response
        if "markdown_content" in data:
            lines = data["markdown_content"].split("\n")
            return self._format_lines_output(lines, is_preview=is_preview)

        if "content" in data:
            content = str(data["content"])
            if is_preview and len(content) > _MAX_WEB_PREVIEW_LEN:
                return FormattedOutput(
                    content=escape_markup(content[:_MAX_WEB_PREVIEW_LEN]),
                    truncation="more",
                )
            return FormattedOutput(content=escape_markup(content))

        # Generic dict - show key fields
        lines = []
        max_keys = 3 if is_preview else len(data)
        for k, v in list(data.items())[:max_keys]:
            v_str = str(v)
            if is_preview and len(v_str) > _MAX_WEB_CONTENT_LEN:
                v_str = v_str[:_MAX_WEB_CONTENT_LEN] + "..."
            lines.append(f"  {k}: {escape_markup(v_str)}")
        truncation = None
        if is_preview and len(data) > max_keys:
            truncation = f"{len(data) - max_keys} more"
        return FormattedOutput(content="\n".join(lines), truncation=truncation)

    def _format_web_search_results(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, results: list, *, is_preview: bool
    ) -> FormattedOutput:
        """Format web search results.

        Returns:
            FormattedOutput with search results and optional truncation info.
        """
        if not results:
            return FormattedOutput(content="[dim]No results[/dim]")
        lines = []
        max_results = 3 if is_preview else len(results)
        for r in results[:max_results]:
            title = r.get("title", "")
            url = r.get("url", "")
            lines.extend(
                [
                    f"  [bold]{escape_markup(title)}[/bold]",
                    f"  [dim]{escape_markup(url)}[/dim]",
                ]
            )
        truncation = None
        if is_preview and len(results) > max_results:
            truncation = f"{len(results) - max_results} more results"
        return FormattedOutput(content="\n".join(lines), truncation=truncation)

    def _format_lines_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, lines: list[str], *, is_preview: bool
    ) -> FormattedOutput:
        """Format a list of lines with optional preview truncation.

        Returns:
            FormattedOutput with lines content and optional truncation info.
        """
        max_lines = 4 if is_preview else len(lines)
        content = "\n".join(escape_markup(line) for line in lines[:max_lines])
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

        formatted_lines = [escape_markup(line) for line in lines[:max_lines]]
        content = "\n".join(formatted_lines)

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
                "[dim italic]click or Ctrl+E to collapse[/dim italic]"
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
                    hint = (
                        f"[dim]{ellipsis} {result.truncation} "
                        "— click or Ctrl+E to expand[/dim]"
                    )
                else:
                    hint = "[dim italic]click or Ctrl+E to expand[/dim italic]"
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


class DiffMessage(Static):
    """Widget displaying a diff with syntax highlighting."""

    DEFAULT_CSS = """
    DiffMessage {
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $surface;
        border: solid $primary;
    }

    DiffMessage .diff-header {
        text-style: bold;
        margin-bottom: 1;
    }

    DiffMessage .diff-add {
        color: #10b981;
        background: #10b98120;
    }

    DiffMessage .diff-remove {
        color: #ef4444;
        background: #ef444420;
    }

    DiffMessage .diff-context {
        color: $text-muted;
    }

    DiffMessage .diff-hunk {
        color: $secondary;
        text-style: bold;
    }
    """

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
                f"[bold]File: {escape_markup(self._file_path)}[/bold]",
                classes="diff-header",
            )

        # Render the diff with enhanced formatting
        rendered = format_diff_textual(self._diff_content, max_lines=100)
        yield Static(rendered)

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border = ("ascii", "cyan")


class ErrorMessage(Static):
    """Widget displaying an error message."""

    DEFAULT_CSS = """
    ErrorMessage {
        height: auto;
        padding: 1;
        margin: 1 0;
        background: #7f1d1d;
        color: white;
        border-left: thick $error;
    }
    """

    def __init__(self, error: str, **kwargs: Any) -> None:
        """Initialize an error message.

        Args:
            error: The error message
            **kwargs: Additional arguments passed to parent
        """
        # Store raw content for serialization
        self._content = error
        # Use Text object to combine styled prefix with unstyled error content
        text = Text("Error: ", style="bold red")
        text.append(error)
        super().__init__(text, **kwargs)

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border_left = ("ascii", "red")


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
        margin: 1 0;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, message: str | Text, **kwargs: Any) -> None:
        """Initialize a system message.

        Args:
            message: The system message as a string or pre-styled Rich Text.
            **kwargs: Additional arguments passed to parent
        """
        # Store raw content for serialization
        self._content = message
        # Use Text object to safely render message without markup parsing
        content = (
            message if isinstance(message, Text) else Text(message, style="dim italic")
        )
        super().__init__(content, **kwargs)

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open Rich-style hyperlinks on single click."""
        open_style_link(event)
