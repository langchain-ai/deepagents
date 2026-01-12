"""Message widgets for deepagents-cli."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownStream

from deepagents_cli.ui import format_tool_display
from deepagents_cli.widgets.diff import format_diff_textual

if TYPE_CHECKING:
    from textual.app import ComposeResult

# Maximum number of tool arguments to display inline
_MAX_INLINE_ARGS = 3


class UserMessage(Static):
    """Widget displaying a user message."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0;
        background: $surface;
        border-left: thick $primary;
    }

    UserMessage .user-prefix {
        color: $primary;
        text-style: bold;
    }

    UserMessage .user-content {
        margin-left: 1;
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

    def compose(self) -> ComposeResult:
        """Compose the user message layout."""
        # Use Text object to combine styled prefix with unstyled user content
        text = Text()
        text.append("> ", style="bold #10b981")
        text.append(self._content)
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
        margin: 1 0;
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

    def compose(self) -> ComposeResult:
        """Compose the assistant message layout."""
        yield Markdown("", id="assistant-content")

    def on_mount(self) -> None:
        """Store reference to markdown widget."""
        self._markdown = self.query_one("#assistant-content", Markdown)

    def _get_markdown(self) -> Markdown:
        """Get the markdown widget, querying if not cached."""
        if self._markdown is None:
            self._markdown = self.query_one("#assistant-content", Markdown)
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        """Ensure the markdown stream is initialized."""
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
    Press Ctrl+O to expand/collapse the full output.
    """

    DEFAULT_CSS = """
    ToolCallMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0;
        background: $surface;
        border-left: thick $secondary;
    }

    ToolCallMessage .tool-header {
        color: $secondary;
        text-style: bold;
    }

    ToolCallMessage .tool-args {
        color: $text-muted;
        margin-left: 2;
    }

    ToolCallMessage .tool-status {
        margin-left: 2;
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
        margin-left: 2;
        margin-top: 1;
        padding: 1;
        background: $surface-darken-1;
        color: $text-muted;
        max-height: 20;
        overflow-y: auto;
    }

    ToolCallMessage .tool-output-preview {
        margin-left: 2;
        color: $text-muted;
    }

    ToolCallMessage .tool-output-hint {
        margin-left: 2;
        color: $primary;
        text-style: italic;
    }
    """

    # Max lines/chars to show in preview mode
    _PREVIEW_LINES = 3
    _PREVIEW_CHARS = 200

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
        self._status = "pending"
        self._output: str = ""
        self._expanded: bool = False

    def compose(self) -> ComposeResult:
        """Compose the tool call message layout."""
        tool_label = format_tool_display(self._tool_name, self._args)
        yield Static(
            f"[bold yellow]Tool:[/bold yellow] {tool_label}",
            classes="tool-header",
        )
        args = self._filtered_args()
        if args:
            args_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:_MAX_INLINE_ARGS])
            if len(args) > _MAX_INLINE_ARGS:
                args_str += ", ..."
            yield Static(f"({args_str})", classes="tool-args")
        yield Static(
            "[yellow]Pending...[/yellow]",
            classes="tool-status pending",
            id="status",
        )
        # Output area - hidden initially, shown when output is set
        # Use markup=False for output content to prevent Rich markup injection
        yield Static("", classes="tool-output-preview", id="output-preview", markup=False)
        yield Static("", classes="tool-output-hint", id="output-hint")  # hint uses our markup
        yield Static("", classes="tool-output", id="output-full", markup=False)

    def on_mount(self) -> None:
        """Hide output areas initially."""
        try:
            self.query_one("#output-preview").display = False
            self.query_one("#output-hint").display = False
            self.query_one("#output-full").display = False
        except NoMatches:
            pass

    def set_success(self, result: str = "") -> None:
        """Mark the tool call as successful.

        Args:
            result: Tool output/result to display
        """
        self._status = "success"
        self._output = result
        try:
            status = self.query_one("#status", Static)
            status.remove_class("pending", "error")
            status.add_class("success")
            status.update("[green]✓ Success[/green]")
        except NoMatches:
            pass
        self._update_output_display()

    def set_error(self, error: str) -> None:
        """Mark the tool call as failed.

        Args:
            error: Error message
        """
        self._status = "error"
        self._output = error
        try:
            status = self.query_one("#status", Static)
            status.remove_class("pending", "success")
            status.add_class("error")
            status.update("[red]✗ Error[/red]")
        except NoMatches:
            pass
        # Always show full error - errors should be visible
        self._expanded = True
        self._update_output_display()

    def set_rejected(self) -> None:
        """Mark the tool call as rejected by user."""
        self._status = "rejected"
        try:
            status = self.query_one("#status", Static)
            status.remove_class("pending", "success", "error")
            status.add_class("rejected")
            status.update("[yellow]✗ Rejected[/yellow]")
        except NoMatches:
            pass

    def toggle_output(self) -> None:
        """Toggle between preview and full output display."""
        if not self._output:
            return
        self._expanded = not self._expanded
        self._update_output_display()

    def _update_output_display(self) -> None:
        """Update the output display based on expanded state."""
        if not self._output:
            return

        try:
            preview = self.query_one("#output-preview", Static)
            hint = self.query_one("#output-hint", Static)
            full = self.query_one("#output-full", Static)

            output_stripped = self._output.strip()
            lines = output_stripped.split("\n")
            total_lines = len(lines)
            total_chars = len(output_stripped)

            # Truncate if too many lines OR too many characters
            needs_truncation = (
                total_lines > self._PREVIEW_LINES or total_chars > self._PREVIEW_CHARS
            )

            if self._expanded:
                # Show full output
                preview.display = False
                hint.display = False
                full.update(self._output)
                full.display = True
            else:
                # Show preview
                full.display = False
                if needs_truncation:
                    # Truncate by lines first, then by chars
                    if total_lines > self._PREVIEW_LINES:
                        preview_text = "\n".join(lines[: self._PREVIEW_LINES])
                    else:
                        preview_text = output_stripped

                    # Also truncate by chars if still too long
                    if len(preview_text) > self._PREVIEW_CHARS:
                        preview_text = preview_text[: self._PREVIEW_CHARS] + "..."

                    preview.update(preview_text)
                    preview.display = True

                    # Show expand hint
                    hint.update("[dim]... (Ctrl+O to expand)[/dim]")
                    hint.display = True
                elif output_stripped:
                    # Output fits in preview, just show it
                    preview.update(output_stripped)
                    preview.display = True
                    hint.display = False
                else:
                    preview.display = False
                    hint.display = False
        except NoMatches:
            pass

    @property
    def has_output(self) -> bool:
        """Check if this tool message has output to display."""
        return bool(self._output)

    def _filtered_args(self) -> dict[str, Any]:
        """Filter large tool args for display."""
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
        """Compose the diff message layout."""
        if self._file_path:
            yield Static(f"[bold]File: {self._file_path}[/bold]", classes="diff-header")

        # Render the diff with enhanced formatting
        rendered = format_diff_textual(self._diff_content, max_lines=100)
        yield Static(rendered)


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
        # Use Text object to combine styled prefix with unstyled error content
        text = Text("Error: ", style="bold red")
        text.append(error)
        super().__init__(text, **kwargs)


class SystemMessage(Static):
    """Widget displaying a system message."""

    DEFAULT_CSS = """
    SystemMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize a system message.

        Args:
            message: The system message
            **kwargs: Additional arguments passed to parent
        """
        # Use Text object to safely render message without markup parsing
        super().__init__(Text(message, style="dim italic"), **kwargs)
