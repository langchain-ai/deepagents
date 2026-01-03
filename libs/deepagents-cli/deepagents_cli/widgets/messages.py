"""Message widgets for deepagents-cli."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.widgets import Markdown, Static

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
        yield Static("[bold cyan]>[/bold cyan] " + self._content)


class AssistantMessage(Vertical):
    """Widget displaying an assistant message with markdown support."""

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

    def compose(self) -> ComposeResult:
        """Compose the assistant message layout."""
        yield Markdown(self._content, id="assistant-content")

    def on_mount(self) -> None:
        """Store reference to markdown widget."""
        self._markdown = self.query_one("#assistant-content", Markdown)

    async def append_content(self, text: str) -> None:
        """Append content to the message (for streaming).

        Args:
            text: Text to append
        """
        self._content += text
        if self._markdown:
            await self._markdown.update(self._content)

    async def set_content(self, content: str) -> None:
        """Set the full message content.

        Args:
            content: The markdown content to display
        """
        self._content = content
        if self._markdown:
            await self._markdown.update(content)


class ToolCallMessage(Vertical):
    """Widget displaying a tool call."""

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
        self._status = "pending"

    def compose(self) -> ComposeResult:
        """Compose the tool call message layout."""
        tool_label = format_tool_display(self._tool_name, self._args)
        yield Static(
            f"[bold yellow]Tool:[/bold yellow] {tool_label}",
            classes="tool-header",
        )
        args = self._filtered_args()
        if args:
            args_str = ", ".join(
                f"{k}={v!r}" for k, v in list(args.items())[:_MAX_INLINE_ARGS]
            )
            if len(args) > _MAX_INLINE_ARGS:
                args_str += ", ..."
            yield Static(f"({args_str})", classes="tool-args")
        yield Static(
            "[yellow]Pending...[/yellow]",
            classes="tool-status pending",
            id="status",
        )

    def set_success(self, result: str = "") -> None:
        """Mark the tool call as successful.

        Args:
            result: Optional result message
        """
        self._status = "success"
        try:
            status = self.query_one("#status", Static)
            status.remove_class("pending", "error")
            status.add_class("success")
            msg = "[green]Success[/green]"
            if result:
                msg += f": {result[:100]}"
            status.update(msg)
        except LookupError:
            pass

    def set_error(self, error: str) -> None:
        """Mark the tool call as failed.

        Args:
            error: Error message
        """
        self._status = "error"
        try:
            status = self.query_one("#status", Static)
            status.remove_class("pending", "success")
            status.add_class("error")
            status.update(f"[red]Error:[/red] {error[:100]}")
        except LookupError:
            pass

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
        super().__init__(f"[bold red]Error:[/bold red] {error}", **kwargs)


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
        super().__init__(f"[dim]{message}[/dim]", **kwargs)
