"""Tool-specific approval widgets for HITL display."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.widgets import Markdown, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

# Constants for display limits
_MAX_VALUE_LEN = 200
_MAX_LINES = 30
_MAX_DIFF_LINES = 50
_MAX_PREVIEW_LINES = 20


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", r"\[").replace("]", r"\]")


class ToolApprovalWidget(Vertical):
    """Base class for tool approval widgets."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the tool approval widget with data."""
        super().__init__(classes="tool-approval-widget")
        self.data = data

    def compose(self) -> ComposeResult:
        """Default compose - override in subclasses."""
        yield Static("Tool details not available", classes="approval-description")


class GenericApprovalWidget(ToolApprovalWidget):
    """Generic approval widget for unknown tools."""

    def compose(self) -> ComposeResult:
        """Compose the generic tool display."""
        for key, value in self.data.items():
            if value is None:
                continue
            value_str = str(value)
            if len(value_str) > _MAX_VALUE_LEN:
                hidden = len(value_str) - _MAX_VALUE_LEN
                value_str = value_str[:_MAX_VALUE_LEN] + f"... ({hidden} more chars)"
            yield Static(f"{key}: {value_str}", markup=False, classes="approval-description")


class WriteFileApprovalWidget(ToolApprovalWidget):
    """Approval widget for write_file - shows file content with syntax highlighting."""

    def compose(self) -> ComposeResult:
        """Compose the file content display with syntax highlighting."""
        file_path = self.data.get("file_path", "")
        content = self.data.get("content", "")
        file_extension = self.data.get("file_extension", "text")

        # File path header
        yield Static(f"File: {file_path}", markup=False, classes="approval-file-path")
        yield Static("")

        # Content with syntax highlighting via Markdown code block
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines > _MAX_LINES:
            # Truncate for display
            shown_lines = lines[:_MAX_LINES]
            remaining = total_lines - _MAX_LINES
            truncated_content = "\n".join(shown_lines) + f"\n... ({remaining} more lines)"
            yield Markdown(f"```{file_extension}\n{truncated_content}\n```")
        else:
            yield Markdown(f"```{file_extension}\n{content}\n```")


class EditFileApprovalWidget(ToolApprovalWidget):
    """Approval widget for edit_file - shows clean diff with colors."""

    DEFAULT_CSS = """
    EditFileApprovalWidget .diff-removed-line {
        color: #ff6b6b;
        background: #3d1f1f;
    }
    EditFileApprovalWidget .diff-added-line {
        color: #69db7c;
        background: #1f3d1f;
    }
    EditFileApprovalWidget .diff-context-line {
        color: #888888;
    }
    EditFileApprovalWidget .diff-stats {
        color: #888888;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the diff display with colored additions and deletions."""
        file_path = self.data.get("file_path", "")
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")

        # Calculate stats first for header
        additions, deletions = self._count_stats(diff_lines, old_string, new_string)

        # File path header with stats
        stats_str = self._format_stats(additions, deletions)
        yield Static(f"[bold cyan]File:[/bold cyan] {file_path}  {stats_str}")
        yield Static("")

        if not diff_lines and not old_string and not new_string:
            yield Static("No changes to display", classes="approval-description")
            return

        # Render content
        if diff_lines:
            yield from self._render_diff_lines_only(diff_lines)
        else:
            yield from self._render_strings_only(old_string, new_string)

    def _count_stats(
        self, diff_lines: list[str], old_string: str, new_string: str
    ) -> tuple[int, int]:
        """Count additions and deletions from diff data."""
        if diff_lines:
            additions = sum(
                1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1 for line in diff_lines if line.startswith("-") and not line.startswith("---")
            )
        else:
            additions = new_string.count("\n") + 1 if new_string else 0
            deletions = old_string.count("\n") + 1 if old_string else 0
        return additions, deletions

    def _format_stats(self, additions: int, deletions: int) -> str:
        """Format stats as colored string."""
        parts = []
        if additions:
            parts.append(f"[green]+{additions}[/green]")
        if deletions:
            parts.append(f"[red]-{deletions}[/red]")
        return " ".join(parts)

    def _render_diff_lines_only(self, diff_lines: list[str]) -> ComposeResult:
        """Render unified diff lines without returning stats."""
        lines_shown = 0

        for line in diff_lines:
            if lines_shown >= _MAX_DIFF_LINES:
                yield Static(f"[dim]... ({len(diff_lines) - lines_shown} more lines)[/dim]")
                break

            if line.startswith(("@@", "---", "+++")):
                continue

            widget = self._render_diff_line(line)
            if widget:
                yield widget
                lines_shown += 1

    def _render_strings_only(self, old_string: str, new_string: str) -> ComposeResult:
        """Render old/new strings without returning stats."""
        if old_string:
            yield Static("[bold red]Removing:[/bold red]")
            yield from self._render_string_lines(old_string, is_addition=False)
            yield Static("")

        if new_string:
            yield Static("[bold green]Adding:[/bold green]")
            yield from self._render_string_lines(new_string, is_addition=True)

    def _render_diff_line(self, line: str) -> Static | None:
        """Render a single diff line with appropriate styling."""
        content = _escape_markup(line[1:] if len(line) > 1 else "")

        if line.startswith("-"):
            return Static(f"[on #3d1f1f][red]- {content}[/red][/on #3d1f1f]")
        if line.startswith("+"):
            return Static(f"[on #1f3d1f][green]+ {content}[/green][/on #1f3d1f]")
        if line.startswith(" "):
            return Static(f"[dim]  {content}[/dim]")
        if line.strip():
            return Static(line, markup=False)
        return None

    def _render_string_lines(self, text: str, *, is_addition: bool) -> ComposeResult:
        """Render lines from a string with appropriate styling."""
        lines = text.split("\n")
        style = "[on #1f3d1f][green]+" if is_addition else "[on #3d1f1f][red]-"
        end_style = "[/green][/on #1f3d1f]" if is_addition else "[/red][/on #3d1f1f]"

        for line in lines[:_MAX_PREVIEW_LINES]:
            escaped = _escape_markup(line)
            yield Static(f"{style} {escaped}{end_style}")

        if len(lines) > _MAX_PREVIEW_LINES:
            remaining = len(lines) - _MAX_PREVIEW_LINES
            yield Static(f"[dim]... ({remaining} more lines)[/dim]")


class BashApprovalWidget(ToolApprovalWidget):
    """Approval widget for bash/shell commands."""

    def compose(self) -> ComposeResult:
        """Compose the bash command display with syntax highlighting."""
        command = self.data.get("command", "")
        description = self.data.get("description", "")

        if description:
            yield Static(description, markup=False, classes="approval-description")
            yield Static("")

        # Show command with bash syntax highlighting
        yield Markdown(f"```bash\n{command}\n```")
