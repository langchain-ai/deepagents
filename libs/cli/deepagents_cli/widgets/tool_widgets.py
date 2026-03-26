"""Tool-specific approval widgets for HITL display."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Markdown, Static

from deepagents_cli import theme

if TYPE_CHECKING:
    from textual.app import ComposeResult

# Constants for display limits
_MAX_VALUE_LEN = 200
_MAX_LINES = 30
_MAX_DIFF_LINES = 50
_MAX_PREVIEW_LINES = 20


def _format_stats(additions: int, deletions: int) -> Content:
    """Format addition/deletion stats as styled Content.

    Args:
        additions: Number of added lines.
        deletions: Number of removed lines.

    Returns:
        Styled Content showing additions and deletions.
    """
    colors = theme.get_theme_colors()
    parts: list[str | tuple[str, str] | Content] = []
    if additions:
        parts.append((f"+{additions}", colors.success))
    if deletions:
        if parts:
            parts.append(" ")
        parts.append((f"-{deletions}", colors.error))
    return Content.assemble(*parts) if parts else Content("")


def _file_header(
    file_path: str, additions: int = 0, deletions: int = 0
) -> ComposeResult:
    """Yield the `File:` path header with optional `+N -M` stats.

    Args:
        file_path: Path to the file being modified.
        additions: Number of added lines.
        deletions: Number of removed lines.

    Yields:
        Static widgets for the file path header and a spacer line.
    """
    stats = _format_stats(additions, deletions)
    yield Static(
        Content.assemble(
            Content.from_markup("[bold cyan]File:[/bold cyan] $path  ", path=file_path),
            stats,
        )
    )
    yield Static("")


def _count_diff_stats(
    diff_lines: list[str], old_string: str, new_string: str
) -> tuple[int, int]:
    """Count additions and deletions from diff data.

    Args:
        diff_lines: Unified diff output lines.
        old_string: Original text being replaced (fallback when no diff).
        new_string: Replacement text (fallback when no diff).

    Returns:
        Tuple of (additions count, deletions count).
    """
    if diff_lines:
        additions = sum(
            1
            for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1
            for line in diff_lines
            if line.startswith("-") and not line.startswith("---")
        )
    else:
        additions = new_string.count("\n") + 1 if new_string else 0
        deletions = old_string.count("\n") + 1 if old_string else 0
    return additions, deletions


class ToolApprovalWidget(Vertical):
    """Base class for tool approval widgets."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the tool approval widget with data."""
        super().__init__(classes="tool-approval-widget")
        self.data = data

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Default compose - override in subclasses.

        Yields:
            Static widget with placeholder message.
        """
        yield Static("Tool details not available", classes="approval-description")


class GenericApprovalWidget(ToolApprovalWidget):
    """Generic approval widget for unknown tools."""

    def compose(self) -> ComposeResult:
        """Compose the generic tool display.

        Yields:
            Static widgets displaying each key-value pair from tool data.
        """
        for key, value in self.data.items():
            if value is None:
                continue
            value_str = str(value)
            if len(value_str) > _MAX_VALUE_LEN:
                hidden = len(value_str) - _MAX_VALUE_LEN
                value_str = value_str[:_MAX_VALUE_LEN] + f"... ({hidden} more chars)"
            yield Static(
                f"{key}: {value_str}", markup=False, classes="approval-description"
            )


class WriteFileApprovalWidget(ToolApprovalWidget):
    """Approval widget for write_file - shows file content with syntax highlighting."""

    def compose(self) -> ComposeResult:
        """Compose the file content display with syntax highlighting.

        Yields:
            Widgets displaying file path header and syntax-highlighted content.
        """
        file_path = self.data.get("file_path", "")
        content = self.data.get("content", "")
        file_extension = self.data.get("file_extension", "text")

        # Content with syntax highlighting via Markdown code block
        lines = content.split("\n")
        total_lines = len(lines)

        # File header with line count
        yield from _file_header(file_path, additions=total_lines if content else 0)

        if total_lines > _MAX_LINES:
            # Truncate for display
            shown_lines = lines[:_MAX_LINES]
            remaining = total_lines - _MAX_LINES
            truncated_content = (
                "\n".join(shown_lines) + f"\n... ({remaining} more lines)"
            )
            yield Markdown(f"```{file_extension}\n{truncated_content}\n```")
        else:
            yield Markdown(f"```{file_extension}\n{content}\n```")


class EditFileApprovalWidget(ToolApprovalWidget):
    """Approval widget for edit_file - shows clean diff with colors."""

    def compose(self) -> ComposeResult:
        """Compose the diff display with colored additions and deletions.

        Yields:
            Widgets displaying file path, stats, and colored diff lines.
        """
        file_path = self.data.get("file_path", "")
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")

        additions, deletions = _count_diff_stats(diff_lines, old_string, new_string)
        yield from _file_header(file_path, additions, deletions)

        if not diff_lines and not old_string and not new_string:
            yield Static("No changes to display", classes="approval-description")
        elif diff_lines:
            # Render content
            yield from self._render_diff_lines_only(diff_lines)
        else:
            yield from self._render_strings_only(old_string, new_string)

    def _render_diff_lines_only(self, diff_lines: list[str]) -> ComposeResult:
        """Render unified diff lines without returning stats.

        Yields:
            Static widgets for each diff line with appropriate styling.
        """
        lines_shown = 0

        for line in diff_lines:
            if lines_shown >= _MAX_DIFF_LINES:
                yield Static(
                    Content.styled(
                        f"... ({len(diff_lines) - lines_shown} more lines)", "dim"
                    )
                )
                break

            if line.startswith(("@@", "---", "+++")):
                continue

            widget = self._render_diff_line(line)
            if widget:
                yield widget
                lines_shown += 1

    def _render_strings_only(self, old_string: str, new_string: str) -> ComposeResult:
        """Render old/new strings without returning stats.

        Yields:
            Static widgets showing removed and added content with styling.
        """
        colors = theme.get_theme_colors()
        if old_string:
            yield Static(Content.styled("Removing:", f"bold {colors.error}"))
            yield from self._render_string_lines(old_string, is_addition=False)
            yield Static("")

        if new_string:
            yield Static(Content.styled("Adding:", f"bold {colors.success}"))
            yield from self._render_string_lines(new_string, is_addition=True)

    @staticmethod
    def _render_diff_line(line: str) -> Static | None:
        """Render a single diff line with appropriate styling.

        Returns:
            Static widget with styled diff line, or None for empty/skipped lines.
        """
        raw = line[1:] if len(line) > 1 else ""

        if line.startswith("-"):
            return Static(
                Content.from_markup("- $text", text=raw), classes="diff-removed"
            )
        if line.startswith("+"):
            return Static(
                Content.from_markup("+ $text", text=raw), classes="diff-added"
            )
        if line.startswith(" "):
            return Static(
                Content.from_markup("  $text", text=raw), classes="diff-context"
            )
        if line.strip():
            return Static(line, markup=False)
        return None

    @staticmethod
    def _render_string_lines(text: str, *, is_addition: bool) -> ComposeResult:
        """Render lines from a string with appropriate styling.

        Yields:
            Static widgets for each line with addition or deletion styling.
        """
        lines = text.split("\n")
        sign = "+" if is_addition else "-"
        cls = "diff-added" if is_addition else "diff-removed"

        for line in lines[:_MAX_PREVIEW_LINES]:
            yield Static(Content.from_markup(f"{sign} $text", text=line), classes=cls)

        if len(lines) > _MAX_PREVIEW_LINES:
            remaining = len(lines) - _MAX_PREVIEW_LINES
            yield Static(Content.styled(f"... ({remaining} more lines)", "dim"))
