"""Tool-specific approval widgets for HITL display."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Markdown, Static

from deepagents_code import theme
from deepagents_code.diff_utils import DiffStats, count_diff_change_lines
from deepagents_code.file_ops import is_sensitive_file_path
from deepagents_code.tui.widgets.diff import compose_diff_lines, format_diff_stats

if TYPE_CHECKING:
    from textual.app import ComposeResult

_CREDENTIAL_NOTICE = "Contents hidden — file may contain credentials"

# Constants for display limits
_MAX_VALUE_LEN = 200
_MAX_LINES = 30
_MAX_DIFF_LINES = 50
_MAX_PREVIEW_LINES = 20

_NO_STATS = DiffStats(additions=0, deletions=0)
"""Stand-in for a header with no counts to show."""


def format_display_content(content: object) -> str:
    """Coerce arbitrary tool-arg content into a displayable string.

    Strings pass through unchanged; other values are JSON-formatted for
    readability, falling back to `str()` when serialization fails.

    Returns:
        A string safe to render in an approval widget.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, indent=2)
    except (TypeError, ValueError, RecursionError):
        return str(content)


def _file_header(file_path: str, stats: DiffStats = _NO_STATS) -> ComposeResult:
    """Yield the `File:` path header with optional `+N -M` stats.

    Args:
        file_path: Path to the file being modified.
        stats: Line counts for the change; zeros render no counts.

    Yields:
        Static widgets for the file path header and a spacer line.
    """
    yield Static(
        Content.assemble(
            Content.from_markup("[bold cyan]File:[/bold cyan] $path  ", path=file_path),
            format_diff_stats(stats),
        )
    )
    yield Static("")


def _count_diff_stats(
    diff_lines: list[str],
    old_string: str,
    new_string: str,
    stats: DiffStats | None = None,
) -> DiffStats:
    """Resolve the counts to show above an approval diff.

    Args:
        diff_lines: Unified diff output lines.
        old_string: Original text being replaced (fallback when no diff).
        new_string: Replacement text (fallback when no diff).
        stats: Authoritative counts from the preview's producer, taken before
            the body was clipped. Always preferred where supplied, because the
            `delete` preview is built with `max_lines=100`, so recounting
            `diff_lines` would describe the excerpt rather than the change — and
            that number is what the user approves when destroying a file. The
            edit path's fragment fallback builds `diff_lines` uncapped and
            supplies none, so the recount below is exact there.
            (`ApprovalPreview.stats` documents the same rule across all three
            producers; this function serves only two of them.)

    Returns:
        Line counts for the change.
    """
    if stats is not None:
        return stats
    if diff_lines:
        return count_diff_change_lines(diff_lines)
    return DiffStats(
        additions=new_string.count("\n") + 1 if new_string else 0,
        deletions=old_string.count("\n") + 1 if old_string else 0,
    )


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
        content = format_display_content(self.data.get("content", ""))
        file_extension = self.data.get("file_extension", "text")

        # Never render the contents of credential files (e.g. `.env`).
        if is_sensitive_file_path(file_path):
            yield from _file_header(file_path)
            yield Static(Content.styled(_CREDENTIAL_NOTICE, "dim"))
        else:
            # Content with syntax highlighting via Markdown code block
            lines = content.split("\n")
            total_lines = len(lines)

            # File header with line count
            yield from _file_header(
                file_path,
                DiffStats(additions=total_lines if content else 0, deletions=0),
            )

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
        old_string = format_display_content(self.data.get("old_string", ""))
        new_string = format_display_content(self.data.get("new_string", ""))

        stats = _count_diff_stats(
            diff_lines, old_string, new_string, self.data.get("stats")
        )
        yield from _file_header(file_path, stats)

        # Never render the diff of credential files (e.g. `.env`); the stats
        # above still convey that a change happened without exposing content.
        if is_sensitive_file_path(file_path):
            yield Static(Content.styled(_CREDENTIAL_NOTICE, "dim"))
        elif not diff_lines and not old_string and not new_string:
            yield Static("No changes to display", classes="approval-description")
        elif diff_lines:
            # The gutter shows only when the renderer marks the diff's numbers
            # as the file's. `EditFileRenderer` sets it for the full-file
            # preview diff; its fragment fallback diffs `old_string` against
            # `new_string`, whose numbers are fragment-relative and would
            # assert locations that are simply wrong. `DeleteFileRenderer`
            # leaves it off too — its diff covers the whole file from line 1,
            # so the numbers are correct but say nothing: every line is going.
            yield from compose_diff_lines(
                "\n".join(diff_lines),
                max_lines=_MAX_DIFF_LINES,
                path=file_path,
                before=old_string,
                after=new_string,
                show_numbers=bool(self.data.get("show_numbers")),
            )
        else:
            yield from self._render_strings_only(old_string, new_string)

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
