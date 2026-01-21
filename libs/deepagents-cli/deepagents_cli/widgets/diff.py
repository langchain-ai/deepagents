"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", r"\[").replace("]", r"\]")


def format_diff_textual(diff: str, max_lines: int | None = 100) -> str:
    """Format a unified diff with line numbers and colors.

    Uses standard Rich color names that work with any theme.

    Args:
        diff: Unified diff string
        max_lines: Maximum number of diff lines to show (None for unlimited)

    Returns:
        Rich-formatted diff string with line numbers
    """
    if not diff:
        return "[dim]No changes detected[/dim]"

    lines = diff.splitlines()

    # Compute stats first
    additions = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    deletions = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))

    # Find max line number for width calculation
    max_line = 0
    for line in lines:
        if m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            max_line = max(max_line, int(m.group(1)), int(m.group(2)))
    width = max(3, len(str(max_line + len(lines))))

    formatted = []

    # Add stats header
    stats_parts = []
    if additions:
        stats_parts.append(f"[green]+{additions}[/green]")
    if deletions:
        stats_parts.append(f"[red]-{deletions}[/red]")
    if stats_parts:
        formatted.append(" ".join(stats_parts))
        formatted.append("")  # Blank line after stats

    old_num = new_num = 0
    line_count = 0

    for line in lines:
        if max_lines and line_count >= max_lines:
            formatted.append(f"\n[dim]... ({len(lines) - line_count} more lines)[/dim]")
            break

        # Skip file headers (--- and +++)
        if line.startswith(("---", "+++")):
            continue

        # Handle hunk headers - just update line numbers, don't display
        if m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            old_num, new_num = int(m.group(1)), int(m.group(2))
            continue

        # Handle diff lines
        content = line[1:] if line else ""
        escaped_content = _escape_markup(content)

        if line.startswith("-"):
            # Deletion - red
            formatted.append(
                f"[red bold]▌[/red bold][dim]{old_num:>{width}}[/dim] "
                f"[red]{escaped_content}[/red]"
            )
            old_num += 1
            line_count += 1
        elif line.startswith("+"):
            # Addition - green
            formatted.append(
                f"[green bold]▌[/green bold][dim]{new_num:>{width}}[/dim] "
                f"[green]{escaped_content}[/green]"
            )
            new_num += 1
            line_count += 1
        elif line.startswith(" "):
            # Context line - dim
            formatted.append(f"[dim]│{old_num:>{width}}[/dim]  {escaped_content}")
            old_num += 1
            new_num += 1
            line_count += 1
        elif line.strip() == "...":
            # Truncation marker
            formatted.append("[dim]...[/dim]")
            line_count += 1

    return "\n".join(formatted)


class EnhancedDiff(Vertical):
    """Widget for displaying a unified diff with syntax highlighting."""

    DEFAULT_CSS = """
    EnhancedDiff {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: round gray;
    }

    EnhancedDiff .diff-title {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }

    EnhancedDiff .diff-content {
        height: auto;
    }

    EnhancedDiff .diff-stats {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        diff: str,
        title: str = "Diff",
        max_lines: int | None = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize the diff widget."""
        super().__init__(**kwargs)
        self._diff = diff
        self._title = title
        self._max_lines = max_lines
        self._stats = self._compute_stats()

    def _compute_stats(self) -> tuple[int, int]:
        """Compute additions and deletions count."""
        additions = 0
        deletions = 0
        for line in self._diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        return additions, deletions

    def compose(self) -> ComposeResult:
        """Compose the diff widget layout."""
        yield Static(f"═══ {self._title} ═══", classes="diff-title")

        formatted = format_diff_textual(self._diff, self._max_lines)
        yield Static(formatted, classes="diff-content")

        additions, deletions = self._stats
        if additions or deletions:
            stats_parts = []
            if additions:
                stats_parts.append(f"[green]+{additions}[/green]")
            if deletions:
                stats_parts.append(f"[red]-{deletions}[/red]")
            yield Static(" ".join(stats_parts), classes="diff-stats")
