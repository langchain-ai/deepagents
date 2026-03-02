"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.widgets import Static

from deepagents_cli.config import CharsetMode, _detect_charset_mode, get_glyphs

if TYPE_CHECKING:
    from textual.app import ComposeResult


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text.

    Args:
        text: Text that may contain Rich markup

    Returns:
        Escaped text safe for Rich rendering
    """
    # Escape brackets that could be interpreted as markup
    return text.replace("[", r"\[").replace("]", r"\]")


def format_diff_textual(diff: str, max_lines: int | None = 100) -> str:
    """Format a unified diff with line numbers and colors.

    Args:
        diff: Unified diff string
        max_lines: Maximum number of diff lines to show (None for unlimited)

    Returns:
        Rich-formatted diff string with line numbers
    """
    if not diff:
        return "[dim]No changes detected[/dim]"

    glyphs = get_glyphs()
    lines = diff.splitlines()

    # Compute stats first
    additions = sum(
        1 for ln in lines if ln.startswith("+") and not ln.startswith("+++")
    )
    deletions = sum(
        1 for ln in lines if ln.startswith("-") and not ln.startswith("---")
    )

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
        formatted.extend([" ".join(stats_parts), ""])  # Blank line after stats

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

        # Handle diff lines - use gutter bar instead of +/- prefix
        content = line[1:] if line else ""
        escaped_content = _escape_markup(content)

        if line.startswith("-"):
            # Deletion - red gutter bar, subtle red background
            gutter = f"[red bold]{glyphs.gutter_bar}[/red bold]"
            line_num = f"[dim]{old_num:>{width}}[/dim]"
            content = f"[on #2d1515]{escaped_content}[/on #2d1515]"
            formatted.append(f"{gutter}{line_num} {content}")
            old_num += 1
            line_count += 1
        elif line.startswith("+"):
            # Addition - green gutter bar, subtle green background
            gutter = f"[green bold]{glyphs.gutter_bar}[/green bold]"
            line_num = f"[dim]{new_num:>{width}}[/dim]"
            content = f"[on #152d15]{escaped_content}[/on #152d15]"
            formatted.append(f"{gutter}{line_num} {content}")
            new_num += 1
            line_count += 1
        elif line.startswith(" "):
            # Context line - dim gutter
            formatted.append(
                f"[dim]{glyphs.box_vertical}{old_num:>{width}}[/dim]  {escaped_content}"
            )
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
        border: round $primary;
    }

    EnhancedDiff .diff-title {
        color: $primary;
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
        """Initialize the diff widget.

        Args:
            diff: Unified diff string
            title: Title to display above the diff
            max_lines: Maximum number of diff lines to show
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._diff = diff
        self._title = title
        self._max_lines = max_lines
        self._stats = self._compute_stats()

    def _compute_stats(self) -> tuple[int, int]:
        """Compute additions and deletions count.

        Returns:
            Tuple of (additions count, deletions count).
        """
        additions = 0
        deletions = 0
        for line in self._diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        return additions, deletions

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border = ("ascii", "cyan")

    def compose(self) -> ComposeResult:
        """Compose the diff widget layout.

        Yields:
            Widgets for title, formatted diff content, and stats.
        """
        glyphs = get_glyphs()
        h = glyphs.box_double_horizontal
        yield Static(
            f"[bold cyan]{h}{h}{h} {self._title} {h}{h}{h}[/bold cyan]",
            classes="diff-title",
        )

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
