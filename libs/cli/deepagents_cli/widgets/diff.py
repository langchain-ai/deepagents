"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from deepagents_cli.theme import ThemeColors


def format_diff_textual(
    diff: str,
    max_lines: int | None = 100,
    *,
    colors: ThemeColors | None = None,
) -> Content:
    """Format a unified diff with line numbers and colors.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff lines to show (None for unlimited).
        colors: Theme color set for backgrounds. Falls back to `DARK_COLORS`.

    Returns:
        Styled `Content` with line numbers and color-coded diff lines.
    """
    if not diff:
        return Content.styled("No changes detected", "dim")

    c = colors if colors is not None else theme.DARK_COLORS
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

    formatted: list[str | Content] = []

    # Add stats header
    stats_parts: list[str | tuple[str, str] | Content] = []
    if additions:
        stats_parts.append((f"+{additions}", "green"))
    if deletions:
        if stats_parts:
            stats_parts.append(" ")
        stats_parts.append((f"-{deletions}", "red"))
    if stats_parts:
        formatted.extend([Content.assemble(*stats_parts), ""])  # Blank line after stats

    old_num = new_num = 0
    line_count = 0

    for line in lines:
        if max_lines and line_count >= max_lines:
            formatted.append(
                Content.styled(f"\n... ({len(lines) - line_count} more lines)", "dim")
            )
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

        if line.startswith("-"):
            # Deletion - red gutter bar, subtle red background
            formatted.append(
                Content.assemble(
                    (f"{glyphs.gutter_bar}", "red bold"),
                    (f"{old_num:>{width}}", "dim"),
                    " ",
                    Content.styled(content, f"on {c.diff_remove_bg}"),
                )
            )
            old_num += 1
            line_count += 1
        elif line.startswith("+"):
            # Addition - green gutter bar, subtle green background
            formatted.append(
                Content.assemble(
                    (f"{glyphs.gutter_bar}", "green bold"),
                    (f"{new_num:>{width}}", "dim"),
                    " ",
                    Content.styled(content, f"on {c.diff_add_bg}"),
                )
            )
            new_num += 1
            line_count += 1
        elif line.startswith(" "):
            # Context line - dim gutter
            formatted.append(
                Content.assemble(
                    (f"{glyphs.box_vertical}{old_num:>{width}}", "dim"),
                    f"  {content}",
                )
            )
            old_num += 1
            new_num += 1
            line_count += 1
        elif line.strip() == "...":
            # Truncation marker
            formatted.append(Content.styled("...", "dim"))
            line_count += 1
        else:
            # Unrecognized diff line (e.g., "\ No newline at end of file")
            formatted.append(Content.styled(line, "dim"))
            line_count += 1

    return Content("\n").join(formatted)


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
        if is_ascii_mode():
            self.styles.border = ("ascii", "cyan")

    def compose(self) -> ComposeResult:
        """Compose the diff widget layout.

        Yields:
            Widgets for title, formatted diff content, and stats.
        """
        glyphs = get_glyphs()
        h = glyphs.box_double_horizontal
        yield Static(
            Content.styled(f"{h}{h}{h} {self._title} {h}{h}{h}", "bold cyan"),
            classes="diff-title",
        )

        formatted = format_diff_textual(
            self._diff, self._max_lines, colors=theme.get_theme_colors(self)
        )
        yield Static(formatted, classes="diff-content")

        additions, deletions = self._stats
        if additions or deletions:
            content_parts: list[str | tuple[str, str]] = []
            if additions:
                content_parts.append((f"+{additions}", "green"))
            if deletions:
                if content_parts:
                    content_parts.append(" ")
                content_parts.append((f"-{deletions}", "red"))
            yield Static(Content.assemble(*content_parts), classes="diff-stats")
