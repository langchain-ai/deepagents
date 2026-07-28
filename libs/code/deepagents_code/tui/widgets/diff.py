"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from textual.color import Color
from textual.containers import Vertical
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode

if TYPE_CHECKING:
    from textual.app import ComposeResult

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)")
"""Matches a unified-diff hunk header, capturing the old and new start lines."""

_TOKEN_RE = re.compile(r"\w+|\s+|.")
"""Splits a line into words, whitespace runs, and single other characters."""

_EMPHASIS_ALPHA = 0.22
"""Tint strength for changed words, composited over the row's own tint."""

_SIMILARITY_FLOOR = 0.4
"""Below this token similarity a `-`/`+` pair is treated as an unrelated rewrite."""

_MAX_EMPHASIS_LEN = 400
"""Longer lines skip word-level emphasis so the token matcher stays cheap."""

_MIN_NUMBER_WIDTH = 2
"""Narrowest line-number column, so short files still align."""

_ContentPart = str | tuple[str, str | TStyle] | Content
"""A single piece accepted by `Content.assemble`."""

_Range = tuple[int, int]
"""Half-open character range within a diff line."""


@dataclass(frozen=True)
class _Row:
    """One renderable row parsed out of a unified diff."""

    kind: str
    """`context`, `added`, `removed`, `separator`, or `note`."""

    text: str
    """Row content with the diff marker stripped."""

    number: int
    """Line number in the file this row belongs to (0 when not applicable)."""


@dataclass(frozen=True)
class _Palette:
    """Theme-derived styles for the marker column and changed words."""

    added_marker: str
    removed_marker: str
    added_emphasis: TStyle
    removed_emphasis: TStyle


def count_diff_changes(diff: str) -> tuple[int, int]:
    """Count added and removed lines in a unified diff.

    Args:
        diff: Unified diff string.

    Returns:
        Tuple of (additions, deletions), excluding `---`/`+++` file headers.
    """
    additions = 0
    deletions = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


def diff_stats_content(additions: int, deletions: int) -> Content:
    """Build the compact `+N -M` change summary.

    Args:
        additions: Number of added lines.
        deletions: Number of removed lines.

    Returns:
        `Content` with the non-zero sides colored, empty when both are zero.
    """
    colors = theme.get_theme_colors()
    parts: list[_ContentPart] = []
    if additions:
        parts.append((f"+{additions}", colors.success))
    if deletions:
        if parts:
            parts.append(" ")
        parts.append((f"-{deletions}", colors.error))
    return Content.assemble(*parts)


def compose_diff_lines(
    diff: str,
    max_lines: int | None = 100,
    *,
    show_stats: bool = True,
) -> ComposeResult:
    """Yield per-line Static widgets for a unified diff.

    Rows render as a dim line number, a colored `-`/`+` marker, and the line
    content. Added/removed rows get a CSS class (`.diff-line-added`,
    `.diff-line-removed`) so their background tint is driven by CSS variables
    and updates automatically on theme change. Within a related `-`/`+` pair,
    only the words that actually changed carry a stronger tint.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff lines to show (None for unlimited).
        show_stats: Whether to prepend the `+N -M` summary row. Pass `False`
            when the caller already shows the counts in its own header.

    Yields:
        Static widgets — one per diff line — with appropriate CSS classes.
    """
    if not diff:
        yield Static(Content.styled("No changes detected", "dim"))
    else:
        yield from _compose_diff_content(diff, max_lines, show_stats=show_stats)


def _compose_diff_content(
    diff: str,
    max_lines: int | None,
    *,
    show_stats: bool,
) -> ComposeResult:
    """Yield styled diff line widgets for non-empty diff content.

    Args:
        diff: Non-empty unified diff string.
        max_lines: Maximum number of diff lines to show (None for unlimited).
        show_stats: Whether to prepend the `+N -M` summary row.

    Yields:
        Static widgets for the optional stats header and individual diff lines.
    """
    if show_stats:
        additions, deletions = count_diff_changes(diff)
        if additions or deletions:
            yield Static(diff_stats_content(additions, deletions))

    rows = _parse_rows(diff.splitlines())
    emphasis = _emphasis_by_row(rows)
    palette = _palette()
    width = max(_MIN_NUMBER_WIDTH, len(str(max((r.number for r in rows), default=0))))

    for index, row in enumerate(rows):
        if max_lines is not None and index >= max_lines:
            yield Static(
                Content.styled(f"\n... ({len(rows) - index} more lines)", "dim")
            )
            break
        yield _row_widget(row, emphasis.get(index, []), width, palette)


def _parse_rows(lines: list[str]) -> list[_Row]:
    """Convert unified-diff lines into renderable rows.

    File headers are dropped and hunk headers become separators (except the
    first) so consecutive hunks read as distinct blocks.

    Args:
        lines: Lines of a unified diff.

    Returns:
        Rows in render order, each carrying its file line number.
    """
    rows: list[_Row] = []
    old = new = 0
    seen_hunk = False
    for line in lines:
        if line.startswith(("---", "+++")):
            continue
        if match := _HUNK_RE.match(line):
            old, new = int(match.group(1)), int(match.group(2))
            if seen_hunk:
                rows.append(_Row("separator", "", 0))
            seen_hunk = True
        elif line.startswith("-"):
            rows.append(_Row("removed", line[1:], old))
            old += 1
        elif line.startswith("+"):
            rows.append(_Row("added", line[1:], new))
            new += 1
        elif line.startswith(" "):
            rows.append(_Row("context", line[1:], old))
            old += 1
            new += 1
        elif line.strip() == "...":
            rows.append(_Row("separator", "", 0))
        else:
            rows.append(_Row("note", line, 0))
    return rows


def _palette() -> _Palette:
    """Build marker colors and emphasis tints from the active theme.

    Returns:
        Palette used by every row of the current render pass.
    """
    colors = theme.get_theme_colors()
    return _Palette(
        added_marker=colors.success,
        removed_marker=colors.error,
        added_emphasis=TStyle(
            background=Color.parse(colors.success).with_alpha(_EMPHASIS_ALPHA)
        ),
        removed_emphasis=TStyle(
            background=Color.parse(colors.error).with_alpha(_EMPHASIS_ALPHA)
        ),
    )


def _row_widget(
    row: _Row,
    ranges: list[_Range],
    width: int,
    palette: _Palette,
) -> Static:
    """Render one parsed row as a `Static`.

    Args:
        row: Parsed diff row.
        ranges: Character ranges within `row.text` to emphasize.
        width: Width of the line-number column.
        palette: Marker colors and emphasis tints for the active theme.

    Returns:
        A `Static` widget, classed so CSS can tint added/removed rows.
    """
    glyphs = get_glyphs()
    if row.kind == "separator":
        return Static(Content.styled(f"{'':>{width}} {glyphs.ellipsis}", "dim"))
    if row.kind == "note":
        return Static(Content.from_markup("[dim]$text[/dim]", text=row.text))
    if row.kind == "added":
        body = _emphasized(row.text, ranges, palette.added_emphasis)
        return Static(
            _row_content(row.number, width, "+", palette.added_marker, body),
            classes="diff-line-added",
        )
    if row.kind == "removed":
        body = _emphasized(row.text, ranges, palette.removed_emphasis)
        return Static(
            _row_content(row.number, width, "-", palette.removed_marker, body),
            classes="diff-line-removed",
        )
    return Static(_row_content(row.number, width, " ", "dim", [row.text]))


def _row_content(
    number: int,
    width: int,
    marker: str,
    marker_style: str,
    body: list[_ContentPart],
) -> Content:
    """Assemble a diff row: dim line number, marker column, then content.

    Args:
        number: File line number for this row.
        width: Width of the line-number column.
        marker: `-`, `+`, or a space for context rows.
        marker_style: Style applied to the marker.
        body: Pre-split content parts, some of which may be emphasized.

    Returns:
        The row's renderable `Content`.
    """
    return Content.assemble(
        (f"{number:>{width}}", "dim"),
        " ",
        (marker, marker_style),
        " ",
        *body,
    )


def _emphasized(
    text: str, ranges: list[_Range], emphasis: TStyle
) -> list[_ContentPart]:
    """Split `text` so the given ranges render with the emphasis tint.

    Args:
        text: Row content.
        ranges: Ordered, non-overlapping character ranges to emphasize.
        emphasis: Style applied to the emphasized ranges.

    Returns:
        Parts ready to pass to `Content.assemble`.
    """
    if not ranges:
        return [text]
    parts: list[_ContentPart] = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            parts.append(text[cursor:start])
        parts.append((text[start:end], emphasis))
        cursor = end
    if cursor < len(text):
        parts.append(text[cursor:])
    return parts


def _emphasis_by_row(rows: list[_Row]) -> dict[int, list[_Range]]:
    """Map row index to the character ranges that changed within that row.

    Rows are paired only when a run of removed rows is followed by an equally
    long run of added rows — the "these lines were modified" case, where lining
    the two up one-to-one is unambiguous. Lopsided rewrites keep the plain
    whole-row tint.

    Args:
        rows: Parsed diff rows.

    Returns:
        Ranges to emphasize, keyed by row index. Rows without a related
        counterpart are absent.
    """
    ranges: dict[int, list[_Range]] = {}
    index = 0
    while index < len(rows):
        removed = _run_length(rows, index, "removed")
        added = _run_length(rows, index + removed, "added")
        if removed and removed == added:
            for offset in range(removed):
                old_index = index + offset
                new_index = index + removed + offset
                old, new = _emphasis_ranges(rows[old_index].text, rows[new_index].text)
                if old:
                    ranges[old_index] = old
                if new:
                    ranges[new_index] = new
        index += max(1, removed + added)
    return ranges


def _run_length(rows: list[_Row], start: int, kind: str) -> int:
    """Count consecutive rows of `kind` starting at `start`.

    Args:
        rows: Parsed diff rows.
        start: Index to start counting from.
        kind: Row kind to match.

    Returns:
        Length of the run (0 when `rows[start]` is a different kind).
    """
    length = 0
    for row in rows[start:]:
        if row.kind != kind:
            break
        length += 1
    return length


def _emphasis_ranges(old: str, new: str) -> tuple[list[_Range], list[_Range]]:
    """Find the changed character ranges within a related `-`/`+` pair.

    Lines that share too little content are reported as fully changed (no
    ranges) so an unrelated rewrite isn't peppered with highlights.

    Args:
        old: Removed line content.
        new: Added line content.

    Returns:
        Tuple of (ranges in `old`, ranges in `new`).
    """
    if not old or not new or max(len(old), len(new)) > _MAX_EMPHASIS_LEN:
        return [], []
    old_tokens = _TOKEN_RE.findall(old)
    new_tokens = _TOKEN_RE.findall(new)
    matcher = SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    if matcher.quick_ratio() < _SIMILARITY_FLOOR:
        return [], []
    old_offsets = _token_offsets(old_tokens)
    new_offsets = _token_offsets(new_tokens)
    old_ranges: list[_Range] = []
    new_ranges: list[_Range] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            old_ranges.append((old_offsets[i1], old_offsets[i2]))
        if j2 > j1:
            new_ranges.append((new_offsets[j1], new_offsets[j2]))
    if _spans_whole_line(old_ranges, old) and _spans_whole_line(new_ranges, new):
        return [], []
    return old_ranges, new_ranges


def _token_offsets(tokens: list[str]) -> list[int]:
    """Return the character offset of each token, plus the end offset.

    Args:
        tokens: Tokens covering a line in order.

    Returns:
        `len(tokens) + 1` cumulative offsets.
    """
    offsets = [0]
    for token in tokens:
        offsets.append(offsets[-1] + len(token))
    return offsets


def _spans_whole_line(ranges: list[_Range], text: str) -> bool:
    """Report whether `ranges` cover all of `text`.

    Args:
        ranges: Ordered, non-overlapping ranges.
        text: The line the ranges refer to.

    Returns:
        `True` when emphasis would add nothing over the row tint.
    """
    return sum(end - start for start, end in ranges) >= len(text)


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
        self._stats = count_diff_changes(diff)

    def on_mount(self) -> None:
        """Set border style based on charset mode."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.primary)

    def compose(self) -> ComposeResult:
        """Compose the diff widget layout.

        Yields:
            Widgets for title, formatted diff content, and stats.
        """
        colors = theme.get_theme_colors(self)
        glyphs = get_glyphs()
        h = glyphs.box_double_horizontal
        yield Static(
            Content.styled(
                f"{h}{h}{h} {self._title} {h}{h}{h}", f"bold {colors.primary}"
            ),
            classes="diff-title",
        )

        yield from compose_diff_lines(self._diff, self._max_lines, show_stats=False)

        additions, deletions = self._stats
        if additions or deletions:
            yield Static(diff_stats_content(additions, deletions), classes="diff-stats")
