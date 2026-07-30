"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

from textual.containers import Vertical
from textual.content import Content
from textual.highlight import highlight
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode

if TYPE_CHECKING:
    from textual.app import ComposeResult

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)")
"""Matches a unified-diff hunk header, capturing the old and new start lines."""

_TOKEN_RE = re.compile(r"\w+|\s+|.")
"""Splits a line into words, whitespace runs, and single other characters."""

_SIMILARITY_FLOOR = 0.4
"""Below this token similarity a `-`/`+` pair is treated as an unrelated rewrite."""

_MAX_EMPHASIS_LEN = 400
"""Longer lines skip word-level emphasis so the token matcher stays cheap."""

_MAX_HIGHLIGHT_CHARS = 400_000
"""Files larger than this render unhighlighted rather than stall the lexer."""

_GUTTERS = {
    "added": "$text-success 80% on $success 20%",
    "removed": "$text-error 80% on $error 20%",
    "context": "$foreground 30% on $foreground 3%",
}
"""Line-number gutter styles, tinted a shade stronger than the row itself."""

_MARKERS = {"added": ("+", "$text-success"), "removed": ("-", "$text-error")}
"""Marker glyph and color for changed rows."""

_EMPHASIS = {"added": "on $success 30%", "removed": "on $error 30%"}
"""Tint for changed words, composited over the row's own tint."""

_ContentPart = str | tuple[str, str] | Content
_Range = tuple[int, int]


class _Row(NamedTuple):
    """A diff row: `kind` is context, added, removed, separator, or note."""

    kind: str
    text: str
    number: int


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
    path: str = "",
    before: str = "",
    after: str = "",
) -> ComposeResult:
    """Yield per-line Static widgets for a unified diff.

    Rows render as a tinted line number, a colored `-`/`+` marker, and the
    syntax-highlighted line content. Added/removed lines get a CSS class
    (`.diff-line-added`, `.diff-line-removed`) so background colors are driven
    by CSS variables and update automatically on theme change. Within a related
    `-`/`+` pair, only the words that actually changed carry a stronger tint.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff lines to show (None for unlimited).
        path: Path of the diffed file, used to pick a syntax highlighter.
        before: Full file content the diff starts from, for syntax highlighting.
        after: Full file content the diff arrives at, for syntax highlighting.

    Yields:
        Static widgets — one per diff line — with appropriate CSS classes.
    """
    if not diff:
        yield Static(Content.styled("No changes detected", "dim"))
    else:
        yield from _compose_diff_content(diff, max_lines, path, before, after)


def _compose_diff_content(
    diff: str,
    max_lines: int | None,
    path: str,
    before: str,
    after: str,
) -> ComposeResult:
    """Yield styled diff line widgets for non-empty diff content.

    Args:
        diff: Non-empty unified diff string.
        max_lines: Maximum number of diff lines to show (None for unlimited).
        path: Path of the diffed file, used to pick a syntax highlighter.
        before: Full file content the diff starts from, for syntax highlighting.
        after: Full file content the diff arrives at, for syntax highlighting.

    Yields:
        Static widgets for individual diff lines.
    """
    glyphs = get_glyphs()
    rows = _parse_rows(diff.splitlines())
    hidden = 0 if max_lines is None else max(0, len(rows) - max_lines)
    rows = rows[: len(rows) - hidden]
    emphasis = _emphasis_by_row(rows)
    highlighted = _highlighted_rows(rows, path, before, after)
    width = max(2, len(str(max((row.number for row in rows), default=0))))

    for index, row in enumerate(rows):
        if row.kind == "separator":
            yield Static(
                Content.styled(glyphs.hunk_break, "bold $text-primary"),
                classes="diff-hunk-break",
            )
            continue
        if row.kind == "note":
            yield Static(Content.from_markup("[dim]$text[/dim]", text=row.text))
            continue
        number = (f"{row.number:>{width}}", _GUTTERS[row.kind])
        body = highlighted.get(index) or Content(row.text)
        marker, marker_style = _MARKERS.get(row.kind, (" ", ""))
        for start, end in emphasis.get(index, []):
            body = body.stylize(_EMPHASIS[row.kind], start, end)
        yield Static(
            Content.assemble(number, " ", (marker, marker_style), " ", body),
            classes=f"diff-line-{row.kind}" if row.kind != "context" else "",
        )
    if hidden:
        yield Static(Content.styled(f"\n... ({hidden} more lines)", "dim"))


def _highlighted_rows(
    rows: list[_Row], path: str, before: str, after: str
) -> dict[int, Content]:
    """Syntax-highlight the diff body, keyed by row index.

    Each row is highlighted as a line of its whole file rather than as a line of
    the hunk: a hunk is an arbitrary fragment, so a lexer reading one alone
    mistakes the closing `\"\"\"` of a docstring for an opening one and paints
    every line after it as a string. Only the text up to the last line the diff
    references is lexed, so cost tracks the hunks' position, not the file size.

    Args:
        rows: Parsed diff rows.
        path: Path of the diffed file, used to pick a syntax highlighter.
        before: Full file content the diff starts from.
        after: Full file content the diff arrives at.

    Returns:
        Highlighted content per row index. Rows whose highlighted text does not
        match the diff exactly are left out, which also covers file content that
        has moved on since the diff was computed.
    """
    if not path:
        return {}
    highlighted: dict[int, Content] = {}
    # Context rows are numbered against the old file, and are identical in both.
    for kinds, code in ((("removed", "context"), before), (("added",), after)):
        wanted = {row.number: i for i, row in enumerate(rows) if row.kind in kinds}
        if not wanted or not code:
            continue
        head = "\n".join(code.splitlines()[: max(wanted)])
        if len(head) > _MAX_HIGHLIGHT_CHARS:
            continue
        # `tab_size=0` leaves tabs alone; expanding them would shift the offsets.
        lines = highlight(head, path=path, tab_size=0).split("\n")
        for number, index in wanted.items():
            line = lines[number - 1] if 0 < number <= len(lines) else None
            if line is not None and line.plain == rows[index].text:
                highlighted[index] = line
    return highlighted


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
    runs: list[tuple[str, int, int]] = []
    start = 0
    for kind, group in groupby(row.kind for row in rows):
        size = len(list(group))
        runs.append((kind, start, size))
        start += size

    ranges: dict[int, list[_Range]] = {}
    for (kind, old_start, size), (next_kind, new_start, next_size) in pairwise(runs):
        if kind != "removed" or next_kind != "added" or size != next_size:
            continue
        for offset in range(size):
            old_index = old_start + offset
            new_index = new_start + offset
            old, new = _emphasis_ranges(rows[old_index].text, rows[new_index].text)
            if old:
                ranges[old_index] = old
            if new:
                ranges[new_index] = new
    return ranges


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
    old_offsets = [0, *accumulate(len(token) for token in old_tokens)]
    new_offsets = [0, *accumulate(len(token) for token in new_tokens)]
    old_ranges: list[_Range] = []
    new_ranges: list[_Range] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            old_ranges.append((old_offsets[i1], old_offsets[i2]))
        if j2 > j1:
            new_ranges.append((new_offsets[j1], new_offsets[j2]))
    old_covered = sum(end - start for start, end in old_ranges) >= len(old)
    new_covered = sum(end - start for start, end in new_ranges) >= len(new)
    # Emphasis adds nothing over the row tint when it spans both whole lines.
    if old_covered and new_covered:
        return [], []
    return old_ranges, new_ranges


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

        yield from compose_diff_lines(self._diff, self._max_lines)

        additions, deletions = self._stats
        if additions or deletions:
            yield Static(diff_stats_content(additions, deletions), classes="diff-stats")
