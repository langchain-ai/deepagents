"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, Literal, NamedTuple

from textual.content import Content
from textual.highlight import highlight
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import get_glyphs
from deepagents_code.diff_utils import file_header_indexes

if TYPE_CHECKING:
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)")
"""Matches a unified-diff hunk header, capturing the old and new start lines."""

_TOKEN_RE = re.compile(r"\w+|\s+|.")
"""Splits a line into words, whitespace runs, and single other characters."""

_SIMILARITY_FLOOR = 0.4
"""Below this token similarity a `-`/`+` pair is treated as an unrelated rewrite.

Measured over non-whitespace tokens only. Counting whitespace would let the
indentation and spacing that almost any two lines of the same file share carry
the score over the floor, which defeats the point of having one.
"""

_MAX_EMPHASIS_LEN = 400
"""Longer lines skip word-level emphasis so the token matcher stays cheap."""

_MAX_HIGHLIGHT_CHARS = 400_000
"""Skip highlighting a side whose lexed prefix exceeds this, rather than stall.

Measured on the prefix actually lexed — up to the last line the diff references
— not on the file, so a hunk near the top of a huge file still highlights. The
two sides are checked independently, so one may highlight while the other does
not.
"""

_ContentPart = str | tuple[str, str] | Content
_Range = tuple[int, int]
_DiffRowKind = Literal["context", "added", "removed"]
_ChangedRowKind = Literal["added", "removed"]
_RowKind = Literal["context", "added", "removed", "separator", "truncated", "note"]

_GUTTERS: dict[_DiffRowKind, str] = {
    "added": "$text-success 80% on $success 20%",
    "removed": "$text-error 80% on $error 20%",
    "context": "$foreground 30% on $foreground 3%",
}
"""Line-number gutter styles, tinted a shade stronger than the row itself."""

_MARKERS: dict[_ChangedRowKind, tuple[str, str]] = {
    "added": ("+", "$text-success"),
    "removed": ("-", "$text-error"),
}
"""Marker glyph and color for changed rows."""

_EMPHASIS: dict[_ChangedRowKind, str] = {
    "added": "on $success 30%",
    "removed": "on $error 30%",
}
"""Tint for changed words, composited over the row's own tint."""

"""Closed set of row kinds, so a typo is a type error rather than a `KeyError`.

The style tables are keyed by the narrower row kinds they support, and a new
kind must be handled in `_compose_diff_content` before it reaches them.
"""


class _Row(NamedTuple):
    """One rendered line of a diff."""

    kind: _RowKind
    """Which of the six row shapes this is; drives styling and gutter."""

    text: str
    """Line content without its diff marker; empty for separator/truncated."""

    number: int
    """File line number, or `0` for separator, truncated, and note rows."""


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
        Static widgets, one per rendered row. Not one per input line: file
        headers and the first hunk header produce nothing, and a truncation
        footer may be appended. Only added/removed rows carry a CSS class.
    """
    if not diff:
        yield Static(Content.styled("No changes detected", "dim"))
    else:
        yield from _compose_diff_content(diff, max_lines, path, before, after)


def highlight_source_prefixes(diff: str, before: str, after: str) -> tuple[str, str]:
    """Keep only the bounded source prefixes needed to highlight a diff.

    Args:
        diff: Unified diff string.
        before: Full content before the change.
        after: Full content after the change.

    Returns:
        Before and after prefixes, or an empty side when its prefix exceeds the
        highlighting limit.
    """
    rows = _parse_rows(diff.splitlines())
    before_line = max(
        (row.number for row in rows if row.kind in {"removed", "context"}),
        default=0,
    )
    after_line = max(
        (row.number for row in rows if row.kind == "added"),
        default=0,
    )
    return (
        _highlight_source_prefix(before, before_line),
        _highlight_source_prefix(after, after_line),
    )


def _highlight_source_prefix(source: str, line: int) -> str:
    """Return the highlightable prefix ending at `line`."""
    if not source or line <= 0:
        return ""
    prefix = "\n".join(source.splitlines()[:line])
    return prefix if len(prefix) <= _MAX_HIGHLIGHT_CHARS else ""


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
        if row.kind == "truncated":
            # Distinct from a hunk break: the rest of the diff is missing, not
            # merely skipped. The header's counts still report the full change.
            yield Static(Content.styled("... diff truncated", "dim"))
            continue
        if row.kind == "note":
            yield Static(Content.from_markup("[dim]$text[/dim]", text=row.text))
            continue
        number = (f"{row.number:>{width}}", _GUTTERS[row.kind])
        body = highlighted.get(index) or Content(row.text)
        marker, marker_style = _MARKERS.get(row.kind, (" ", ""))
        if emphasis_style := _EMPHASIS.get(row.kind):
            for start, end in emphasis.get(index, []):
                body = body.stylize(emphasis_style, start, end)
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
        Highlighted content per row index. A row is included only when its
        highlighted text matches the diff exactly, so emphasis offsets computed
        against the row text stay valid. That also drops rows whose file
        content has since moved on, though it cannot reliably detect that —
        a mismatch here is usually a lexer artifact, not a stale diff.
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
        lines = _highlight_lines(head, path)
        if lines is None:
            continue
        for number, index in wanted.items():
            line = lines[number - 1] if 0 < number <= len(lines) else None
            if line is not None and line.plain == rows[index].text:
                highlighted[index] = line
    return highlighted


@lru_cache(maxsize=8)
def _highlight_lines(code: str, path: str) -> tuple[Content, ...] | None:
    """Lex `code` as `path` and return its highlighted lines.

    Cached because a turn that edits one file repeatedly re-lexes the same
    prefix on every mount, and this runs synchronously on the render path. The
    cache is keyed on the content itself, so a stale entry is impossible; it is
    small because the entries are whole file prefixes.

    Args:
        code: File prefix to lex.
        path: Path the content came from, used to pick a lexer.

    Returns:
        One `Content` per line, or `None` if the lexer failed — highlighting is
        decorative, so a bad lexer must degrade to plain text rather than take
        down the `compose()` that is rendering the diff.
    """
    try:
        # `tab_size=0` leaves tabs alone; expanding them would shift the offsets.
        return tuple(highlight(code, path=path, tab_size=0).split("\n"))
    except Exception:
        logger.debug("Syntax highlighting failed for %s", path, exc_info=True)
        return None


def _parse_rows(lines: list[str]) -> list[_Row]:
    """Convert unified-diff lines into renderable rows.

    File headers are dropped and hunk headers become separators (except the
    first) so consecutive hunks read as distinct blocks. An upstream `...`
    truncation marker becomes its own `truncated` row so a diff that was cut
    short cannot be mistaken for one that merely skips between hunks.

    Args:
        lines: Lines of a unified diff.

    Returns:
        Rows in render order. `number` is the file line number for content
        rows and `0` for `separator`, `truncated`, and `note` rows.
    """
    rows: list[_Row] = []
    header_indexes = file_header_indexes(lines)
    old = new = 0
    seen_hunk = False
    for index, line in enumerate(lines):
        if index in header_indexes:
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
            rows.append(_Row("truncated", "", 0))
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


def _is_related(old_tokens: list[str], new_tokens: list[str]) -> bool:
    """Whether two token sequences are alike enough to emphasize word-by-word.

    Scored on non-whitespace tokens only. On short code lines the shared
    indentation, spaces, and punctuation otherwise dominate the score, so an
    unrelated rewrite clears the floor and gets peppered with highlights.

    `quick_ratio` is an upper bound and runs first as a cheap reject; only
    survivors pay for the real `ratio`.

    Args:
        old_tokens: Tokens of the removed line.
        new_tokens: Tokens of the added line.

    Returns:
        True when the pair scores at or above `_SIMILARITY_FLOOR`.
    """
    old_words = [token for token in old_tokens if token.strip()]
    new_words = [token for token in new_tokens if token.strip()]
    if not old_words or not new_words:
        return False
    matcher = SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    if matcher.quick_ratio() < _SIMILARITY_FLOOR:
        return False
    return matcher.ratio() >= _SIMILARITY_FLOOR


def _emphasis_ranges(old: str, new: str) -> tuple[list[_Range], list[_Range]]:
    """Find the changed character ranges within a related `-`/`+` pair.

    Returns no ranges — leaving the row its uniform tint — in four cases:
    either side is blank, either side is longer than `_MAX_EMPHASIS_LEN` (a
    cost bound on the token matcher), or the two are too dissimilar to be a
    modification of one another (`_is_related`), or every character on both
    sides would be emphasized.

    Args:
        old: Removed line content.
        new: Added line content.

    Returns:
        Tuple of (ranges in `old`, ranges in `new`); empty lists mean the row
        keeps its uniform tint.
    """
    if not old or not new or max(len(old), len(new)) > _MAX_EMPHASIS_LEN:
        return [], []
    old_tokens = _TOKEN_RE.findall(old)
    new_tokens = _TOKEN_RE.findall(new)
    if not _is_related(old_tokens, new_tokens):
        return [], []
    matcher = SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
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
