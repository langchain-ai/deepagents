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

from deepagents_code.config import get_glyphs
from deepagents_code.diff_utils import file_header_indexes

if TYPE_CHECKING:
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)")
_TOKEN_RE = re.compile(r"\w+|\s+|.")

_SIMILARITY_FLOOR = 0.4
_MAX_EMPHASIS_LEN = 400
_MAX_HIGHLIGHT_CHARS = 400_000

_Range = tuple[int, int]
_DiffRowKind = Literal["context", "added", "removed"]
_ChangedRowKind = Literal["added", "removed"]
_RowKind = Literal["context", "added", "removed", "separator", "truncated", "note"]

_GUTTERS: dict[_DiffRowKind, str] = {
    "added": "$text-success 80% on $success 20%",
    "removed": "$text-error 80% on $error 20%",
    "context": "$foreground 30% on $foreground 3%",
}

_MARKERS: dict[_ChangedRowKind, tuple[str, str]] = {
    "added": ("+", "$text-success"),
    "removed": ("-", "$text-error"),
}

_EMPHASIS: dict[_ChangedRowKind, str] = {
    "added": "on $success 30%",
    "removed": "on $error 30%",
}


class _Row(NamedTuple):
    """One rendered line of a diff."""

    kind: _RowKind
    text: str
    number: int


def compose_diff_lines(
    diff: str,
    max_lines: int | None = 100,
    *,
    path: str = "",
    before: str = "",
    after: str = "",
) -> ComposeResult:
    """Yield syntax-highlighted widgets for a unified diff.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff lines to show (None for unlimited).
        path: Path of the diffed file, used to pick a syntax highlighter.
        before: Full file content the diff starts from, for syntax highlighting.
        after: Full file content the diff arrives at, for syntax highlighting.

    Yields:
        One `Static` per rendered row.
    """
    if not diff:
        yield Static(Content.styled("No changes detected", "dim"))
    else:
        yield from _compose_diff_content(diff, max_lines, path, before, after)


def highlight_source_prefixes(diff: str, before: str, after: str) -> tuple[str, str]:
    """Keep the bounded source prefixes needed to highlight a diff.

    Args:
        diff: Unified diff string.
        before: Full content before the change.
        after: Full content after the change.

    Returns:
        Before and after prefixes, with oversized sides omitted.
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
    """Yield styled widgets for a non-empty diff."""
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
    """Return diff rows highlighted with whole-file lexer state."""
    if not path:
        return {}
    highlighted: dict[int, Content] = {}
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
    """Return highlighted source lines, or `None` if lexing fails."""
    try:
        return tuple(highlight(code, path=path, tab_size=0).split("\n"))
    except Exception:
        logger.debug("Syntax highlighting failed for %s", path, exc_info=True)
        return None


def _parse_rows(lines: list[str]) -> list[_Row]:
    """Return renderable rows parsed from unified-diff lines."""
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
    """Return changed ranges for equal-length removed/added runs."""
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
    """Return whether two lines are similar enough for word emphasis."""
    old_words = [token for token in old_tokens if token.strip()]
    new_words = [token for token in new_tokens if token.strip()]
    if not old_words or not new_words:
        return False
    matcher = SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    return (
        matcher.quick_ratio() >= _SIMILARITY_FLOOR
        and matcher.ratio() >= _SIMILARITY_FLOOR
    )


def _emphasis_ranges(old: str, new: str) -> tuple[list[_Range], list[_Range]]:
    """Return changed ranges within a related removed/added pair."""
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
    if old_covered and new_covered:
        return [], []
    return old_ranges, new_ranges
