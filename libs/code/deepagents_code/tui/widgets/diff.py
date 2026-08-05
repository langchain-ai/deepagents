"""Renderers turning a unified diff into one `Static` per row.

Rows carry a line-number gutter, a `+`/`-` marker, syntax highlighting lifted
from whole-file lexer state, and word-level emphasis on the spans that actually
changed between a paired removed/added line.
"""

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
from deepagents_code.diff_utils import (
    DIFF_TRUNCATION_MARKER,
    HUNK_RE,
    DiffStats,
    file_header_indexes,
    split_diff_lines,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+|\s+|.")
"""Splits a line into word / whitespace / single-character tokens.

Total over any string, so `"".join(findall(s)) == s` and token offsets index
back into the original line.
"""

_SIMILARITY_FLOOR = 0.4
"""Minimum word-level similarity before a removed/added pair gets emphasis.

Below this the two lines are treated as unrelated rewrites, where emphasising
"changed" spans would just tint the whole line and add noise.
"""

_MAX_EMPHASIS_LEN = 400
"""Longest line eligible for word emphasis.

`SequenceMatcher` over per-character tokens is quadratic, so minified JS or
single-line JSON would stall the compose path. Longer lines render unemphasised.
"""

_MAX_HIGHLIGHT_CHARS = 100_000
"""Largest source prefix worth lexing for syntax highlighting.

Above this the side is skipped and its rows render as plain text.

The prefix has to start at line 1 — the lexer needs the preceding source to know
whether the changed lines sit inside a string or comment — so its size is set by
how far into the file the edit is, not by how much of it is rendered. That makes
this constant the bound on two separate costs.

Lexing runs synchronously in `compose`, measured at roughly 0.72 ms per 1,000
characters (~70 ms at this limit, per side, plus a one-off ~150 ms to build the
lexer on the session's first diff). Both sides of a first-time diff are lexed
back to back, so the limit buys a worst case of about 150 ms of blocked message
pump rather than 70.

The prefix is also retained per message by `MessageData` so a rehydrated diff can
re-highlight, at up to this many characters per side. Raising it slows the diff
mount and grows the transcript's memory in step, and the transcript has no cap.
"""

_Range = tuple[int, int]
_DiffRowKind = Literal["context", "added", "removed"]
_RowKind = Literal["context", "added", "removed", "separator", "truncated", "note"]

# A changed row's color builds up in three tiers of the same hue, each darker
# than the last: the row background from `.diff-line-added`/`.diff-line-removed`
# in `app.tcss` (10%), the gutter below (20%), and the words that actually
# changed (30%, applied per-span in `_compose_diff_content`). Keep them ordered
# that way — equal tiers flatten the row and lose the distinction.
#
# All three maps are keyed over a total row kind so a new one fails type-checking
# here rather than silently rendering with no marker or emphasis.
_GUTTERS: dict[_DiffRowKind, str] = {
    "added": "$text-success 80% on $success 20%",
    "removed": "$text-error 80% on $error 20%",
    "context": "$foreground 30% on $foreground 3%",
}

_MARKERS: dict[_DiffRowKind, tuple[str, str]] = {
    "added": ("+", "$text-success"),
    "removed": ("-", "$text-error"),
    "context": (" ", ""),
}

_EMPHASIS: dict[_DiffRowKind, str] = {
    "added": "on $success 30%",
    "removed": "on $error 30%",
    "context": "",
}


class _Row(NamedTuple):
    """One rendered line of a diff.

    Attributes:
        kind: What the row represents. `context`/`added`/`removed` are numbered
            source lines; `separator`/`truncated`/`note` are decorations.
        text: The line with its diff marker stripped. Empty for `separator` and
            `truncated`.
        number: For `added`, the line number in the *new* file; for `context`
            and `removed`, the line number in the *old* file. `0` for
            decoration rows, where it carries no meaning.
    """

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
    show_numbers: bool = True,
) -> ComposeResult:
    """Yield syntax-highlighted widgets for a unified diff.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of *rendered rows* to show (None for
            unlimited). Rows are not diff lines: file and hunk headers are
            dropped and hunk separators added, so this does not correspond to a
            line count in `diff`. Rows are dropped from the end, which can split
            a removed/added run and drop word emphasis on the surviving half.
        path: Path of the diffed file, used to pick a syntax highlighter.
        before: Source aligned to the diff's *old* line numbers. May be a
            truncated prefix or empty; rows whose text does not match the
            lexed source are left unhighlighted (logged at debug).
        after: Source aligned to the diff's *new* line numbers, same contract.
        show_numbers: Whether to render the line-number gutter. Pass `False`
            when the diff's line numbers are not the file's — e.g. a diff of
            edit fragments, whose hunks always start at 1.

    Yields:
        One `Static` per rendered row, plus a trailing count when rows were
        dropped to fit `max_lines`. An empty `diff` yields a single "no changes"
        row; both callers already distinguish that case in their own headers, so
        this is a defensive fallback rather than the live path.
    """
    if not diff:
        yield Static(Content.styled("No changes detected", "dim"))
    else:
        yield from _compose_diff_content(
            diff, max_lines, path, before, after, show_numbers=show_numbers
        )


def format_diff_stats(stats: DiffStats) -> Content:
    """Format addition/deletion counts as styled `+N -M` content.

    Takes the pair as a `DiffStats` rather than two ints so the counts cannot be
    transposed on the way to the places the user reads them — the `DiffMessage`
    header and the approval prompt's `File:` header.

    Args:
        stats: Line counts for the change.

    Returns:
        Styled content, empty when both counts are zero.
    """
    colors = theme.get_theme_colors()
    parts: list[str | tuple[str, str] | Content] = []
    if stats.additions:
        parts.append((f"+{stats.additions}", colors.success))
    if stats.deletions:
        if parts:
            parts.append(" ")
        parts.append((f"-{stats.deletions}", colors.error))
    return Content.assemble(*parts) if parts else Content("")


def highlight_source_prefixes(diff: str, before: str, after: str) -> tuple[str, str]:
    """Keep the bounded source prefixes needed to highlight a diff.

    Idempotent, and rehydration depends on it: `DiffMessage.__init__` calls this
    on whatever it is handed, which is the full file from the live path but an
    already-trimmed prefix from `MessageData`. Re-trimming a prefix must return it
    unchanged, so any future trimming rule has to stay keyed on the diff's line
    numbers rather than on a count relative to the input.

    Args:
        diff: Unified diff string.
        before: Content before the change — the whole file, or a prefix this
            function previously returned.
        after: Content after the change, same contract.

    Returns:
        Before and after prefixes, with oversized sides omitted.
    """
    rows = _parse_rows(split_diff_lines(diff))
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
    # Never split more than the limit itself. An edit near the end of a large
    # file asks for a prefix that is going to be rejected anyway, and splitting
    # the whole source first would allocate a near-full copy of the file per
    # side, per compose, only to throw it away. One char past the limit is
    # enough to tell "fits" from "does not".
    oversized = len(source) > _MAX_HIGHLIGHT_CHARS
    lines = (source[: _MAX_HIGHLIGHT_CHARS + 1] if oversized else source).splitlines()
    # With a truncated head, `line` is only reached within the limit when a
    # further line follows it — otherwise the last entry is a partial line and
    # the real prefix runs past the limit.
    if oversized and len(lines) <= line:
        return ""
    prefix = "\n".join(lines[:line])
    return prefix if len(prefix) <= _MAX_HIGHLIGHT_CHARS else ""


def _compose_diff_content(
    diff: str,
    max_lines: int | None,
    path: str,
    before: str,
    after: str,
    *,
    show_numbers: bool = True,
) -> ComposeResult:
    """Yield styled widgets for a non-empty diff."""
    glyphs = get_glyphs()
    rows = _parse_rows(split_diff_lines(diff))
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
        body = highlighted.get(index) or Content(row.text)
        marker, marker_style = _MARKERS[row.kind]
        if emphasis_style := _EMPHASIS[row.kind]:
            for start, end in emphasis.get(index, []):
                body = body.stylize(emphasis_style, start, end)
        parts: list[Content | str | tuple[str, str]] = []
        if show_numbers:
            parts += [(f"{row.number:>{width}}", _GUTTERS[row.kind]), " "]
        parts += [(marker, marker_style), " ", body]
        yield Static(
            Content.assemble(*parts),
            classes=f"diff-line-{row.kind}" if row.kind != "context" else "",
        )
    if hidden:
        yield Static(Content.styled(f"\n... ({hidden} more lines)", "dim"))


def _highlighted_rows(
    rows: list[_Row], path: str, before: str, after: str
) -> dict[int, Content]:
    """Return a `{row index: highlighted content}` map.

    Each side is lexed from the start of the supplied source through the last
    referenced line so multi-line constructs (docstrings, block comments) resolve
    correctly rather than reopening at the hunk boundary. That holds only as far
    as the caller's source really is file-aligned: the approval path passes edit
    fragments, where the lexer does start mid-file and the drift check below
    mostly rejects the result. Rows outside the prefix, or whose text has drifted
    from the source, are omitted and render as plain text.

    Assumes a single-file diff, as `before`/`after` are one file's contents: rows
    are matched to source by line number, which restarts per file in a multi-file
    diff and would collide.
    """
    if not path:
        return {}
    highlighted: dict[int, Content] = {}
    for kinds, code in ((("removed", "context"), before), (("added",), after)):
        wanted = {row.number: i for i, row in enumerate(rows) if row.kind in kinds}
        if not wanted or not code:
            continue
        head = _highlight_source_prefix(code, max(wanted))
        if not head:
            continue
        lines = _highlight_lines(head, path)
        if lines is None:
            continue
        drifted = 0
        for number, index in wanted.items():
            line = lines[number - 1] if 0 < number <= len(lines) else None
            if line is None:
                continue
            if line.plain != rows[index].text:
                drifted += 1
                continue
            highlighted[index] = line
        if drifted:
            # The source no longer matches the diff it came with — a stale
            # rehydration, or `before`/`after` belonging to another file.
            # Rendering plain is right, but it also hides a real misalignment,
            # so leave a trace. Once per side at warning rather than per row at
            # debug: a whole drifted side reports thousands of rows, and debug
            # sits below both the default level and the in-app console's ring
            # buffer, so the trace was invisible where it mattered.
            logger.warning(
                "Highlight source drifted from diff at %s (%s): %d of %d rows",
                path,
                "/".join(kinds),
                drifted,
                len(wanted),
            )
    return highlighted


@lru_cache(maxsize=4)
def _highlight_lines(code: str, path: str) -> tuple[Content, ...] | None:
    """Return highlighted source lines, or `None` if lexing fails.

    Cached because scrolling rebuilds a `DiffMessage` from `MessageData` on every
    pass, and each mount would otherwise re-lex both sides. Two entries per diff,
    so this holds the last two — the scrolling case it exists for.

    Sized small on purpose. `_MAX_HIGHLIGHT_CHARS` bounds the *input*, not what is
    retained: an entry is one `Content` per line, each carrying a span list, and
    measures several times its source. Nothing clears this cache, so its cost is
    held for the process lifetime — size it against measured entries, not against
    the character limit.

    Failures are cached too: a file whose lexer cannot parse it will not parse on
    the next scroll either, and retrying would pay the cost to fail again.
    """
    try:
        return tuple(highlight(code, path=path, tab_size=0).split("\n"))
    except (ValueError, LookupError) as e:
        # No usable lexer for this path — expected for unknown extensions.
        logger.debug("No usable lexer for %s: %s", path, e)
        return None
    except Exception:
        # Anything else is a bug here or a Textual API change, not a missing
        # lexer. Highlighting is cosmetic, so still degrade to plain text, but
        # say so at a level that will actually be seen.
        logger.warning(
            "Syntax highlighting failed unexpectedly for %s", path, exc_info=True
        )
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
        if match := HUNK_RE.match(line):
            old, new = int(match.group(1)), int(match.group(3))
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
        elif line.strip() == DIFF_TRUNCATION_MARKER:
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
    # `quick_ratio` is a cheap upper bound on `ratio`, so a failure there rules
    # the pair out without running the full match.
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
    # No total-coverage bail-out is needed: `_is_related` has already found
    # shared word tokens, so `get_opcodes` always yields at least one `equal`
    # block and the ranges can never span the whole line on both sides.
    return old_ranges, new_ranges
