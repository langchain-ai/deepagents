"""Renderers turning a unified diff into one `Static` per row.

Rows carry a line-number gutter, a `+`/`-` marker, syntax highlighting lifted
from whole-file lexer state, and word-level emphasis on the spans that actually
changed between a paired removed/added line.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args

from textual.content import Content
from textual.geometry import Offset
from textual.highlight import highlight
from textual.selection import Selection
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import get_glyphs
from deepagents_code.diff_utils import (
    HUNK_RE,
    DiffStats,
    file_header_indexes,
    is_truncation_marker,
    split_diff_lines,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.widget import Widget

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

`SequenceMatcher` is quadratic in token count, and `_TOKEN_RE` degenerates to
one token per character on punctuation-dense lines — so minified JS or
single-line JSON would stall the compose path. Longer lines render unemphasised.
"""

MAX_HIGHLIGHT_CHARS = 100_000
"""Largest source prefix worth lexing for syntax highlighting.

Above this the side is skipped and its rows render as plain text.

The prefix has to start at line 1 — the lexer needs the preceding source to know
whether the changed lines sit inside a string or comment — so its size is set by
how far into the file the edit is, not by how much of it is rendered. That makes
this constant the bound on two separate costs.

Lexing runs synchronously in `compose`. Measured 2026-07 on an M-series Mac,
CPython 3.12: roughly 0.72 ms per 1,000 characters, so ~70 ms per side at this
limit, plus a one-off ~150 ms to build the lexer on the session's first diff.
Both sides are lexed back to back, so the steady-state worst case is ~140 ms of
blocked message pump and the session's first diff is ~290 ms. Re-measure before
trusting these; they set the limit but nothing enforces them.

The prefix is also retained per message by `MessageData` so a rehydrated diff can
re-highlight, at up to this many characters per side. Raising it slows the diff
mount and grows the transcript's memory in step, and the transcript has no cap.
"""

_Range = tuple[int, int]
_DiffRowKind = Literal["context", "added", "removed"]
_RowKind = Literal["context", "added", "removed", "separator", "truncated", "note"]


@dataclass(frozen=True, kw_only=True)
class _RowStyle:
    """How one kind of source row renders.

    Keyword-only because all four fields are `str`: positional construction
    lets `marker_style` and `emphasis` swap into a wrong-but-plausible render
    that nothing would flag. Same hazard `DiffStats` is `kw_only` to rule out,
    and the reason this is a dataclass rather than a `NamedTuple` — it is only
    ever read by attribute, so nothing needs the tuple behavior.

    Attributes:
        gutter: Style for the line-number column.
        marker: The `+`/`-`/space shown between gutter and text.
        marker_style: Style for `marker`.
        emphasis: Style laid over the spans that changed, empty when the kind
            takes no emphasis.
    """

    gutter: str
    marker: str
    marker_style: str
    emphasis: str


# A changed row's color builds up in three tiers of the same hue, each darker
# than the last: the row background from `.diff-line-added`/`.diff-line-removed`
# in `app.tcss` (10%), the gutter (20%), and the words that actually changed
# (30%, applied per-span in `_compose_diff_content`). Keep them ordered that way
# — equal tiers flatten the row and lose the distinction.
#
# Keyed over `_DiffRowKind`. Decoration kinds live only in the wider `_RowKind`
# and are handled by the early `continue`s in `_compose_diff_content`, which is
# what narrows `row.kind` to a valid key. One map rather than three so a kind
# cannot be added to some of them and missed in the rest.
_ROW_STYLES: dict[_DiffRowKind, _RowStyle] = {
    "added": _RowStyle(
        gutter="$text-success 80% on $success 20%",
        marker="+",
        marker_style="$text-success",
        emphasis="on $success 30%",
    ),
    "removed": _RowStyle(
        gutter="$text-error 80% on $error 20%",
        marker="-",
        marker_style="$text-error",
        emphasis="on $error 30%",
    ),
    "context": _RowStyle(
        gutter="$foreground 30% on $foreground 3%",
        marker=" ",
        marker_style="",
        emphasis="",
    ),
}

# A `dict` literal is not checked for exhaustiveness — neither mypy nor ty flags
# a missing key, and `_ROW_STYLES[row.kind]` type-checks fine while raising
# `KeyError` at render time. Check it at import so a new numbered row kind fails
# loudly on startup instead of on whichever diff first happens to contain one.
# A raise rather than an `assert`, which `-O` strips and ruff bans.
if _missing_row_styles := set(get_args(_DiffRowKind)) - _ROW_STYLES.keys():
    _msg = f"_ROW_STYLES is missing entries for {sorted(_missing_row_styles)}"
    raise RuntimeError(_msg)


# Which row kinds are read from which side's source, keyed by `_Row.number`.
#
# Follows the numbering in `_Row.number`: only removed rows are numbered in the
# old file. Named once because `highlight_source_prefixes` sizes each prefix and
# `_highlighted_rows` reads from it, and a row kind sized against one side but
# read from the other loses its highlighting with nothing to explain why.
#
# A comment rather than a docstring under `_AFTER_KINDS`, which would attribute
# the shared rule to that name alone and leave `_BEFORE_KINDS` reading as
# undocumented.
_BEFORE_KINDS: tuple[_DiffRowKind, ...] = ("removed",)
_AFTER_KINDS: tuple[_DiffRowKind, ...] = ("added", "context")


class _Row(NamedTuple):
    """One rendered line of a diff.

    Attributes:
        kind: What the row represents. `context`/`added`/`removed` are numbered
            source lines; `separator`/`truncated`/`note` are decorations.
        text: The line with its diff marker stripped. Empty for `separator` and
            `truncated`.
        number: Line number in the file the user can still open — the *new* file
            for `added` and `context`, the *old* file for `removed`, which is
            the only kind that no longer exists in the new one. Numbering
            context from the old file instead would make every row after an
            insertion disagree with the file on disk, and repeat numbers the
            added rows had just used. `None` for decoration rows, which have no
            line to name.
    """

    kind: _RowKind
    text: str
    number: int | None


class _DiffRowStatic(Static):
    """A numbered diff row whose gutter is excluded from text selections.

    The gutter (line number, `+`/`-` marker, and the spaces around them) is
    decorative: a copy taken from a diff should hold the source text, so a
    paste into an editor does not need the numbers stripped back out. The
    exclusion is applied to the stored `Selection` itself — see
    `clamp_selection` — because Textual paints the selection highlight from
    that same geometry, and a `get_selection` override would leave the gutter
    visually selected while absent from the copy.
    """

    def __init__(self, content: Content, prefix_len: int, **kwargs: Any) -> None:
        """Initialize the row.

        Args:
            content: The row's full content, gutter included.
            prefix_len: Cell width of the leading gutter (number, marker, and
                their separating spaces).
            **kwargs: Forwarded to `Static`.
        """
        super().__init__(content, **kwargs)
        self.selection_prefix = prefix_len

    selection_prefix: int
    """Cells at the row's left edge a selection must not cover."""


def clamp_selection(widget: Widget, selection: Selection) -> Selection | None:
    """Return `selection` shifted past a diff row's gutter, if one is set.

    Every form a selection can take over a single-line row covers the gutter
    unless its start is moved past it:

    - `Selection(None, None)` — the row sits mid-selection. Textual extracts
      the row's full text, so the start must move to the gutter's end even
      though no endpoint lands here.
    - `Selection(None, end)` — entered from above; same move, plus an `end`
      still inside the gutter means nothing selectable is covered, reported
      as `None` so the row drops out of the screen's selection map.
    - `Selection(start, None)` / `Selection(start, end)` — pull any endpoint
      inside the gutter forward to its end; a range that then collapses
      (wholly gutter) is `None`.

    An endpoint counts as "inside the gutter" only on the row's first visual
    line (`y == 0`). A row wrapped by Textual continues at column 0 of each
    following visual line, where the gutter no longer exists — an `x` there
    already indexes source text, so a continuation coordinate must pass through
    untouched or a drag starting on a continuation would skip its first
    gutter-width characters, and one ending within them would drop source the
    user visibly selected.

    Args:
        widget: The row the selection applies to. Anything that is not a
            `_DiffRowStatic` is returned unchanged.
        selection: The geometry Textual computed for this row.

    Returns:
        The clamped selection, the original selection, or `None` when the
        covered range lies entirely in the gutter and the row should drop out
        of the screen's selection map.
    """
    if not isinstance(widget, _DiffRowStatic):
        return selection
    prefix = widget.selection_prefix
    start, end = selection.start, selection.end
    if start is None:
        if end is not None and end.y == 0 and end.x <= prefix:
            return None
        start = Offset(prefix, 0)
    elif start.y == 0 and start.x < prefix:
        start = Offset(prefix, start.y)
    if end is not None and end.y == 0 and end.x <= prefix:
        end = Offset(prefix, end.y)
    if end is not None and end.transpose <= start.transpose:
        return None
    return Selection(start, end)


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
            line count in `diff`. Rows are dropped from the end; emphasis is
            computed after the drop, so splitting a removed/added run costs the
            whole run its word emphasis, not just the clipped half.
        path: Path of the diffed file, used to pick a syntax highlighter.
        before: Source aligned to the diff's *old* line numbers. May be a
            truncated prefix or empty; rows whose text does not match the lexed
            source are left unhighlighted (logged once per side at warning).
        after: Source aligned to the diff's *new* line numbers, same contract.
            Context rows are read from here, not from `before` — see
            `_Row.number`.
        show_numbers: Whether to render the line-number gutter. Pass `False`
            when the diff's line numbers are not the file's — e.g. a diff of
            edit fragments, whose hunks always start at 1.

    Yields:
        One `Static` per rendered row, plus a trailing count when rows were
        dropped to fit `max_lines`. An empty `diff` yields a single "no changes"
        row; both callers already handle that case in their own output, so this
        is a defensive fallback rather than the live path.
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
    numbers rather than on a count relative to the input, and has to survive the
    split/join round trip — see the trailing-newline case in
    `_highlight_source_prefix`. `test_trimming_a_prefix_again_returns_it_unchanged`
    pins this.

    Args:
        diff: Unified diff string.
        before: Content before the change — the whole file, or a prefix this
            function previously returned.
        after: Content after the change, same contract.

    Returns:
        Before and after prefixes, with oversized sides omitted.
    """
    rows = _parse_rows(split_diff_lines(diff))
    before_line = _max_number(rows, _BEFORE_KINDS)
    after_line = _max_number(rows, _AFTER_KINDS)
    return (
        _highlight_source_prefix(before, before_line),
        _highlight_source_prefix(after, after_line),
    )


def _max_number(rows: list[_Row], kinds: tuple[_DiffRowKind, ...]) -> int:
    """Return the highest line number among rows of `kinds`, or 0 for none."""
    return max((row.number or 0 for row in rows if row.kind in kinds), default=0)


def _highlight_source_prefix(source: str, line: int) -> str:
    """Return the highlightable prefix ending at `line`."""
    if not source or line <= 0:
        return ""
    # Never split more than the limit itself. An edit near the end of a large
    # file asks for a prefix that is going to be rejected anyway, and splitting
    # the whole source first would allocate a near-full copy of the file per
    # side, per compose, only to throw it away. One char past the limit is
    # enough to tell "fits" from "does not".
    oversized = len(source) > MAX_HIGHLIGHT_CHARS
    lines = (source[: MAX_HIGHLIGHT_CHARS + 1] if oversized else source).splitlines()
    # With a truncated head, `line` is only reached within the limit when a
    # further line follows it — otherwise the last entry is a partial line and
    # the real prefix runs past the limit.
    if oversized and len(lines) <= line:
        return ""
    kept = lines[:line]
    prefix = "\n".join(kept)
    if kept and not kept[-1]:
        # Re-splitting drops a trailing empty line, because a terminating
        # newline yields no final entry — so without this the next trim would
        # see one line fewer and return a shorter prefix. Joining with `"\n"`
        # rather than keeping the original terminators is deliberate: it
        # normalizes `\r`, U+2028 and the rest that `splitlines()` breaks on but
        # `Content.split("\n")` does not, keeping row numbers aligned to the
        # lexed output.
        prefix += "\n"
    return prefix if len(prefix) <= MAX_HIGHLIGHT_CHARS else ""


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
    total = len(rows)
    if max_lines is not None:
        rows = rows[:max_lines]
    hidden = total - len(rows)
    emphasis = _emphasis_by_row(rows)
    highlighted = _highlighted_rows(rows, path, before, after)
    width = max(2, len(str(max((row.number or 0 for row in rows), default=0))))

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
        style = _ROW_STYLES[row.kind]
        if style.emphasis:
            for start, end in emphasis.get(index, []):
                body = body.stylize(style.emphasis, start, end)
        parts: list[Content | str | tuple[str, str]] = []
        numbered = show_numbers and row.number is not None
        if numbered:
            parts += [(f"{row.number:>{width}}", style.gutter), " "]
        parts += [(style.marker, style.marker_style), " ", body]
        # The selectable prefix is everything before the source text: the
        # padded number and a space, plus the marker and a space.
        prefix_len = (width + 1 if numbered else 0) + 2
        yield _DiffRowStatic(
            Content.assemble(*parts),
            prefix_len,
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
    correctly rather than reopening at the hunk boundary. Rows outside the
    prefix, or whose text has drifted from the source, are omitted and render as
    plain text.

    That holds only as far as the caller's source really is file-aligned. The
    approval prompt's main path now passes full before/after contents, but its
    fallback still passes edit fragments: the lexer starts mid-file and treats
    the fragment as if it began at line 1. The drift check below does not catch
    this — the fragment's diff is generated *from* those same strings, so the
    row text matches and every row is highlighted. A fragment cut from inside a
    docstring or block comment is therefore colored as code, and nothing
    detects it. Cosmetic, and confined to the approval prompt's fallback.

    Assumes a single-file diff, as `before`/`after` are one file's contents: rows
    are matched to source by line number, which restarts per file in a multi-file
    diff and would collide.
    """
    if not path:
        return {}
    highlighted: dict[int, Content] = {}
    for kinds, code in ((_BEFORE_KINDS, before), (_AFTER_KINDS, after)):
        wanted = {
            row.number: i
            for i, row in enumerate(rows)
            if row.kind in kinds and row.number is not None
        }
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
def _highlight_lines_cached(code: str, path: str) -> tuple[Content, ...] | None:
    """Return highlighted source lines, or `None` if lexing fails.

    Cached because scrolling rebuilds a `DiffMessage` from `MessageData` on every
    pass, and each mount would otherwise re-lex both sides. Two entries per diff,
    so `maxsize=4` holds the last two diffs — the scrolling case it exists for.

    Sized small on purpose. `MAX_HIGHLIGHT_CHARS` bounds the *input*, not what is
    retained: an entry is one `Content` per line, each carrying a span list, and
    measures several times its source. Nothing clears this cache, so its cost is
    held for the process lifetime — size it against measured entries, not against
    the character limit.

    *Expected* failures are cached too: a file whose lexer cannot parse it will
    not parse on the next scroll either, and retrying would pay the cost to fail
    again. Unexpected ones deliberately propagate to `_highlight_lines`, which
    handles them outside the cache — memoizing a genuine bug would log it once
    per `(code, path)` and then hide it for the rest of the process.
    """
    try:
        return tuple(highlight(code, path=path, tab_size=0).split("\n"))
    except (ValueError, LookupError) as e:
        # No usable lexer. Not reachable through an unknown extension —
        # `highlight` guesses a lexer rather than raising, so `m.unknownext`
        # and a bare `noext` both return styled output. This covers a lexer
        # that fails on the content itself, and any future `highlight` that
        # stops guessing; debug rather than warning because degrading to plain
        # text is a complete, if plainer, render.
        logger.debug("No usable lexer for %s: %s", path, e)
        return None


def _highlight_lines(code: str, path: str) -> tuple[Content, ...] | None:
    """Return highlighted source lines, or `None` if highlighting fails.

    Wraps the cache rather than living inside it so an unexpected failure stays
    visible on every attempt. `lru_cache` does not memoize raised exceptions, so
    letting them escape `_highlight_lines_cached` is what keeps the retry.

    Returns:
        One `Content` per line, or `None` when the source could not be lexed.
    """
    try:
        return _highlight_lines_cached(code, path)
    except Exception:
        # Anything not caught inside is a bug here or a Textual API change, not
        # a missing lexer. Highlighting is cosmetic, so still degrade to plain
        # text, but say so at a level that will actually be seen.
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
                rows.append(_Row("separator", "", None))
            seen_hunk = True
        elif line.startswith("-"):
            rows.append(_Row("removed", line[1:], old))
            old += 1
        elif line.startswith("+"):
            rows.append(_Row("added", line[1:], new))
            new += 1
        elif line.startswith(" "):
            # Numbered from `new`, not `old` — see `_Row.number`. Both walkers
            # still advance: `old` is what the *next* removed row is numbered
            # from.
            rows.append(_Row("context", line[1:], new))
            old += 1
            new += 1
        elif is_truncation_marker(line):
            # Checked after the marker prefixes above, so a context or added
            # line whose own text is `...` stays a source row. Reordering these
            # branches would render it as "diff truncated".
            rows.append(_Row("truncated", "", None))
        else:
            rows.append(_Row("note", line, None))
    return rows


def _emphasis_by_row(rows: list[_Row]) -> dict[int, list[_Range]]:
    """Return changed ranges for equal-length removed/added runs.

    A removed run is paired with the added run that immediately follows it, row
    by row in order, and only when the two are the same length — with no
    one-to-one correspondence there is nothing to diff a row against. Any other
    row kind between them (including a `note`, which is what a "no newline at
    end of file" marker parses to) breaks the adjacency and leaves the pair
    unemphasised.
    """
    runs = [
        (kind, [index for index, _ in group])
        for kind, group in groupby(enumerate(rows), key=lambda pair: pair[1].kind)
    ]
    ranges: dict[int, list[_Range]] = {}
    for (kind, old_indexes), (next_kind, new_indexes) in pairwise(runs):
        if (
            kind != "removed"
            or next_kind != "added"
            or len(old_indexes) != len(new_indexes)
        ):
            continue
        for old_index, new_index in zip(old_indexes, new_indexes, strict=True):
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
