"""Enhanced diff widget for displaying unified diffs."""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from rich.cells import cell_len
from textual.containers import Vertical
from textual.content import Content
from textual.highlight import highlight
from textual.strip import Strip
from textual.visual import Visual
from textual.widget import Widget
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.geometry import Offset, Size
    from textual.selection import Selection

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)")
"""Matches a unified-diff hunk header, capturing the old and new start lines."""

_HUNK_COUNTS_RE = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))?")
"""Matches a hunk header, capturing its optional old and new line counts."""

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

_TINTS = {"added": "on $success 10%", "removed": "on $error 10%"}
"""Whole-row tint for changed lines.

Applied with `stylize_before` so the gutter and word-level tints, which are
added afterwards, composite on top of it. This lives here rather than in a CSS
class because the diff body is one widget: a widget has a single background,
so a per-row tint has to travel with the row's own content.
"""

_TRUNCATED_NOTE = "... diff truncated"
"""Body text of a `truncated` row."""

_NO_CHANGES_NOTE = "No changes detected"
"""Body text shown in place of an empty diff."""

_ContentPart = str | tuple[str, str] | Content
_Range = tuple[int, int]
_RowKind = Literal["context", "added", "removed", "separator", "truncated", "note"]
"""Closed set of row kinds, so a typo is a type error rather than a `KeyError`.

`_GUTTERS`, `_MARKERS`, and `_EMPHASIS` are keyed by a subset of these, and a
new kind must be handled in `_compose_diff_content` before it reaches them.
"""


class _Row(NamedTuple):
    """One rendered line of a diff."""

    kind: _RowKind
    """Which of the six row shapes this is; drives styling and gutter."""

    text: str
    """Line content without its diff marker; empty for separator/truncated."""

    number: int
    """File line number, or `0` for separator, truncated, and note rows."""


def count_diff_changes(diff: str) -> tuple[int, int]:
    """Count added and removed lines in a unified diff.

    Args:
        diff: Unified diff string.

    Returns:
        Tuple of (additions, deletions), excluding `---`/`+++` file headers.
    """
    lines = diff.splitlines()
    header_indexes = _file_header_indexes(lines)
    additions = 0
    deletions = 0
    for index, line in enumerate(lines):
        if index in header_indexes:
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _file_header_indexes(lines: list[str]) -> set[int]:
    """Locate paired file headers immediately preceding a hunk.

    Prefixes alone are ambiguous: removing content that begins with `--` also
    produces a diff row beginning with `---`. A real unified-diff file header
    is the paired `---`/`+++` prelude to a hunk, so use that surrounding
    structure to distinguish metadata from content.

    Args:
        lines: Lines of a unified diff.

    Returns:
        Indexes of file-header lines.
    """
    indexes: set[int] = set()
    old_remaining = new_remaining = 0
    inside_hunk = False
    for index, line in enumerate(lines):
        if match := _HUNK_COUNTS_RE.match(line):
            old_remaining = int(match.group(1) or 1)
            new_remaining = int(match.group(2) or 1)
            inside_hunk = bool(old_remaining or new_remaining)
            continue
        if inside_hunk:
            if line.startswith("-"):
                old_remaining -= 1
            elif line.startswith("+"):
                new_remaining -= 1
            elif line.startswith(" "):
                old_remaining -= 1
                new_remaining -= 1
            inside_hunk = old_remaining > 0 or new_remaining > 0
            continue
        if index + 2 >= len(lines):
            continue
        if (
            line.startswith("--- ")
            and lines[index + 1].startswith("+++ ")
            and _HUNK_RE.match(lines[index + 2])
        ):
            indexes.update((index, index + 1))
    return indexes


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


def _hard_wrap(text: str, width: int) -> list[_Range]:
    """Split `text` into character spans, each at most `width` cells wide.

    A greedy fold measured in cells, so a double-width character is never split
    across the boundary. A character wider than the whole column is emitted on a
    line of its own and overflows, which keeps the walk making progress.

    Textual breaks on word boundaries and restarts every line at column zero,
    and `line_pad` insets both sides equally, so neither can produce a diff's
    hanging indent. cc_src and codex hand-roll this same fold for the same
    reason; there is no framework primitive for it.

    Args:
        text: Line to wrap.
        width: Column width in cells.

    Returns:
        `(start, end)` character offsets, one per visual line. Never empty, so
        a blank line still occupies a row.
    """
    width = max(1, width)
    if not text:
        return [(0, 0)]
    spans: list[_Range] = []
    start = used = 0
    for index, char in enumerate(text):
        size = cell_len(char)
        if used + size > width and index > start:
            spans.append((start, index))
            start, used = index, 0
        used += size
    spans.append((start, len(text)))
    return spans


class DiffBody(Widget):
    """A unified diff rendered as a single widget.

    Rows render as a tinted line number, a colored `-`/`+` marker, and the
    syntax-highlighted line content, with only the words that actually changed
    carrying a stronger tint within a related `-`/`+` pair.

    The whole body is one widget rather than one widget per row. A Textual
    widget is a message pump with its own task, style map, and layout node, so
    a hundred-row diff otherwise costs a hundred of each; and only one widget
    can own the wrapping, which is what lets a wrapped line keep its gutter
    column instead of restarting at column zero. Rows are laid out against the
    current width once and each row's strips are built on first paint, so
    scrolling past a long diff only pays for the rows actually shown.
    """

    DEFAULT_CSS = """
    DiffBody {
        width: 1fr;
        height: auto;
        text-wrap: nowrap;
        text-overflow: clip;
    }
    """
    """Wrapping is done here, so Textual must not also wrap or fold a row."""

    def __init__(
        self,
        diff: str,
        max_lines: int | None = 100,
        *,
        path: str = "",
        before: str = "",
        after: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize the diff body.

        Args:
            diff: Unified diff string.
            max_lines: Maximum number of diff rows to show (None for unlimited).
            path: Path of the diffed file, used to pick a syntax highlighter.
            before: Full file content the diff starts from, for highlighting.
            after: Full file content the diff arrives at, for highlighting.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        rows = _parse_rows(diff.splitlines()) if diff else []
        hidden = 0 if max_lines is None else max(0, len(rows) - max_lines)
        rows = rows[: len(rows) - hidden]
        # Both are keyed by index into the visible rows, so they have to be
        # computed before any footer row shifts the indexing.
        self._emphasis = _emphasis_by_row(rows)
        self._highlighted = _highlighted_rows(rows, path, before, after)
        if hidden:
            rows = [*rows, _Row("note", "", 0), _Row("note", _more_note(hidden), 0)]
        elif not rows:
            rows = [_Row("note", _NO_CHANGES_NOTE, 0)]
        self._rows = rows
        self._number_width = max(
            2, len(str(max((row.number for row in rows), default=0)))
        )
        self._gutter_width = self._number_width + 3
        self._layout_width = -1
        self._spans: list[list[_Range]] = []
        self._starts: list[int] = []
        self._strips: dict[int, list[Strip]] = {}

    @property
    def rows(self) -> list[_Row]:
        """The rows this widget renders, including any footer row."""
        return self._rows

    def notify_style_update(self) -> None:
        """Drop rendered strips so a theme change repaints in the new colors.

        Every tint is a CSS-variable style string resolved when the row is
        rendered, so cached strips hold the colors of the theme that was active
        when they were built.
        """
        super().notify_style_update()
        self._strips.clear()

    def get_content_height(
        self,
        container: Size,  # noqa: ARG002
        viewport: Size,  # noqa: ARG002
        width: int,
    ) -> int:
        """Return the number of visual lines the diff occupies at `width`.

        Args:
            container: Size of the container widget.
            viewport: Size of the viewport.
            width: Width the content will be rendered at.

        Returns:
            Total visual line count, wrapping included.
        """
        return self._layout(width)

    def render_line(self, y: int) -> Strip:
        """Render one visual line.

        Args:
            y: Line offset within the widget.

        Returns:
            The rendered strip, with any selection applied.
        """
        self._layout(self.size.width)
        blank = Strip.blank(self.size.width, self.visual_style.rich_style)
        index = bisect_right(self._starts, y) - 1
        if not 0 <= index < len(self._rows):
            return blank
        strips = self._row_strips(index)
        offset = y - self._starts[index]
        strip = strips[offset] if 0 <= offset < len(strips) else blank
        return self._select(strip, y)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Extract the selected source, without the gutter and without wraps.

        Copying a diff should yield something that can be pasted back, so the
        widget coordinates are mapped to offsets in the original rows rather
        than read off the rendered lines. Reading the lines would carry the line
        numbers, the `-`/`+` markers, and the terminal's wrap points into the
        clipboard.

        Args:
            selection: Selection in widget coordinates.

        Returns:
            Tuple of the selected text and its line ending, or `None` before
            the first layout, when there are no coordinates to map against.
        """
        if not self._starts:
            return None
        last = len(self._rows) - 1
        first_index, first_offset = (
            self._locate(selection.start) if selection.start else (0, 0)
        )
        last_index, last_offset = (
            self._locate(selection.end)
            if selection.end
            else (last, len(self._row_plain(last)))
        )
        if first_index == last_index:
            return self._row_plain(first_index)[first_offset:last_offset], "\n"
        parts = [self._row_plain(first_index)[first_offset:]]
        parts.extend(
            self._row_plain(index) for index in range(first_index + 1, last_index)
        )
        parts.append(self._row_plain(last_index)[:last_offset])
        return "\n".join(parts), "\n"

    def _layout(self, width: int) -> int:
        """Wrap every row to `width`, if that is not already the current layout.

        Args:
            width: Content width in cells.

        Returns:
            Total visual line count across all rows.
        """
        width = max(1, width)
        if width == self._layout_width:
            return self._starts[-1]
        self._layout_width = width
        self._strips.clear()
        self._spans = []
        self._starts = []
        total = 0
        for index, row in enumerate(self._rows):
            if row.kind == "separator":
                spans = [(0, 0)]
            else:
                gutter = self._gutter_width if row.kind in _GUTTERS else 0
                spans = _hard_wrap(self._row_plain(index), width - gutter)
            self._spans.append(spans)
            self._starts.append(total)
            total += len(spans)
        self._starts.append(total)
        return total

    def _row_plain(self, index: int) -> str:
        """Return a row's body text, without its gutter.

        Kept separate from `_row_body` because the layout pass needs the text of
        every row while only the rows actually painted need styling.

        Args:
            index: Row index.

        Returns:
            The row's text as it appears after the gutter.
        """
        row = self._rows[index]
        if row.kind == "separator":
            return get_glyphs().hunk_break
        if row.kind == "truncated":
            return _TRUNCATED_NOTE
        return row.text

    def _row_body(self, index: int) -> Content:
        """Return a row's styled body, without its gutter.

        Args:
            index: Row index.

        Returns:
            Syntax-highlighted content carrying any word-level emphasis, or
            plain text for rows that are not diff content.
        """
        row = self._rows[index]
        if row.kind == "separator":
            return Content.styled(self._row_plain(index), "bold $text-primary")
        if row.kind in {"truncated", "note"}:
            return Content.styled(self._row_plain(index), "dim")
        body = self._highlighted.get(index) or Content(row.text)
        for start, end in self._emphasis.get(index, []):
            body = body.stylize(_EMPHASIS[row.kind], start, end)
        return body

    def _row_lines(self, index: int) -> list[Content]:
        """Build a row's visual lines at the current layout width.

        The gutter is drawn on the first line and reserved as blank on the rest,
        so a wrapped line stays in the content column instead of restarting at
        column zero. Changed rows are padded to the full width before being
        tinted so the tint reaches the right edge.

        Args:
            index: Row index.

        Returns:
            One `Content` per visual line.
        """
        row = self._rows[index]
        body = self._row_body(index)
        width = self._layout_width
        if row.kind == "separator":
            indent = max(0, (width - body.cell_length) // 2)
            return [Content(" " * indent) + body]
        spans = self._spans[index]
        if row.kind not in _GUTTERS:
            return [body[start:end] for start, end in spans]
        marker, marker_style = _MARKERS.get(row.kind, (" ", ""))
        gutter = Content.assemble(
            (f"{row.number:>{self._number_width}}", _GUTTERS[row.kind]),
            " ",
            (marker, marker_style),
            " ",
        )
        blank = Content(" " * self._gutter_width)
        lines = [
            (gutter if position == 0 else blank) + body[start:end]
            for position, (start, end) in enumerate(spans)
        ]
        if tint := _TINTS.get(row.kind):
            lines = [
                (line + Content(" " * max(0, width - line.cell_length))).stylize_before(
                    tint
                )
                for line in lines
            ]
        return lines

    def _row_strips(self, index: int) -> list[Strip]:
        """Render a row to strips, caching the result for the current width.

        Args:
            index: Row index.

        Returns:
            One strip per visual line of the row.
        """
        strips = self._strips.get(index)
        if strips is None:
            content = Content("\n").join(self._row_lines(index))
            strips = Visual.to_strips(
                self,
                content,
                self._layout_width,
                None,
                self.visual_style,
                apply_selection=False,
                pad=True,
            )
            self._strips[index] = strips
        return strips

    def _select(self, strip: Strip, y: int) -> Strip:
        """Apply the selection tint to the selected span of a line.

        `Visual.to_strips` resolves a selection against the visual it is given,
        and each row is rendered as its own visual whose line numbering starts
        at zero. Selection is tracked in widget coordinates, so it is applied
        here instead, where `y` is absolute.

        Args:
            strip: The rendered line.
            y: Line offset within the widget.

        Returns:
            The strip, tinted where it is selected.
        """
        selection = self.text_selection
        if selection is None:
            return strip
        span = selection.get_span(y)
        if span is None:
            return strip
        start, end = span
        if end == -1:
            end = strip.cell_length
        if start >= end:
            return strip
        before, selected, after = strip.divide([start, end])
        return Strip.join([before, selected.apply_style(self.selection_style), after])

    def _locate(self, offset: Offset) -> tuple[int, int]:
        """Map a widget coordinate to a row index and offset within that row.

        Args:
            offset: Coordinate in widget space, with `x` measured in cells.

        Returns:
            Tuple of row index and character offset into that row's text.
        """
        column, y = offset
        index = min(max(bisect_right(self._starts, y) - 1, 0), len(self._rows) - 1)
        spans = self._spans[index]
        start, end = spans[min(max(y - self._starts[index], 0), len(spans) - 1)]
        if self._rows[index].kind in _GUTTERS:
            column -= self._gutter_width
        text = self._row_plain(index)
        cursor = start
        used = 0
        while cursor < end and used < column:
            used += cell_len(text[cursor])
            cursor += 1
        return index, cursor


def _more_note(hidden: int) -> str:
    """Build the footer text for a diff cut short by `max_lines`.

    Args:
        hidden: Number of rows not rendered.

    Returns:
        The footer text.
    """
    return f"... ({hidden} more lines)"


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
    header_indexes = _file_header_indexes(lines)
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

    Returns no ranges — leaving the row its uniform tint — in three cases:
    either side is blank, either side is longer than `_MAX_EMPHASIS_LEN` (a
    cost bound on the token matcher), or the two are too dissimilar to be a
    modification of one another (`_is_related`).

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


class EnhancedDiff(Vertical):
    """Widget for displaying a unified diff in a titled, bordered box.

    Unused as of this writing — `DiffMessage` is what the transcript mounts.
    Note it composes without a `path`, so its rows are never syntax
    highlighted.
    """

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

        yield DiffBody(self._diff, self._max_lines)

        additions, deletions = self._stats
        if additions or deletions:
            yield Static(diff_stats_content(additions, deletions), classes="diff-stats")
