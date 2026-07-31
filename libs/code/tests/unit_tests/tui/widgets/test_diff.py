"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection

from deepagents_code.tui.widgets import diff as diff_module
from deepagents_code.tui.widgets.diff import (
    _EMPHASIS,
    _MAX_EMPHASIS_LEN,
    _MAX_HIGHLIGHT_CHARS,
    _TINTS,
    DiffBody,
    count_diff_changes,
)

if TYPE_CHECKING:
    import pytest

_LEXER_FAILURE = "lexer exploded"
"""Message raised by the stubbed lexer in `test_lexer_failure_degrades...`."""

_WIDE = 200
"""A layout width no fixture line reaches, so nothing wraps unless asked."""


def _body(
    diff: str, max_lines: int | None = 100, *, width: int = _WIDE, **kwargs: str
) -> DiffBody:
    """Build a `DiffBody` and lay it out, without mounting it in an app.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff rows to show.
        width: Layout width in cells.
        **kwargs: Forwarded to `DiffBody` (`path`, `before`, `after`).

    Returns:
        A laid-out widget, ready to be asked for rows.
    """
    widget = DiffBody(diff, max_lines, **kwargs)
    widget._layout(width)
    return widget


def _contents(widget: DiffBody) -> list[Content]:
    """Render each row of a laid-out widget to a single `Content`.

    Rows that wrap are joined back with newlines so a test can assert against
    a row without caring how many visual lines it occupies.

    Args:
        widget: A laid-out `DiffBody`.

    Returns:
        One `Content` per row, in render order.
    """
    return [
        Content("\n").join(widget._row_lines(index))
        for index in range(len(widget.rows))
    ]


def _rendered(diff: str, max_lines: int | None = 100, **kwargs: str) -> list[Content]:
    """Build, lay out, and render a diff in one step.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff rows to show.
        **kwargs: Forwarded to `DiffBody`.

    Returns:
        One `Content` per row.
    """
    return _contents(_body(diff, max_lines, **kwargs))


def _texts(contents: list[Content]) -> list[str]:
    """Extract each row's plain text, ignoring styles.

    Changed rows are padded to the layout width so their tint reaches the right
    edge, so the trailing padding is stripped as a rendering artifact.

    Args:
        contents: Rendered rows.

    Returns:
        The plain text of each row, in order.
    """
    return [content.plain.rstrip() for content in contents]


def _body_spans(content: Content, text: str) -> list[object]:
    """Return the spans falling inside a row's content, ignoring its gutter.

    Every row styles its line number and `-`/`+` marker, and a changed row
    carries a whole-row tint anchored at offset zero, so a bare `.spans` check
    can never be empty and would silently pass.

    Args:
        content: Rendered row content.
        text: The row's line text, used to locate where the body begins.

    Returns:
        Spans starting at or after the body offset.
    """
    start = content.plain.index(text)
    return [span for span in content.spans if span.start >= start]


def _emphasis_spans(contents: list[Content]) -> list[str]:
    """Collect the substrings carrying a word-level emphasis tint.

    Keyed off `_EMPHASIS` rather than a hard-coded style string so a retint
    does not silently turn these assertions into no-ops.

    Args:
        contents: Rendered diff rows.

    Returns:
        Emphasized substrings, in render order.
    """
    styles = set(_EMPHASIS.values())
    found: list[str] = []
    for content in contents:
        found.extend(
            content.plain[span.start : span.end]
            for span in content.spans
            if span.style in styles
        )
    return found


# A diff exercising file headers, a hunk header, and context/add/remove lines.
_SAMPLE_DIFF = (
    "--- a/f.py\n"
    "+++ b/f.py\n"
    "@@ -10,3 +12,4 @@ def f():\n"
    " ctx\n"
    "-removed\n"
    "+added1\n"
    "+added2"
)


class TestDiffBody:
    """Rendering behavior of `DiffBody`."""

    def test_empty_diff_reports_no_changes(self) -> None:
        """An empty diff yields a single 'No changes detected' row."""
        assert _texts(_rendered("")) == ["No changes detected"]

    def test_change_counts_exclude_file_headers(self) -> None:
        """`+++`/`---` headers are not counted as additions/deletions."""
        # Two additions (added1, added2), one deletion (removed) — headers
        # `+++ b/f.py` and `--- a/f.py` must not inflate the counts.
        assert count_diff_changes(_SAMPLE_DIFF) == (2, 1)

    def test_content_starting_with_header_markers_is_counted_and_rendered(
        self,
    ) -> None:
        """Changed `--`/`++` content is not mistaken for file metadata."""
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n---old value\n+++new value"

        assert count_diff_changes(diff) == (1, 1)
        texts = _texts(_rendered(diff))
        assert any(text.endswith("--old value") for text in texts)
        assert any(text.endswith("++new value") for text in texts)

    def test_header_shaped_content_before_another_hunk_is_still_content(
        self,
    ) -> None:
        """Hunk counts disambiguate changed rows next to a later hunk."""
        diff = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1 +1 @@\n--- old value\n+++ new value\n"
            "@@ -10 +10 @@\n-old tail\n+new tail"
        )

        assert count_diff_changes(diff) == (2, 2)
        texts = _texts(_rendered(diff))
        assert any(text.endswith("-- old value") for text in texts)
        assert any(text.endswith("++ new value") for text in texts)

    def test_file_and_hunk_headers_are_not_rendered_as_rows(self) -> None:
        """File headers and hunk headers don't appear as diff rows."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        # No rendered row should contain the raw header markers.
        assert not any("a/f.py" in text or "b/f.py" in text for text in texts)
        assert not any(text.startswith("@@") for text in texts)

    def test_hunk_header_drives_line_numbers(self) -> None:
        """Old/new line numbers track from the hunk header start values."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        # Locate rows by their content.
        ctx = next(text for text in texts if "ctx" in text)
        removed = next(text for text in texts if "removed" in text)
        added1 = next(text for text in texts if "added1" in text)
        added2 = next(text for text in texts if "added2" in text)
        # Hunk starts at old=10, new=12. Context uses the old counter (10);
        # the deletion follows at old=11; additions use the new counter,
        # which advanced past the context line to 13 then 14.
        assert "10" in ctx
        assert "11" in removed
        assert "13" in added1
        assert "14" in added2

    def test_changed_rows_are_tinted_and_context_rows_are_not(self) -> None:
        """Added/removed rows carry a whole-row tint; context rows do not."""
        contents = _rendered(_SAMPLE_DIFF)
        tints = {
            text: {span.style for span in content.spans} & set(_TINTS.values())
            for text, content in zip(_texts(contents), contents)
        }
        added = next(t for text, t in tints.items() if "added1" in text)
        removed = next(t for text, t in tints.items() if "removed" in text)
        context = next(t for text, t in tints.items() if "ctx" in text)
        assert added == {_TINTS["added"]}
        assert removed == {_TINTS["removed"]}
        assert context == set()

    def test_row_tint_reaches_the_right_edge(self) -> None:
        """A changed row is padded so its tint spans the full width."""
        content = next(c for c in _rendered("@@ -1 +1 @@\n-a\n+b") if "+ b" in c.plain)
        tint = next(span for span in content.spans if span.style == _TINTS["added"])
        assert content.cell_length == _WIDE
        assert (tint.start, tint.end) == (0, _WIDE)

    def test_content_columns_align_across_line_types(self) -> None:
        """Context/added/removed rows start their content at the same column."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        ctx = next(text for text in texts if "ctx" in text)
        removed = next(text for text in texts if "removed" in text)
        added1 = next(text for text in texts if "added1" in text)
        # The gutter glyph, right-aligned line number, and separator must be
        # the same width on every row so the diff body lines up vertically.
        assert ctx.index("ctx") == removed.index("removed") == added1.index("added1")

    def test_max_lines_truncates_with_marker(self) -> None:
        """Beyond `max_lines`, a truncation marker replaces remaining rows."""
        diff = "\n".join(["@@ -1,5 +1,5 @@", *(f"+line{i}" for i in range(5))])
        texts = _texts(_rendered(diff, max_lines=2))
        # 2 rendered rows + a blank spacer + the truncation marker.
        assert any("more lines" in text for text in texts)
        rendered_rows = [t for t in texts if "line" in t and "more lines" not in t]
        assert len(rendered_rows) == 2

    def test_only_changed_words_are_emphasized(self) -> None:
        """Within a paired `-`/`+` row, only differing words get an extra tint."""
        diff = "@@ -1 +1 @@\n-value = compute(old_arg)\n+value = compute(new_arg)"
        added = next(c for c in _rendered(diff) if "new_arg" in c.plain)
        emphasized = [
            added.plain[span.start : span.end]
            for span in added.spans
            if span.style == _EMPHASIS["added"]
        ]
        assert emphasized == ["new_arg"]

    def test_rows_are_highlighted_with_whole_file_lexer_state(self) -> None:
        """Highlighting reads the file, not the hunk, so string state is right."""
        after = 'def f():\n    """Doc.\n\n    More.\n    """\n    if x:\n\tpass\n'
        # The hunk opens on the line *closing* a docstring. Lexed on its own
        # that `\"\"\"` reads as the start of a string, and every line after it
        # is painted as one — the "everything is green" failure.
        diff = '@@ -5 +5,3 @@\n     """\n+    if x:\n+\tpass'
        rows = _rendered(diff, path="m.py", after=after)
        keyword = next(row for row in rows if "if x:" in row.plain)
        assert any(span.style == "$text-accent" for span in keyword.spans)
        # Tabs must survive unexpanded, or the emphasis offsets would misalign.
        assert any(row.plain.rstrip().endswith("\tpass") for row in rows)


class TestWrapping:
    """Long lines fold into the content column rather than back to column zero."""

    # Long enough to wrap twice at the narrow width used below. The leading
    # indentation matters: it is content, and must land after the gutter.
    _LONG = "    args = ['--no-mcp', '--no-interpreter', '--shell-allow-list', 'pwd']"

    def _wrapped(self, text: str, width: int) -> tuple[list[str], int]:
        """Render a single long added row and return its visual lines.

        Args:
            text: Line content to add.
            width: Layout width in cells.

        Returns:
            Tuple of the row's visual lines, with trailing tint padding
            stripped, and the width of the gutter each line reserves.
        """
        widget = _body(f"@@ -1 +1 @@\n+{text}", width=width)
        lines = [line.plain.rstrip() for line in widget._row_lines(0)]
        return lines, widget._gutter_width

    def test_continuation_lines_reserve_the_gutter(self) -> None:
        """Wrapped lines start in the content column, not at column zero."""
        lines, gutter = self._wrapped(self._LONG, 40)
        assert len(lines) > 1
        # The first line carries the number and marker; the rest carry neither,
        # but hold the column open so the code stays aligned under itself.
        assert lines[0].startswith(" 1 + ")
        for line in lines[1:]:
            assert line[:gutter] == " " * gutter

    def test_wrapping_is_character_level_and_fills_the_width(self) -> None:
        """Folding on characters uses the full column, unlike word wrapping."""
        width = 40
        lines, _ = self._wrapped(self._LONG, width)
        # Every line but the last is filled to the edge; a word-wrapped line
        # would stop early at the last space that fits.
        for line in lines[:-1]:
            assert len(line) == width

    def test_reassembling_the_visual_lines_recovers_the_source(self) -> None:
        """No character is dropped or duplicated by the fold."""
        lines, gutter = self._wrapped(self._LONG, 40)
        assert "".join(line[gutter:] for line in lines) == self._LONG

    def test_wide_characters_are_not_split_across_the_boundary(self) -> None:
        """A double-width glyph moves to the next line rather than being cut."""
        text = "日本語" * 12
        lines, gutter = self._wrapped(text, 20)
        for line in lines:
            assert Content(line[gutter:]).cell_length <= 20 - gutter
        assert "".join(line[gutter:] for line in lines) == text

    def test_height_accounts_for_wrapping(self) -> None:
        """The reported height is visual lines, not rows."""
        narrow = _body(f"@@ -1 +1 @@\n+{self._LONG}", width=40)
        wide = _body(f"@@ -1 +1 @@\n+{self._LONG}", width=_WIDE)
        assert narrow._layout(40) > wide._layout(_WIDE) == 1

    def test_emphasis_survives_a_wrap(self) -> None:
        """Word-level tints computed on the unwrapped row still land correctly."""
        old = f"{self._LONG} old_tail"
        new = f"{self._LONG} new_tail"
        widget = _body(f"@@ -1 +1 @@\n-{old}\n+{new}", width=40)
        added = Content("\n").join(widget._row_lines(1))
        emphasized = [
            added.plain[span.start : span.end]
            for span in added.spans
            if span.style == _EMPHASIS["added"]
        ]
        assert emphasized == ["new_tail"]


class TestSelection:
    """Copying a diff should yield source, not the rendering of one."""

    _DIFF = "@@ -1,2 +1,2 @@\n ctx line\n-old value\n+new value"

    def test_selection_excludes_the_gutter(self) -> None:
        """Line numbers and markers stay out of the clipboard."""
        widget = _body(self._DIFF, width=40)
        selected = widget.get_selection(Selection(Offset(5, 0), Offset(13, 0)))
        assert selected == ("ctx line", "\n")

    def test_selection_spanning_rows_joins_them(self) -> None:
        """A multi-row selection returns each row's text, newline separated."""
        widget = _body(self._DIFF, width=40)
        selected = widget.get_selection(Selection(Offset(5, 1), Offset(14, 2)))
        assert selected == ("old value\nnew value", "\n")

    def test_selection_within_a_wrapped_row_returns_unwrapped_source(self) -> None:
        """The terminal's wrap points do not survive into the copied text."""
        long = "value = " + "abcdefghij" * 6
        widget = _body(f"@@ -1 +1 @@\n+{long}", width=30)
        assert len(widget._row_lines(0)) > 1
        # From the start of the row to the end of its last visual line.
        selected = widget.get_selection(Selection(Offset(5, 0), Offset(30, 2)))
        assert selected == (long, "\n")

    def test_selection_before_layout_is_declined(self) -> None:
        """Without a layout there are no coordinates to map against."""
        widget = DiffBody(self._DIFF)
        assert widget.get_selection(Selection(Offset(0, 0), Offset(5, 0))) is None


class TestEmphasisGuards:
    """Each bailout that suppresses word-level emphasis must actually fire.

    Without these, deleting a guard or widening its threshold leaves the suite
    green — the happy-path tests only prove emphasis appears, never that it is
    withheld when the heuristic says it should be.
    """

    def test_unrelated_rewrite_gets_no_word_emphasis(self) -> None:
        """A pair sharing too little content keeps its uniform row tint."""
        diff = "@@ -1 +1 @@\n-x = 1\n+completely_different_name_here = 999"
        assert _emphasis_spans(_rendered(diff)) == []

    def test_pair_differing_in_every_word_gets_no_emphasis(self) -> None:
        """Whitespace must not carry a wholly-changed pair over the floor."""
        diff = "@@ -1 +1 @@\n-aaa bbb ccc\n+zzz yyy xxx"
        assert _emphasis_spans(_rendered(diff)) == []

    def test_very_long_lines_skip_emphasis(self) -> None:
        """Past `_MAX_EMPHASIS_LEN` the token matcher is not run at all."""
        # Many words with exactly one changed: near-identical, so neither the
        # similarity floor nor the whole-line-coverage guard can suppress this.
        # Only the length bailout can, which is what makes the test isolating.
        words = [f"word{i}" for i in range(_MAX_EMPHASIS_LEN // 6 + 2)]
        old = " ".join(words)
        new = " ".join(["CHANGED", *words[1:]])
        assert len(old) > _MAX_EMPHASIS_LEN
        diff = f"@@ -1 +1 @@\n-{old}\n+{new}"
        assert _emphasis_spans(_rendered(diff)) == []

    def test_a_shorter_line_with_the_same_shape_is_emphasized(self) -> None:
        """Control for the above: under the ceiling, the one word is tinted."""
        words = [f"word{i}" for i in range(5)]
        old = " ".join(words)
        new = " ".join(["CHANGED", *words[1:]])
        assert len(old) < _MAX_EMPHASIS_LEN
        diff = f"@@ -1 +1 @@\n-{old}\n+{new}"
        assert _emphasis_spans(_rendered(diff)) == ["word0", "CHANGED"]

    def test_unequal_run_lengths_are_not_paired(self) -> None:
        """One removed line against two added lines is not a modification."""
        diff = "@@ -1 +1,2 @@\n-value = 1\n+value = 2\n+value = 3"
        assert _emphasis_spans(_rendered(diff)) == []

    def test_equal_run_lengths_are_paired(self) -> None:
        """The counterpart to the above: equal runs still emphasize."""
        diff = "@@ -1,2 +1,2 @@\n-value = 1\n-other = 1\n+value = 2\n+other = 2"
        # Both sides are emphasized, removed rows first in render order.
        assert _emphasis_spans(_rendered(diff)) == ["1", "1", "2", "2"]


class TestHighlightGuards:
    """Fallbacks that drop syntax highlighting rather than misrender it."""

    def test_row_not_matching_the_file_is_left_unhighlighted(self) -> None:
        """Content that moved on since the diff renders plain, not misaligned."""
        # The diff adds `if x:` at line 2 but the file has something else
        # there, so the lexed line cannot be trusted to line up with the row.
        # Addition-only, so word-level emphasis cannot muddy the span check.
        stale = "def f():\n    return 0\n"
        diff = "@@ -2 +2 @@\n+    if x:"
        rows = _rendered(diff, path="m.py", after=stale)
        added = next(row for row in rows if "if x:" in row.plain)
        assert _body_spans(added, "if x:") == []

    def test_file_over_the_char_ceiling_renders_unhighlighted(self) -> None:
        """A prefix past `_MAX_HIGHLIGHT_CHARS` skips the lexer entirely."""
        filler = "\n".join("x = 1" for _ in range(_MAX_HIGHLIGHT_CHARS // 5))
        after = f"{filler}\nif y:\n"
        line_number = after.count("\n") - 1
        diff = f"@@ -{line_number} +{line_number} @@\n+if y:"
        rows = _rendered(diff, path="m.py", after=after)
        added = next(row for row in rows if "if y:" in row.plain)
        assert _body_spans(added, "if y:") == []

    def test_lexer_failure_degrades_to_plain_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising lexer must not escape `compose()` and kill the widget."""

        def _boom(*_args: object, **_kwargs: object) -> Content:
            raise RuntimeError(_LEXER_FAILURE)

        monkeypatch.setattr(diff_module, "highlight", _boom)
        diff_module._highlight_lines.cache_clear()
        rows = _rendered("@@ -1 +1 @@\n+if x:", path="m.py", after="if x:\n")
        diff_module._highlight_lines.cache_clear()
        assert any("if x:" in row.plain for row in rows)


class TestRowKinds:
    """Rows that are neither added, removed, nor context."""

    def test_second_hunk_is_introduced_by_a_separator(self) -> None:
        """Consecutive hunks read as distinct blocks."""
        diff = "@@ -1 +1 @@\n-a\n+b\n@@ -50 +50 @@\n-c\n+d"
        widget = _body(diff)
        assert [row.kind for row in widget.rows].count("separator") == 1

    def test_separator_is_centered(self) -> None:
        """The hunk break sits in the middle of the row, not at its left edge."""
        widget = _body("@@ -1 +1 @@\n-a\n+b\n@@ -50 +50 @@\n-c\n+d", width=41)
        index = next(i for i, row in enumerate(widget.rows) if row.kind == "separator")
        line = widget._row_lines(index)[0].plain
        assert line.index(line.strip()) == 20

    def test_truncation_marker_is_distinct_from_a_hunk_break(self) -> None:
        """A cut-short diff must not read as one that merely skips ahead."""
        widget = _body("@@ -1 +1 @@\n-a\n+b\n...")
        kinds = [row.kind for row in widget.rows]
        assert "truncated" in kinds
        assert "separator" not in kinds
        assert any("truncated" in text for text in _texts(_contents(widget)))

    def test_unrecognized_lines_render_as_notes(self) -> None:
        r"""`\ No newline at end of file` and friends stay visible."""
        diff = "@@ -1 +1 @@\n-a\n+b\n\\ No newline at end of file"
        assert any("No newline" in text for text in _texts(_rendered(diff)))


class TestCountDiffChanges:
    """Header counts must survive content that looks like diff metadata."""

    def test_content_lines_beginning_with_dashes_are_counted(self) -> None:
        """`---`/`+++` inside a hunk body are content, not file headers."""
        diff = (
            "--- front.md (before)\n"
            "+++ front.md (after)\n"
            "@@ -1,3 +1,3 @@\n"
            "----\n"
            "++++\n"
            " title: a"
        )
        assert count_diff_changes(diff) == (1, 1)

    def test_real_file_headers_are_still_excluded(self) -> None:
        """The paired prelude to a hunk stays metadata."""
        diff = "--- a.py (before)\n+++ a.py (after)\n@@ -1 +1 @@\n-a\n+b"
        assert count_diff_changes(diff) == (1, 1)
