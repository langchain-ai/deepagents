"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.widgets import Static

from deepagents_code.tui.widgets import diff as diff_module
from deepagents_code.tui.widgets.diff import (
    _EMPHASIS,
    _MAX_EMPHASIS_LEN,
    _MAX_HIGHLIGHT_CHARS,
    compose_diff_lines,
    count_diff_changes,
)

if TYPE_CHECKING:
    import pytest
    from textual.app import ComposeResult
    from textual.content import Content

_LEXER_FAILURE = "lexer exploded"
"""Message raised by the stubbed lexer in `test_lexer_failure_degrades...`."""


def _rendered(diff: str, max_lines: int | None = 100) -> list[Static]:
    """Materialize the diff widgets produced for `diff`.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff lines to show.

    Returns:
        The list of `Static` widgets yielded by `compose_diff_lines`.
    """
    return [w for w in compose_diff_lines(diff, max_lines) if isinstance(w, Static)]


def _plain(widget: Static) -> str:
    """Return the plain text a diff widget renders, ignoring styles.

    The diff renderer builds every widget from a `Content` instance, so the
    `render()` result is narrowed back to `Content` to read its `.plain`.

    Args:
        widget: A `Static` widget produced by the diff renderer.

    Returns:
        The widget's rendered text without style markup.
    """
    return cast("Content", widget.render()).plain


def _contents(widgets: ComposeResult) -> list[Content]:
    """Render each composed widget to its `Content`.

    Args:
        widgets: Result of a `compose_diff_lines` call.

    Returns:
        One `Content` per `Static` produced.
    """
    return [cast("Content", w.render()) for w in widgets if isinstance(w, Static)]


def _body_spans(content: Content, text: str) -> list[object]:
    """Return the spans falling inside a row's content, ignoring its gutter.

    Every row styles its line number and `-`/`+` marker, so a bare `.spans`
    check can never be empty and would silently pass.

    Args:
        content: Rendered row content.
        text: The row's line text, used to locate where the body begins.

    Returns:
        Spans starting at or after the body offset.
    """
    start = content.plain.index(text)
    return [s for s in content.spans if s.start >= start]


def _emphasis_spans(widgets: list[Static]) -> list[str]:
    """Collect the substrings carrying a word-level emphasis tint.

    Keyed off `_EMPHASIS` rather than a hard-coded style string so a retint
    does not silently turn these assertions into no-ops.

    Args:
        widgets: Rendered diff row widgets.

    Returns:
        Emphasized substrings, in render order.
    """
    styles = set(_EMPHASIS.values())
    found: list[str] = []
    for widget in widgets:
        content = cast("Content", widget.render())
        found.extend(
            content.plain[s.start : s.end] for s in content.spans if s.style in styles
        )
    return found


def _texts(widgets: list[Static]) -> list[str]:
    """Extract the plain text of each widget, ignoring styles.

    Args:
        widgets: Widgets produced by the diff renderer.

    Returns:
        The plain-text rendering of each widget, in order.
    """
    return [_plain(w) for w in widgets]


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


class TestComposeDiffLines:
    """Rendering behavior of `compose_diff_lines`."""

    def test_empty_diff_reports_no_changes(self) -> None:
        """An empty diff yields a single 'No changes detected' row."""
        texts = _texts(_rendered(""))
        assert texts == ["No changes detected"]

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
        """File headers and hunk headers don't appear as diff-line widgets."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        # No rendered row should contain the raw header markers.
        assert not any("a/f.py" in t or "b/f.py" in t for t in texts)
        assert not any(t.startswith("@@") for t in texts)

    def test_hunk_header_drives_line_numbers(self) -> None:
        """Old/new line numbers track from the hunk header start values."""
        widgets = _rendered(_SAMPLE_DIFF)
        texts = _texts(widgets)
        # Locate rows by their content.
        ctx = next(t for t in texts if "ctx" in t)
        removed = next(t for t in texts if "removed" in t)
        added1 = next(t for t in texts if "added1" in t)
        added2 = next(t for t in texts if "added2" in t)
        # Hunk starts at old=10, new=12. Context uses the old counter (10);
        # the deletion follows at old=11; additions use the new counter,
        # which advanced past the context line to 13 then 14.
        assert "10" in ctx
        assert "11" in removed
        assert "13" in added1
        assert "14" in added2

    def test_added_and_removed_rows_get_css_classes(self) -> None:
        """Added/removed rows carry CSS classes; context rows do not."""
        classes = {_plain(w): set(w.classes) for w in _rendered(_SAMPLE_DIFF)}
        added = next(c for t, c in classes.items() if "added1" in t)
        removed = next(c for t, c in classes.items() if "removed" in t)
        context = next(c for t, c in classes.items() if "ctx" in t)
        assert "diff-line-added" in added
        assert "diff-line-removed" in removed
        assert context == set()

    def test_content_columns_align_across_line_types(self) -> None:
        """Context/added/removed rows start their content at the same column."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        ctx = next(t for t in texts if "ctx" in t)
        removed = next(t for t in texts if "removed" in t)
        added1 = next(t for t in texts if "added1" in t)
        # The gutter glyph, right-aligned line number, and separator must be
        # the same width on every row so the diff body lines up vertically.
        assert ctx.index("ctx") == removed.index("removed") == added1.index("added1")

    def test_max_lines_truncates_with_marker(self) -> None:
        """Beyond `max_lines`, a truncation marker replaces remaining rows."""
        diff = "\n".join(["@@ -1,5 +1,5 @@", *(f"+line{i}" for i in range(5))])
        texts = _texts(_rendered(diff, max_lines=2))
        # 2 rendered rows + 1 truncation marker.
        assert any("more lines" in t for t in texts)
        rendered_rows = [t for t in texts if "line" in t and "more lines" not in t]
        assert len(rendered_rows) == 2

    def test_only_changed_words_are_emphasized(self) -> None:
        """Within a paired `-`/`+` row, only differing words get an extra tint."""
        diff = "@@ -1 +1 @@\n-value = compute(old_arg)\n+value = compute(new_arg)"
        added = next(w for w in _rendered(diff) if "new_arg" in _plain(w))
        content = cast("Content", added.render())
        emphasized = [
            content.plain[s.start : s.end]
            for s in content.spans
            if s.style == "on $success 30%"
        ]
        assert emphasized == ["new_arg"]

    def test_rows_are_highlighted_with_whole_file_lexer_state(self) -> None:
        """Highlighting reads the file, not the hunk, so string state is right."""
        after = 'def f():\n    """Doc.\n\n    More.\n    """\n    if x:\n\tpass\n'
        # The hunk opens on the line *closing* a docstring. Lexed on its own
        # that `\"\"\"` reads as the start of a string, and every line after it
        # is painted as one — the "everything is green" failure.
        diff = '@@ -5 +5,3 @@\n     """\n+    if x:\n+\tpass'
        rows = [
            cast("Content", w.render())
            for w in compose_diff_lines(diff, path="m.py", after=after)
            if isinstance(w, Static)
        ]
        keyword = next(r for r in rows if "if x:" in r.plain)
        assert any(s.style == "$text-accent" for s in keyword.spans)
        # Tabs must survive unexpanded, or the emphasis offsets would misalign.
        assert any(r.plain.endswith("\tpass") for r in rows)


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
        rows = _contents(compose_diff_lines(diff, path="m.py", after=stale))
        added = next(r for r in rows if "if x:" in r.plain)
        assert _body_spans(added, "if x:") == []

    def test_file_over_the_char_ceiling_renders_unhighlighted(self) -> None:
        """A prefix past `_MAX_HIGHLIGHT_CHARS` skips the lexer entirely."""
        filler = "\n".join("x = 1" for _ in range(_MAX_HIGHLIGHT_CHARS // 5))
        after = f"{filler}\nif y:\n"
        line_number = after.count("\n") - 1
        diff = f"@@ -{line_number} +{line_number} @@\n+if y:"
        rows = _contents(compose_diff_lines(diff, path="m.py", after=after))
        added = next(r for r in rows if "if y:" in r.plain)
        assert _body_spans(added, "if y:") == []

    def test_lexer_failure_degrades_to_plain_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising lexer must not escape `compose()` and kill the widget."""

        def _boom(*_args: object, **_kwargs: object) -> Content:
            raise RuntimeError(_LEXER_FAILURE)

        monkeypatch.setattr(diff_module, "highlight", _boom)
        diff_module._highlight_lines.cache_clear()
        rows = _contents(
            compose_diff_lines("@@ -1 +1 @@\n+if x:", path="m.py", after="if x:\n")
        )
        diff_module._highlight_lines.cache_clear()
        assert any("if x:" in r.plain for r in rows)


class TestRowKinds:
    """Rows that are neither added, removed, nor context."""

    def test_second_hunk_is_introduced_by_a_separator(self) -> None:
        """Consecutive hunks read as distinct blocks."""
        diff = "@@ -1 +1 @@\n-a\n+b\n@@ -50 +50 @@\n-c\n+d"
        assert any("diff-hunk-break" in w.classes for w in _rendered(diff)), _texts(
            _rendered(diff)
        )

    def test_truncation_marker_is_distinct_from_a_hunk_break(self) -> None:
        """A cut-short diff must not read as one that merely skips ahead."""
        widgets = _rendered("@@ -1 +1 @@\n-a\n+b\n...")
        assert any("truncated" in _plain(w) for w in widgets)
        assert not any("diff-hunk-break" in w.classes for w in widgets)

    def test_unrecognized_lines_render_as_notes(self) -> None:
        r"""`\ No newline at end of file` and friends stay visible."""
        diff = "@@ -1 +1 @@\n-a\n+b\n\\ No newline at end of file"
        assert any("No newline" in t for t in _texts(_rendered(diff)))


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
