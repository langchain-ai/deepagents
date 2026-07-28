"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.widgets import Static

from deepagents_code.tui.widgets.diff import compose_diff_lines

if TYPE_CHECKING:
    from textual.content import Content


def _rendered(
    diff: str, max_lines: int | None = 100, *, show_stats: bool = True
) -> list[Static]:
    """Materialize the diff widgets produced for `diff`.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of diff lines to show.
        show_stats: Whether the `+N -M` summary row is included.

    Returns:
        The list of `Static` widgets yielded by `compose_diff_lines`.
    """
    return [
        w
        for w in compose_diff_lines(diff, max_lines, show_stats=show_stats)
        if isinstance(w, Static)
    ]


def _emphasized(widget: Static) -> list[str]:
    """Return the substrings of a row that carry an emphasis span.

    Args:
        widget: A `Static` widget produced by the diff renderer.

    Returns:
        The emphasized substrings, in order.
    """
    content = cast("Content", widget.render())
    return [
        content.plain[span.start : span.end]
        for span in content.spans
        if not isinstance(span.style, str)
    ]


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

    def test_stats_header_excludes_file_headers(self) -> None:
        """`+++`/`---` headers are not counted as additions/deletions."""
        # First widget is the stats header when there are changes.
        header = _texts(_rendered(_SAMPLE_DIFF))[0]
        # Two additions (added1, added2), one deletion (removed) — headers
        # `+++ b/f.py` and `--- a/f.py` must not inflate the counts.
        assert header == "+2 -1"

    def test_stats_header_omits_zero_side(self) -> None:
        """A diff with only additions shows just the `+N` segment."""
        diff = "@@ -1,0 +1,1 @@\n+only addition"
        header = _texts(_rendered(diff))[0]
        assert header == "+1"

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
        # Locate rows by their content (skip the stats header at index 0).
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

    def test_stats_row_can_be_suppressed(self) -> None:
        """`show_stats=False` leaves the counts to the caller's own header."""
        texts = _texts(_rendered(_SAMPLE_DIFF, show_stats=False))
        assert not any(t.startswith("+2") for t in texts)
        assert any("added1" in t for t in texts)

    def test_rows_carry_change_markers(self) -> None:
        """Changed rows are marked with `-`/`+`; context rows are not."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        removed = next(t for t in texts if "removed" in t)
        added1 = next(t for t in texts if "added1" in t)
        ctx = next(t for t in texts if "ctx" in t)
        assert "- removed" in removed
        assert "+ added1" in added1
        assert "-" not in ctx
        assert "+" not in ctx

    def test_second_hunk_is_separated(self) -> None:
        """Each hunk after the first is introduced by a separator row."""
        diff = "@@ -1,1 +1,1 @@\n-a\n+b\n@@ -20,1 +20,1 @@\n-c\n+d"
        texts = _texts(_rendered(diff))
        # One separator only — the first hunk needs no break above it.
        assert sum(t.strip() in {"…", "..."} for t in texts) == 1

    def test_max_lines_truncates_with_marker(self) -> None:
        """Beyond `max_lines`, a truncation marker replaces remaining rows."""
        diff = "\n".join(["@@ -1,5 +1,5 @@", *(f"+line{i}" for i in range(5))])
        texts = _texts(_rendered(diff, max_lines=2))
        # Stats header + 2 rendered rows + 1 truncation marker.
        assert any("more lines" in t for t in texts)
        rendered_rows = [t for t in texts if "line" in t and "more lines" not in t]
        assert len(rendered_rows) == 2


class TestIntraLineEmphasis:
    """Word-level highlighting within paired removed/added rows."""

    def test_only_changed_words_are_emphasized(self) -> None:
        """A one-to-one edit highlights just the words that differ."""
        diff = "@@ -1 +1 @@\n-value = compute(old_arg)\n+value = compute(new_arg)"
        widgets = _rendered(diff)
        removed = next(w for w in widgets if "old_arg" in _plain(w))
        added = next(w for w in widgets if "new_arg" in _plain(w))
        assert _emphasized(removed) == ["old_arg"]
        assert _emphasized(added) == ["new_arg"]

    def test_context_rows_are_never_emphasized(self) -> None:
        """Unchanged rows carry no emphasis spans."""
        diff = "@@ -1,2 +1,2 @@\n ctx\n-a = 1\n+a = 2"
        widgets = _rendered(diff)
        ctx = next(w for w in widgets if "ctx" in _plain(w))
        assert _emphasized(ctx) == []

    def test_unrelated_rewrite_is_not_emphasized(self) -> None:
        """Lines sharing almost nothing fall back to the plain row tint."""
        diff = "@@ -1 +1 @@\n-import os\n+CONSTANT_TABLE = {1: 'x', 2: 'y'}"
        widgets = _rendered(diff)
        added = next(w for w in widgets if "CONSTANT_TABLE" in _plain(w))
        assert _emphasized(added) == []

    def test_lopsided_runs_are_not_paired(self) -> None:
        """When N lines become one, positional pairing would mislead."""
        diff = "@@ -1,3 +1,1 @@\n-a = [\n-    1,\n-]\n+a = [1]"
        widgets = _rendered(diff)
        added = next(w for w in widgets if "a = [1]" in _plain(w))
        assert _emphasized(added) == []
