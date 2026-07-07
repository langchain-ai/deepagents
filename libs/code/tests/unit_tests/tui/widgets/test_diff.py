"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.widgets import Static

from deepagents_code.tui.widgets.diff import compose_diff_lines

if TYPE_CHECKING:
    from textual.content import Content


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

    def test_max_lines_truncates_with_marker(self) -> None:
        """Beyond `max_lines`, a truncation marker replaces remaining rows."""
        diff = "\n".join(["@@ -1,5 +1,5 @@", *(f"+line{i}" for i in range(5))])
        texts = _texts(_rendered(diff, max_lines=2))
        # Stats header + 2 rendered rows + 1 truncation marker.
        assert any("more lines" in t for t in texts)
        rendered_rows = [t for t in texts if "line" in t and "more lines" not in t]
        assert len(rendered_rows) == 2
