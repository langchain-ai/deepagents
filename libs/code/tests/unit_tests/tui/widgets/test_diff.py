"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.widgets import Static

from deepagents_code.diff_utils import count_diff_changes
from deepagents_code.tui.widgets import diff as diff_module
from deepagents_code.tui.widgets.diff import compose_diff_lines

if TYPE_CHECKING:
    import pytest
    from textual.app import ComposeResult
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


def _contents(widgets: ComposeResult) -> list[Content]:
    """Render each composed widget to its `Content`.

    Args:
        widgets: Result of a `compose_diff_lines` call.

    Returns:
        One `Content` per `Static` produced.
    """
    return [cast("Content", w.render()) for w in widgets if isinstance(w, Static)]


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

    def test_show_numbers_false_drops_the_gutter(self) -> None:
        """Callers whose diff is not file-relative can suppress line numbers.

        The approval preview diffs edit *fragments*, so its hunks always start
        at 1; rendering that gutter would assert wrong file line numbers.
        """
        numbered = _texts(_rendered(_SAMPLE_DIFF))
        plain = _texts(
            [
                w
                for w in compose_diff_lines(_SAMPLE_DIFF, 100, show_numbers=False)
                if isinstance(w, Static)
            ]
        )

        assert any(text.lstrip().startswith("10") for text in numbered)
        assert not any(text.strip().startswith(("10", "11", "12")) for text in plain)
        # The marker and body survive; only the gutter is gone.
        assert any(text.strip() == "- removed" for text in plain)
        assert any(text.strip() == "+ added1" for text in plain)

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
        assert any(r.plain.endswith("\tpass") for r in rows)

    def test_lexer_failure_degrades_to_plain_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        msg = "lexer exploded"

        def _boom(*_args: object, **_kwargs: object) -> Content:
            raise RuntimeError(msg)

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
