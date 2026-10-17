"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from rich.cells import get_character_cell_size
from textual.style import Style
from textual.visual import RenderOptions
from textual.widgets import Static

from deepagents_code.config import reset_glyphs_cache
from deepagents_code.tui.widgets import diff as diff_module
from deepagents_code.tui.widgets.diff import _DiffRowStatic, compose_diff_lines

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.app import ComposeResult
    from textual.content import Content
    from textual.selection import Selection
    from textual.strip import Strip


@pytest.fixture(autouse=True)
def _clear_highlight_cache() -> Iterator[None]:
    """Keep the module-level highlight cache from leaking between tests.

    `_highlight_lines_cached` is an `lru_cache` on `(code, path)`, so one test's
    highlighted lines — or its cached lexer *failure* — would otherwise be served
    to the next test using the same snippet.
    """
    diff_module._highlight_lines_cached.cache_clear()
    yield
    diff_module._highlight_lines_cached.cache_clear()


@pytest.fixture(autouse=True)
def _unicode_glyphs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Render every diff row with the Unicode glyph set, regardless of ambient state.

    The diff renderer reads `get_glyphs()`, which is cached process-wide from the
    terminal charset detection. An unrelated test that forces ASCII mode (e.g. in
    `test_charset.py`) would otherwise leave `.` as the continuation glyph and
    break the `…` expectations below, with xdist scheduling deciding which run
    order exposes the leak.
    """
    monkeypatch.setenv("UI_CHARSET_MODE", "unicode")
    reset_glyphs_cache()
    yield
    reset_glyphs_cache()


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


def _emphasized(widget: Static, style: str) -> list[str]:
    """Return the substrings a diff row marks with word-level emphasis.

    Args:
        widget: A rendered diff row.
        style: The emphasis style to look for, e.g. `on $success 30%`.

    Returns:
        The emphasized substrings, in span order.
    """
    content = cast("Content", widget.render())
    return [content.plain[s.start : s.end] for s in content.spans if s.style == style]


def _keyword_spans(content: Content) -> list[object]:
    """Return the syntax-highlighting spans on a row.

    Every rendered row carries gutter and marker spans regardless, so "was this
    row highlighted" has to be asked of the lexer's own style — the accent
    Textual paints keywords with.

    Args:
        content: A rendered diff row.

    Returns:
        The keyword spans, empty when the row rendered as plain text.
    """
    return [s for s in content.spans if s.style == "$text-accent"]


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


def _visual_strips(
    widget: Static,
    width: int,
    selection: Selection | None = None,
    selection_style: Style | None = None,
) -> list[Strip]:
    """Render a diff widget at `width` and return its visual strips."""
    content = cast("Content", widget.render())
    options = RenderOptions(lambda _: Style.null(), {}, selection, selection_style)
    return content.render_strips(width, None, Style.null(), options)


def _offset_at(widget: Static, width: int, x: int, y: int) -> int | None:
    """Resolve a visual cell to a logical offset the way Textual does.

    Mirrors `Compositor.get_widget_and_offset_at`, which reads a segment's
    `offset` metadata and then adds the character index *within* that segment.
    Asserting on the metadata alone cannot tell a correct offset from one that
    happens to share a segment base, so tests go through this instead.
    """
    strip = _visual_strips(widget, width)[y]
    start = end = 0
    for segment in strip:
        end += segment.cell_length
        offset = (segment.style.meta if segment.style else {}).get("offset")
        if offset is None or offset[1] is None:
            start = end
            continue
        if start <= x < end:
            cut = x - start
            size = index = 0
            for character in segment.text:
                if size >= cut:
                    break
                size += get_character_cell_size(character)
                index += 1
            return offset[0] + index
        start = end
    return None


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

    def test_file_and_hunk_headers_are_not_rendered_as_rows(self) -> None:
        """File headers and hunk headers don't appear as diff-line widgets."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        # No rendered row should contain the raw header markers.
        assert not any("a/f.py" in t or "b/f.py" in t for t in texts)
        assert not any(t.startswith("@@") for t in texts)

    def test_content_columns_align_across_line_types(self) -> None:
        """Context/added/removed rows start their content at the same column."""
        texts = _texts(_rendered(_SAMPLE_DIFF))
        ctx = next(t for t in texts if "ctx" in t)
        removed = next(t for t in texts if "removed" in t)
        added1 = next(t for t in texts if "added1" in t)
        # The gutter glyph, right-aligned line number, and separator must be
        # the same width on every row so the diff body lines up vertically.
        assert ctx.index("ctx") == removed.index("removed") == added1.index("added1")

    def test_equal_multi_line_runs_pair_row_by_row_in_order(self) -> None:
        """Row *i* of the removed run pairs with row *i* of the added run.

        A regression that paired only the first row, or paired from the end,
        still emphasizes *something* — so a single-row case cannot catch it.
        """
        diff = (
            "@@ -1,2 +1,2 @@\n"
            "-alpha = old_one\n"
            "-beta = old_two\n"
            "+alpha = new_one\n"
            "+beta = new_two"
        )
        rendered = _rendered(diff)
        pairs = {
            "old_one": "on $error 30%",
            "old_two": "on $error 30%",
            "new_one": "on $success 30%",
            "new_two": "on $success 30%",
        }
        for token, style in pairs.items():
            row = next(w for w in rendered if token in _plain(w))
            assert _emphasized(row, style) == [token], (
                f"{token} paired with the wrong row"
            )

    def test_an_expected_lexer_failure_degrades_and_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `(ValueError, LookupError)` branch, and its negative caching.

        The test above raises `RuntimeError`, which the *outer* wrapper handles
        and deliberately leaves uncached. This is the inner branch, whose
        failures are cached on purpose: a lexer that cannot parse this content
        will not parse it on the next scroll either, and a rebuilt
        `DiffMessage` re-lexes both sides every pass. Asserting the call count
        is what separates the two policies — without it, moving the handler
        inside or outside the cache reads the same.

        Note this branch is not reachable through an unknown extension:
        `highlight` guesses a lexer instead of raising, so the failure has to
        be injected.
        """
        diff_module._highlight_lines_cached.cache_clear()
        calls = 0

        def _no_lexer(*_args: object, **_kwargs: object) -> Content:
            """Fail the way a missing lexer does, counting attempts."""
            nonlocal calls
            calls += 1
            raise LookupError(msg)

        msg = "no lexer for this"
        monkeypatch.setattr(diff_module, "highlight", _no_lexer)
        for _ in range(3):
            rows = _contents(
                compose_diff_lines("@@ -1 +1 @@\n+if x:", path="m.py", after="if x:\n")
            )

        row = next(r for r in rows if "if x:" in r.plain)
        assert not _keyword_spans(row), f"an unlexable row was styled: {row.spans}"
        assert calls == 1, f"an expected failure was re-attempted {calls} times"
        diff_module._highlight_lines_cached.cache_clear()

    def test_source_that_drifted_from_the_diff_renders_plain(self) -> None:
        """Highlighting is only safe while the source still matches the diff.

        A stale rehydration, or `before`/`after` belonging to another file, would
        otherwise paint a row with spans lifted from an unrelated line — colors
        that look authoritative and describe different code. Matching by line
        number cannot detect that; comparing the text can.
        """
        # Line 1 of `after` is not the line the diff says was added there.
        rows = _contents(
            compose_diff_lines(
                "@@ -1 +1 @@\n+if x:", path="m.py", after="something_else = 1\n"
            )
        )
        row = next(r for r in rows if "if x:" in r.plain)
        assert not _keyword_spans(row), f"drifted row was highlighted: {row.spans}"


class TestClampSelection:
    """Diff-row selections exclude the gutter from highlight and copy alike.

    The gutter is dropped by rewriting the stored `Selection` (what Textual
    paints) rather than by intercepting `get_selection` (what Textual copies),
    because painting reads the stored geometry directly. Each test below would
    otherwise pass for the copy while leaving the gutter visually selected.
    """

    @staticmethod
    def _row(text: str) -> _DiffRowStatic:
        """Return the rendered row containing `text`."""
        widget = next(w for w in _rendered(_SAMPLE_DIFF) if text in _plain(w))
        assert isinstance(widget, _DiffRowStatic)
        return widget


class TestRowKinds:
    """Rows that are neither added, removed, nor context."""

    def test_second_hunk_is_introduced_by_a_separator(self) -> None:
        """Consecutive hunks read as distinct blocks."""
        diff = "@@ -1 +1 @@\n-a\n+b\n@@ -50 +50 @@\n-c\n+d"
        assert any("diff-hunk-break" in w.classes for w in _rendered(diff)), _texts(
            _rendered(diff)
        )


class TestFormatDiffStats:
    """Tests for the `+N -M` header fragment."""


class TestGutterNumbersTheFileOnDisk:
    """The gutter names lines in the file the user can still open."""

    _INSERTION = (
        "@@ -1,3 +1,5 @@\n"
        " def f():\n"
        "+    # added a\n"
        "+    # added b\n"
        "     return 1\n"
        " end = True"
    )

    def _numbers(self, diff: str) -> list[str]:
        """Return the leading gutter token of every rendered row.

        Returns:
            One gutter string per row, in render order.
        """
        return [_plain(w).split()[0] for w in _rendered(diff)]
