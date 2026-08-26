"""Unit tests for the unified-diff rendering widget."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pytest
from rich.cells import get_character_cell_size
from textual.geometry import Offset
from textual.selection import Selection
from textual.style import Color, Style
from textual.visual import RenderOptions
from textual.widgets import Static

from deepagents_code.config import ASCII_GLYPHS, get_glyphs, reset_glyphs_cache
from deepagents_code.diff_utils import (
    DiffStats,
    count_diff_change_lines,
    split_diff_lines,
)
from deepagents_code.tui.widgets import diff as diff_module
from deepagents_code.tui.widgets.diff import (
    _DiffRowContent,
    _DiffRowStatic,
    clamp_selection,
    compose_diff_lines,
    format_diff_stats,
    highlight_source_prefixes,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.app import ComposeResult
    from textual.content import Content
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


def _visual_lines(widget: Static, width: int) -> list[str]:
    """Render a diff widget at `width` and return its visual lines."""
    return [strip.text for strip in _visual_strips(widget, width)]


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


def _selected(widget: Static, width: int, x: int, y: int) -> str:
    """Return the source text a drag from cell `(x, y)` to the row end copies."""
    content = cast("Content", widget.render())
    offset = _offset_at(widget, width, x, y)
    return "" if offset is None else content.plain[offset:]


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
        assert count_diff_change_lines(split_diff_lines(_SAMPLE_DIFF)) == DiffStats(
            additions=2, deletions=1
        )

    def test_content_starting_with_header_markers_is_counted_and_rendered(
        self,
    ) -> None:
        """Changed `--`/`++` content is not mistaken for file metadata."""
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n---old value\n+++new value"

        assert count_diff_change_lines(split_diff_lines(diff)) == DiffStats(
            additions=1, deletions=1
        )
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

        assert any(text.lstrip().startswith("11") for text in numbered)
        assert not any(text.strip().startswith(("11", "12", "13")) for text in plain)
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
        # Hunk starts at old=10, new=12. Context is numbered in the *new* file
        # (12) so its number matches the file the user can still open; the
        # deletion keeps its old-file number (11), the only line that no longer
        # exists. Additions follow the new counter past the context line.
        assert "12" in ctx
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

    def test_wrapped_lines_replace_line_numbers_with_ellipsis(self) -> None:
        """Continuation text stays aligned under source, not the gutter."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdefghijkl") if "abc" in _plain(w)
        )

        assert _visual_lines(widget, 12) == ["680 + abcdef", "  …   ghijkl"]

    def test_continuation_gutter_maps_to_the_line_start(self) -> None:
        """Every synthetic gutter cell selects from the line's first source char.

        The gutter is decorative and absent from the logical content, so a
        drag begun anywhere in it must copy the whole visual line rather than
        skipping into the middle of it.
        """
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdefghijkl") if "abc" in _plain(w)
        )
        prefix = cast("_DiffRowStatic", widget).selection_prefix

        gutter = [_offset_at(widget, 12, x, 1) for x in range(prefix)]
        assert gutter == [12] * prefix
        assert _selected(widget, 12, 2, 1) == "ghijkl"

    def test_unnumbered_rows_keep_normal_wrapping(self) -> None:
        """Hiding line numbers must not clip wrapped source text."""
        widget = next(
            w
            for w in compose_diff_lines(
                "@@ -1 +1 @@\n+abcdefghijkl", show_numbers=False
            )
            if isinstance(w, Static) and "abc" in _plain(w)
        )
        content = cast("Content", widget.render())
        lines = _visual_lines(widget, 6)

        assert len(lines) > 1
        assert "".join(lines).replace(" ", "") == "+abcdefghijkl"
        assert content.get_height({}, 6) == len(lines)

    def test_narrow_width_uses_normal_wrapping_without_clipping(self) -> None:
        """Rows narrower than the gutter still render every source cell."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdef") if "abc" in _plain(w)
        )
        content = cast("Content", widget.render())
        lines = _visual_lines(widget, 5)

        assert lines == ["680 +", "abcde", "f"]
        assert content.get_height({}, 5) == len(lines)

    def test_wide_characters_survive_the_narrowest_body_column(self) -> None:
        """A body column too narrow for one wide char falls back, never blanks."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+漢字漢字") if "漢" in _plain(w)
        )
        content = cast("Content", widget.render())

        for width in (7, 8, 9):
            lines = _visual_lines(widget, width)
            assert "".join(lines).count("漢") == 2, width
            assert "".join(lines).count("字") == 2, width
            assert content.get_height({}, width) == len(lines), width

    def test_wide_character_with_zero_width_suffix_survives(self) -> None:
        """A zero-width suffix cannot hide the preceding wide character."""
        text = "漢\ufe0f"
        widget = next(
            w for w in _rendered(f"@@ -680 +680 @@\n+{text}") if "漢" in _plain(w)
        )
        content = cast("Content", widget.render())
        lines = _visual_lines(widget, 7)

        assert "漢" in "".join(lines)
        assert content.get_height({}, 7) == len(lines)

    def test_word_wrapped_rows_report_the_height_they_render(self) -> None:
        """`Static` is `height: auto`, so a short count clips the last lines."""
        widget = next(
            w
            for w in _rendered("@@ -680 +680 @@\n+alpha beta gamma delta epsilon")
            if "alpha" in _plain(w)
        )
        content = cast("Content", widget.render())

        for width in (12, 20, 28):
            lines = _visual_lines(widget, width)
            assert len(lines) > 1, width
            assert content.get_height({}, width) == len(lines), width
            # Drop the synthetic gutter glyph each continuation repeats;
            # at the narrowest width even a single word folds mid-token.
            painted = "".join(lines).replace("…", "").replace(" ", "")
            assert painted == "680+alphabetagammadeltaepsilon", width

    @pytest.mark.parametrize(
        "body",
        [
            "x" * (diff_module._MAX_WRAPPED_OFFSET_CHARS + 1),
            "\t" * (diff_module._MAX_WRAPPED_OFFSET_CHARS // 8 + 1),
        ],
        ids=["plain", "tabs"],
    )
    def test_oversized_rows_do_not_cache_per_character_offsets(self, body: str) -> None:
        """Large rows fall back before allocating custom offset tables."""
        widget = next(
            w
            for w in _rendered(f"@@ -680 +680 @@\n+{body}")
            if isinstance(w, _DiffRowStatic)
        )
        content = cast("_DiffRowContent", widget.render())

        content.get_height({}, 12)
        assert "_body" not in content.__dict__
        assert "_body_offsets" not in content.__dict__

    def test_wrapped_tabbed_lines_keep_all_source_text(self) -> None:
        """Tab expansion cannot make wrapping discard trailing source."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+\tabcdef") if "abc" in _plain(w)
        )

        assert _visual_lines(widget, 12) == [
            "680 +       ",
            "  …     abcd",
            "  …   ef",
        ]

    def test_tabbed_continuations_report_raw_logical_offsets(self) -> None:
        """Offsets index the raw row, not the wider tab-expanded rendering.

        Selections are extracted from `Content.plain`, where a tab is one
        character; reporting expanded indexes made a tab-indented row copy the
        wrong characters and, on its last lines, nothing at all.
        """
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+\tabcdef") if "abc" in _plain(w)
        )
        content = cast("Content", widget.render())

        assert content.plain == "680 + \tabcdef"
        # Visual cell -> the source character actually painted there.
        for width, x, y, expected in ((12, 8, 1, "a"), (12, 6, 2, "e")):
            offset = _offset_at(widget, width, x, y)
            assert offset is not None
            assert offset < len(content.plain)
            assert content.plain[offset] == expected
        assert _selected(widget, 12, 8, 1) == "abcdef"
        assert _selected(widget, 12, 7, 2) == "f"

    def test_unstyled_wide_character_before_tab_keeps_source_offsets(self) -> None:
        """Cell-aware tab expansion keeps painted characters copyable."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+漢\tabcdef") if "漢" in _plain(w)
        )
        content = cast("Content", widget.render())

        for expected in "abcdef":
            y, line = next(
                (y, line)
                for y, line in enumerate(_visual_lines(widget, 12))
                if expected in line
            )
            index = line.index(expected)
            x = sum(get_character_cell_size(char) for char in line[:index])
            offset = _offset_at(widget, 12, x, y)
            assert offset is not None
            assert content.plain[offset] == expected
            if expected == "a":
                assert content.plain[offset:] == "abcdef"

    def test_styled_continuations_keep_per_segment_offsets(self) -> None:
        """Clicks on later styled segments map to their source characters."""
        text = "value = alpha + beta + gamma"
        widget = next(
            w
            for w in compose_diff_lines(
                f"@@ -0,0 +1 @@\n+{text}", path="m.py", after=f"{text}\n"
            )
            if isinstance(w, Static) and "alpha" in _plain(w)
        )
        content = cast("Content", widget.render())
        line = _visual_lines(widget, 18)[1]
        prefix = cast("_DiffRowStatic", widget).selection_prefix

        # Highlighting cuts this line into several segments; each source cell
        # must still resolve to the character drawn in it.
        for x in range(prefix, len(line)):
            offset = _offset_at(widget, 18, x, 1)
            assert offset is not None, x
            assert content.plain[offset] == line[x], x

    def test_ascii_continuation_fits_the_line_number_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ASCII mode uses one dot, preserving the gutter width."""
        monkeypatch.setattr(diff_module, "get_glyphs", lambda: ASCII_GLYPHS)
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdefghijkl") if "abc" in _plain(w)
        )

        assert _visual_lines(widget, 12) == ["680 + abcdef", "  .   ghijkl"]

    def test_selection_style_reaches_wrapped_source(self) -> None:
        """Logical selections remain visibly highlighted after wrapping."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdefghijkl") if "abc" in _plain(w)
        )
        style = Style(background=Color(1, 2, 3))

        continuation = _visual_strips(
            widget, 12, Selection(Offset(12, 0), None), style
        )[1]
        source = tuple(continuation)[-1]
        assert source.style is not None
        assert source.style.bgcolor == style.rich_style.bgcolor

    def test_selection_ending_before_a_continuation_leaves_it_unstyled(self) -> None:
        """A selection on the first line must not paint the whole next one."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdefghijkl") if "abc" in _plain(w)
        )
        style = Style(background=Color(1, 2, 3))

        continuation = _visual_strips(
            widget, 12, Selection(Offset(6, 0), Offset(9, 0)), style
        )[1]
        assert all(
            segment.style is None or segment.style.bgcolor != style.rich_style.bgcolor
            for segment in continuation
        )

    def test_selection_ending_inside_a_continuation_splits_it(self) -> None:
        """Highlighting stops at the endpoint, mid-continuation."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abcdefghijkl") if "abc" in _plain(w)
        )
        style = Style(background=Color(1, 2, 3))

        continuation = _visual_strips(
            widget, 12, Selection(Offset(12, 0), Offset(15, 0)), style
        )[1]
        highlighted = "".join(
            segment.text
            for segment in continuation
            if segment.style is not None
            and segment.style.bgcolor == style.rich_style.bgcolor
        )
        assert highlighted == "ghi"

    def test_unwrapped_lines_keep_the_original_gutter(self) -> None:
        """A row that fits is unchanged by continuation support."""
        widget = next(
            w for w in _rendered("@@ -680 +680 @@\n+abc") if "abc" in _plain(w)
        )

        assert _visual_lines(widget, 12) == ["680 + abc"]

    @pytest.mark.parametrize("text", ["", " ", "abcdefghijkl", "\tabc", "alpha beta"])
    def test_every_visual_line_carries_an_offset(self, text: str) -> None:
        """No visual line may end up with an empty offset list.

        `render_strips` reads `offsets[0]` to point a continuation's synthetic
        gutter at its line, and `get_height` counts the same lines it must
        render. Skipping an offsetless line in only one of them desynchronises
        the two, and `Static` is `height: auto`, so the row grows a blank line.
        """
        widget = next(
            w
            for w in _rendered(f"@@ -680 +680 @@\n+{text}\n+tail")
            if _plain(w).endswith(text)
        )
        content = cast("_DiffRowContent", widget.render())

        for width in range(7, 20):
            lines = content._wrapped(width)
            assert lines
            assert all(line.offsets for line in lines), width
            assert content.get_height({}, width) == len(_visual_lines(widget, width))

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

    def test_removed_side_is_emphasized_too(self) -> None:
        """The pair is only readable when both halves point at the change."""
        diff = "@@ -1 +1 @@\n-value = compute(old_arg)\n+value = compute(new_arg)"
        removed = next(w for w in _rendered(diff) if "old_arg" in _plain(w))
        assert _emphasized(removed, "on $error 30%") == ["old_arg"]

    def test_unrelated_rewrites_get_no_emphasis(self) -> None:
        """Tinting every word is noise, not information.

        Below `_SIMILARITY_FLOOR` the two lines are not versions of each other, so
        "what changed" is the whole line and marking it says nothing.
        """
        diff = "@@ -1 +1 @@\n-import os\n+CONSTANT_TABLE = {1: 'a', 2: 'b'}"
        added = next(w for w in _rendered(diff) if "CONSTANT_TABLE" in _plain(w))
        assert _emphasized(added, "on $success 30%") == []

    def test_unequal_removed_and_added_runs_get_no_emphasis(self) -> None:
        """Pairing by offset only means anything when the runs are the same size.

        With two removals against one addition there is no defensible pairing, so
        emphasis is dropped rather than guessed.
        """
        diff = "@@ -1,2 +1 @@\n-alpha = 1\n-beta = 2\n+alpha = 3"
        added = next(w for w in _rendered(diff) if "alpha = 3" in _plain(w))
        assert _emphasized(added, "on $success 30%") == []

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

    def test_a_note_row_between_a_pair_breaks_emphasis(self) -> None:
        """A "no newline at end of file" marker parses to a `note` row.

        That splits the removed and added runs, so the pair is no longer
        adjacent and emphasis is dropped. Pinned because the shape is a real
        git-diff output, not a hypothetical.
        """
        diff = (
            "@@ -1 +1 @@\n"
            "-value = old_arg\n"
            "\\ No newline at end of file\n"
            "+value = new_arg"
        )
        added = next(w for w in _rendered(diff) if "new_arg" in _plain(w))
        assert _emphasized(added, "on $success 30%") == []

    def test_long_lines_are_not_emphasized(self) -> None:
        """The length guard keeps a quadratic match off the compose path.

        Minified JS or single-line JSON would otherwise stall the UI while
        `SequenceMatcher` runs over per-character tokens.
        """
        old = "a" * (diff_module._MAX_EMPHASIS_LEN + 1)
        new = f"{'a' * diff_module._MAX_EMPHASIS_LEN}b"
        added = next(
            w for w in _rendered(f"@@ -1 +1 @@\n-{old}\n+{new}") if new in _plain(w)
        )
        assert _emphasized(added, "on $success 30%") == []

    @pytest.mark.parametrize(
        ("old", "new", "changed"),
        [
            ('label = "你好世界 old"', 'label = "你好世界 new"', "new"),
            ('label = "😀🎉 old"', 'label = "😀🎉 new"', "new"),
        ],
        ids=["cjk", "astral"],
    )
    def test_emphasis_offsets_hold_for_non_ascii(
        self, old: str, new: str, changed: str
    ) -> None:
        """Token offsets index the line by code point, not by byte.

        `_TOKEN_RE` is total over its input so a running sum of token lengths is a
        valid offset. A non-total pattern, or byte offsets, would pass every
        ASCII test and silently mis-highlight anything wider.
        """
        added = next(
            w for w in _rendered(f"@@ -1 +1 @@\n-{old}\n+{new}") if changed in _plain(w)
        )
        assert _emphasized(added, "on $success 30%") == [changed]

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
        rows = _contents(
            compose_diff_lines("@@ -1 +1 @@\n+if x:", path="m.py", after="if x:\n")
        )
        assert any("if x:" in r.plain for r in rows)

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

    def test_a_drifted_side_warns_once_not_once_per_row(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A whole drifted side would otherwise report thousands of rows.

        The warning is deliberately per side rather than per row: at warning
        level it reaches the in-app console's ring buffer, and one row per line
        would flush everything else out of it.
        """
        added = "\n".join(f"+line_{i} = {i}" for i in range(40))
        with caplog.at_level(logging.WARNING, logger=diff_module.__name__):
            _contents(
                compose_diff_lines(
                    f"@@ -0,0 +1,40 @@\n{added}",
                    path="m.py",
                    after="\n".join(f"unrelated_{i} = {i}" for i in range(40)) + "\n",
                )
            )
        drift_warnings = [
            r for r in caplog.records if "Highlight source drifted" in r.message
        ]
        assert len(drift_warnings) == 1
        assert "40 of 40 rows" in drift_warnings[0].getMessage()

    def test_trimming_a_prefix_again_returns_it_unchanged(self) -> None:
        """Rehydration re-trims an already-trimmed prefix on every mount.

        A rule that shortens the prefix each pass would quietly walk the
        highlightable region back a line at a time. The trailing-empty-line case
        is the one that broke: a terminating newline yields no final entry when
        the prefix is split again.
        """
        diff = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n \n"
        for before, after in [("a\nb\n\n", "a\nB\n\n"), ("a\nb\n", "a\nB\n")]:
            once = highlight_source_prefixes(diff, before, after)
            twice = highlight_source_prefixes(diff, *once)
            assert once == twice, f"not idempotent for {before!r}"

    def test_oversized_source_is_rejected_and_renders_plain(self) -> None:
        """The size guard is the only thing bounding compose-time lexing.

        Past the limit the side is skipped and its rows render plain, rather than
        stalling the message pump lexing a whole large file — twice, once per
        side — inside `compose`. Rejecting the prefix must also not cost a
        near-full copy of the file first: an edit near the end of a large file
        asks for a prefix that is going to be refused anyway, and building it to
        measure it allocated and discarded megabytes per side, per compose.
        """
        line = 199_000
        source = "y = 2\n" * 200_000
        assert len(source) > diff_module.MAX_HIGHLIGHT_CHARS
        assert diff_module._highlight_source_prefix(source, line) == ""
        # A prefix that does fit still comes back whole.
        assert diff_module._highlight_source_prefix(source, 10) == "\n".join(
            ["y = 2"] * 10
        )

        rows = _contents(
            compose_diff_lines(
                f"@@ -{line} +{line} @@\n+y = 2", path="m.py", after=source
            )
        )
        row = next(r for r in rows if "y = 2" in r.plain)
        assert not _keyword_spans(row), (
            f"an oversized source was lexed anyway: {row.spans}"
        )


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

    def test_rows_report_their_gutter_width(self) -> None:
        """Every numbered row knows how wide its gutter is (5 for "13 + ")."""
        # 5 = width 2 + space + marker + space.
        assert self._row("added1").selection_prefix == 5
        assert self._row("ctx").selection_prefix == 5

    def test_unnumbered_rows_report_just_the_marker(self) -> None:
        """`show_numbers=False` still excludes the marker and its space."""
        widget = next(
            w
            for w in compose_diff_lines(_SAMPLE_DIFF, 100, show_numbers=False)
            if isinstance(w, _DiffRowStatic) and "added1" in _plain(w)
        )
        assert widget.selection_prefix == 2  # marker + space

    def test_select_all_starts_past_the_gutter(self) -> None:
        """A mid-selection row clamps its start, or the copy would carry numbers."""
        row = self._row("added1")
        assert clamp_selection(row, Selection(None, None)) == Selection(
            Offset(5, 0), None
        )

    def test_entered_from_above_starts_past_the_gutter(self) -> None:
        """`Selection(None, end)` covers the gutter unless its start moves."""
        row = self._row("added1")
        assert clamp_selection(row, Selection(None, Offset(8, 0))) == Selection(
            Offset(5, 0), Offset(8, 0)
        )

    def test_ending_inside_the_gutter_drops_the_row(self) -> None:
        """A selection reaching only the gutter selects nothing from this row."""
        row = self._row("added1")
        assert clamp_selection(row, Selection(None, Offset(5, 0))) is None
        assert clamp_selection(row, Selection(None, Offset(2, 0))) is None

    def test_starting_inside_the_gutter_moves_the_start(self) -> None:
        """A drag beginning on a line number still selects from the source."""
        row = self._row("added1")
        assert clamp_selection(row, Selection(Offset(2, 0), None)) == Selection(
            Offset(5, 0), None
        )

    def test_wholly_gutter_selection_drops_the_row(self) -> None:
        """A range covering only the gutter leaves nothing to select."""
        row = self._row("added1")
        assert clamp_selection(row, Selection(Offset(1, 0), Offset(4, 0))) is None

    def test_selection_inside_the_source_is_untouched(self) -> None:
        """Endpoints past the gutter keep their exact positions."""
        row = self._row("added1")
        selection = Selection(Offset(7, 0), None)
        assert clamp_selection(row, selection) == selection

    def test_wrapped_row_keeps_logical_continuation_offsets(self) -> None:
        """Continuation selections already point into the logical source row."""
        row = self._row("added1")
        prefix = row.selection_prefix
        start = Offset(prefix + 2, 0)

        assert clamp_selection(row, Selection(start, None)) == Selection(start, None)
        assert clamp_selection(row, Selection(None, start)) == Selection(
            Offset(prefix, 0), start
        )
        selection = Selection(start, Offset(prefix + 4, 0))
        assert clamp_selection(row, selection) == selection

    def test_non_diff_widgets_are_untouched(self) -> None:
        """Other widgets' selections pass through unchanged."""
        selection = Selection(Offset(1, 0), None)
        assert clamp_selection(Static("plain"), selection) == selection

    def test_clamped_selection_copies_without_the_gutter(self) -> None:
        """End to end: the clamped geometry yields source text when extracted."""
        row = self._row("added1")
        clamped = clamp_selection(row, Selection(None, None))
        assert clamped is not None
        assert row.get_selection(clamped) == ("added1", "\n")


class TestRowKinds:
    """Rows that are neither added, removed, nor context."""

    def test_second_hunk_is_introduced_by_a_separator(self) -> None:
        """Consecutive hunks read as distinct blocks."""
        diff = "@@ -1 +1 @@\n-a\n+b\n@@ -50 +50 @@\n-c\n+d"
        assert any("diff-hunk-break" in w.classes for w in _rendered(diff)), _texts(
            _rendered(diff)
        )

    def test_rows_after_a_separator_renumber_from_the_second_hunk(self) -> None:
        """Each hunk header resets the counters to the file it names.

        Asserting only that a separator appears leaves the reset itself
        unpinned: numbering the second hunk continuously from the first would
        render `3, 3` here instead of `50, 50` — plausible-looking, and wrong
        about every line it names.
        """
        diff = "@@ -1,1 +1,1 @@\n-a\n+b\n@@ -50,1 +50,1 @@\n-c\n+d"
        numbered = [
            _plain(w).split()[0]
            for w in _rendered(diff)
            if "diff-hunk-break" not in w.classes
        ]

        assert numbered == ["1", "1", "50", "50"]

    def test_truncation_marker_is_distinct_from_a_hunk_break(self) -> None:
        """A cut-short diff must not read as one that merely skips ahead."""
        widgets = _rendered("@@ -1 +1 @@\n-a\n+b\n...")
        assert any("truncated" in _plain(w) for w in widgets)
        assert not any("diff-hunk-break" in w.classes for w in widgets)

    def test_hunk_break_uses_the_configured_glyph(self) -> None:
        """The separator must go through `Glyphs` so ASCII terminals get `:`."""
        widget = next(
            w
            for w in _rendered("@@ -1 +1 @@\n-a\n+b\n@@ -50 +50 @@\n-c\n+d")
            if "diff-hunk-break" in w.classes
        )
        assert _plain(widget) == get_glyphs().hunk_break

    def test_unrecognized_lines_render_as_unnumbered_notes(self) -> None:
        r"""A line with no diff marker is metadata, not content.

        `\ No newline at end of file` is the common case: it belongs to the
        preceding row rather than being a line of its own, so it gets no gutter
        and no `+`/`-`.
        """
        note = r"\ No newline at end of file"
        widget = next(
            w for w in _rendered(f"@@ -1 +1 @@\n-a\n+b\n{note}") if note in _plain(w)
        )
        assert _plain(widget) == note
        assert not widget.classes


class TestFormatDiffStats:
    """Tests for the `+N -M` header fragment."""

    def test_both_sides_are_separated(self) -> None:
        """The common case: an edit that adds and removes."""
        assert format_diff_stats(DiffStats(additions=3, deletions=2)).plain == "+3 -2"

    def test_additions_only_carry_no_separator(self) -> None:
        """What every `write_file` approval renders.

        An unconditional separator would leave a stray trailing space that no
        both-sided test can see.
        """
        assert format_diff_stats(DiffStats(additions=3, deletions=0)).plain == "+3"

    def test_deletions_only_carry_no_leading_space(self) -> None:
        """What every `delete` approval renders."""
        assert format_diff_stats(DiffStats(additions=0, deletions=2)).plain == "-2"

    def test_zero_renders_nothing(self) -> None:
        """An unchanged file gets no counts rather than `+0 -0`."""
        assert format_diff_stats(DiffStats(additions=0, deletions=0)).plain == ""


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

    def test_context_after_an_insertion_follows_the_new_file(self) -> None:
        """Numbering context from the old file contradicts the file on disk.

        With context taken from the old counter, a two-line insertion rendered
        `1, 2, 3, 2, 3` — the numbers going backwards and repeating the ones the
        added rows had just used, while naming lines that no longer exist at
        those positions.
        """
        assert self._numbers(self._INSERTION) == ["1", "2", "3", "4", "5"]

    def test_removed_rows_keep_their_old_file_numbers(self) -> None:
        """A removed line has no number in the new file, so it keeps the old one."""
        diff = "@@ -1,3 +1,2 @@\n keep\n-gone\n tail"

        assert self._numbers(diff) == ["1", "2", "2"]

    def test_context_is_highlighted_from_the_after_source(self) -> None:
        """Lookup side must follow the numbering, or highlighting silently drops.

        Context rows are numbered in the new file, so they have to be read from
        `after`. Reading them from `before` at a new-file number lands on the
        wrong line, the drift check rejects it, and the row renders plain.
        """
        before = "def f():\n    return 1\nend = True\n"
        after = "def f():\n    # added a\n    # added b\n    return 1\nend = True\n"
        widgets = [
            w
            for w in compose_diff_lines(
                self._INSERTION, 100, path="m.py", before=before, after=after
            )
            if isinstance(w, Static)
        ]

        context = next(w for w in widgets if "end = True" in _plain(w))
        spans = cast("Content", context.render()).spans
        assert spans, "a context row lost its syntax highlighting"

    def test_prefix_sizing_covers_the_context_rows(self) -> None:
        """`highlight_source_prefixes` must size `after` for context, not just added."""
        before = "def f():\n    return 1\nend = True\n"
        after = "def f():\n    # added a\n    # added b\n    return 1\nend = True\n"

        _, after_prefix = highlight_source_prefixes(self._INSERTION, before, after)

        assert "end = True" in after_prefix, (
            "the last context row is outside the lexed prefix"
        )
