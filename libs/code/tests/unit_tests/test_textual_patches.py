"""Tests for the Textual keyboard parser monkey-patch.

See `_textual_patches.py` and Textualize/textual#6378.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from textual import events
from textual._time import get_time
from textual._xterm_parser import XTermParser
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Markdown, Static

from deepagents_code import _textual_patches  # triggers patch
from deepagents_code.tui.widgets.diff import _DiffRowStatic


def _keys_for(sequence: str, *, alt: bool) -> list[tuple[str, str | None]]:
    parser = XTermParser.__new__(XTermParser)
    return [
        (event.key, event.character)
        for event in parser._sequence_to_key_events(sequence, alt=alt)
    ]


class SelectableTextApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("alpha beta gamma", id="msg")


def test_ascii_mode_replaces_every_textual_border_glyph() -> None:
    code = (
        "import deepagents_code._textual_patches\n"
        "from textual._border import BORDER_CHARS, INVISIBLE_EDGE_TYPES\n"
        "ascii_border = BORDER_CHARS['ascii']\n"
        "assert all(border == ascii_border for name, border in "
        "BORDER_CHARS.items() if name not in {*INVISIBLE_EDGE_TYPES, 'blank'})\n"
        "assert all(not character.strip() for edge in BORDER_CHARS['blank'] "
        "for character in edge)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "DEEPAGENTS_CODE_UI_CHARSET_MODE": "ascii"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class SelectableDiffApp(App[None]):
    def compose(self) -> ComposeResult:
        yield _DiffRowStatic(
            Content(" 1 - removed word"), prefix_len=5, id="diff-before"
        )
        yield _DiffRowStatic(Content(" 1 + added word"), prefix_len=5, id="diff-row")


class SelectableMarkdownApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Markdown("alpha **beta** gamma", id="msg")


class SelectableHistoryApp(App[None]):
    def compose(self) -> ComposeResult:
        with Vertical(id="history"):
            yield Static("first message", id="first")
            yield Static("second message", id="second")


class SelectableScrollApp(App[None]):
    CSS = "VerticalScroll { height: 8; }"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="history"):
            for index in range(1, 31):
                yield Static(f"line{index:02d} content", id=f"row{index}")


class TestPatchedWordSelection:
    async def test_shift_click_extends_drag_selection_from_anchor(self) -> None:
        async with SelectableTextApp().run_test() as pilot:
            await pilot.mouse_down("#msg", offset=(0, 0))
            await pilot.mouse_up("#msg", offset=(4, 0))
            assert pilot.app.screen.get_selected_text() == "alpha"

            await pilot.click("#msg", offset=(11, 0), shift=True)

            assert pilot.app.screen.get_selected_text() == "alpha beta g"

    async def test_shift_click_preserves_backward_drag_anchor(self) -> None:
        async with SelectableTextApp().run_test() as pilot:
            await pilot.mouse_down("#msg", offset=(15, 0))
            await pilot.mouse_up("#msg", offset=(11, 0))
            assert pilot.app.screen.get_selected_text() == "gamma"

            await pilot.click("#msg", offset=(0, 0), shift=True)

            assert pilot.app.screen.get_selected_text() == "alpha beta gamma"

    async def test_shift_click_rejects_detached_markdown_anchor(self) -> None:
        async with SelectableMarkdownApp().run_test() as pilot:
            screen = pilot.app.screen
            document = pilot.app.query_one("#msg", Markdown)
            await pilot.mouse_down("#msg", offset=(15, 0))
            await pilot.mouse_up("#msg", offset=(11, 0))
            select_state = screen._select_state
            assert select_state is not None
            anchor_widget = select_state.start.content_widget
            assert anchor_widget is not None

            await document.update("replacement text")
            assert not anchor_widget.is_attached
            await pilot.click("#msg", offset=(0, 0), shift=True)

            assert screen.get_selected_text() is None

    async def test_shift_click_extends_from_anchor_after_scroll(self) -> None:
        async with SelectableScrollApp().run_test(size=(40, 8)) as pilot:
            await pilot.mouse_down("#row1", offset=(0, 0))
            await pilot.mouse_up("#row2", offset=(6, 0))
            history = pilot.app.query_one("#history", VerticalScroll)
            history.scroll_to(y=10, animate=False)
            await pilot.pause()

            await pilot.click("#row14", offset=(6, 0), shift=True)

            selected = pilot.app.screen.get_selected_text()
            assert selected is not None
            assert selected.startswith("line01 content")
            assert selected.endswith("line14")

    async def test_shift_click_ignores_unmodified_click(self) -> None:
        async with SelectableTextApp().run_test() as pilot:
            await pilot.mouse_down("#msg", offset=(0, 0))
            await pilot.mouse_up("#msg", offset=(4, 0))
            assert pilot.app.screen.get_selected_text() == "alpha"

            await pilot.click("#msg", offset=(11, 0))

            assert pilot.app.screen.get_selected_text() is None

    async def test_shift_click_extends_selection_across_widgets(self) -> None:
        async with SelectableHistoryApp().run_test() as pilot:
            await pilot.mouse_down("#first", offset=(6, 0))
            await pilot.mouse_up("#first", offset=(12, 0))
            assert pilot.app.screen.get_selected_text() == "message"

            await pilot.click("#second", offset=(6, 0), shift=True)

            assert pilot.app.screen.get_selected_text() == "message\nsecond"

    async def test_shift_click_without_selection_remains_unselected(self) -> None:
        async with SelectableTextApp().run_test() as pilot:
            await pilot.click("#msg", offset=(7, 0), shift=True)

            assert pilot.app.screen.get_selected_text() is None


class TestDetachedHitGuard:
    """Coverage of the Textualize/textual#6643 crash guard."""

    async def test_mouse_down_on_detached_widget_does_not_crash(self) -> None:
        """A press on a widget pruned since the last repaint must be ignored.

        `Markdown.update` — which `MarkdownStream` runs on every streaming
        assistant message — detaches its old blocks while the compositor still
        reports them as visible. `_detach` is exactly what Textual calls during
        that prune, so calling it directly pins the race window deterministically
        instead of spinning the event loop until it happens to be observed.
        Without the guard, `Screen._forward_event` raises `AttributeError` on the
        detached widget's `None` parent and takes the whole app down.
        """
        async with SelectableMarkdownApp().run_test() as pilot:
            screen = pilot.app.screen
            document = pilot.app.query_one("#msg", Markdown)
            paragraph = document.query("*").first()
            x = paragraph.region.x + 1
            y = paragraph.region.y
            assert screen._compositor.get_widget_and_offset_at(x, y)[0] is paragraph

            paragraph._detach()
            try:
                assert screen.get_widget_and_offset_at(x, y) == (None, None)
                screen._forward_event(
                    events.MouseDown(None, x, y, 0, 0, 1, False, False, False)
                )

                assert screen._select_state is None
            finally:
                # Textual's own teardown asserts every widget still has a
                # parent, so hand the simulated prune victim back to the DOM.
                paragraph._attach(document)

    async def test_attached_widget_hit_is_still_reported(self) -> None:
        """The guard must only drop detached hits, not live ones."""
        async with SelectableTextApp().run_test() as pilot:
            widget = pilot.app.query_one("#msg", Static)
            offset = widget.content_region.offset + Offset(2, 0)

            hit, hit_offset = pilot.app.screen.get_widget_and_offset_at(*offset)

            assert hit is widget
            assert hit_offset == Offset(2, 0)


def test_missing_shift_selection_internals_does_not_break_import() -> None:
    """Missing private classes must skip only the best-effort Shift patch."""
    code = (
        "import textual.selection\n"
        "del textual.selection.SelectEnd\n"
        "del textual.selection.SelectState\n"
        "import deepagents_code._textual_patches\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class TestPatchedSequenceToKeyEvents:
    r"""Targeted coverage of the two interventions in the shim."""

    def test_kitty_extended_key_sequence_unchanged(self) -> None:
        r"""Regression guard: kitty `CSI 13;2u` must still decode natively.

        The patch only intercepts single-byte tuple mappings; extended
        key sequences are handled by the unmodified upstream path.
        """
        assert _keys_for("\x1b[13;2u", alt=False) == [("shift+enter", None)]

    def test_fast_path_double_escape_yields_alt_escape(self) -> None:
        r"""Pin the documented semantic: `\x1b\x1b` emits `alt+escape` immediately.

        Upstream Textual waits the full escape-delay before giving up; the
        fast path short-circuits with zero latency. Any refactor that breaks
        this should fail loudly rather than silently reverting the behavior.
        """
        assert _keys_for("\x1b\x1b", alt=False) == [("alt+escape", None)]

    @pytest.mark.parametrize(
        ("sequence", "key"),
        [
            # Plain press, no associated text.
            ("\x1b[57358u", "caps_lock"),
            # Conformant flags-25 form: modifier + associated text.
            ("\x1b[57358;1;65u", "caps_lock"),
            # Lock bit set in the modifier mask.
            ("\x1b[57358;65;65u", "caps_lock"),
            # Other modifier bits set alongside the lock key.
            ("\x1b[57358;64;65u", "caps_lock"),
            # Alternate-key sub-field (iTerm2): `unicode:shifted`.
            ("\x1b[57358:65;1;65u", "caps_lock"),
            # Event-type sub-field on the modifier field.
            ("\x1b[57358;1:1;65u", "caps_lock"),
            # Num Lock and Scroll Lock use the same encoding family.
            ("\x1b[57360;1;65u", "num_lock"),
            ("\x1b[57359;1;65u", "scroll_lock"),
        ],
    )
    def test_kitty_lock_keys_never_carry_text(self, sequence: str, key: str) -> None:
        r"""Lock keys must decode to a single character-less event.

        Under the kitty protocol with associated-text reporting, terminals
        (notably iTerm2) encode Caps Lock with the letter the next key would
        have produced. Without the patch Textual either types that letter or,
        when `:` sub-fields are present, leaks the raw sequence byte by byte.
        The patch collapses every lock-key sequence to a text-free event.
        """
        assert _keys_for(sequence, alt=False) == [(key, None)]

    def test_kitty_subfield_strip_preserves_normal_keys(self) -> None:
        r"""Alternate-key sub-fields on text keys still decode to the key.

        `CSI 97:65;1;65u` is the `a` key with shifted alternate `A`; only the
        primary code point and associated text matter to Textual. This guards
        against the sub-field strip swallowing real characters.
        """
        assert _keys_for("\x1b[97:65;1;65u", alt=False) == [("A", "A")]

    @pytest.mark.parametrize(
        ("sequence", "key"),
        [
            # `~`-terminated sequence (Delete) with an event-type `:` sub-field.
            ("\x1b[3:3~", "delete"),
            # Cursor key (letter terminator) with a `:` sub-field on the
            # modifier field.
            ("\x1b[1;5:1C", "ctrl+right"),
        ],
    )
    def test_kitty_subfield_strip_handles_non_u_terminators(
        self, sequence: str, key: str
    ) -> None:
        r"""Sub-field stripping covers `~` and letter terminators, not just `u`.

        `_KITTY_SUBFIELD_KEY` matches terminators `[u~ABCDEFHPQRS]`, so F-keys,
        arrows, and Insert/Delete carrying `:` sub-fields are normalized rather
        than leaked byte by byte. Every other test ends in `u`; this pins the
        non-`u` paths against a regex regression that would reintroduce the
        very byte-by-byte leak this patch exists to fix.
        """
        assert _keys_for(sequence, alt=False) == [(key, None)]

    @pytest.mark.parametrize(
        "sequence",
        [
            # iTerm2 Caps Lock toggle: bare upper-case code point, no fields.
            "\x1b[65u",
            # With an explicit "no modifiers" field (value 1).
            "\x1b[65;1u",
            # Upper-case letters across the ASCII range.
            "\x1b[90u",
            # Caps-lock bit present in the modifier mask, still no text.
            "\x1b[67;65u",
        ],
    )
    def test_iterm_caps_lock_toggle_inserts_nothing(self, sequence: str) -> None:
        r"""iTerm2's bare upper-case Caps Lock report must not type.

        iTerm2 encodes the Caps Lock toggle as the upper-case letter that
        would be produced next (`CSI 65 u` → 'A') rather than the kitty
        functional code, with no associated-text field. The kitty spec never
        emits an upper-case primary code point for a real press, so the patch
        treats it as the lock toggle and drops the character.
        """
        assert _keys_for(sequence, alt=False) == [("caps_lock", None)]

    @pytest.mark.parametrize(
        ("sequence", "expected"),
        [
            # Lower-case letters are always real text.
            ("\x1b[97u", [("a", "a")]),
            # Shift+A reported as lower-case primary + shift modifier.
            ("\x1b[97;2u", [("shift+a", None)]),
            # Upper-case primary WITH associated text is a real character
            # (e.g. caps-on typing): the text field disambiguates it.
            ("\x1b[65;1;65u", [("A", "A")]),
            ("\x1b[67;65;67u", [("C", "C")]),
            # Upper-case primary with a real modifier (ctrl) and no text is a
            # genuine press — the `_REAL_MODIFIER_MASK` guard must not drop it.
            ("\x1b[65;5u", [("ctrl+A", None)]),
        ],
    )
    def test_iterm_caps_lock_guard_preserves_real_keys(
        self, sequence: str, expected: list[tuple[str, str | None]]
    ) -> None:
        r"""The Caps Lock guard must not swallow genuine key presses.

        Only a bare upper-case primary code point with no real modifiers and
        no associated text is treated as the toggle; everything else decodes
        normally.
        """
        assert _keys_for(sequence, alt=False) == expected


class TestGutterClampWatcher:
    """Diff gutters stay outside selections from every selection-map update."""
