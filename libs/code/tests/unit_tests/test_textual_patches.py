"""Tests for the Textual keyboard parser monkey-patch.

See `_textual_patches.py` and Textualize/textual#6378.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
from textual._time import get_time
from textual._xterm_parser import XTermParser
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.geometry import Offset
from textual.widgets import Markdown, Static

from deepagents_code import _textual_patches  # triggers patch


def _keys_for(sequence: str, *, alt: bool) -> list[tuple[str, str | None]]:
    parser = XTermParser.__new__(XTermParser)
    return [
        (event.key, event.character)
        for event in parser._sequence_to_key_events(sequence, alt=alt)
    ]


class SelectableTextApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("alpha beta gamma", id="msg")


class SelectableMarkdownApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Markdown("alpha **beta** gamma", id="msg")


class SelectableHistoryApp(App[None]):
    def compose(self) -> ComposeResult:
        with Vertical(id="history"):
            yield Static("first message", id="first")
            yield Static("second message", id="second")


class TestPatchedWordSelection:
    async def test_double_click_selects_word_not_entire_widget(self) -> None:
        async with SelectableTextApp().run_test() as pilot:
            await pilot.double_click("#msg", offset=(7, 0))

            assert pilot.app.screen.get_selected_text() == "beta"

    async def test_double_click_drag_expands_to_word_boundaries(self) -> None:
        async with SelectableTextApp().run_test() as pilot:
            widget = pilot.app.query_one("#msg", Static)
            start = widget.content_region.offset + Offset(1, 0)
            pilot.app._click_chain_last_offset = start
            pilot.app._click_chain_last_time = get_time()

            await pilot.mouse_down("#msg", offset=(1, 0))
            await pilot.mouse_up("#msg", offset=(13, 0))

            assert pilot.app.screen.get_selected_text() == "alpha beta gamma"

    async def test_double_click_falls_back_for_non_text_renderable(self) -> None:
        async with SelectableMarkdownApp().run_test() as pilot:
            await pilot.double_click("#msg", offset=(7, 0))

            assert pilot.app.screen.get_selected_text() is not None

    async def test_triple_click_selects_clicked_widget_not_history(self) -> None:
        async with SelectableHistoryApp().run_test() as pilot:
            await pilot.triple_click("#second", offset=(1, 0))

            assert pilot.app.screen.get_selected_text() == "second message"


class TestPatchedSequenceToKeyEvents:
    r"""Targeted coverage of the two interventions in the shim."""

    def test_reissue_path_preserves_alt_for_enter(self) -> None:
        r"""Correctness fix: `\r` with `alt=True` must emit `alt+enter`.

        Without the patch, the tuple branch in upstream drops `alt` and
        VSCode `sendSequence` shift+enter arrives as bare `enter`.
        """
        assert _keys_for("\r", alt=True) == [("alt+enter", "\r")]

    def test_fast_path_decodes_esc_cr_as_alt_enter(self) -> None:
        r"""Fast path: `\x1b\r` with `alt=False` short-circuits to `alt+enter`.

        Without the fast path, upstream stalls for ~100 ms waiting for
        more bytes before reissuing.
        """
        assert _keys_for("\x1b\r", alt=False) == [("alt+enter", None)]

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

    def test_fast_path_falls_through_when_inner_byte_unmapped(self) -> None:
        r"""`\x1b<printable>` must bypass the fast path and defer to upstream.

        Pins the `isinstance(inner, tuple)` guard — the `.get()` returns
        `None` for unmapped bytes, which must not be treated as an alt key.
        """
        assert _keys_for("\x1bZ", alt=False) == []

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


def test_app_imports_textual_patches_for_side_effect() -> None:
    """`app.py` must import `_textual_patches` for the patch to install.

    Direct-import tests would pass even if the side-effect import were
    removed, so silently breaking shift+enter for VSCode `sendSequence`
    users. A static AST check closes that gap without spawning a subprocess.
    """
    spec = importlib.util.find_spec("deepagents_code.app")
    assert spec is not None
    assert spec.origin is not None

    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "deepagents_code"
        for alias in node.names
    }
    assert "_textual_patches" in imported, (
        "deepagents_code/app.py must import `_textual_patches` as a side "
        "effect; removing it silently breaks shift+enter via VSCode "
        "sendSequence. See `_textual_patches.py` for context."
    )
