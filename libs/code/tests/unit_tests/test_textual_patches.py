"""Tests for the Textual keyboard parser monkey-patch.

See `_textual_patches.py` and Textualize/textual#6378.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

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
