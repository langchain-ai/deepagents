"""Tests for Textual parser monkey-patches.

Import side effect: `deepagents_cli._textual_patches` replaces
`XTermParser._sequence_to_key_events` at import time. These tests
exercise the patched method directly.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from textual._xterm_parser import XTermParser

# Import triggers the monkey-patch.
from deepagents_cli import _textual_patches


def _keys_for(sequence: str, *, alt: bool) -> list[tuple[str, str | None]]:
    """Return `(key, character)` tuples produced by the patched method."""
    parser = XTermParser.__new__(XTermParser)
    return [
        (event.key, event.character)
        for event in parser._sequence_to_key_events(sequence, alt=alt)
    ]


class TestPatchedSequenceToKeyEvents:
    r"""Tests for the alt-preserving parser patch.

    Background: upstream's tuple-branch fast path drops the `alt` flag,
    so `ESC + <byte>` sequences emitted by VSCode's `sendSequence`
    binding for `shift+enter` (e.g. `\x1b\r`) get dispatched as plain
    `enter` instead of `alt+enter`. The patch only alters that path.
    """

    def test_preserves_alt_for_tuple_mapped_enter(self) -> None:
        r"""`\r` maps to `(Keys.Enter,)` — patched path must emit `alt+enter`."""
        assert _keys_for("\r", alt=True) == [("alt+enter", "\r")]

    def test_preserves_alt_for_tuple_mapped_tab(self) -> None:
        r"""`\t` maps to `(Keys.Tab,)` — same bug, same fix."""
        assert _keys_for("\t", alt=True) == [("alt+tab", "\t")]

    def test_preserves_alt_for_tuple_mapped_backspace(self) -> None:
        r"""Backspace (`\x7f`) also maps to a tuple upstream."""
        keys = _keys_for("\x7f", alt=True)
        assert keys == [("alt+backspace", "\x7f")]

    def test_non_alt_tuple_path_unchanged(self) -> None:
        """Plain (non-alt) tuple mappings must still produce the bare key."""
        assert _keys_for("\r", alt=False) == [("enter", "\r")]

    def test_single_char_alt_fallback_unchanged(self) -> None:
        """Printable alt combos already worked upstream — the patch defers."""
        assert _keys_for("a", alt=True) == [("alt+a", "a")]

    def test_extended_key_sequence_unchanged(self) -> None:
        """Kitty-encoded shift+enter (`CSI 13;2u`) goes through the original path.

        The patch only intercepts single-byte tuple-mapped sequences;
        CSI sequences never reach the tuple branch, so they must still
        be decoded by the unmodified extended-key regex.
        """
        assert _keys_for("\x1b[13;2u", alt=False) == [("shift+enter", None)]


class TestTwoCharFastPath:
    r"""Tests for the 2-char `\x1b<byte>` fast path.

    Without the fast path, the parser waits ~100 ms (`ESCAPE_DELAY`)
    for more bytes after seeing `\x1b\r` before giving up and reissuing.
    The fast path short-circuits directly to `alt+<key>` so the parser
    breaks out of its loop immediately.
    """

    def test_esc_cr_yields_alt_enter_without_alt_flag(self) -> None:
        r"""`\x1b\r` on the first-pass path (alt=False, 2 chars) emits `alt+enter`."""
        assert _keys_for("\x1b\r", alt=False) == [("alt+enter", None)]

    def test_esc_lf_yields_alt_ctrl_j(self) -> None:
        r"""`\x1b\n` produces `alt+ctrl+j`.

        LF maps to `(Keys.ControlJ,)` in `ANSI_SEQUENCES_KEYS`, so the
        fast path prepends `alt+` to the upstream single-byte mapping.
        """
        assert _keys_for("\x1b\n", alt=False) == [("alt+ctrl+j", None)]

    def test_esc_tab_yields_alt_tab(self) -> None:
        r"""Every tuple-mapped byte gets the fast path, not just Enter."""
        assert _keys_for("\x1b\t", alt=False) == [("alt+tab", None)]

    def test_esc_backspace_yields_alt_backspace(self) -> None:
        r"""`\x1b\x7f` — the other common ESC-prefixed single byte."""
        assert _keys_for("\x1b\x7f", alt=False) == [("alt+backspace", None)]

    def test_esc_printable_falls_through(self) -> None:
        r"""`\x1b<printable>` (e.g. `\x1ba`) is handled by the original path.

        Printable bytes aren't in `ANSI_SEQUENCES_KEYS` as tuples, so the
        fast path must defer — the upstream single-char branch already
        honors `alt` for these.
        """
        assert _keys_for("\x1ba", alt=False) == []

    def test_longer_sequence_ignored_by_fast_path(self) -> None:
        r"""`\x1b[A` (cursor up) must not be misinterpreted as alt-anything."""
        assert _keys_for("\x1b[A", alt=False) == [("up", None)]

    def test_esc_esc_yields_alt_escape(self) -> None:
        r"""`\x1b\x1b` is the intentional semantic change: immediate `alt+escape`.

        Upstream waits for the full escape-delay before giving up; the
        fast path short-circuits to `alt+escape` with zero latency,
        matching crossterm / Node TTY. Flagged as load-bearing in the
        patch module docstring.
        """
        assert _keys_for("\x1b\x1b", alt=False) == [("alt+escape", None)]


class TestPatchInstalled:
    """Guardrail against silent upstream drift.

    The patch is applied as an import-time side effect. If a Textual
    minor bump renames the private API we patch, our defensive
    `try/except` swallows the failure — tests elsewhere still pass
    because they call the patched function directly. This assertion
    catches that situation loudly.
    """

    def test_xterm_parser_method_points_at_our_replacement(self) -> None:
        """`XTermParser._sequence_to_key_events` must be our patched callable."""
        # Access via `getattr` because the name is only defined when the
        # defensive `try/except ImportError` at module load succeeded;
        # ty can't narrow that statically.
        patched = getattr(_textual_patches, "_sequence_to_key_events_with_alt", None)
        assert patched is not None, "patch module failed to bind replacement"
        assert XTermParser._sequence_to_key_events is patched

    def test_patch_applied_flag_is_true_after_import(self) -> None:
        """`PATCH_APPLIED` is the programmatic signal callers check on mount."""
        assert _textual_patches.PATCH_APPLIED is True


class TestImportSiteIntegration:
    """Guardrails that the patch is actually installed via `app.py`.

    The unit tests above import `_textual_patches` themselves, so they
    cannot detect a regression where the side-effect import is removed
    from `app.py`. This class uses a subprocess with a fresh interpreter
    so it only sees the patch if some module in `deepagents_cli.app`'s
    import graph triggers it.
    """

    def test_importing_app_installs_the_patch(self) -> None:
        r"""`import deepagents_cli.app` must leave `XTermParser` patched.

        If a refactor drops the `_textual_patches` import from `app.py`,
        this test fails loudly. Runs in a subprocess to bypass the
        current interpreter's already-patched state.
        """
        script = textwrap.dedent(
            """
            import sys
            import deepagents_cli.app  # noqa: F401
            from textual._xterm_parser import XTermParser

            assert (
                "deepagents_cli._textual_patches" in sys.modules
            ), "patch module not imported by app.py's import graph"
            assert (
                XTermParser._sequence_to_key_events.__name__
                == "_sequence_to_key_events_with_alt"
            ), "XTermParser._sequence_to_key_events is not our replacement"
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"subprocess assertion failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
