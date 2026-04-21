"""Monkey-patches for Textual bugs that block key-handling parity with other TUIs.

Each patch is scoped tightly so we defer to upstream behaviour for every
case that isn't known-broken. Remove a patch as soon as its upstream fix
lands and the Textual pin is bumped.

Importing this module applies the patches as a side effect. Import it
once, early, from a module that always runs before `App()` is
instantiated (see `app.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual._ansi_sequences import (
    ANSI_SEQUENCES_KEYS,  # noqa: PLC2701
    IGNORE_SEQUENCE,  # noqa: PLC2701
)
from textual._xterm_parser import XTermParser  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Patch: preserve the `alt` modifier when a single-byte sequence maps to a
# tuple in `ANSI_SEQUENCES_KEYS`.
#
# Upstream `XTermParser._sequence_to_key_events` drops the `alt` flag on the
# tuple-branch fast path, so `ESC + <byte>` sequences that VSCode's
# `sendSequence` binding emits (e.g. `\x1b\r` for shift+enter, when the
# user has `"text": "\r"` in `keybindings.json`) are dispatched as
# plain `Key("enter")` instead of `Key("alt+enter")`.
#
# Affects ~32 single-byte keys that map to tuples: Enter, Space,
# Backspace, Tab, and all Ctrl+letter. Only breaks for terminals that
# fall back to legacy ESC-prefix encoding (VSCode's integrated terminal
# when `sendSequence` bypasses xterm.js, GNOME Console, etc.) —
# kitty-protocol terminals (Ghostty, kitty, recent iTerm2) send
# `CSI 13;3u` and go through the extended-key regex path, which is
# correct.
#
# crossterm (used by codex, claude-code) and the Node TTY parsers
# (used by opencode) both decode `ESC + <byte>` as `Alt+<byte>`
# unconditionally, which is why those tools accept the same VSCode
# binding out of the box.
#
# A secondary symptom of the same family of bugs is a ~100 ms input
# lag on these sequences: the parser calls `_sequence_to_key_events`
# with the full 2-byte sequence `\x1b\r` on the first pass, gets back
# an empty iterator (neither the extended-key regex, `ANSI_SEQUENCES_KEYS`,
# nor the single-char fallback matches), then waits
# `constants.ESCAPE_DELAY` (100 ms) on the off chance more bytes arrive
# before finally reissuing via `reissue_sequence_as_keys`. Adding a
# 2-char fast path lets the parser break out of its inner loop on
# first pass, eliminating the lag.
#
# Upstream tracking:
#   - Textualize/textual#6378 (issue, 2026-02-18)
#   - Textualize/textual#6379 (open fix PR, +6/-1 lines — alt-preservation only;
#     does not address the ESCAPE_DELAY lag on 2-char ESC-prefix sequences)
# Remove this patch once #6379 merges *and* the lag is separately
# resolved, then bump the Textual pin.
# ---------------------------------------------------------------------------

_original_sequence_to_key_events = XTermParser._sequence_to_key_events


def _emit_alt_keys(keys: tuple, character: str | None) -> Iterable[events.Key]:
    """Yield Key events with an `alt+` prefix applied to each tuple member.

    `keys` is a tuple of `textual.keys.Keys` members; each exposes a
    `.value` string.
    """
    for key in keys:
        yield events.Key(f"alt+{key.value}", character)


_ESC_PREFIX_LEN = 2
"""Length of the `ESC + <byte>` legacy alt-modifier encoding."""


def _sequence_to_key_events_with_alt(
    self: XTermParser, sequence: str, alt: bool = False
) -> Iterable[events.Key]:
    r"""Drop-in replacement for `XTermParser._sequence_to_key_events`.

    Two scoped interventions, both about the `ESC + <byte>` legacy
    encoding of alt-modified keys:

    - **2-char fast path** (`alt=False`, `sequence == "\x1b<byte>"`):
      short-circuit with `alt+<key>` so the parser breaks out of its
      inner loop on the first pass instead of stalling for
      `constants.ESCAPE_DELAY` (100 ms) waiting for more bytes.
    - **Reissue path** (`alt=True`, single-byte sequence, tuple
      mapping): preserve the `alt` flag that the upstream tuple branch
      silently drops (#6378). Matches crossterm / Node TTY behaviour.

    Defers to the original method for every other case.

    Yields:
        Key events, with the `alt` modifier preserved for the known-buggy path.
    """
    if not alt and len(sequence) == _ESC_PREFIX_LEN and sequence[0] == "\x1b":
        inner_keys = ANSI_SEQUENCES_KEYS.get(sequence[1])
        if inner_keys is not IGNORE_SEQUENCE and isinstance(inner_keys, tuple):
            yield from _emit_alt_keys(inner_keys, None)
            return
    if alt:
        keys = ANSI_SEQUENCES_KEYS.get(sequence)
        if keys is not IGNORE_SEQUENCE and isinstance(keys, tuple):
            character = sequence if len(sequence) == 1 else None
            yield from _emit_alt_keys(keys, character)
            return
    yield from _original_sequence_to_key_events(self, sequence, alt=alt)


XTermParser._sequence_to_key_events = _sequence_to_key_events_with_alt  # ty: ignore[invalid-assignment]
