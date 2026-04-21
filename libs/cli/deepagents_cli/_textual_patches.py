"""Monkey-patches for Textual bugs that block key-handling parity with other TUIs.

Each patch is scoped tightly so we defer to upstream behavior for every
case that isn't known-broken. Remove a patch as soon as its upstream fix
lands and the Textual pin is bumped.

Importing this module applies the patches as a side effect. Import it
once, early, from a module that always runs before `App()` is
instantiated (see `app.py`). If the Textual internals we patch have
been renamed or removed (e.g. after a minor version bump), the module
logs a warning, writes the same message to stderr, and no-ops instead
of crashing the CLI. Check `PATCH_APPLIED` after import if a later
caller needs to surface a user-visible warning (e.g. a toast on app
mount).
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

PATCH_APPLIED: bool = False
"""Whether the `XTermParser._sequence_to_key_events` monkey-patch installed.

`False` when either the import-time lookup or the attribute assignment
failed (e.g. after a Textual pin bump renamed the private API). A
warning is also printed to stderr before `App()` starts so the
failure is visible to anyone launching the CLI from a shell; the flag
is exposed for callers that want to surface a follow-up signal (e.g.
an app-mount toast).
"""


def _report_patch_failure(reason: str) -> None:
    """Log and stderr-print a patch-skipped warning.

    The Textual app takes over the terminal before Python's "last
    resort" handler can surface a `logger.warning` to the user, so we
    also print to stderr at import time (before `App()` is
    instantiated). The stderr message lands above the alternate-screen
    switch and is visible to anyone running the CLI from a shell.
    """
    message = (
        f"deepagents-cli: Textual keyboard parser patch skipped ({reason}). "
        "Shift+Enter via VSCode sendSequence may not insert a newline; "
        "check whether the Textual pin was bumped across a private-API rename."
    )
    logger.warning("%s", message)
    print(message, file=sys.stderr)  # noqa: T201


# ---------------------------------------------------------------------------
# Patch: preserve the `alt` modifier when a single-byte sequence maps to a
# tuple in `ANSI_SEQUENCES_KEYS`.
#
# Upstream `XTermParser._sequence_to_key_events` drops the `alt` flag on the
# tuple-branch fast path, so `ESC + <byte>` sequences that VSCode's
# `sendSequence` binding emits (e.g. `\x1b\r` for shift+enter, when the
# user has `"text": "\r"` in `keybindings.json`) are dispatched as
# plain `Key("enter")` instead of `Key("alt+enter")`.
#
# Affects all 35 single-byte keys mapped to tuples in `ANSI_SEQUENCES_KEYS`
# — Enter, Space, Backspace, Tab, Escape (and Shift+Escape), plus every
# Ctrl+<letter|@|\|]|^|_>. Only breaks for terminals that fall back to
# legacy ESC-prefix encoding (VSCode's integrated terminal when
# `sendSequence` bypasses xterm.js, GNOME Console, etc.) — kitty-protocol
# terminals (Ghostty, kitty, recent iTerm2) send `CSI 13;<modifier>u`
# (e.g. `;2u` for shift+enter, `;3u` for alt+enter) and go through the
# extended-key regex path, which is correct.
#
# crossterm (used by codex, claude-code) and the Node TTY parsers
# (used by opencode) both decode `ESC + <byte>` as `Alt+<byte>`
# unconditionally, which is why those tools accept the same VSCode
# binding out of the box.
#
# A secondary symptom of the same family of bugs is a ~100 ms input lag
# on these sequences: when the parser receives the full 2-byte sequence
# on its first pass it finds no match and then blocks for Textual's
# escape-delay constant on the off chance more bytes arrive, before
# finally reissuing as legacy alt keys. A 2-char fast path short-circuits
# the match so the parser breaks out of its inner loop immediately.
#
# Intentional semantic change worth calling out: `\x1b\x1b` (ESC ESC) now
# dispatches as `alt+escape` with no delay. Upstream waits the full
# escape-delay before giving up. This matches crossterm/Node TTY and is
# expected, not a regression.
#
# Upstream tracking (check before removing this file):
#   - https://github.com/Textualize/textual/issues/6378
# Remove this patch once the upstream fix merges *and* the lag is
# separately resolved, then bump the Textual pin.
# ---------------------------------------------------------------------------

_ESC_PREFIX_LEN = 2
"""Length of an `ESC + <byte>` sequence: one escape byte plus the payload
byte whose `ANSI_SEQUENCES_KEYS` mapping we look up for the fast path."""


try:
    from textual import events
    from textual._ansi_sequences import (
        ANSI_SEQUENCES_KEYS,  # noqa: PLC2701
        IGNORE_SEQUENCE,  # noqa: PLC2701
    )
    from textual._xterm_parser import XTermParser  # noqa: PLC2701

    _original_sequence_to_key_events = XTermParser._sequence_to_key_events
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    _report_patch_failure(str(exc))
else:

    def _emit_alt_keys(keys: tuple, character: str | None) -> Iterable[events.Key]:
        """Yield Key events with an `alt+` prefix applied to each tuple member.

        `keys` is a tuple of `textual.keys.Keys` members; each exposes a
        `.value` string.
        """
        for key in keys:
            yield events.Key(f"alt+{key.value}", character)

    def _sequence_to_key_events_with_alt(
        self: XTermParser, sequence: str, alt: bool = False
    ) -> Iterable[events.Key]:
        r"""Drop-in replacement for `XTermParser._sequence_to_key_events`.

        Two scoped interventions, both about the `ESC + <byte>` legacy
        encoding of alt-modified keys:

        - **2-char fast path** (`alt=False`, `sequence == "\x1b<byte>"`):
            short-circuit with `alt+<key>` so the parser breaks out of its
            inner loop on the first pass instead of stalling for Textual's
            escape-delay constant waiting for more bytes.
        - **Reissue path** (`alt=True`, single-byte sequence, tuple
            mapping): preserve the `alt` flag that the upstream tuple branch
            silently drops (Textualize/textual#6378). Matches crossterm /
            Node TTY behavior.

        Defers to the original method for every other case.

        Yields:
            Key events, with the `alt` modifier preserved for the
            known-buggy path.
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

    try:
        XTermParser._sequence_to_key_events = _sequence_to_key_events_with_alt  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        # Guards against future upstream hardening (`__slots__`, frozen
        # class, descriptor hooks) that would reject attribute writes on
        # `XTermParser` even though the imports succeeded.
        _report_patch_failure(f"assignment rejected: {exc}")
    else:
        PATCH_APPLIED = True
