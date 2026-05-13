r"""Preserve the `alt` modifier on legacy `ESC + <byte>` sequences.

Upstream `XTermParser._sequence_to_key_events` drops the `alt` flag on
the tuple-branch fast path, so VSCode's `sendSequence` shift+enter
binding (which writes `\x1b\r` to the PTY) arrives as bare `enter`
instead of `alt+enter`. Tracked in Textualize/textual#6378. Remove this
file and the Textual pin comment in `pyproject.toml` when that lands.

Imported for side effect from `app.py` before any `App()` is created.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_ESC_PREFIX_LEN = 2

try:
    from textual import events
    from textual._ansi_sequences import (
        ANSI_SEQUENCES_KEYS,  # noqa: PLC2701
        IGNORE_SEQUENCE,  # noqa: PLC2701
    )
    from textual._xterm_parser import XTermParser  # noqa: PLC2701

    _original = XTermParser._sequence_to_key_events
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    logger.warning("Textual keyboard parser patch skipped: %s", exc)
else:

    def _emit_alt(keys: tuple, character: str | None) -> Iterable[events.Key]:
        for key in keys:
            yield events.Key(f"alt+{key.value}", character)

    def _sequence_to_key_events_with_alt(
        self: XTermParser, sequence: str, alt: bool = False
    ) -> Iterable[events.Key]:
        # Fast path: \x1b<byte> on first pass. Short-circuits the ~100 ms
        # escape-delay wait when both bytes arrive together. Semantic side
        # effect: \x1b\x1b dispatches as `alt+escape` with no delay, matching
        # crossterm and Node TTY.
        if not alt and len(sequence) == _ESC_PREFIX_LEN and sequence[0] == "\x1b":
            inner = ANSI_SEQUENCES_KEYS.get(sequence[1])
            if inner is not IGNORE_SEQUENCE and isinstance(inner, tuple):
                yield from _emit_alt(inner, None)
                return
        # Correctness fix (Textualize/textual#6378): preserve `alt` on the
        # reissue path for single-byte tuple mappings.
        if alt:
            keys = ANSI_SEQUENCES_KEYS.get(sequence)
            if keys is not IGNORE_SEQUENCE and isinstance(keys, tuple):
                character = sequence if len(sequence) == 1 else None
                yield from _emit_alt(keys, character)
                return
        yield from _original(self, sequence, alt=alt)

    try:
        XTermParser._sequence_to_key_events = _sequence_to_key_events_with_alt  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning("Textual keyboard parser patch assignment rejected: %s", exc)
