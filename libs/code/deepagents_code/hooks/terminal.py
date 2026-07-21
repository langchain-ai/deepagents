"""Terminal escape-sequence validation for hook output."""

from __future__ import annotations

import re

# OSC 0/1/2/9/99/777 terminated by BEL or ST, plus bare BEL.
_ALLOWED_SEQUENCE = re.compile(
    r"(?:"
    r"\x1b\](?:0|1|2|9|99|777);[^\x00-\x1f\x7f-\x9f]*(?:\x07|\x1b\\)"
    r"|"
    r"\x07"
    r")+"
)


def validate_terminal_sequence(value: str) -> str | None:
    """Return `value` when it is composed only of allowlisted sequences.

    Allowed tokens are OSC `0`/`1`/`2`/`9`/`99`/`777` (BEL or ST terminated)
    and bare BEL. Any other escape or control content rejects the entire value.

    Args:
        value: Candidate `terminalSequence` payload.

    Returns:
        The original string when valid, otherwise `None`.
    """
    if not value:
        return None
    return value if _ALLOWED_SEQUENCE.fullmatch(value) is not None else None
