"""Terminal capability detection.

Probes the attached terminal for optional features so the CLI can
adapt its UI (e.g. show `Shift+Enter` as the newline shortcut when the
kitty keyboard protocol is supported, and `Ctrl+J` otherwise).

Detection runs on first access and is cached for the lifetime of the
process. Callers must resolve capabilities before Textual's driver
acquires stdin — running the query while Textual owns the tty will
race with its own keypress stream.
"""

from __future__ import annotations

import logging
import os
import re
import select
import sys
from functools import cache

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT_SECONDS = 0.1
"""Per-read timeout for the tty query. The first response byte typically
arrives within a few ms; 100 ms is a generous ceiling that still keeps
startup snappy on misbehaving terminals."""

_MAX_RESPONSE_BYTES = 256
"""Cap total bytes read from the tty. Real responses are ~20 bytes;
this guards against a hostile or broken terminal streaming indefinitely."""

_KITTY_RESPONSE_RE = re.compile(rb"\x1b\[\?\d*u")
"""Matches a kitty-protocol flags-query reply (`CSI ? <flags> u`).

A terminal that supports the protocol emits this sequence in response to
`CSI ? u`. Terminals that do not support it stay silent for the flags
query and only reply to the primary device attributes (DA1) query that
follows."""


@cache
def supports_kitty_keyboard_protocol() -> bool:
    """Return whether the attached terminal speaks the kitty keyboard protocol.

    The protocol disambiguates modified keys like `shift+enter` and
    `ctrl+enter` that legacy xterm-style terminals collapse onto plain
    `enter`. Supporting terminals reply to a `CSI ? u` query with
    `CSI ? <flags> u`.

    Cached for the process lifetime. Writes to stdout and reads from
    stdin, so call during startup — before Textual acquires the tty.

    Returns:
        `True` when a valid response is received, `False` otherwise
        (no tty, non-POSIX platform, or unsupported terminal).
    """
    if sys.platform == "win32":
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        import termios  # POSIX-only
        import tty
    except ImportError:
        return False

    fd = sys.stdin.fileno()
    try:
        original = termios.tcgetattr(fd)
    except termios.error as exc:
        logger.debug("kitty kbd probe: tcgetattr failed: %s", exc)
        return False

    buffer = b""
    try:
        tty.setcbreak(fd)
        os.write(sys.stdout.fileno(), b"\x1b[?u\x1b[c")
        while len(buffer) < _MAX_RESPONSE_BYTES:
            ready, _, _ = select.select([fd], [], [], _QUERY_TIMEOUT_SECONDS)
            if not ready:
                break
            chunk = os.read(fd, 64)
            if not chunk:
                break
            buffer += chunk
            # DA1 reply ends with `c` and always follows the (optional)
            # kitty reply, so its arrival means we have everything.
            if buffer.endswith(b"c"):
                break
    except OSError as exc:
        logger.debug("kitty kbd probe: query failed: %s", exc)
        return False
    finally:
        # `tcsetattr` can itself fail (fd closed mid-probe, EIO on a
        # disappearing pty). Best-effort: log and swallow so a flaky
        # restore doesn't crash startup.
        try:
            termios.tcsetattr(fd, termios.TCSANOW, original)
        except (termios.error, OSError) as exc:
            logger.debug("kitty kbd probe: tcsetattr restore failed: %s", exc)

    return _parse_kitty_response(buffer)


def _parse_kitty_response(data: bytes) -> bool:
    """Return whether `data` contains a kitty-protocol flags-query reply.

    Split out for unit testing without a real tty.
    """
    return _KITTY_RESPONSE_RE.search(data) is not None
