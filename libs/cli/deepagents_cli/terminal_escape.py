"""Best-effort writer for terminal escape/control sequences.

Centralizes the "fire and forget" pattern the CLI uses for cosmetic terminal
control (OSC 9;4 taskbar progress today; eventually OSC 52 clipboard and the
iTerm2 cursor guide). Writes prefer `/dev/tty` so output reaches the terminal
even when stdout/stderr are redirected, fall back to `sys.__stderr__`, and
never raise — cosmetic control output must not crash the app.

Set `DEEPAGENTS_CLI_NO_TERMINAL_ESCAPE=1` to disable all output (useful for
unsupported terminals or noisy logs).
"""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import sys
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TextIO

logger = logging.getLogger(__name__)

_DISABLE_ENV_VAR = "DEEPAGENTS_CLI_NO_TERMINAL_ESCAPE"
_PROGRESS_MIN = 0
_PROGRESS_MAX = 100


class TerminalProgressState(StrEnum):
    """`OSC 9;4` progress states.

    See https://learn.microsoft.com/en-us/windows/terminal/tutorials/progress-bar-sequences.
    """

    CLEAR = "0"
    NORMAL = "1"
    ERROR = "2"
    INDETERMINATE = "3"
    WARNING = "4"


def _is_disabled() -> bool:
    """Return whether terminal-escape output is opt-out disabled."""
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def _open_tty() -> TextIO | None:
    """Return an open `/dev/tty` handle, or `None` if unavailable."""
    try:
        return pathlib.Path("/dev/tty").open("w", encoding="utf-8")
    except OSError:
        return None


def _is_stream_tty(stream: TextIO | None) -> bool:
    """Return whether `stream` is a real TTY."""
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except (ValueError, OSError):
        return False


def write_terminal_escape(sequence: str) -> bool:
    r"""Best-effort write of a terminal control sequence.

    Prefers `/dev/tty` so the sequence reaches the terminal even when stdout
    or stderr are redirected. Falls back to `sys.__stderr__` only if it is a
    TTY. Returns `False` (no-op) when output is disabled or no TTY is reachable.

    Args:
        sequence: Raw escape sequence to write, including leading `\x1b`/`ESC`
            and terminator.

    Returns:
        `True` if the sequence was written and flushed without error.
    """
    if _is_disabled() or not sequence:
        return False
    tty = _open_tty()
    if tty is not None:
        try:
            with tty:
                tty.write(sequence)
                tty.flush()
        except OSError as exc:
            logger.debug("terminal_escape /dev/tty write failed: %s", exc)
        else:
            return True
    stderr = sys.__stderr__
    if stderr is not None and _is_stream_tty(stderr):
        try:
            stderr.write(sequence)
            stderr.flush()
        except (OSError, ValueError) as exc:
            logger.debug("terminal_escape stderr write failed: %s", exc)
            return False
        return True
    return False


def write_osc(command: str, payload: str = "", *, st: bool = False) -> bool:
    r"""Write an `OSC <command>;<payload>` sequence.

    Args:
        command: The numeric OSC command (e.g. ``"9;4"`` for taskbar progress).
        payload: Optional semicolon-joined payload appended after the command.
        st: When `True`, terminate with String Terminator (`ESC \`) instead of
            the default BEL (`\a`). BEL matches the Windows Terminal docs and
            works on most terminals; VTE-derived terminals may prefer ST.

    Returns:
        `True` if the sequence was written.
    """
    body = f"{command};{payload}" if payload else command
    terminator = "\x1b\\" if st else "\a"
    return write_terminal_escape(f"\x1b]{body}{terminator}")


_progress_active = False


def _validate_progress(progress: int | None, state: TerminalProgressState) -> int:
    """Clamp/normalize `progress` for a given `state`.

    Determinate states (`NORMAL`, `ERROR`, `WARNING`) clamp to `[0, 100]`;
    `INDETERMINATE` and `CLEAR` always emit `0`.

    Args:
        progress: Raw percentage, or `None`.
        state: The OSC 9;4 progress state.

    Returns:
        The normalized progress integer to emit.
    """
    if state in {TerminalProgressState.CLEAR, TerminalProgressState.INDETERMINATE}:
        return 0
    if progress is None:
        return 0
    return max(_PROGRESS_MIN, min(_PROGRESS_MAX, int(progress)))


def set_terminal_progress(
    progress: int | None = None,
    *,
    state: TerminalProgressState = TerminalProgressState.NORMAL,
) -> bool:
    """Set the terminal's `OSC 9;4` progress indicator.

    Unsupported terminals silently ignore the sequence, so callers can fire
    this unconditionally without runtime probing.

    Args:
        progress: Percentage `0-100` for determinate states. Ignored for
            `INDETERMINATE` and `CLEAR`.
        state: One of `TerminalProgressState`.

    Returns:
        `True` if the sequence was written.
    """
    value = _validate_progress(progress, state)
    payload = f"{state.value};{value}"
    written = write_osc("9;4", payload)
    global _progress_active  # noqa: PLW0603
    if written and state is not TerminalProgressState.CLEAR:
        if not _progress_active:
            atexit.register(_atexit_clear)
        _progress_active = True
    elif state is TerminalProgressState.CLEAR:
        _progress_active = False
    return written


def clear_terminal_progress() -> bool:
    """Clear the terminal's progress indicator.

    Emits `OSC 9;4;0;0`.

    Returns:
        `True` if the sequence was written.
    """
    return set_terminal_progress(state=TerminalProgressState.CLEAR)


def _atexit_clear() -> None:
    """`atexit` hook that clears any leftover progress indicator."""
    if _progress_active:
        clear_terminal_progress()
