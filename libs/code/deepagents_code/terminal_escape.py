"""Best-effort writer for terminal escape/control sequences.

Centralizes the "fire and forget" pattern the app uses for cosmetic terminal
control (OSC 9;4 taskbar progress, OSC 11 background, OSC 52 clipboard).
Writes prefer `/dev/tty` so output reaches the terminal even when stdout/stderr
are redirected, fall back to `sys.__stderr__`, and never raise — cosmetic
control output must not crash the app.

It is also the single place that knows a terminal multiplexer may be in the
way: `write_osc` wraps its sequence for tmux, which otherwise drops every OSC
its panes emit.

Set `DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE=1` to disable all output (useful for
unsupported terminals or noisy logs).
"""

from __future__ import annotations

import atexit
import logging
import pathlib
import sys
import threading
from enum import StrEnum
from typing import TYPE_CHECKING

from deepagents_code._env_vars import NO_TERMINAL_ESCAPE, is_env_truthy

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TextIO

logger = logging.getLogger(__name__)

_TMUX_PASSTHROUGH_PREFIX = "\x1bPtmux;"
"""Device Control String that tells tmux to forward the payload verbatim."""

_TMUX_PASSTHROUGH_SUFFIX = "\x1b\\"
"""String Terminator closing the passthrough DCS."""

_PROGRESS_MIN = 0
"""Lower clamp bound for determinate `OSC 9;4` progress percentages."""

_PROGRESS_MAX = 100
"""Upper clamp bound for determinate `OSC 9;4` progress percentages."""


class TerminalProgressState(StrEnum):
    """`OSC 9;4` progress states.

    See https://learn.microsoft.com/en-us/windows/terminal/tutorials/progress-bar-sequences.
    """

    CLEAR = "0"
    """Remove any progress indicator. Percentage is ignored."""

    NORMAL = "1"
    """Determinate progress shown with the default (success) color."""

    ERROR = "2"
    """Determinate progress shown with the error/red color."""

    INDETERMINATE = "3"
    """Activity in progress with no known percentage; renders as a pulse."""

    WARNING = "4"
    """Determinate progress shown with the warning/yellow color."""


def _is_disabled() -> bool:
    """Return whether terminal-escape output is opt-out disabled."""
    return is_env_truthy(NO_TERMINAL_ESCAPE)


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
    TTY.

    Returns `False` (no-op) when output is disabled or no TTY is reachable.

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
        except (OSError, UnicodeError) as exc:
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


def wrap_for_multiplexer(sequence: str, env: Mapping[str, str] | None = None) -> str:
    r"""Wrap `sequence` so tmux forwards it to the terminal it is hosted in.

    tmux swallows operating-system commands from its panes: an unwrapped
    `OSC 9;4` or `OSC 52` never reaches the outer terminal. Wrapping in tmux's
    passthrough DCS forwards the payload verbatim — provided the user has
    turned on `allow-passthrough`, which `dcode doctor` reports. Every escape
    inside the payload has to be doubled or tmux ends the DCS early.

    Args:
        sequence: Raw escape sequence intended for the outer terminal.
        env: Environment mapping to read. Defaults to the process environment.

    Returns:
        The wrapped sequence inside tmux, otherwise `sequence` unchanged.
    """
    from deepagents_code.multiplexer import inside_tmux

    if not sequence or not inside_tmux(env):
        return sequence
    doubled = sequence.replace("\x1b", "\x1b\x1b")
    return f"{_TMUX_PASSTHROUGH_PREFIX}{doubled}{_TMUX_PASSTHROUGH_SUFFIX}"


def write_osc(command: str, payload: str = "", *, st: bool = False) -> bool:
    r"""Write an `OSC <command>;<payload>` sequence.

    Every OSC the app emits addresses the terminal the user is looking at, so
    the sequence is wrapped for a multiplexer when one is in the way.

    Args:
        command: The numeric OSC command (e.g. `"9;4"` for taskbar progress).
        payload: Optional semicolon-joined payload appended after the command.
        st: When `True`, terminate with String Terminator (`ESC \`) instead of
            the default BEL (`\a`).

            BEL matches the Windows Terminal docs and works on most terminals;
            VTE-derived terminals may prefer ST.

    Returns:
        `True` if the sequence was written.
    """
    body = f"{command};{payload}" if payload else command
    terminator = "\x1b\\" if st else "\a"
    return write_terminal_escape(wrap_for_multiplexer(f"\x1b]{body}{terminator}"))


_progress_active = False
_terminal_background_active = False
_atexit_registered = False
_atexit_lock = threading.Lock()


def _ensure_atexit_registered() -> None:
    """Register terminal-state cleanup exactly once."""
    global _atexit_registered  # noqa: PLW0603

    with _atexit_lock:
        if not _atexit_registered:
            atexit.register(_atexit_clear)
            _atexit_registered = True


def _validate_progress(progress: int | None, state: TerminalProgressState) -> int:
    """Clamp/normalize `progress` for a given `state`.

    Determinate states (`NORMAL`, `ERROR`, `WARNING`) clamp to `[0, 100]`;
    `INDETERMINATE` and `CLEAR` always emit `0`. A non-`None` `progress`
    supplied with `CLEAR`/`INDETERMINATE` is dropped with a debug log so
    misuse stays observable without raising on a cosmetic write path. A
    `progress` that can't be coerced to `int` is treated the same way.

    Args:
        progress: Raw percentage, or `None`.
        state: The OSC 9;4 progress state.

    Returns:
        The normalized progress integer to emit.
    """
    if state in {TerminalProgressState.CLEAR, TerminalProgressState.INDETERMINATE}:
        if progress is not None and progress != 0:
            logger.debug(
                "terminal_progress: ignoring progress=%r for state=%s",
                progress,
                state.name,
            )
        return 0
    if progress is None:
        return 0
    try:
        coerced = int(progress)
    except (TypeError, ValueError) as exc:
        logger.debug(
            "terminal_progress: non-numeric progress=%r ignored (%s)", progress, exc
        )
        return 0
    return max(_PROGRESS_MIN, min(_PROGRESS_MAX, coerced))


def set_terminal_progress(
    progress: int | None = None,
    *,
    state: TerminalProgressState = TerminalProgressState.NORMAL,
) -> bool:
    """Set the terminal's `OSC 9;4` progress indicator.

    Fires unconditionally — terminals that don't recognize `OSC 9;4` silently
    ignore the sequence. Set `DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE=1` to opt out
    entirely.

    Args:
        progress: Percentage `0-100` for determinate states. Ignored for
            `INDETERMINATE` and `CLEAR`.
        state: One of `TerminalProgressState`.

    Returns:
        `True` if the sequence was written.
    """
    global _progress_active  # noqa: PLW0603

    value = _validate_progress(progress, state)
    payload = f"{state.value};{value}"
    written = write_osc("9;4", payload)
    if written and state is not TerminalProgressState.CLEAR:
        _ensure_atexit_registered()
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


def set_terminal_background(color: str) -> bool:
    """Set the terminal's dynamic default background color with `OSC 11`.

    This is cosmetic and intentionally best-effort. Terminals that don't
    support `OSC 11` ignore it; `OSC 111` is emitted from `atexit` to restore
    the default background when this call succeeds.

    Args:
        color: Terminal color payload, usually a CSS-style hex color such as
            `#11121D`.

    Returns:
        `True` if the sequence was written.
    """
    global _terminal_background_active  # noqa: PLW0603

    if not color:
        return False
    written = write_osc("11", color, st=True)
    if written:
        _ensure_atexit_registered()
        _terminal_background_active = True
    return written


def reset_terminal_background() -> bool:
    """Reset the terminal's dynamic default background color with `OSC 111`.

    Returns:
        `True` if the sequence was written.
    """
    global _terminal_background_active  # noqa: PLW0603

    written = write_osc("111", st=True)
    if written:
        _terminal_background_active = False
    return written


def _atexit_clear() -> None:
    """`atexit` hook that clears any leftover terminal state."""
    if _progress_active:
        try:
            clear_terminal_progress()
        except Exception:
            logger.warning("Failed to clear terminal progress at exit", exc_info=True)
    if _terminal_background_active:
        try:
            reset_terminal_background()
        except Exception:
            logger.warning("Failed to reset terminal background at exit", exc_info=True)
