"""Protect the Textual viewport from unmanaged macOS stderr writes.

Suppressing fd 2 also suppresses Textual itself, so this module owns both
halves of that trade: `TerminalStderrGuard` hides the native writes, and
`stdout_driver_class` re-routes Textual's own frames to the surviving stdout
stream. Neither is correct without the other.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.app import App
    from textual.driver import Driver

logger = logging.getLogger(__name__)


def stdout_driver_class() -> type[Driver] | None:
    """Return a Textual Unix driver class that renders through stdout.

    Textual's `LinuxDriver` writes every frame to `sys.__stderr__`, while
    `TerminalStderrGuard` points fd 2 at `/dev/null` — so under an active guard
    the stock driver renders the whole TUI into the void. The returned subclass
    aims driver output at `sys.__stdout__` instead, which the guard leaves
    alone and which `_stderr_targets_stdout` has already proven is the same
    terminal. Callers must not install the guard without also applying this.

    Like the patches in `_textual_patches`, this reaches into private Textual
    state and so refuses to guess when the ground shifts: it returns `None`
    when stdout cannot carry the TUI or when the caller's own driver choice
    must win, and the subclass raises rather than rendering blind if Textual
    ever stops keeping its output stream in `_file`.

    Returns:
        The driver class, or `None` when the caller should leave both the
        driver and stderr alone.
    """
    from textual import constants

    if constants.DRIVER is not None:
        # An explicit TEXTUAL_DRIVER is a deliberate override; silently
        # swapping it for ours would strand the user with no thread to pull.
        logger.info(
            "Leaving stderr unsuppressed: TEXTUAL_DRIVER=%s renders to stderr.",
            constants.DRIVER,
        )
        return None

    stdout = sys.__stdout__
    if stdout is None or stdout.closed:
        # An active guard only proves fd 1 is a tty; the Python-level object
        # can still be missing. Writing frames to it would kill Textual's
        # writer thread and then deadlock its full write queue.
        logger.warning(
            "Leaving stderr unsuppressed: sys.__stdout__ is %s.",
            "None" if stdout is None else "closed",
        )
        return None

    from textual.drivers.linux_driver import LinuxDriver

    class StdoutLinuxDriver(LinuxDriver):
        """`LinuxDriver` whose frames go to stdout instead of stderr.

        Reassigning `_file` after `super().__init__()` is safe because Textual
        reads it only in `start_application_mode`, to build the writer thread.
        """

        def __init__(
            self,
            app: App,
            *,
            debug: bool = False,
            mouse: bool = True,
            size: tuple[int, int] | None = None,
        ) -> None:
            super().__init__(app, debug=debug, mouse=mouse, size=size)
            if not hasattr(self, "_file"):
                # Textual moved its output stream. Fail loudly while the
                # caller's `finally` can still restore fd 2 to report it —
                # assigning a dead attribute would leave a black screen.
                msg = (
                    "Textual's LinuxDriver no longer stores its output stream "
                    "in `_file`, so the stderr guard cannot route the TUI to "
                    "stdout. Update stdout_driver_class for the new internals."
                )
                raise RuntimeError(msg)
            self._file = stdout

    return StdoutLinuxDriver


class TerminalStderrGuard:
    """Suppress native stderr writes while the TUI owns the terminal."""

    def __init__(self, *, enabled: bool = False) -> None:
        self._enabled = enabled
        self._saved_stderr: int | None = None
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def install(cls) -> TerminalStderrGuard:
        """Install suppression when stdout and stderr share a macOS terminal.

        Callers rendering a Textual app must pair an active guard with
        `App.driver_class = stdout_driver_class()`; the stock driver writes to
        the very stderr this suppresses, so the TUI would be invisible.

        Returns:
            The installed guard, which may be inactive on unsupported streams.
        """
        guard = cls(enabled=sys.platform == "darwin" and _stderr_targets_stdout())
        guard.resume()
        return guard

    @property
    def active(self) -> bool:
        """Whether stderr is currently suppressed."""
        with self._lock:
            return self._saved_stderr is not None

    def pause(self) -> None:
        """Restore stderr while terminal ownership is released."""
        with self._lock:
            if self._saved_stderr is None:
                return
            os.dup2(self._saved_stderr, 2)
            saved_stderr = self._saved_stderr
            self._saved_stderr = None
            os.close(saved_stderr)

    def resume(self) -> None:
        """Suppress stderr after terminal ownership returns."""
        with self._lock:
            if not self._enabled or self._closed or self._saved_stderr is not None:
                return
            saved_stderr = os.dup(2)
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(devnull, 2)
                finally:
                    os.close(devnull)
            except BaseException:
                os.close(saved_stderr)
                raise
            self._saved_stderr = saved_stderr

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Restore stderr for the duration of a terminal handoff.

        Yields:
            Control while stderr targets the caller's terminal.
        """
        was_active = self.active
        if was_active:
            self.pause()
        try:
            yield
        finally:
            if was_active:
                self.resume()

    def close(self) -> None:
        """Restore stderr permanently when the TUI session ends."""
        self.pause()
        with self._lock:
            self._closed = True


def _stderr_targets_stdout() -> bool:
    """Return whether stdout and stderr are the same terminal."""
    if not os.isatty(1) or not os.isatty(2):
        return False
    try:
        stdout_stat = os.fstat(1)
        stderr_stat = os.fstat(2)
    except OSError:
        return False
    return (stdout_stat.st_dev, stdout_stat.st_ino) == (
        stderr_stat.st_dev,
        stderr_stat.st_ino,
    )
