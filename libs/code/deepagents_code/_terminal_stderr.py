"""Protect the Textual viewport from unmanaged macOS stderr writes."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.app import App
    from textual.driver import Driver


def stdout_driver_class() -> type[Driver]:
    """Return a Textual Unix driver that renders through stdout."""
    from textual.drivers.linux_driver import LinuxDriver

    class StdoutLinuxDriver(LinuxDriver):
        def __init__(
            self,
            app: App,
            *,
            debug: bool = False,
            mouse: bool = True,
            size: tuple[int, int] | None = None,
        ) -> None:
            super().__init__(app, debug=debug, mouse=mouse, size=size)
            self._file = sys.__stdout__

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
