"""Poll Windows subprocess pipes without creating blocking reader threads."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import IO, TextIO, cast

if sys.platform == "win32":
    import _winapi
    import msvcrt

_READ_SIZE = 32_768
_POLL_INTERVAL = 0.01
_ERROR_BROKEN_PIPE = 109


def _peek_pipe(descriptor: int) -> int:
    """Return available pipe bytes without consuming or waiting for output."""
    if sys.platform == "win32":
        return _winapi.PeekNamedPipe(msvcrt.get_osfhandle(descriptor), 0)[-2]
    msg = "Windows pipe polling is only available on Windows"
    raise OSError(msg)


def _read_available(pipe: IO[str]) -> bytes | None:
    """Read available bytes, returning None for no data and empty bytes for EOF."""
    try:
        available = _peek_pipe(pipe.fileno())
        return os.read(pipe.fileno(), min(available, _READ_SIZE)) if available else None
    except OSError as error:
        if getattr(error, "winerror", None) == _ERROR_BROKEN_PIPE:
            return b""
        raise


class WindowsProcessReader:
    """Collect captured text output without competing readers holding pipe locks.

    The caller owns termination and pipe cleanup if communication raises. Repeated
    calls preserve captured bytes, including incomplete encoded characters.
    """

    def __init__(self, process: subprocess.Popen[str]) -> None:
        """Retain a newly created text-mode process before any pipe reads."""
        self._process = process
        self._pipes = (process.stdout, process.stderr)
        self._buffers = (bytearray(), bytearray())

    def _read(self) -> bool:
        """Read one bounded chunk from each open pipe without starving either."""
        progress = False
        for pipe, buffer in zip(self._pipes, self._buffers, strict=True):
            if pipe is None or pipe.closed:
                continue
            data = _read_available(pipe)
            if data is not None:
                progress = True
                buffer.extend(data)
                if not data:
                    pipe.close()
        return progress

    def _output(self) -> tuple[str, str]:
        """Decode complete streams with subprocess-compatible newline handling."""

        def decode(data: bytearray, pipe: IO[str] | None) -> str:
            if pipe is None:
                return ""
            stream = cast("TextIO", pipe)
            text = data.decode(stream.encoding, stream.errors or "strict")
            return text.replace("\r\n", "\n").replace("\r", "\n")

        return decode(self._buffers[0], self._pipes[0]), decode(self._buffers[1], self._pipes[1])

    def communicate(self, *, timeout: float) -> tuple[str, str]:
        """Collect output and reap the direct process within a polling deadline.

        EOF is required on both pipes even if the shell has already exited, so
        output inherited by descendants is retained during ordinary execution.

        Args:
            timeout: Maximum seconds for this communication attempt.

        Returns:
            Decoded stdout and stderr after both pipes reach EOF and the process exits.

        Raises:
            subprocess.TimeoutExpired: If output or process completion exceeds the deadline.
            OSError: If reading a captured pipe fails.
            UnicodeError: If captured output cannot be decoded.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self._process.args, timeout)
            progress = self._read()
            if all(pipe is None or pipe.closed for pipe in self._pipes) and self._process.poll() is not None:
                return self._output()
            if not progress:
                time.sleep(min(remaining, _POLL_INTERVAL))
