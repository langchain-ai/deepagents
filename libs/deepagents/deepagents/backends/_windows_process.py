"""Poll Windows subprocess pipes without creating blocking reader threads."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import IO, TextIO, cast

if sys.platform == "win32":
    import _winapi
    import msvcrt

logger = logging.getLogger(__name__)

_READ_SIZE = 32_768
_POLL_INTERVAL = 0.01

_END_OF_STREAM_ERRORS = frozenset({109, 232, 233})
"""Windows error codes that mean the write end of a pipe is gone.

`ERROR_BROKEN_PIPE` (109), `ERROR_NO_DATA` (232) and `ERROR_PIPE_NOT_CONNECTED`
(233). Windows selects one of the three based on how far the teardown of the
write end has gone, so all three mean end of stream. Windows does not
distinguish a pipe that broke early from one that reached its natural end, so
this reader cannot detect truncation.
"""


def _peek_pipe(descriptor: int) -> int:
    """Return available pipe bytes without consuming or waiting for output.

    `PeekNamedPipe` returns `(available, left_in_message)` when it is asked for
    zero bytes, so the byte count is the second element from the end.

    Args:
        descriptor: File descriptor of a pipe opened by `subprocess`.

    Returns:
        The number of bytes that a read can take immediately.

    Raises:
        OSError: If this is not Windows, or if the pipe cannot be peeked.
    """
    # Keep the literal `sys.platform` test. The type checker uses it to remove
    # the Windows-only imports on other platforms.
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
        # `_peek_pipe` raises this OSError without a `winerror`, so a non-Windows
        # caller always reaches the `raise` below.
        winerror = getattr(error, "winerror", None)
        if winerror in _END_OF_STREAM_ERRORS:
            logger.debug("Windows pipe reached end of stream with error %s", winerror)
            return b""
        raise


class WindowsProcessReader:
    """Read both captured pipes by polling instead of with reader threads.

    `subprocess.communicate` starts a thread for each pipe on Windows. Those
    threads hold the pipe until the write end closes, so a caller cannot close
    the pipes to stop waiting for a descendant that inherited them. This reader
    polls instead, which leaves the caller free to close the pipes at any time.

    The process must be in text mode, because `_output` reads the encoding from
    each pipe. Nothing may read the pipes before this reader is created.
    Repeated calls to `communicate` keep the bytes already read, including an
    incomplete encoded character split across two reads.

    The caller owns termination and pipe cleanup if `communicate` raises.
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

    def _output(self, *, lenient: bool = False) -> tuple[str, str]:
        """Decode the streams with subprocess-compatible newline handling.

        Args:
            lenient: Replace undecodable bytes instead of raising. Use this for a
                buffer that is not at a stream boundary, because the last read can
                end in the middle of an encoded character.

        Returns:
            The decoded stdout and stderr.
        """

        def decode(data: bytearray, pipe: IO[str] | None) -> str:
            if pipe is None:
                return ""
            stream = cast("TextIO", pipe)
            errors = "replace" if lenient else (stream.errors or "strict")
            text = data.decode(stream.encoding, errors)
            return text.replace("\r\n", "\n").replace("\r", "\n")

        return decode(self._buffers[0], self._pipes[0]), decode(self._buffers[1], self._pipes[1])

    def communicate(self, *, timeout: float) -> tuple[str, str]:
        """Collect output and reap the direct process within a polling deadline.

        EOF is required on both pipes even if the shell has already exited.
        Descendants that inherited the pipes can still write to them, and their
        output is kept.

        Args:
            timeout: Maximum seconds for this communication attempt.

        Returns:
            Decoded stdout and stderr after both pipes reach EOF and the process exits.

        Raises:
            subprocess.TimeoutExpired: If output or process completion exceeds the
                deadline. The exception carries the output read so far, so a
                caller that gives up keeps it.
            OSError: If reading a captured pipe fails.
            UnicodeError: If captured output cannot be decoded.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The buffers can end mid-character here, so decode leniently.
                # A decoding error must not replace the timeout.
                stdout, stderr = self._output(lenient=True)
                raise subprocess.TimeoutExpired(self._process.args, timeout, output=stdout, stderr=stderr)
            progress = self._read()
            if all(pipe is None or pipe.closed for pipe in self._pipes) and self._process.poll() is not None:
                return self._output()
            if not progress:
                time.sleep(min(remaining, _POLL_INTERVAL))
