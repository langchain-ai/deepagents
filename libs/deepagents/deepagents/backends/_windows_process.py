"""Poll Windows subprocess pipes without creating blocking reader threads."""

from __future__ import annotations

import codecs
import io
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


def _make_decoder(pipe: IO[str] | None) -> io.IncrementalNewlineDecoder | None:
    r"""Build an incremental decoder matching one text-mode pipe.

    The decoder holds back the bytes of a character that a read cut in half, and
    `IncrementalNewlineDecoder` applies the same universal-newline translation
    `subprocess` gives a fully buffered stream, including a `\r\n` pair split
    across two reads.
    """
    if pipe is None:
        return None
    stream = cast("TextIO", pipe)
    decoder = codecs.getincrementaldecoder(stream.encoding)(stream.errors or "strict")
    return io.IncrementalNewlineDecoder(decoder, translate=True)


class WindowsProcessReader:
    """Read both captured pipes by polling instead of with reader threads.

    `subprocess.communicate` starts a thread for each pipe on Windows. Those
    threads hold the pipe until the write end closes, so a caller cannot close
    the pipes to stop waiting for a descendant that inherited them. This reader
    polls instead, which leaves the caller free to close the pipes at any time.

    The process must be in text mode, because the reader takes the encoding and
    error handler from each pipe. Nothing may read the pipes before this reader
    is created. Each pipe is decoded incrementally as it is read, so repeated
    calls to `communicate` keep the text already decoded, and an encoded
    character split across two reads is held until the rest of it arrives.

    The caller owns termination and pipe cleanup if `communicate` raises.
    """

    def __init__(self, process: subprocess.Popen[str]) -> None:
        """Retain a newly created text-mode process before any pipe reads."""
        self._process = process
        self._pipes = (process.stdout, process.stderr)
        self._decoders = tuple(_make_decoder(pipe) for pipe in self._pipes)
        self._texts = ["", ""]

    def _read(self) -> bool:
        """Read one bounded chunk from each open pipe without starving either."""
        progress = False
        for index, (pipe, decoder) in enumerate(zip(self._pipes, self._decoders, strict=True)):
            if pipe is None or decoder is None or pipe.closed:
                continue
            data = _read_available(pipe)
            if data is not None:
                progress = True
                # Empty bytes mean EOF, which is where a still-incomplete
                # character is a decoding error rather than a short read.
                self._texts[index] += decoder.decode(data, final=not data)
                if not data:
                    pipe.close()
        return progress

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
                deadline. The exception carries the text decoded so far, so a
                caller that gives up keeps it.
            OSError: If reading a captured pipe fails.
            UnicodeError: If captured output cannot be decoded.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    self._process.args,
                    timeout,
                    output=self._texts[0],
                    stderr=self._texts[1],
                )
            progress = self._read()
            if all(pipe is None or pipe.closed for pipe in self._pipes) and self._process.poll() is not None:
                return self._texts[0], self._texts[1]
            if not progress:
                time.sleep(min(remaining, _POLL_INTERVAL))
