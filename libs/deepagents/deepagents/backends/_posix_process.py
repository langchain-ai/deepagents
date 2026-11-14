"""Poll POSIX subprocess pipes without copying output on each timeout."""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from typing import IO, TextIO, cast

_READ_SIZE = 32_768


def _decode_output(pipe: IO[str] | None, chunks: list[bytes]) -> str:
    """Apply the text-mode pipe's encoding, error handler, and newline policy."""
    if pipe is None:
        return ""
    stream = cast("TextIO", pipe)
    output = b"".join(chunks).decode(stream.encoding, stream.errors or "strict")
    return output.replace("\r\n", "\n").replace("\r", "\n")


class PosixProcessReader:
    """Collect text-mode output across polling attempts, joining only at EOF.

    Unlike `Popen.communicate`, a timed-out attempt does not build a snapshot of
    all output read so far. Chunks stay buffered for the next attempt, including
    partial encoded characters and newline sequences. Nothing else may read the
    pipes while this reader is in use.

    The caller owns termination and pipe cleanup if `communicate` raises.
    """

    def __init__(self, process: subprocess.Popen[str]) -> None:
        """Retain a newly created text-mode process before any pipe reads."""
        self._process = process
        self._pipes = (process.stdout, process.stderr)
        self._chunks: list[list[bytes]] = [[], []]

    def _read(self, selector: selectors.BaseSelector, timeout: float) -> None:
        """Drain one bounded chunk per ready pipe without starving either."""
        for key, _ in selector.select(timeout):
            data = os.read(key.fd, _READ_SIZE)
            if data:
                self._chunks[key.data].append(data)
            else:
                selector.unregister(key.fd)
                pipe = self._pipes[key.data]
                if pipe is not None:
                    pipe.close()

    def communicate(self, *, timeout: float) -> tuple[str, str]:
        """Collect output and reap the direct process within a polling deadline.

        Both pipes must reach EOF, even if the shell exits before a descendant
        finishes writing. Closed pipes still require waiting for the shell.

        Args:
            timeout: Maximum seconds for this communication attempt.

        Returns:
            Decoded stdout and stderr after both pipes reach EOF and the process exits.

        Raises:
            subprocess.TimeoutExpired: If output or process completion exceeds
                the deadline. Captured output is retained without a snapshot.
            OSError: If reading a captured pipe fails.
            UnicodeError: If captured output cannot be decoded.
        """
        deadline = time.monotonic() + timeout
        # Scope the selector to one attempt so cancellation cannot leak its fd.
        with selectors.DefaultSelector() as selector:
            for index, pipe in enumerate(self._pipes):
                if pipe is not None and not pipe.closed:
                    selector.register(pipe, selectors.EVENT_READ, index)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self._process.args, timeout)
                self._read(selector, remaining)
        self._process.wait(timeout=max(0, deadline - time.monotonic()))
        return _decode_output(self._pipes[0], self._chunks[0]), _decode_output(self._pipes[1], self._chunks[1])
