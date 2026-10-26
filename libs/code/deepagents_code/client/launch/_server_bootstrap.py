"""Run the POSIX server while watching the launching client's lifetime.

The inherited pipe is never written to: EOF means every client-side writer has
closed. Keep this module stdlib-only so the guard starts before server imports.
"""

from __future__ import annotations

import os
import runpy
import selectors
import signal
import sys
import threading
import time
from pathlib import Path


class _ParentGuard:
    """Own the pipe reader and stop the server when its client disappears."""

    def __init__(self, reader: int, timeout: float) -> None:
        # Popen passes this descriptor through one exec, never through the next.
        os.set_inheritable(reader, False)
        self._reader = reader
        self._timeout = timeout
        self._lock = threading.Lock()
        self._parent_exited = False
        self._dispatched = False

    def start(self) -> None:
        """Start watching before importing or dispatching the CLI."""
        threading.Thread(
            target=self._watch, name="server-parent-guard", daemon=True
        ).start()

    def begin_dispatch(self) -> bool:
        """Claim dispatch unless EOF has already arrived.

        Returns:
            Whether the client is still alive at the dispatch boundary.
        """
        with self._lock:
            if self._parent_exited:
                return False
            # Check synchronously too: the watcher may not have run yet. Only
            # the watcher closes the reader, under this same lock, so its number
            # cannot be reused while this check is in progress.
            with selectors.DefaultSelector() as selector:
                selector.register(self._reader, selectors.EVENT_READ)
                if selector.select(timeout=0):
                    return False
            self._dispatched = True
            return True

    def _watch(self) -> None:
        try:
            while os.read(self._reader, 1):
                pass
        finally:
            with self._lock:
                self._parent_exited = True
                os.close(self._reader)
                dispatched = self._dispatched
            if dispatched:
                self._terminate()

    def _terminate(self) -> None:
        """Allow ordinary signal shutdown, then enforce the same grace period."""
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)
        # This runs independently of the main thread's server loop and remains
        # active if the CLI returns but interpreter teardown blocks on a thread
        # or an atexit callback. Do not cancel it when run_module returns.
        time.sleep(self._timeout)
        os.kill(pid, signal.SIGKILL)


def _main() -> None:
    reader, *held = (int(fd) for fd in sys.argv[1].split(","))
    for fd in held:
        # Keep the session lock in this process only, so a long-lived
        # grandchild cannot hold the install in use after the server exits.
        os.set_inheritable(fd, False)
    guard = _ParentGuard(reader, float(sys.argv[2]))
    guard.start()
    if not guard.begin_dispatch():
        return
    cache_dir = Path(sys.prefix) / "__pycache__"
    if os.access(cache_dir if cache_dir.exists() else sys.prefix, os.W_OK):
        sys.pycache_prefix = str(cache_dir)
        sys.dont_write_bytecode = False
    sys.argv = ["langgraph_cli", *sys.argv[3:]]
    runpy.run_module("langgraph_cli", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    _main()
