"""Where a slot's REPL work runs: a dedicated worker thread, or the caller."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

ExecutionMode = Literal["worker", "inline"]


class ReplWorker(Protocol):
    """The surface `_ThreadREPL` needs; `quickjs_rs.ThreadWorker` satisfies it."""

    def run_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run `coro` to completion, blocking the caller."""

    def run_async(self, coro: Coroutine[Any, Any, Any]) -> Awaitable[Any]:
        """Run `coro`; return something the caller's loop can await."""

    def close(self) -> None:
        """Release held resources. Idempotent."""


def _loop_running_here() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class InlineWorker:
    """Run REPL work on the calling thread and event loop, with no thread hop.

    `ThreadWorker` returns results through `asyncio.wrap_future`, which wakes
    the caller's loop with `call_soon_threadsafe`; a loop that only its owner
    drives never sees that wake-up, and the eval hangs. Here `run_async` is a
    plain `await` on the caller's loop, and `run_sync` runs on the calling
    thread: in place when a loop is already running there (setup work never
    suspends), otherwise on one private loop per worker.

    The caller takes on what the dedicated thread used to guarantee: one slot
    is not used concurrently from several threads, a long eval blocks the
    caller, one slot's evals go through either the sync or the async API
    (`quickjs_rs` binds a context's asyncio state to the first loop that
    drives it), and PTC tools are async where the host cannot be woken from
    an executor thread.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()

    def run_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run `coro` to completion on the calling thread."""
        if not _loop_running_here():
            return self._run_on_private_loop(coro)
        # Under a running loop the work cannot yield, so it must finish on
        # its first `send`; anything that suspends is a caller bug.
        try:
            coro.send(None)
        except StopIteration as done:
            return done.value
        msg = (
            "InlineWorker.run_sync was called from a running event loop with "
            "work that suspends; use the async REPL API from async code"
        )
        try:
            coro.close()
        except RuntimeError as exc:
            # Parked inside a TaskGroup, which cannot be closed synchronously.
            raise RuntimeError(msg) from exc
        raise RuntimeError(msg)

    async def run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Await `coro` on the caller's loop."""
        task = asyncio.current_task()
        cancelling = task.cancelling() if task is not None else 0
        result = await coro
        # A cancellation the JS swallowed still reaches the caller, as it
        # would through `wrap_future`.
        if task is not None and task.cancelling() > cancelling:
            raise asyncio.CancelledError
        return result

    def close(self) -> None:
        """Close the private loop, if any. Idempotent."""
        with self._lock:
            loop = self._loop
            if loop is None:
                return
            if loop.is_running():
                msg = "InlineWorker.close() was called from work running on its loop"
                raise RuntimeError(msg)
            self._loop = None
        if _loop_running_here():
            # Another loop owns this thread; `close()` alone still releases
            # the selector and shuts the default executor down without waiting.
            loop.close()
            return
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()  # executor shutdown without waiting, like ThreadWorker

    def _run_on_private_loop(self, coro: Coroutine[Any, Any, Any]) -> Any:
        # One loop for the worker's lifetime: `quickjs_rs` binds a context's
        # wake-up event to the first loop that waits on it. The lock also
        # serializes sync callers on different threads.
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(coro)
