"""Where a slot's REPL work runs: a dedicated worker thread, or the caller."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

ExecutionMode = Literal["worker", "inline"]


class ReplWorker(Protocol):
    """Runs REPL coroutines for one slot.

    `quickjs_rs.ThreadWorker` satisfies this by construction; `InlineWorker`
    is the alternative that stays on the caller's thread.
    """

    def run_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run `coro` to completion and return its result, blocking the caller."""

    def run_async(self, coro: Coroutine[Any, Any, Any]) -> Awaitable[Any]:
        """Schedule `coro` and return something the caller's loop can await."""

    def close(self) -> None:
        """Release whatever the worker holds. Idempotent."""


def _loop_running_here() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class InlineWorker:
    """Run REPL work on the calling thread and event loop.

    `ThreadWorker` hosts each slot on its own OS thread and hands results
    back through `asyncio.wrap_future`, which wakes the caller's loop with
    `call_soon_threadsafe`. That assumes the caller's loop can be woken from
    another thread. Some loops cannot: a Temporal workflow's loop, for one,
    only runs while its workflow task is being processed, so the wake-up
    never lands and the eval hangs. `InlineWorker` removes the hop: async
    work is a plain `await` on the caller's loop, and synchronous work runs
    on the calling thread.

    The caller takes over what the dedicated thread used to guarantee:

    * One slot must not be used concurrently from several threads.
      Synchronous calls made without a running loop are serialized on a
      private loop owned by this worker; concurrent async calls interleave
      on the caller's loop exactly as they would on the worker loop.
    * A long eval blocks the calling thread, and its loop, until it ends.
    * Synchronous calls made *from* a running loop's thread must not
      suspend. The middleware only makes such calls for slot setup, PTC
      installation, snapshots, and close, none of which await anything.
    * Drive one slot's evals through either the sync API or the async API,
      not both: `quickjs_rs` keeps loop-bound asyncio state on a context,
      and alternating between the private loop and the caller's loop trips
      it. (Non-suspending setup calls are fine on either.) The middleware
      honours this on its own, since a slot lives for one turn and
      LangGraph drives a turn one way.
    * On a host that cannot be woken from other threads, PTC tools should
      be async: a sync tool runs in an executor thread, and its completion
      is exactly the cross-thread wake-up being avoided.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()

    def run_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run `coro` to completion on the calling thread."""
        if not _loop_running_here():
            return self._run_on_private_loop(coro)
        # Called synchronously from inside a running loop: there is no way
        # to yield to that loop from here, so the work has to finish without
        # suspending. A coroutine that never awaits completes on its first
        # `send`; anything else is a caller bug, reported instead of hung.
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
            # A coroutine parked inside an `asyncio.TaskGroup` cannot be
            # closed synchronously ("coroutine ignored GeneratorExit"); the
            # caller's mistake is still the suspension, so report that.
            raise RuntimeError(msg) from exc
        raise RuntimeError(msg)

    async def run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Await `coro` on the caller's loop."""
        task = asyncio.current_task()
        cancelling = task.cancelling() if task is not None else 0
        result = await coro
        # A cancellation the coroutine swallowed (JS may catch the host-side
        # error) must still reach the caller, as it does when the result
        # crosses `asyncio.wrap_future` from a worker thread.
        if task is not None and task.cancelling() > cancelling:
            raise asyncio.CancelledError
        return result

    def close(self) -> None:
        """Close the private loop, if one was ever created. Idempotent."""
        with self._lock:
            loop = self._loop
            if loop is None:
                return
            if loop.is_running():
                msg = "InlineWorker.close() was called from work running on its loop"
                raise RuntimeError(msg)
            self._loop = None
        if _loop_running_here():
            # Another loop owns this thread, so ours cannot spin for its
            # shutdown bookkeeping. `close()` still releases the selector and
            # tells the default executor to shut down without waiting.
            loop.close()
            return
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            # `close()` shuts the default executor down without waiting, as
            # `ThreadWorker.close()` does; waiting for it here could block on
            # a tool still running in an executor thread.
            loop.close()

    def _run_on_private_loop(self, coro: Coroutine[Any, Any, Any]) -> Any:
        # One loop for the worker's lifetime rather than `asyncio.run` per
        # call: `quickjs_rs` keeps a wake-up `asyncio.Event` on the context
        # across evals, and that event binds to the first loop that waits
        # on it. The lock also serializes sync callers on different threads,
        # which the worker thread used to do by construction.
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(coro)
