"""Process-local lifecycle hooks for resources owned by the agent graph."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager

from starlette.applications import Starlette

CleanupCallback = Callable[[], Awaitable[None]]


class ServerResources:
    """Own asynchronous cleanup callbacks for process-lifetime server resources.

    Resource construction can run in worker threads while shutdown runs on the
    server event loop. Registrations are therefore protected by a thread lock and
    atomically transferred to an ``AsyncExitStack`` during cleanup.
    """

    def __init__(self) -> None:
        """Create an empty resource registry."""
        self._lock = threading.Lock()
        self._cleanups: list[CleanupCallback] = []

    def add_cleanup(self, callback: CleanupCallback) -> None:
        """Register an asynchronous callback for reverse-order cleanup.

        Args:
            callback: Zero-argument asynchronous cleanup callback.
        """
        with self._lock:
            self._cleanups.append(callback)

    async def aclose(self) -> None:
        """Consume and await every registered cleanup callback."""
        with self._lock:
            cleanups = self._cleanups
            self._cleanups = []

        async with AsyncExitStack() as stack:
            for cleanup in cleanups:
                stack.push_async_callback(cleanup)


server_resources = ServerResources()


@asynccontextmanager
async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
    """Close process-lifetime resources when the LangGraph server shuts down."""
    try:
        yield
    finally:
        await server_resources.aclose()


app = Starlette(lifespan=_lifespan)
