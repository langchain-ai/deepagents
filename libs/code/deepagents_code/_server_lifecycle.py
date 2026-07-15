"""Process-local lifecycle hooks for resources owned by the agent graph."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from starlette.applications import Starlette

CleanupCallback = Callable[[], Awaitable[None]]

_cleanup_lock = threading.Lock()
_browser_cleanup: CleanupCallback | None = None


def register_browser_cleanup(callback: CleanupCallback) -> None:
    """Register the browser cleanup callback exactly once for this process.

    Graph construction runs in ``asyncio.to_thread``, while the lifespan runs on
    the server event loop, so registry access is protected by a thread lock.

    Args:
        callback: Async cleanup callback owned by the browser middleware.

    Raises:
        RuntimeError: If a browser cleanup callback is already registered.
    """
    global _browser_cleanup  # noqa: PLW0603

    with _cleanup_lock:
        if _browser_cleanup is not None:
            msg = "Browser cleanup is already registered"
            raise RuntimeError(msg)
        _browser_cleanup = callback


def _consume_browser_cleanup() -> CleanupCallback | None:
    """Atomically remove and return the registered browser cleanup callback.

    Returns:
        The registered callback, or ``None`` when no callback is registered.
    """
    global _browser_cleanup  # noqa: PLW0603

    with _cleanup_lock:
        callback = _browser_cleanup
        _browser_cleanup = None
        return callback


@asynccontextmanager
async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
    """Await browser cleanup on server shutdown without suppressing failures."""
    try:
        yield
    finally:
        callback = _consume_browser_cleanup()
        if callback is not None:
            await callback()


app = Starlette(lifespan=_lifespan)
