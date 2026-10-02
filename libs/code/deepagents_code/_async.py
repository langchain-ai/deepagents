"""Async task helpers shared by server operations."""

from __future__ import annotations

import asyncio


async def _join_task_deferring_cancellation[T](
    task: asyncio.Task[T],
) -> asyncio.CancelledError | None:
    """Join a settlement task while retaining the first cancellation edge.

    The caller inspects `task.result()` before re-raising the cancellation.

    Returns:
        The cancellation to re-raise after settlement, or `None`.
    """
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
    return cancellation
