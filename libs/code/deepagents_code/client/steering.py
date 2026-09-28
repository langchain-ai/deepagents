"""Coordinate native run interruption without overlapping terminal renderers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from deepagents_code.input import MediaTracker


@dataclass(frozen=True)
class SteeringInput:
    """A submitted prompt and its independently owned attachments."""

    text: str
    media: MediaTracker | None = None


@dataclass
class SteeringControl:
    """One pending user steer and the current stream's registration barrier."""

    pending: asyncio.Queue[SteeringInput] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1)
    )
    registered: asyncio.Event = field(default_factory=asyncio.Event)
    detached: bool = False
    accepting: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    unsent: SteeringInput | None = None
    cancelled: bool = field(default=False, init=False)

    def submit(self, text: str, media: MediaTracker | None = None) -> bool:
        """Return whether one steer was accepted without blocking the UI."""
        if self.cancelled or not self.accepting or self.pending.full():
            return False
        self.pending.put_nowait(
            SteeringInput(text, media.snapshot() if media is not None else None)
        )
        self.accepting = False
        return True


class SteeredError(Exception):
    """Transfer the renderer to a replacement run with a new user prompt."""

    def __init__(self, request: SteeringInput) -> None:
        """Retain the submitted prompt and attachments across stream teardown."""
        self.request = request
        super().__init__(request.text)


async def steerable_stream[T](
    stream: AsyncGenerator[T, None], control: SteeringControl
) -> AsyncGenerator[T, None]:
    """Yield stream events until a steer replaces the run.

    Raises:
        SteeredError: After the original run has been registered.
    """

    async def next_steer() -> SteeringInput:
        # Until the replacement registers, `unsent` still owns the previous
        # handoff. Leave additional input queued so a failed registration can
        # recover both requests and their attachments independently.
        if control.detached:
            await control.registered.wait()
        return await control.pending.get()

    next_chunk: asyncio.Task[T] | None = None
    steer = asyncio.create_task(next_steer())
    registered = asyncio.create_task(control.registered.wait())
    try:
        while True:
            next_chunk = asyncio.create_task(anext(stream))
            await asyncio.wait((next_chunk, steer), return_when=asyncio.FIRST_COMPLETED)
            if steer.done():
                await asyncio.wait(
                    (next_chunk, registered), return_when=asyncio.FIRST_COMPLETED
                )
                if control.registered.is_set():
                    control.detached = True
                    control.unsent = steer.result()
                    raise SteeredError(steer.result())
            try:
                yield await next_chunk
            except StopAsyncIteration:
                if steer.done():
                    control.unsent = steer.result()
                    if control.registered.is_set():
                        control.detached = True
                        raise SteeredError(steer.result()) from None
                break
    finally:
        control.accepting = False
        for task in (next_chunk, steer, registered):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (next_chunk, steer, registered) if task is not None),
            return_exceptions=True,
        )
        await stream.aclose()
        if steer.done() and not steer.cancelled() and not control.detached:
            control.unsent = steer.result()
