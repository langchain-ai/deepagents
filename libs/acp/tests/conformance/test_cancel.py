"""Conformance: `session/cancel` and stop reasons."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import pytest
from acp.exceptions import RequestError
from langchain_core.tools import StructuredTool

from tests.conformance.harness import (
    Harness,
    call,
    check_tool_call,
    deep_agent,
    echo,
    reply,
    scripted_model,
)

if TYPE_CHECKING:
    from acp.schema import PromptResponse


class ToolNeverStartedError(Exception):
    """The turn did not reach the blocking tool, so cancellation was never exercised."""


class CancelIgnoredError(Exception):
    """The turn was still running when the post-cancel deadline passed."""


async def _wait_until_started(turn: asyncio.Task[PromptResponse], started: asyncio.Event) -> None:
    waiter = asyncio.create_task(started.wait())
    try:
        await asyncio.wait({turn, waiter}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if not started.is_set():
        if turn.done():
            turn.result()
        msg = "the turn never reached the blocking tool"
        raise ToolNeverStartedError(msg)


async def _stop(turn: asyncio.Task[PromptResponse]) -> None:
    if turn.done():
        return
    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn


async def test_cancel_during_a_blocked_tool_returns_cancelled_promptly() -> None:
    started = asyncio.Event()

    async def wait_forever(reason: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tool = StructuredTool.from_function(coroutine=wait_forever, name="wait_forever")
    model = scripted_model(reply("", call("wait_forever", {"reason": "test"})))
    harness = Harness.start(deep_agent(model, tools=[tool]))
    session_id = await harness.new_session()

    turn = asyncio.create_task(harness.prompt(session_id, "Wait"))
    try:
        await _wait_until_started(turn, started)
        await harness.cancel(session_id)
        # Never cancel the turn here: the server's cancel must end it on its own.
        done, _ = await asyncio.wait({turn}, timeout=2)
        if not done:
            msg = "the turn was still running 2s after session/cancel"
            raise CancelIgnoredError(msg)
        assert turn.result().stop_reason == "cancelled"
    finally:
        await _stop(turn)


async def test_cancel_fails_the_open_tool_call_and_frees_the_session() -> None:
    started = asyncio.Event()

    async def wait_forever(reason: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tool = StructuredTool.from_function(coroutine=wait_forever, name="wait_forever")
    model = scripted_model(
        reply("", call("echo", {"text": "hi"}, call_id="before-cancel")),
        reply("", call("wait_forever", {"reason": "test"})),
        reply("", call("echo", {"text": "hi"}, call_id="after-cancel")),
        reply("Again"),
    )
    harness = Harness.start(deep_agent(model, tools=[echo, tool]))
    session_id = await harness.new_session()

    turn = asyncio.create_task(harness.prompt(session_id, "Wait"))
    try:
        await _wait_until_started(turn, started)
        with pytest.raises(RequestError):
            await harness.prompt(session_id, "Busy?")
        await harness.cancel(session_id)
        assert (await asyncio.wait_for(turn, timeout=2)).stop_reason == "cancelled"
    finally:
        await _stop(turn)

    check_tool_call(harness.client.tool_call(session_id, "call-wait_forever"), "failed")
    assert (await harness.prompt(session_id, "Again")).stop_reason == "end_turn"
    check_tool_call(harness.client.tool_call(session_id, "before-cancel"), "completed")
    check_tool_call(harness.client.tool_call(session_id, "call-wait_forever"), "failed")
    check_tool_call(harness.client.tool_call(session_id, "after-cancel"), "completed")
