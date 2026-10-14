from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.breakout import BREAKOUT_HANDLER, breakout
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage
from tests.channels.test_slack import _channel
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("title", "instructions"),
    [(" ", "task"), ("title", ""), ("x" * 161, "task"), ("title", "x" * 12001)],
)
async def test_invalid_input_does_not_create_thread(title: str, instructions: str) -> None:
    async def fail(_title: str, _instructions: str) -> str:
        pytest.fail("invalid input reached host")

    token = BREAKOUT_HANDLER.set(fail)
    try:
        result = await breakout.ainvoke({"title": title, "instructions": instructions})
        assert "must contain" in result
    finally:
        BREAKOUT_HANDLER.reset(token)


async def test_subagent_and_unbound_run_cannot_break_out() -> None:
    assert "unavailable" in await breakout.ainvoke({"title": "task", "instructions": "work"})
    token = _IN_SUBAGENT.set(True)
    try:
        assert "subagents" in await breakout.ainvoke({"title": "task", "instructions": "work"})
    finally:
        _IN_SUBAGENT.reset(token)


@pytest.mark.parametrize("sender", ["UOPERATOR", "UGUEST"])
async def test_breakout_keeps_parent_active_and_routes_child(tmp_path: Path, sender: str) -> None:
    channel, gateway, _, _ = _channel(tmp_path)
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("C1:100.1", "block", sender_id=sender))
        await asyncio.sleep(0)
        parent = agent.requests[0]
        assert parent.breakout_handler is not None
        response = await parent.breakout_handler("Investigate", "independent task")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        child = agent.requests[1]
        assert "started" in response
        assert child.conversation_id != parent.conversation_id
        assert child.text == "independent task"
        assert child.metadata["sender_id"] == sender
        assert child.metadata["tool_approval_operator"] is (sender == "UOPERATOR")
        assert not agent.released.is_set()
        assert not agent.recoveries
        assert gateway.posts[:2] == [
            ("C1", "Investigate", None),
            ("C1", "independent task", "1700000000.000001"),
        ]
        assert "unavailable" in await parent.breakout_handler("Again", "task")
        assert child.breakout_handler is not None
        assert "unavailable" in await child.breakout_handler("Recurse", "task")
        await asyncio.wait_for(host._tasks[child.conversation_id], timeout=2)
        assert any(
            post[1] == "reply:independent task" and post[2] == "1700000000.000001"
            for post in gateway.posts
        )
    finally:
        agent.released.set()
        await host.stop()


async def test_unsupported_channel_and_stale_turn(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("chat", "block", sender_id="sender"))
        await asyncio.sleep(0)
        handler = agent.requests[0].breakout_handler
        assert handler is not None
        assert "only on Slack" in await handler("title", "task")
        await host.receive_message(channel, ChannelMessage("chat", "/stop", sender_id="sender"))
        assert "unavailable" in await handler("title", "task")
        assert len(agent.requests) == 1
    finally:
        await host.stop()


async def test_creation_failure_never_starts_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel, _, _, _ = _channel(tmp_path)
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])

    async def fail(_conversation_id: str, _title: str, _instructions: str) -> str:
        msg = "transport failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(channel, "create_breakout", fail)
    await host.start()
    try:
        await host.receive_message(
            channel, ChannelMessage("C1:100.1", "block", sender_id="UOPERATOR")
        )
        await asyncio.sleep(0)
        handler = agent.requests[0].breakout_handler
        assert handler is not None
        assert "no task was started" in await handler("title", "task")
        assert len(agent.requests) == 1
    finally:
        agent.released.set()
        await host.stop()
