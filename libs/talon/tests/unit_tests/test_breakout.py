from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.breakout import BREAKOUT_HANDLER, breakout
from deepagents_talon.channels.discord import DiscordChannel, _DiscordPyGateway
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage
from tests.channels.test_discord import RecordingGateway, _make_config
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
        assert "only on Slack and Discord" in await handler("title", "task")
        await host.receive_message(channel, ChannelMessage("chat", "/stop", sender_id="sender"))
        assert "unavailable" in await handler("title", "task")
        assert len(agent.requests) == 1
    finally:
        await host.stop()


@pytest.mark.parametrize("provider", ["slack", "discord"])
async def test_creation_failure_never_starts_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    channel, _, _, _ = _channel(tmp_path)
    if provider == "discord":
        channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())
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


@pytest.mark.parametrize("sender", ["operator", "guest"])
async def test_discord_breakout_routes_independent_child(tmp_path: Path, sender: str) -> None:
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("123", "block", sender_id=sender))
        await asyncio.sleep(0)
        parent = agent.requests[0]
        assert parent.breakout_handler is not None
        assert "started" in await parent.breakout_handler("Investigate", "independent task")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        child = agent.requests[1]
        assert child.conversation_id != parent.conversation_id
        assert child.metadata["sender_id"] == sender
        assert child.metadata["tool_approval_operator"] is (sender == "operator")
        assert not agent.released.is_set()
        await asyncio.wait_for(host._tasks[child.conversation_id], timeout=2)
        assert ("456", "reply:independent task") in gateway.sent_text
        assert gateway.breakouts == [("123", "Investigate", "independent task")]
    finally:
        agent.released.set()
        await host.stop()


@pytest.mark.parametrize("public_thread", [False, True])
async def test_discord_gateway_creates_thread_without_mentions(*, public_thread: bool) -> None:
    credential = "test"
    gateway = _DiscordPyGateway(token=credential, connect_timeout_seconds=1)
    channel = MagicMock(spec=discord.TextChannel)
    message = MagicMock()
    thread = MagicMock(spec=discord.Thread)
    thread.id = 456
    thread.send = AsyncMock()
    message.create_thread = AsyncMock(return_value=thread)
    channel.send = AsyncMock(return_value=message)
    source = channel
    if public_thread:
        source = MagicMock(spec=discord.Thread)
        source.type = discord.ChannelType.public_thread
        source.parent = channel
    gateway._fetch_channel = AsyncMock(return_value=source)
    assert await gateway.create_breakout("123", "title", "x" * 2100) == "456"
    assert channel.send.call_args.kwargs["allowed_mentions"].everyone is False
    assert len(thread.send.call_args_list) == 2
    assert all(
        call.kwargs["allowed_mentions"].users is False for call in thread.send.call_args_list
    )


@pytest.mark.parametrize("kind", ["dm", "private", "news"])
async def test_discord_gateway_rejects_unsupported_sources(kind: str) -> None:
    credential = "test"
    gateway = _DiscordPyGateway(token=credential, connect_timeout_seconds=1)
    source = MagicMock(spec=discord.DMChannel if kind == "dm" else discord.Thread)
    source.type = (
        discord.ChannelType.private_thread if kind == "private" else discord.ChannelType.news_thread
    )
    gateway._fetch_channel = AsyncMock(return_value=source)
    with pytest.raises(discord.ClientException, match="guild text channel"):
        await gateway.create_breakout("123", "title", "task")
    source.send.assert_not_called()
