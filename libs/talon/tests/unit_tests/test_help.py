from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage, SendResult
from tests.conftest import RecordingChannel
from tests.test_host import (
    ApprovalAgent,
    AuthorizationAgent,
    BlockingAgent,
    _config,
    _wait_for_request,
    _wait_for_sent_count,
)

if TYPE_CHECKING:
    from pathlib import Path


class StalledHelpChannel(RecordingChannel):
    def __init__(self) -> None:
        super().__init__()
        self.help_started = asyncio.Event()
        self.release_help = asyncio.Event()

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        if "/help" in text:
            self.help_started.set()
            await self.release_help.wait()
        return await super().send_message(conversation_id, text)


@pytest.mark.parametrize("stop", [False, True])
async def test_stalled_help_allows_conversation_progress(tmp_path: Path, *, stop: bool) -> None:
    channel = StalledHelpChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    help_task = None
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
        await _wait_for_request(agent, "block")
        help_task = asyncio.create_task(
            host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/help"))
        )
        async with asyncio.timeout(1):
            await channel.help_started.wait()
            if stop:
                await host.receive_message(
                    channel, ChannelMessage(conversation_id="chat", text="/stop")
                )
            else:
                agent.released.set()
            await _wait_for_sent_count(channel, 1)
        expected = "Stopped current run." if stop else "reply:block"
        assert channel.sent == [("chat", expected)]
        assert not help_task.done()
    finally:
        channel.release_help.set()
        if help_task is not None:
            await help_task
        await host.stop()


@pytest.mark.parametrize("command", ["/help", " /HELP ", "/help@TestBot"])
async def test_help_replies_without_invoking_agent(tmp_path: Path, command: str) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text=command))
        assert agent.requests == []
        assert len(channel.sent) == 1
        conversation_id, text = channel.sent[0]
        assert conversation_id == "chat"
        for topic in ("/help", "/new", "/stop", "/mcp-reload", "MCP", "OAuth", "callback URL"):
            assert topic in text
    finally:
        await host.stop()


async def test_help_preserves_active_work(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
        await _wait_for_request(agent, "block")
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/help"))
        agent.released.set()
        await _wait_for_sent_count(channel, 2)
        assert channel.sent[-1] == ("chat", "reply:block")
        assert [request.text for request in agent.requests] == ["block"]
        assert agent.recoveries == []
    finally:
        await host.stop()


@pytest.mark.parametrize("authorization", [False, True])
async def test_help_preserves_pending_reply(tmp_path: Path, *, authorization: bool) -> None:
    channel = RecordingChannel(provider="telegram")
    agent = AuthorizationAgent() if authorization else ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        for text in ("start", "/help"):
            await host.receive_message(
                channel,
                ChannelMessage(conversation_id="chat", text=text, sender_id="operator"),
            )
            await _wait_for_sent_count(channel, 1 if text == "start" else 2)
        reply = "http://localhost:3000/callback?code=test&state=test" if authorization else "yes"
        await host.receive_message(
            channel,
            ChannelMessage(conversation_id="chat", text=reply, sender_id="operator"),
        )
        expected = "authorization:completed" if authorization else "decision:approve"
        await _wait_for_sent_count(channel, 4 if authorization else 3)
        assert channel.sent[-1] == ("chat", expected)
        assert [request.text for request in agent.requests] == ["start"]
    finally:
        await host.stop()
