from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_request, _wait_for_sent_count

if TYPE_CHECKING:
    from pathlib import Path


class LocalReloadableAgent(BlockingAgent):
    def __init__(self, *, fail_reload: bool = False) -> None:
        super().__init__()
        self.fail_reload = fail_reload
        self.reloads = 0

    async def reload_local_tools(self) -> None:
        self.reloads += 1
        if self.fail_reload:
            msg = "private-tool-source-secret"
            raise RuntimeError(msg)


@pytest.mark.parametrize("command", ["/tools-reload", " /TOOLS-RELOAD ", "/tools-reload@TestBot"])
async def test_tools_reload_replies_without_invoking_agent(tmp_path: Path, command: str) -> None:
    channel = RecordingChannel()
    agent = LocalReloadableAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text=command))
        assert agent.reloads == 1
        assert agent.requests == []
        assert channel.sent == [("chat", "Reloaded local Python tools.")]
    finally:
        await host.stop()


async def test_tools_reload_hides_import_errors(tmp_path: Path, caplog) -> None:
    channel = RecordingChannel()
    agent = LocalReloadableAgent(fail_reload=True)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(
            channel, ChannelMessage(conversation_id="chat", text="/tools-reload")
        )
        assert channel.sent == [("chat", "Could not reload local Python tools. Check Talon logs.")]
        assert "private-tool-source-secret" not in channel.sent[0][1]
        assert "Local Python tool reload failed" in caplog.text
        assert agent.requests == []
    finally:
        await host.stop()


async def test_tools_reload_reports_unsupported_runtime(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(
            channel, ChannelMessage(conversation_id="chat", text="/tools-reload")
        )
        assert channel.sent == [("chat", "Local Python tool reload is unavailable.")]
        assert agent.requests == []
    finally:
        await host.stop()


async def test_tools_reload_preserves_active_turn_and_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = LocalReloadableAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
        await _wait_for_request(agent, "block")
        original_id = agent.requests[0].conversation_id
        async with asyncio.timeout(1):
            await host.receive_message(
                channel, ChannelMessage(conversation_id="chat", text="/tools-reload")
            )
        assert agent.reloads == 1
        assert channel.sent == [("chat", "Reloaded local Python tools.")]
        assert [request.text for request in agent.requests] == ["block"]
        assert agent.recoveries == []
        agent.released.set()
        await _wait_for_sent_count(channel, 2)
        assert channel.sent[-1] == ("chat", "reply:block")
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="next"))
        await _wait_for_sent_count(channel, 3)
        assert [request.conversation_id for request in agent.requests] == [original_id, original_id]
        assert channel.sent[-1] == ("chat", "reply:next")
    finally:
        agent.released.set()
        await host.stop()
