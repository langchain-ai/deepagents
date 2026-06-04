from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage, ChannelStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class RecordingChannel:
    def __init__(self) -> None:
        self.handler: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, str]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def set_message_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        self.handler = handler

    async def send_message(self, conversation_id: str, text: str) -> None:
        self.sent.append((conversation_id, text))

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        self.sent.append((conversation_id, f"{message_id}:{text}"))

    async def status(self) -> ChannelStatus:
        return ChannelStatus(provider="test", connected=True)


class RecordingScheduler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class BlockingAgent:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.requests: list[AgentRequest] = []
        self.released = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text == "block":
            await self.released.wait()
        return AgentResult(text=f"reply:{request.text}")


def _config(tmp_path: Path) -> TalonConfig:
    return TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)


async def test_host_starts_and_stops_components(tmp_path: Path) -> None:
    channel = RecordingChannel()
    scheduler = RecordingScheduler()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel], scheduler=scheduler)

    await host.start()
    await host.stop()

    assert agent.started is True
    assert agent.stopped is True
    assert scheduler.started is True
    assert scheduler.stopped is True
    assert channel.started is True
    assert channel.stopped is True
    assert channel.handler is not None


async def test_host_serializes_messages_per_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    first = asyncio.create_task(
        host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block")),
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second")),
    )
    await asyncio.sleep(0)

    assert [request.text for request in agent.requests] == ["block"]

    agent.released.set()
    await asyncio.gather(first, second)
    await host.stop()

    assert [request.text for request in agent.requests] == ["block", "second"]
    assert channel.sent == [("chat", "reply:block"), ("chat", "reply:second")]


async def test_stop_cancels_in_flight_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    running = asyncio.create_task(
        host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block")),
    )
    await _wait_for_request(agent, "block")

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/stop"))
    with contextlib.suppress(asyncio.CancelledError):
        await running
    await host.stop()

    assert channel.sent == [("chat", "Stopped current run.")]


async def _wait_for_request(agent: BlockingAgent, text: str) -> None:
    for _ in range(100):
        if any(request.text == text for request in agent.requests):
            return
        await asyncio.sleep(0)
    msg = f"agent did not receive request: {text}"
    raise AssertionError(msg)
