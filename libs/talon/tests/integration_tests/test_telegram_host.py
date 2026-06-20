from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents_talon.__main__ import _channels
from deepagents_talon.channels.telegram import TelegramChannel
from deepagents_talon.channels.whatsapp import WhatsAppChannel
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage, ChannelStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class RecordingChannel:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.handler: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self.started = False
        self.stopped = False
        self.status_report = ChannelStatus(provider=provider, connected=True, detail="connected")

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def set_message_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        self.handler = handler

    async def send_message(self, conversation_id: str, text: str) -> None:
        pass

    async def send_media(self, conversation_id: str, media: object) -> None:
        pass

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        pass

    async def status(self) -> ChannelStatus:
        return self.status_report

    async def receive(self, text: str, *, conversation_id: str = "chat") -> None:
        if self.handler is None:
            msg = "channel handler was not registered"
            raise AssertionError(msg)
        await self.handler(ChannelMessage(conversation_id=conversation_id, text=text))


class EchoAgent:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(text=f"reply:{request.text}")


def test_channels_factory_returns_both_when_both_enabled(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_ENABLED": "1",
            "DEEPAGENTS_TALON_TELEGRAM_ENABLED": "1",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
        },
        base_home=tmp_path,
    )

    channels = _channels(config, whatsapp=False, telegram=False)

    assert len(channels) == 2
    assert isinstance(channels[0], WhatsAppChannel)
    assert isinstance(channels[1], TelegramChannel)


def test_channels_factory_returns_only_telegram(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_ENABLED": "1",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
        },
        base_home=tmp_path,
    )

    channels = _channels(config, whatsapp=False, telegram=False)

    assert len(channels) == 1
    assert isinstance(channels[0], TelegramChannel)


def test_channels_factory_returns_only_whatsapp(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_ENABLED": "1",
        },
        base_home=tmp_path,
    )

    channels = _channels(config, whatsapp=False, telegram=False)

    assert len(channels) == 1
    assert isinstance(channels[0], WhatsAppChannel)


def test_channels_factory_returns_neither(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "assistant"},
        base_home=tmp_path,
    )

    channels = _channels(config, whatsapp=False, telegram=False)

    assert channels == ()


def test_channels_factory_accepts_cli_flags(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
        },
        base_home=tmp_path,
    )

    channels = _channels(config, whatsapp=True, telegram=True)

    assert len(channels) == 2


async def test_simultaneous_channels_coexist_without_interference(tmp_path: Path) -> None:
    whatsapp_channel = RecordingChannel("whatsapp")
    telegram_channel = RecordingChannel("telegram")
    agent = EchoAgent()
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "assistant"},
        base_home=tmp_path,
    )
    host = TalonHost(
        config=config,
        agent=agent,
        channels=[whatsapp_channel, telegram_channel],
    )

    await host.start()
    await whatsapp_channel.receive("hello from whatsapp", conversation_id="wa-chat")
    await telegram_channel.receive("hello from telegram", conversation_id="tg-chat")
    await _drain()
    await host.stop()

    assert whatsapp_channel.started
    assert whatsapp_channel.stopped
    assert telegram_channel.started
    assert telegram_channel.stopped
    assert len(agent.requests) == 2
    assert agent.requests[0].text == "hello from whatsapp"
    assert agent.requests[0].metadata["channel"] == "whatsapp"
    assert agent.requests[1].text == "hello from telegram"
    assert agent.requests[1].metadata["channel"] == "telegram"


async def test_stop_command_cancels_per_channel(tmp_path: Path) -> None:
    whatsapp_channel = RecordingChannel("whatsapp")
    telegram_channel = RecordingChannel("telegram")
    agent = EchoAgent()
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "assistant"},
        base_home=tmp_path,
    )
    host = TalonHost(
        config=config,
        agent=agent,
        channels=[whatsapp_channel, telegram_channel],
    )

    await host.start()
    await whatsapp_channel.receive("/stop", conversation_id="wa-chat")
    await _drain()
    await host.stop()

    assert len(agent.requests) == 0


async def _drain() -> None:
    for _ in range(100):
        await asyncio.sleep(0)
