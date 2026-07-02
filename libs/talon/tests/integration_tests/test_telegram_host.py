from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents_talon.__main__ import _channels
from deepagents_talon.channels.telegram import TelegramChannel
from deepagents_talon.channels.whatsapp import WhatsAppChannel
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult
from tests.conftest import RecordingChannel

if TYPE_CHECKING:
    from pathlib import Path


class EchoAgent:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []
        self.history: dict[str, list[str]] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        history = self.history.setdefault(request.conversation_id, [])
        seen = len(history)
        history.append(request.text)
        return AgentResult(text=f"seen:{seen}:{request.text}")


def test_channels_factory_selects_configured_channels(tmp_path: Path) -> None:
    cases: tuple[
        tuple[dict[str, str], bool, bool, str | None, tuple[type[object], ...]],
        ...,
    ] = (
        (
            {
                "DEEPAGENTS_TALON_WHATSAPP_ENABLED": "1",
                "DEEPAGENTS_TALON_TELEGRAM_ENABLED": "1",
                "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
                "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
            },
            False,
            False,
            None,
            (WhatsAppChannel, TelegramChannel),
        ),
        (
            {
                "DEEPAGENTS_TALON_TELEGRAM_ENABLED": "1",
                "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
                "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
            },
            False,
            False,
            None,
            (TelegramChannel,),
        ),
        (
            {"DEEPAGENTS_TALON_WHATSAPP_ENABLED": "1"},
            False,
            False,
            None,
            (WhatsAppChannel,),
        ),
        ({}, False, False, None, ()),
        (
            {
                "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
                "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
            },
            False,
            False,
            "telegram",
            (TelegramChannel,),
        ),
        ({}, False, False, "whatsapp", (WhatsAppChannel,)),
        (
            {
                "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "test-token",
                "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
            },
            True,
            True,
            None,
            (WhatsAppChannel, TelegramChannel),
        ),
    )

    for env, whatsapp, telegram, selected_channel, expected_types in cases:
        config = TalonConfig.from_env(
            {"AGENT_ASSISTANT_ID": "assistant", **env},
            base_home=tmp_path,
        )

        channels = _channels(
            config,
            whatsapp=whatsapp,
            telegram=telegram,
            selected_channel=selected_channel,
        )

        assert tuple(type(channel) for channel in channels) == expected_types


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
    await whatsapp_channel.receive("hello from whatsapp", conversation_id="chat")
    await telegram_channel.receive("hello from telegram", conversation_id="chat")
    await _drain()
    await host.stop()

    assert whatsapp_channel.started
    assert whatsapp_channel.stopped
    assert telegram_channel.started
    assert telegram_channel.stopped
    assert len(agent.requests) == 2
    assert agent.requests[0].conversation_id != agent.requests[1].conversation_id
    assert agent.requests[0].text == "hello from whatsapp"
    assert agent.requests[0].metadata["channel"] == "whatsapp"
    assert agent.requests[1].text == "hello from telegram"
    assert agent.requests[1].metadata["channel"] == "telegram"
    assert whatsapp_channel.sent == [("chat", "seen:0:hello from whatsapp")]
    assert telegram_channel.sent == [("chat", "seen:0:hello from telegram")]


async def _drain() -> None:
    for _ in range(100):
        await asyncio.sleep(0)
