from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.discord import (
    DiscordChannel,
    DiscordChannelConfig,
    _DiscordInboundInteraction,
    _DiscordInboundMessage,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult

if TYPE_CHECKING:
    from pathlib import Path


class EchoAgent:
    """Minimal runtime: echoes request text so replies are identifiable."""

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(text=f"echo:{request.text}")

    async def recover_interrupted(self, conversation_id: str) -> None:
        del conversation_id


class StubGateway:
    """Delivers inbound Discord events without a Gateway connection."""

    def __init__(self) -> None:
        self.bot_id = "bot-1"
        self.sent_text: list[tuple[str, str]] = []
        self._handle_interaction = None
        self._handle_message = None

    async def start(
        self, *, handle_message, handle_reaction, handle_connection, handle_interaction
    ):
        del handle_reaction, handle_connection
        self._handle_message = handle_message
        self._handle_interaction = handle_interaction

    async def stop(self):
        pass

    async def send_message(self, channel_id, text):
        self.sent_text.append((channel_id, text))
        return "m1"

    async def send_file(self, channel_id, file_path, *, content):
        del channel_id, file_path, content
        return "m1"

    async def edit_message(self, channel_id, message_id, text):
        del channel_id, message_id, text

    async def trigger_typing(self, channel_id):
        del channel_id


class CapturingResponder:
    """Records how one interaction was answered."""

    def __init__(self) -> None:
        self.rejects: list[str] = []
        self.sends: list[str] = []
        self.defers = 0

    async def reject(self, text):
        self.rejects.append(text)

    async def defer(self):
        self.defers += 1

    async def send(self, text):
        self.sends.append(text)
        return "followup-1"


async def _drain() -> None:
    for _ in range(80):
        await asyncio.sleep(0)


def _host(tmp_path: Path) -> tuple[TalonHost, StubGateway, DiscordChannel]:
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant"}, base_home=tmp_path)
    gateway = StubGateway()
    channel = DiscordChannel(
        DiscordChannelConfig(
            bot_token="test-token",  # noqa: S106  # inert test token
            inbound_media_dir=tmp_path / "inbound",
            exposure=ChannelExposure(
                mode=ExposureMode.SELF,
                operator_ids=frozenset({"op-1"}),
            ),
        ),
        gateway=gateway,
    )
    host = TalonHost(config=config, agent=EchoAgent(), channels=[channel])
    return host, gateway, channel


async def test_slash_command_runs_the_host_command_and_answers_the_interaction(
    tmp_path: Path,
) -> None:
    """A `/new` interaction must drive the real host command and reply through Discord."""
    host, gateway, _channel = _host(tmp_path)
    responder = CapturingResponder()
    await host.start()

    await gateway._handle_interaction(
        _DiscordInboundInteraction(
            command="new",
            channel_id="chan-1",
            sender_id="op-1",
            interaction_id="int-1",
            is_dm=True,
            responder=responder,
        ),
    )
    await _drain()
    await host.stop()

    assert responder.defers == 1
    assert responder.sends == ["Started a fresh conversation."]
    assert gateway.sent_text == []


async def test_slash_command_new_starts_a_fresh_thread(tmp_path: Path) -> None:
    """`/new` from an interaction must advance the reset counter like the typed command."""
    host, gateway, _channel = _host(tmp_path)
    agent = host.agent
    await host.start()

    async def message(text: str) -> None:
        await gateway._handle_message(
            _DiscordInboundMessage(
                channel_id="chan-1",
                message_id="m",
                sender_id="op-1",
                text=text,
                is_dm=True,
                from_self=False,
            ),
        )
        await _drain()

    await message("before")
    await gateway._handle_interaction(
        _DiscordInboundInteraction(
            command="new",
            channel_id="chan-1",
            sender_id="op-1",
            interaction_id="int-1",
            is_dm=True,
            responder=CapturingResponder(),
        ),
    )
    await _drain()
    await message("after")
    await host.stop()

    threads = [request.conversation_id for request in agent.requests]
    assert len(threads) == 2
    assert threads[0] != threads[1], "expected /new to move the conversation to a new thread"


async def test_slash_command_stop_reports_no_running_work(tmp_path: Path) -> None:
    host, gateway, _channel = _host(tmp_path)
    responder = CapturingResponder()
    await host.start()

    await gateway._handle_interaction(
        _DiscordInboundInteraction(
            command="stop",
            channel_id="chan-1",
            sender_id="op-1",
            interaction_id="int-1",
            is_dm=True,
            responder=responder,
        ),
    )
    await _drain()
    await host.stop()

    assert responder.sends == ["No in-flight run to stop."]


async def test_slash_command_help_answers_the_interaction(tmp_path: Path) -> None:
    host, gateway, _channel = _host(tmp_path)
    responder = CapturingResponder()
    await host.start()

    await gateway._handle_interaction(
        _DiscordInboundInteraction(
            command="help",
            channel_id="chan-1",
            sender_id="op-1",
            interaction_id="int-1",
            is_dm=True,
            responder=responder,
        ),
    )
    await _drain()
    await host.stop()

    assert len(responder.sends) == 1
    assert "/new — Stop current work and start a fresh conversation." in responder.sends[0]
    assert gateway.sent_text == []


async def test_unauthorized_slash_command_never_reaches_the_agent(tmp_path: Path) -> None:
    host, gateway, _channel = _host(tmp_path)
    agent = host.agent
    responder = CapturingResponder()
    await host.start()

    await gateway._handle_interaction(
        _DiscordInboundInteraction(
            command="new",
            channel_id="chan-1",
            sender_id="intruder",
            interaction_id="int-1",
            is_dm=True,
            responder=responder,
        ),
    )
    await _drain()
    await host.stop()

    assert responder.rejects
    assert responder.defers == 0
    assert responder.sends == []
    assert agent.requests == []


async def test_agent_reply_is_not_captured_by_an_interaction(tmp_path: Path) -> None:
    """An ordinary turn's reply must go to the channel, not into a slash command."""
    host, gateway, _channel = _host(tmp_path)
    await host.start()

    await gateway._handle_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="hello",
            is_dm=True,
            from_self=False,
        ),
    )
    await _drain()
    await host.stop()

    assert gateway.sent_text == [("chan-1", "echo:hello")]
