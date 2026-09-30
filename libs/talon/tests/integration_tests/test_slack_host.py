from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.slack import (
    SlackChannel,
    SlackChannelConfig,
    _SlackInboundCommand,
    _SlackInboundMessage,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ToolApprovalRequest

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


class ApprovalAgent(EchoAgent):
    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        assert request.approval_handler is not None
        decision = await request.approval_handler(
            ToolApprovalRequest(
                conversation_id=request.conversation_id,
                interrupt_id="interrupt-1",
                action_requests=({"name": "dangerous_tool", "args": {}},),
            )
        )
        return AgentResult(text=f"decision:{decision}")


class StubGateway:
    """Delivers inbound Slack events without a Socket Mode connection."""

    def __init__(self) -> None:
        self.bot_id = "UBOT"
        self.posts: list[tuple[str, str, str | None]] = []
        self.context: list[tuple[str, str]] = []
        self.handle_message = None
        self.handle_command = None

    async def start(self, *, handle_message, handle_reaction, handle_connection, handle_command):
        del handle_reaction, handle_connection
        self.handle_message = handle_message
        self.handle_command = handle_command

    async def stop(self):
        pass

    async def post_message(self, channel_id, text, *, thread_ts):
        self.posts.append((channel_id, text, thread_ts))
        return f"1700000100.{len(self.posts):06d}"

    async def thread_context(self, channel_id, thread_ts, before_ts):
        del channel_id, thread_ts, before_ts
        return self.context

    async def upload_file(self, channel_id, file_path, *, thread_ts, comment):
        del channel_id, file_path, thread_ts, comment

    async def update_message(self, channel_id, ts, text):
        del channel_id, ts, text


class CapturingResponder:
    def __init__(self) -> None:
        self.rejects: list[str] = []
        self.sends: list[str] = []

    async def reject(self, text):
        self.rejects.append(text)

    async def send(self, text):
        self.sends.append(text)


async def _drain() -> None:
    for _ in range(80):
        await asyncio.sleep(0)


def _host(tmp_path: Path) -> tuple[TalonHost, StubGateway]:
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant"}, base_home=tmp_path)
    gateway = StubGateway()
    channel = SlackChannel(
        SlackChannelConfig(
            bot_token="xoxb-test",  # noqa: S106  # inert test token
            app_token="xapp-test",  # noqa: S106  # inert test token
            inbound_media_dir=tmp_path / "inbound",
            exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"UOP"})),
        ),
        gateway=gateway,
    )
    return TalonHost(config=config, agent=EchoAgent(), channels=[channel]), gateway


def _mention(ts: str, *, thread_ts: str | None = None, text: str = "hi") -> _SlackInboundMessage:
    return _SlackInboundMessage(
        channel_id="C1",
        ts=ts,
        thread_ts=thread_ts,
        sender_id="UOP",
        text=text,
        is_dm=False,
    )


async def test_channel_mention_is_answered_in_its_thread(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    await host.start()
    await gateway.handle_message(_mention("1700000000.000100"))
    await _drain()
    await host.stop()

    assert gateway.posts == [("C1", "echo:hi", "1700000000.000100")]


async def test_thread_follow_up_keeps_the_thread_conversation(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    agent = host.agent
    await host.start()
    await gateway.handle_message(_mention("1700000000.000100", text="first"))
    await _drain()
    await gateway.handle_message(
        _mention("1700000000.000200", thread_ts="1700000000.000100", text="second"),
    )
    await _drain()
    await gateway.handle_message(_mention("1700000000.000300", text="other thread"))
    await _drain()
    await host.stop()

    threads = [request.conversation_id for request in agent.requests]
    assert threads[0] == threads[1]
    assert threads[2] != threads[0]
    assert [thread_ts for _, _, thread_ts in gateway.posts] == [
        "1700000000.000100",
        "1700000000.000100",
        "1700000000.000300",
    ]
    assert [request.metadata["history_chat"] for request in agent.requests] == ["C1"] * 3


async def test_talon_help_is_answered_through_the_command(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    responder = CapturingResponder()
    await host.start()
    await gateway.handle_command(
        _SlackInboundCommand(
            command="help",
            channel_id="D1",
            sender_id="UOP",
            trigger_id="trig-1",
            responder=responder,
        ),
    )
    await _drain()
    await host.stop()

    assert len(responder.sends) == 1
    assert "/new" in responder.sends[0]
    assert gateway.posts == []


async def test_thread_context_reaches_model_without_replacing_current_message(
    tmp_path: Path,
) -> None:
    host, gateway = _host(tmp_path)
    gateway.context = [("UOP", "please schedule hello in one minute")]
    await host.start()
    try:
        await gateway.handle_message(
            _mention("1700000000.000200", thread_ts="1700000000.000100", text="do it"),
        )
        await _drain()
        assert len(host.agent.requests) == 1
        prompt = host.agent.requests[0].text
        assert "UOP: please schedule hello in one minute" in prompt
        assert prompt.endswith("Current message from UOP: do it")
    finally:
        await host.stop()


@pytest.mark.parametrize(
    ("command", "reply"),
    [
        ("/new", "Started a fresh conversation."),
        ("/stop", "No in-flight run to stop."),
        ("/help", "/new"),
        ("/model", "Model switching is unavailable."),
    ],
)
async def test_thread_context_does_not_interfere_with_commands(
    tmp_path: Path, command: str, reply: str
) -> None:
    host, gateway = _host(tmp_path)
    gateway.context = [("UOP", "an earlier request")]
    await host.start()
    try:
        await gateway.handle_message(
            _mention("1700000000.000200", thread_ts="1700000000.000100", text=command),
        )
        await _drain()
        assert host.agent.requests == []
        assert len(gateway.posts) == 1
        assert reply.lower() in gateway.posts[0][1].lower()
    finally:
        await host.stop()


@pytest.mark.parametrize(("reply", "decision"), [("yes", "approve"), ("no", "reject")])
async def test_thread_context_does_not_interfere_with_approval_replies(
    tmp_path: Path, reply: str, decision: str
) -> None:
    host, gateway = _host(tmp_path)
    host.agent = ApprovalAgent()
    gateway.context = [("UOP", "an earlier request")]
    await host.start()
    try:
        await gateway.handle_message(
            _mention("1700000000.000200", thread_ts="1700000000.000100", text="run"),
        )
        await _drain()
        assert "Tool approval required." in gateway.posts[0][1]
        await gateway.handle_message(
            _mention("1700000000.000300", thread_ts="1700000000.000100", text=reply),
        )
        await _drain()
        assert len(host.agent.requests) == 1
        assert gateway.posts[-1] == ("C1", f"decision:{decision}", "1700000000.000100")
    finally:
        await host.stop()
