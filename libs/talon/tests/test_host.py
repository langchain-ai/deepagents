from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, cast

from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    ChannelMedia,
    ChannelMessage,
    ChannelReaction,
    SendResult,
    ToolApprovalRequest,
)
from tests.conftest import RecordingChannel

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from deepagents_talon.authorization import (
    AuthorizationBinding,
    AuthorizationCompleted,
    AuthorizationURL,
    CallbackURLRequested,
)


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
        self.recoveries: list[str] = []
        self.released = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def recover_interrupted(self, conversation_id: str) -> None:
        self.recoveries.append(conversation_id)

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text == "block":
            await self.released.wait()
        return AgentResult(text=f"reply:{request.text}")


class FailingRecoveryAgent(BlockingAgent):
    async def recover_interrupted(self, conversation_id: str) -> None:
        self.recoveries.append(conversation_id)
        message = "persistence failed"
        raise RuntimeError(message)


class CancellationResistantAgent(BlockingAgent):
    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text == "block":
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                assert task is not None
                task.uncancel()
                await self.released.wait()
        return AgentResult(text=f"reply:{request.text}")


class HistoryAgent:
    def __init__(self) -> None:
        self.history: dict[str, list[str]] = {}
        self.requests: list[AgentRequest] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def recover_interrupted(self, conversation_id: str) -> None:
        self.history.setdefault(conversation_id, []).append("[recovered]")

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        history = self.history.setdefault(request.conversation_id, [])
        seen = len(history)
        history.append(request.text)
        return AgentResult(text=f"seen:{seen}")


class VoiceTranscriber:
    async def transcribe(self, message: ChannelMessage) -> str | None:
        del message
        return "transcribed voice"


class MediaAgent(BlockingAgent):
    def __init__(self, image: Path | str) -> None:
        super().__init__()
        self.image = str(image)

    async def invoke(self, request: AgentRequest) -> AgentResult:
        del request
        return AgentResult(text=f"Here is the image.\n\n![chart]({self.image})")


class ApprovalAgent(BlockingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.approvals: list[ToolApprovalRequest] = []

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.approval_handler is None:
            msg = "approval handler was missing"
            raise TypeError(msg)
        approval = ToolApprovalRequest(
            conversation_id=request.conversation_id,
            interrupt_id="interrupt-1",
            action_requests=(
                {
                    "name": "dangerous_tool",
                    "args": {"path": "/secret"},
                },
            ),
        )
        self.approvals.append(approval)
        decision = await request.approval_handler(approval)
        return AgentResult(text=f"decision:{decision}")


class AuthorizationAgent(BlockingAgent):
    def __init__(self, *, terminal: bool = False) -> None:
        super().__init__()
        self.callbacks: list[str] = []
        self.terminal = terminal

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.authorization_handler is None:
            msg = "authorization handler was missing"
            raise TypeError(msg)
        binding = AuthorizationBinding(
            server_name="notion",
            invocation_id="tool-call-1",
            expires_at=asyncio.get_running_loop().time() + 30,
        )
        await request.authorization_handler(
            AuthorizationURL(
                binding=binding,
                url="https://auth.example/authorize?state=sensitive-state",
            )
        )
        callback = await request.authorization_handler(CallbackURLRequested(binding=binding))
        assert callback is not None
        self.callbacks.append(callback)
        await request.authorization_handler(
            AuthorizationCompleted(binding=binding, terminal=self.terminal)
        )
        return AgentResult(text="authorization:completed")


class CompletionFailingChannel(RecordingChannel):
    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        if text == "MCP server `notion` is authorized.":
            return SendResult(success=False, error="permanent failure")
        return await super().send_message(conversation_id, text)


class ExpiringAuthorizationAgent(BlockingAgent):
    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        assert request.authorization_handler is not None
        binding = AuthorizationBinding(
            server_name="notion",
            invocation_id="tool-call-expiring",
            expires_at=asyncio.get_running_loop().time() + 0.01,
        )
        await request.authorization_handler(
            AuthorizationURL(binding=binding, url="https://auth.example/authorize")
        )
        with pytest.raises(TimeoutError):
            await request.authorization_handler(CallbackURLRequested(binding=binding))
        return AgentResult(text="authorization:expired")


def _config(tmp_path: Path, env: dict[str, str] | None = None) -> TalonConfig:
    return TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test", **(env or {})}, base_home=tmp_path)


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


@pytest.mark.parametrize("provider", ["whatsapp", "telegram", "discord"])
async def test_channel_authorization_intercepts_bound_callback_outside_model(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    provider: str,
) -> None:
    channel = RecordingChannel(provider=provider)
    other_channel = RecordingChannel(provider="telegram" if provider == "whatsapp" else "whatsapp")
    agent = AuthorizationAgent()
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[channel, other_channel],
    )
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    callback = "http://localhost:3118/callback?code=sensitive-code&state=sensitive-state"
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text=callback, sender_id="attacker"),
    )
    assert agent.callbacks == []
    await host.receive_message(
        other_channel,
        ChannelMessage(conversation_id="chat", text=callback, sender_id="operator"),
    )
    assert agent.callbacks == []
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text=f"<{callback}>", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 4)
    await host.stop()

    assert [request.text for request in agent.requests] == ["login"]
    assert agent.callbacks == [callback]
    assert "sensitive-code" not in caplog.text
    assert "sensitive-state" not in caplog.text
    assert "Only the operator" in channel.sent[1][1]
    assert other_channel.sent == [("chat", "No matching MCP authorization request is pending.")]
    assert channel.sent[-2:] == [
        ("chat", "MCP server `notion` is authorized."),
        ("chat", "authorization:completed"),
    ]


async def test_terminal_channel_authorization_suppresses_redundant_agent_result(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel(provider="telegram")
    agent = AuthorizationAgent(terminal=True)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    callback = "http://localhost:3000/callback?code=secret-code&state=secret-state"
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text=callback, sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent == [
        (
            "chat",
            "Authorization required for MCP server `notion`.\n"
            "Open this link and approve access:\n"
            "https://auth.example/authorize?state=sensitive-state\n"
            "Then paste the full callback URL here.",
        ),
        ("chat", "MCP server `notion` is authorized."),
    ]


async def test_terminal_authorization_preserves_agent_result_when_notice_fails(
    tmp_path: Path,
) -> None:
    channel = CompletionFailingChannel(provider="telegram")
    agent = AuthorizationAgent(terminal=True)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    callback = "http://localhost:3000/callback?code=secret-code&state=secret-state"
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text=callback, sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent[-1] == ("chat", "authorization:completed")


async def test_stop_cancels_pending_channel_authorization(tmp_path: Path) -> None:
    channel = RecordingChannel(provider="telegram")
    agent = AuthorizationAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="/stop", sender_id="operator"),
    )

    assert host._pending_authorizations == {}
    assert channel.sent[-1] == ("chat", "Stopped current run.")
    await host.stop()


async def test_channel_authorization_expires_and_cleans_pending_state(tmp_path: Path) -> None:
    channel = RecordingChannel(provider="whatsapp")
    agent = ExpiringAuthorizationAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await asyncio.sleep(0.02)
    await _wait_for_sent_count(channel, 2)

    assert channel.sent[-1] == ("chat", "authorization:expired")
    assert host._pending_authorizations == {}
    assert host._authorization_flows == {}
    await host.stop()


async def test_late_authorization_callback_expires_without_cancelling_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wait_without_timeout(awaitable, **options: float):
        assert options.keys() == {"timeout"}
        return await awaitable

    monkeypatch.setattr("deepagents_talon.host.asyncio.wait_for", wait_without_timeout)
    channel = RecordingChannel(provider="whatsapp")
    agent = ExpiringAuthorizationAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await asyncio.sleep(0.02)
    await host.receive_message(
        channel,
        ChannelMessage(
            conversation_id="chat",
            text="http://localhost:3000/callback?code=late&state=late",
            sender_id="operator",
        ),
    )
    await _wait_for_sent_count(channel, 2)

    assert channel.sent[-1] == ("chat", "authorization:expired")
    assert host._pending_authorizations == {}
    await host.stop()


async def test_host_interrupts_active_turn_and_continues_same_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await _wait_for_request(agent, "second")
    await _wait_for_sent_count(channel, 1)
    await host.stop()

    assert [request.text for request in agent.requests] == ["block", "second"]
    assert agent.recoveries == ["chat"]
    assert channel.sent == [("chat", "reply:second")]


async def test_typing_indicator_refreshes_during_long_agent_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.host._TYPING_REFRESH_SECONDS", 0.01)
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await _wait_for_typing_count(channel, 3)

    agent.released.set()
    await _wait_for_sent_count(channel, 1)
    await host.stop()

    assert len(channel.typing_calls) >= 3
    assert set(channel.typing_calls) == {"chat"}


async def test_stop_cancels_in_flight_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/stop"))
    await host.stop()

    assert channel.sent == [("chat", "Stopped current run.")]


async def test_stop_keeps_ack_when_recovery_fails(tmp_path: Path, caplog) -> None:
    channel = RecordingChannel()
    agent = FailingRecoveryAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/stop"))
    await host.stop()

    assert agent.recoveries == ["chat"]
    assert channel.sent == [("chat", "Stopped current run.")]
    assert "Failed to recover interrupted conversation" in caplog.text


async def test_new_command_starts_fresh_conversation_thread(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = HistoryAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="first"))
    await _wait_for_sent_count(channel, 1)
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/new"))
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await _wait_for_sent_count(channel, 3)
    await host.stop()

    assert [request.text for request in agent.requests] == ["first", "second"]
    assert agent.requests[0].conversation_id == "chat"
    assert agent.requests[1].conversation_id.startswith("chat:talon-reset:")
    assert channel.sent == [
        ("chat", "seen:0"),
        ("chat", "Started a fresh conversation."),
        ("chat", "seen:0"),
    ]


async def test_new_command_accepts_telegram_bot_command_suffix(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = HistoryAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/new@TestBot"))
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="hello"))
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert [request.text for request in agent.requests] == ["hello"]
    assert agent.requests[0].conversation_id.startswith("chat:talon-reset:")
    assert channel.sent == [
        ("chat", "Started a fresh conversation."),
        ("chat", "seen:0"),
    ]


async def test_new_command_cancels_in_flight_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/new"))
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert [request.text for request in agent.requests] == ["block", "second"]
    assert agent.requests[1].conversation_id.startswith("chat:talon-reset:")
    assert channel.sent == [
        ("chat", "Started a fresh conversation."),
        ("chat", "reply:second"),
    ]


async def test_new_recovers_old_thread_before_reset(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/new"))
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await _wait_for_request(agent, "second")
    await host.stop()

    assert agent.recoveries == ["chat"]
    assert agent.requests[1].conversation_id.startswith("chat:talon-reset:")


async def test_recovery_failure_starts_replacement_with_metadata(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = FailingRecoveryAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await _wait_for_request(agent, "second")
    await host.stop()

    assert agent.requests[1].metadata["interruption_recovery"] == "failed"


async def test_cancellation_timeout_blocks_until_host_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.host._CANCEL_TIMEOUT_SECONDS", 0.01)
    channel = RecordingChannel()
    agent = CancellationResistantAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="third"))
    assert [request.text for request in agent.requests] == ["block"]
    assert len(channel.sent) == 2

    agent.released.set()
    await asyncio.sleep(0)
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="fourth"))
    assert [request.text for request in agent.requests] == ["block"]
    assert len(channel.sent) == 3
    assert "chat" in host._blocked
    await host.stop()


async def test_host_sends_markdown_media_refs_as_channel_media(tmp_path: Path) -> None:
    channel = RecordingChannel()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "result.png"
    image.write_bytes(b"image")
    agent = MediaAgent("result.png")
    host = TalonHost(
        config=_config(tmp_path, {"DEEPAGENTS_TALON_WORKSPACE": str(workspace)}),
        agent=agent,
        channels=[channel],
    )
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="draw"))
    await _wait_for_sent_count(channel, 1)
    await host.stop()

    assert channel.media == [
        (
            "chat",
            ChannelMedia(path=image.resolve(), media_type="image", caption="Here is the image."),
        ),
    ]


async def test_host_rejects_markdown_media_outside_workspace(tmp_path: Path) -> None:
    channel = RecordingChannel()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    agent = MediaAgent(outside)
    host = TalonHost(
        config=_config(tmp_path, {"DEEPAGENTS_TALON_WORKSPACE": str(workspace)}),
        agent=agent,
        channels=[channel],
    )
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="draw"))
    await _wait_for_sent_count(channel, 1)
    await host.stop()

    assert channel.media == []
    assert channel.sent == [("chat", "Here is the image.\n\n_(Could not attach: chart.)_")]


async def test_host_passes_inbound_photo_as_model_content(tmp_path: Path) -> None:
    channel = RecordingChannel()
    image = tmp_path / "inbound.png"
    image.write_bytes(b"image-bytes")
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(
            conversation_id="chat",
            text="look",
            metadata={
                "media_type": "image",
                "media_paths": [str(image)],
                "media_mime_types": ["image/png"],
            },
        ),
    )
    await _wait_for_request(agent, "look")
    await host.stop()

    content = cast("list[dict[str, object]]", agent.requests[0].metadata["model_content"])
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image_url"


async def test_host_passes_inbound_video_path_in_text(tmp_path: Path) -> None:
    channel = RecordingChannel()
    video = tmp_path / "inbound.mp4"
    video.write_bytes(b"video-bytes")
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(
            conversation_id="chat",
            text="watch this",
            metadata={
                "media_type": "video",
                "media_paths": [str(video)],
                "media_mime_types": ["video/mp4"],
            },
        ),
    )
    await _wait_for_request(agent, f"watch this\n\n_(Received video attachment: {video}.)_")
    await host.stop()

    request = agent.requests[0]
    assert "unsupported" not in request.text
    assert request.metadata["media_type"] == "video"
    assert request.metadata["media_paths"] == [str(video)]


async def test_host_routes_tool_approval_reply_to_pending_run(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="approve", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert len(agent.requests) == 1
    assert agent.approvals[0].action_requests[0]["name"] == "dangerous_tool"
    assert "Tool approval required." in channel.sent[0][1]
    assert "`dangerous_tool`" in channel.sent[0][1]
    assert '{"path": "/secret"}' in channel.sent[0][1]
    assert channel.sent[1] == ("chat", "decision:approve")


async def test_host_routes_tool_approval_emoji_reply_to_pending_run(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="👍🏽", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert "Reply `👍` / `approve`" in channel.sent[0][1]
    assert channel.sent[1] == ("chat", "decision:approve")


async def test_host_routes_tool_approval_emoji_reply_denial(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="👎️", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent[1] == ("chat", "decision:reject")


async def test_host_keeps_tool_approval_scoped_to_original_sender(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="approve", sender_id="other"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="maybe", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 3)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="deny", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 4)
    await host.stop()

    assert [request.text for request in agent.requests] == ["run", "maybe"]
    assert agent.recoveries == ["chat"]
    assert channel.sent[1] == (
        "chat",
        "Only the operator who started this run can approve or deny it.",
    )
    assert "Tool approval required." in channel.sent[2][1]
    assert channel.sent[3] == ("chat", "decision:reject")


async def test_host_routes_tool_approval_reaction_to_prompt_message(tmp_path: Path) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👍🏽",
        message_id="approval-prompt",
        sender_id="operator",
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert len(agent.requests) == 1
    assert channel.reaction_handler is not None
    assert channel.sent[1] == ("chat", "decision:approve")


async def test_host_routes_tool_approval_reaction_denial(
    tmp_path: Path,
    caplog,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    caplog.set_level("INFO", logger="deepagents_talon.host")
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👎️",
        message_id="approval-prompt",
        sender_id="operator",
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    event = _talon_events(caplog, "tool_approval.reaction")[0]
    assert event["decision"] == "reject"
    assert event["match_status"] == "matched"
    assert event["resolution"] == "operator_reaction"
    assert channel.sent[1] == ("chat", "decision:reject")


async def test_host_logs_tool_approval_reaction_without_sensitive_values(
    tmp_path: Path,
    caplog,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt-private"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    caplog.set_level("INFO", logger="deepagents_talon.host")
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(
            conversation_id="chat-private",
            text="run private user text",
            sender_id="sender-private",
        ),
    )
    await _wait_for_sent_count(channel, 1)
    await host.receive_reaction(
        channel,
        ChannelReaction(
            conversation_id="chat-private",
            message_id="approval-prompt-private",
            emoji="👍",
            sender_id="sender-private",
            metadata={"raw": "RAW_PROVIDER_METADATA"},
        ),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    event = _talon_events(caplog, "tool_approval.reaction")[0]
    assert event["provider"] == "test"
    assert event["emoji"] == "👍"
    assert event["decision"] == "approve"
    assert event["match_status"] == "matched"
    assert event["resolution"] == "operator_reaction"
    assert event["channel_conversation_ref"] != "chat-private"
    assert event["prompt_message_ref"] != "approval-prompt-private"
    assert event["reacting_sender_ref"] != "sender-private"
    assert "raw_channel_conversation_id" not in event
    assert "raw_prompt_message_id" not in event
    assert "raw_reacting_sender_id" not in event
    assert "/secret" not in caplog.text
    assert "Tool approval required." not in caplog.text
    assert "run private user text" not in caplog.text
    assert "RAW_PROVIDER_METADATA" not in caplog.text
    assert "chat-private" not in caplog.text
    assert "approval-prompt-private" not in caplog.text
    assert "sender-private" not in caplog.text


async def test_host_logs_raw_reaction_ids_only_when_enabled(
    tmp_path: Path,
    caplog,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt-private"
    agent = ApprovalAgent()
    host = TalonHost(
        config=_config(tmp_path, {"DEEPAGENTS_TALON_APPROVAL_LOG_RAW_IDS": "true"}),
        agent=agent,
        channels=[channel],
    )
    caplog.set_level("INFO", logger="deepagents_talon.host")
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(
            conversation_id="chat-private",
            text="run",
            sender_id="sender-private",
        ),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👍",
        message_id="approval-prompt-private",
        sender_id="sender-private",
        conversation_id="chat-private",
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    event = _talon_events(caplog, "tool_approval.reaction")[0]
    assert event["raw_channel_conversation_id"] == "chat-private"
    assert event["raw_prompt_message_id"] == "approval-prompt-private"
    assert event["raw_reacting_sender_id"] == "sender-private"


async def test_host_ignores_tool_approval_reaction_on_unrelated_message(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👍",
        message_id="unrelated",
        sender_id="operator",
    )
    await asyncio.sleep(0)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="deny", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent[1] == ("chat", "decision:reject")


async def test_host_logs_ignored_tool_approval_reaction_attempt(
    tmp_path: Path,
    caplog,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    caplog.set_level("INFO", logger="deepagents_talon.host")
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👍",
        message_id="other-message",
        sender_id="operator",
    )
    await asyncio.sleep(0)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="deny", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    event = _talon_events(caplog, "tool_approval.reaction")[0]
    assert event["decision"] == "approve"
    assert event["match_status"] == "ignored"
    assert event["resolution"] == "message_mismatch"
    assert channel.sent[1] == ("chat", "decision:reject")


async def test_host_ignores_tool_approval_reaction_from_different_sender(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👍",
        message_id="approval-prompt",
        sender_id="other",
    )
    await asyncio.sleep(0)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="deny", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent[1] == ("chat", "decision:reject")


async def test_host_ignores_senderless_tool_approval_reaction_when_sender_known(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    channel.next_message_id = "approval-prompt"
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction("👍", message_id="approval-prompt")
    await asyncio.sleep(0)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="deny", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent[1] == ("chat", "decision:reject")


async def test_host_ignores_tool_approval_reaction_without_prompt_message_id(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    await channel.receive_reaction(
        "👍",
        message_id="approval-prompt",
        sender_id="operator",
    )
    await asyncio.sleep(0)
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="approve", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert channel.sent[1] == ("chat", "decision:approve")


async def test_host_logs_reaction_without_pending_approval(tmp_path: Path, caplog) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    caplog.set_level("INFO", logger="deepagents_talon.host")
    await host.start()

    await host.receive_reaction(
        channel,
        ChannelReaction(
            conversation_id="chat-private",
            message_id="approval-prompt-private",
            emoji="🙂",
            sender_id="sender-private",
            metadata={"raw": "RAW_PROVIDER_METADATA"},
        ),
    )
    await host.stop()

    event = _talon_events(caplog, "tool_approval.reaction")[0]
    assert event["decision"] is None
    assert event["match_status"] == "ignored"
    assert event["resolution"] == "no_pending_approval"
    assert "RAW_PROVIDER_METADATA" not in caplog.text


async def test_host_runs_scheduled_job_and_delivers_result(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "test" / "cron")
    job = store.create_job(
        prompt="scheduled prompt",
        schedule=CronSchedule.parse("in 5m"),
        origin=CronOrigin(conversation_id="chat"),
    )
    await host.start()

    text = await host.run_scheduled_job(job)
    await host.deliver_scheduled_result(channel, job, text)
    await host.stop()

    assert [request.text for request in agent.requests] == ["scheduled prompt"]
    assert agent.requests[0].metadata["trigger"] == "cron"
    assert channel.sent == [("chat", "reply:scheduled prompt")]


async def test_scheduled_job_runs_while_interactive_turn_remains_active(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "test" / "cron")
    job = store.create_job(
        prompt="scheduled prompt",
        schedule=CronSchedule.parse("in 5m"),
        origin=CronOrigin(conversation_id="chat"),
    )
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    text = await asyncio.wait_for(host.run_scheduled_job(job), timeout=1)

    assert text == "reply:scheduled prompt"
    assert [request.text for request in agent.requests] == ["block", "scheduled prompt"]
    assert agent.recoveries == []
    assert agent.requests[0].conversation_id == "chat"
    assert agent.requests[1].conversation_id == f"{job.id}:talon-cron"
    assert not host._tasks["chat"].done()
    agent.released.set()
    await host.stop()


async def test_host_transcribes_voice_before_agent(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[channel],
        voice_transcriber=VoiceTranscriber(),
    )
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(
            conversation_id="chat",
            text="",
            metadata={"media_type": "voice", "voice_path": "voice.ogg"},
        ),
    )
    await _wait_for_request(agent, "transcribed voice")
    await host.stop()

    assert [request.text for request in agent.requests] == ["transcribed voice"]
    assert agent.requests[0].metadata["voice_transcribed"] is True


async def _wait_for_request(agent: BlockingAgent, text: str) -> None:
    for _ in range(100):
        if any(request.text == text for request in agent.requests):
            return
        await asyncio.sleep(0)
    msg = f"agent did not receive request: {text}"
    raise AssertionError(msg)


async def _wait_for_sent_count(channel: RecordingChannel, count: int) -> None:
    for _ in range(100):
        if len(channel.sent) >= count:
            return
        await asyncio.sleep(0)
    msg = f"channel sent {len(channel.sent)} message(s), expected {count}"
    raise AssertionError(msg)


async def _wait_for_typing_count(channel: RecordingChannel, count: int) -> None:
    for _ in range(200):
        if len(channel.typing_calls) >= count:
            return
        await asyncio.sleep(0.01)
    msg = f"channel received {len(channel.typing_calls)} typing call(s), expected {count}"
    raise AssertionError(msg)


def _talon_events(caplog, event: str) -> list[dict[str, object]]:
    return [
        payload
        for message in caplog.messages
        if message.startswith("talon_event ")
        for payload in [json.loads(message.removeprefix("talon_event "))]
        if payload.get("event") == event
    ]
