from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, cast

from deepagents_talon.background import BackgroundSubagents
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.host import (
    _BACKGROUND_FOLLOW_UP,
    _SCHEDULED_FOLLOW_UP,
    TalonHost,
    _BackgroundRoute,
    _save_conversation_resets,
)
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
    DeviceCode,
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


class ExplodingAgent(BlockingAgent):
    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        message = "sensitive upstream detail"
        raise RuntimeError(message)


class FailingStopAgent(BlockingAgent):
    async def stop(self) -> None:
        self.stopped = True
        message = "agent stop failed"
        raise RuntimeError(message)


class StubBackground:
    def __init__(self) -> None:
        self.pending: set[str] = set()
        self.requeued: list[str] = []

    def owners(self) -> set[str]:
        return set(self.pending)

    def results(self, owner: str) -> dict[str, str]:
        return {f"{owner}-task": "result"} if owner in self.pending else {}

    async def cancel(self, owner: str | None = None) -> bool:
        self.pending.discard(owner) if owner else self.pending.clear()
        return True

    def requeue(self, results: object) -> None:
        self.requeued.extend(results)  # type: ignore[arg-type]


class ArchiveAgent(BlockingAgent):
    def __init__(self, *, failures: int = 0) -> None:
        super().__init__()
        self.history_enabled = True
        self.cleared: list[tuple[str, str]] = []
        self.failures = failures

    async def clear_history(self, channel: str, chat: str) -> None:
        if self.failures > 0:
            self.failures -= 1
            message = "archive backend unavailable"
            raise RuntimeError(message)
        self.cleared.append((channel, chat))


class RoutedAgent(BlockingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.background = StubBackground()


class BackgroundAgent(BlockingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.background = BackgroundSubagents()


class FailingStartChannel(RecordingChannel):
    async def start(self) -> None:
        message = "channel start failed"
        raise RuntimeError(message)


class PartiallyStartingChannel(RecordingChannel):
    """Channel that acquires a resource and then fails, like the WhatsApp bridge."""

    def __init__(self, provider: str = "broken") -> None:
        super().__init__(provider=provider)
        self.bridge_running = False

    async def start(self) -> None:
        self.started = True
        self.bridge_running = True
        message = "bridge did not become ready"
        raise RuntimeError(message)

    async def stop(self) -> None:
        self.bridge_running = False
        await super().stop()


class FailingRecoveryAgent(BlockingAgent):
    async def recover_interrupted(self, conversation_id: str) -> None:
        self.recoveries.append(conversation_id)
        message = "persistence failed"
        raise RuntimeError(message)


class ReloadableAgent(BlockingAgent):
    def __init__(self, *, fail_reload: bool = False) -> None:
        super().__init__()
        self.fail_reload = fail_reload
        self.reloads = 0

    async def reload_mcp_configuration(self) -> None:
        self.reloads += 1
        if self.fail_reload:
            message = "sensitive reload failure"
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


class DeviceAuthorizationAgent(BlockingAgent):
    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        assert request.authorization_handler is not None
        binding = AuthorizationBinding(
            server_name="github",
            invocation_id="tool-call-device",
            expires_at=asyncio.get_running_loop().time() + 30,
        )
        await request.authorization_handler(
            DeviceCode(
                binding=binding,
                verification_uri="https://github.com/login/device",
                user_code="ABCD-1234",
            )
        )
        await self.released.wait()
        return AgentResult(text="authorization:completed")


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


async def test_follow_up_preserves_pending_device_authorization(tmp_path: Path) -> None:
    channel = RecordingChannel(provider="telegram")
    agent = DeviceAuthorizationAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="login", sender_id="operator"),
    )
    await _wait_for_sent_count(channel, 1)
    active = host._tasks["telegram:chat"]
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="cancel it", sender_id="attacker"),
    )
    assert channel.sent[-1] == (
        "chat",
        "Only the operator who started this authorization can complete it.",
    )
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="any update?", sender_id="operator"),
    )

    assert host._tasks["telegram:chat"] is active
    assert not active.done()
    assert [request.text for request in agent.requests] == ["login"]
    assert channel.sent[-1] == (
        "chat",
        "Complete authorization for MCP server `github` in your browser, or send `/stop`.",
    )

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="/stop", sender_id="operator"),
    )
    assert active.cancelled()
    assert host._authorization_flows == {}
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
    assert agent.recoveries == ["test:chat"]
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

    assert agent.recoveries == ["test:chat"]
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
    assert agent.requests[0].conversation_id == "test:chat"
    assert agent.requests[1].conversation_id.startswith("test:chat:talon-reset:")
    assert channel.sent == [
        ("chat", "seen:0"),
        ("chat", "Started a fresh conversation."),
        ("chat", "seen:0"),
    ]


async def test_new_command_remains_active_after_restart(tmp_path: Path) -> None:
    config = _config(tmp_path)
    channel = RecordingChannel()
    first_agent = HistoryAgent()
    first_host = TalonHost(config=config, agent=first_agent, channels=[channel])
    await first_host.start()

    await first_host.receive_message(channel, ChannelMessage(conversation_id="chat", text="first"))
    await _wait_for_sent_count(channel, 1)
    await first_host.receive_message(channel, ChannelMessage(conversation_id="chat", text="/new"))
    await first_host.stop()

    restarted_channel = RecordingChannel()
    restarted_agent = HistoryAgent()
    restarted_host = TalonHost(
        config=config,
        agent=restarted_agent,
        channels=[restarted_channel],
    )
    await restarted_host.start()
    await restarted_host.receive_message(
        restarted_channel,
        ChannelMessage(conversation_id="chat", text="second"),
    )
    await _wait_for_sent_count(restarted_channel, 1)
    await restarted_host.stop()

    assert restarted_agent.requests[0].conversation_id.startswith("test:chat:talon-reset:")
    assert restarted_channel.sent == [("chat", "seen:0")]


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
    assert agent.requests[0].conversation_id.startswith("test:chat:talon-reset:")
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
    assert agent.requests[1].conversation_id.startswith("test:chat:talon-reset:")
    assert channel.sent == [
        ("chat", "Started a fresh conversation."),
        ("chat", "reply:second"),
    ]


async def test_mcp_reload_command_reloads_without_invoking_agent(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = ReloadableAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="/mcp-reload@TestBot"),
    )
    await host.stop()

    assert agent.reloads == 1
    assert agent.requests == []
    assert channel.sent == [("chat", "Reloaded MCP configuration.")]


async def test_mcp_reload_command_hides_reload_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    channel = RecordingChannel()
    agent = ReloadableAgent(fail_reload=True)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="/mcp-reload"),
    )
    await host.stop()

    assert channel.sent == [
        ("chat", "Could not reload MCP configuration. Check Talon logs."),
    ]
    assert "MCP configuration reload failed" in caplog.text
    assert "sensitive reload failure" not in channel.sent[0][1]


async def test_mcp_reload_command_reports_unavailable_runtime(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="/mcp-reload"),
    )
    await host.stop()

    assert agent.requests == []
    assert channel.sent == [("chat", "MCP configuration reload is unavailable.")]


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

    assert agent.recoveries == ["test:chat"]
    assert agent.requests[1].conversation_id.startswith("test:chat:talon-reset:")


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
    assert "test:chat" in host._blocked
    await host.stop()


class DelegatingCronAgent(BlockingAgent):
    """Agent that delegates on a scheduled run and answers the follow-up turn."""

    def __init__(self, *, follow_up: str = "digest", history: bool = False) -> None:
        super().__init__()
        self.background = StubBackground()
        self.follow_up = follow_up
        self.history_enabled = history
        self.fires = 0

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text.startswith(_BACKGROUND_FOLLOW_UP):
            self.background.pending.discard(request.conversation_id)
            return AgentResult(text=self.follow_up)
        self.fires += 1
        if self.fires == 1:
            self.background.pending.add(request.conversation_id)
        return AgentResult(text="[SILENT]")


class BlockingFollowUpCronAgent(DelegatingCronAgent):
    """Agent whose background follow-up turn blocks until released."""

    async def invoke(self, request: AgentRequest) -> AgentResult:
        if request.text.startswith(_BACKGROUND_FOLLOW_UP):
            self.requests.append(request)
            await self.released.wait()
            return AgentResult(text=self.follow_up)
        return await super().invoke(request)


class RouteAssertingCronAgent(BlockingAgent):
    """Agent that records whether its route existed, then fails the run."""

    def __init__(self) -> None:
        super().__init__()
        self.background = StubBackground()
        self.host: TalonHost | None = None
        self.routed_at_invoke: list[bool] = []

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        assert self.host is not None
        self.routed_at_invoke.append(request.conversation_id in self.host._background_routes)
        self.background.pending.add(request.conversation_id)
        message = "scheduled run exploded"
        raise RuntimeError(message)


def _cron_job(tmp_path: Path, *, prompt: str = "scan the market", channel: str | None = None):
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "test" / "cron")
    return store.create_job(
        prompt=prompt,
        schedule=CronSchedule.parse("in 5m"),
        origin=CronOrigin(conversation_id="chat", channel=channel),
    )


async def test_cron_background_result_reaches_the_job_origin_chat(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = DelegatingCronAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    cron_id = f"{job.id}:talon-cron"
    await host.start()
    try:
        assert await host.run_scheduled_job(job) == "[SILENT]"
        assert channel.sent == []

        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks[cron_id], 2)

        assert channel.sent == [("chat", "digest")]
        assert agent.requests[-1].conversation_id == cron_id
        assert agent.requests[-1].text.startswith(_BACKGROUND_FOLLOW_UP)
    finally:
        await host.stop()


@pytest.mark.parametrize("origin_channel", [None, "test"])
async def test_cron_background_turn_keeps_the_job_identity(
    tmp_path: Path, origin_channel: str | None
) -> None:
    channel = RecordingChannel()
    agent = DelegatingCronAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path, channel=origin_channel)
    await host.start()
    try:
        await host.run_scheduled_job(job)
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks[f"{job.id}:talon-cron"], 2)

        metadata = agent.requests[-1].metadata
        assert metadata["trigger"] == "cron"
        assert metadata["cron_job_id"] == job.id
        assert metadata["origin_conversation_id"] == "chat"
        # `_same_origin_scope` compares the channel too, so a follow-up turn that
        # reported the resolved provider here could not find its own job.
        assert metadata["channel"] == origin_channel
        # Identical to the run's own metadata in every field the runtime reads, so
        # the follow-up turn cannot behave as a different kind of turn.
        run_metadata = agent.requests[0].metadata
        assert {key: metadata[key] for key in run_metadata} == dict(run_metadata)
    finally:
        await host.stop()


async def test_cron_background_turn_gets_no_operator_handlers(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = DelegatingCronAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    await host.start()
    try:
        await host.run_scheduled_job(job)
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks[f"{job.id}:talon-cron"], 2)

        # An approval prompt would be keyed by the cron thread but answered in the
        # chat thread, and the wait on it has no timeout.
        assert agent.requests[-1].approval_handler is None
        assert agent.requests[-1].authorization_handler is None
    finally:
        await host.stop()


async def test_cron_background_turn_is_not_archived_into_the_chat(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = DelegatingCronAgent(history=True)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    await host.start()
    try:
        await host.run_scheduled_job(job)
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks[f"{job.id}:talon-cron"], 2)

        assert "history_chat" not in agent.requests[-1].metadata
        assert "history_channel" not in agent.requests[-1].metadata
    finally:
        await host.stop()


@pytest.mark.parametrize("reply", ["[SILENT]", "nothing moved [SILENT]"])
async def test_cron_background_reply_is_suppressed_when_silent(
    tmp_path: Path, reply: str, caplog
) -> None:
    channel = RecordingChannel()
    agent = DelegatingCronAgent(follow_up=reply)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    await host.start()
    try:
        with caplog.at_level(logging.INFO, logger="deepagents_talon.host"):
            await host.run_scheduled_job(job)
            await host._dispatch_background_results()
            await asyncio.wait_for(host._tasks[f"{job.id}:talon-cron"], 2)

        assert channel.sent == []
        events = _talon_events(caplog, "cron.background_suppressed")
        assert [event["job_id"] for event in events] == [job.id]
    finally:
        await host.stop()


async def test_cron_route_is_registered_before_the_run(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = RouteAssertingCronAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    agent.host = host
    job = _cron_job(tmp_path)
    await host.start()
    try:
        with pytest.raises(RuntimeError, match="scheduled run exploded"):
            await host.run_scheduled_job(job)

        # The scheduler swallows this raise, so a run that fails after delegating
        # still needs the route its subagent will be delivered through.
        assert agent.routed_at_invoke == [True]
        assert f"{job.id}:talon-cron" in host._background_routes
    finally:
        await host.stop()


async def test_cron_route_is_dropped_when_the_job_delegates_nothing(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    await host.start()
    try:
        assert await host.run_scheduled_job(job) == "reply:scan the market"
        assert f"{job.id}:talon-cron" in host._background_routes

        await host._dispatch_background_results()

        assert host._background_routes == {}
        assert host._locks == {}
        assert dict(host._generations) == {}
    finally:
        await host.stop()


async def test_cron_route_addresses_the_channel_matching_the_job_origin(tmp_path: Path) -> None:
    first = RecordingChannel(provider="whatsapp")
    second = RecordingChannel(provider="telegram")
    agent = DelegatingCronAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[first, second])
    job = _cron_job(tmp_path, channel="telegram")
    await host.start()
    try:
        await host.run_scheduled_job(job)
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks[f"{job.id}:talon-cron"], 2)

        assert first.sent == []
        assert second.sent == [("chat", "digest")]
    finally:
        await host.stop()


async def test_second_cron_run_preempts_a_pending_background_turn(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingFollowUpCronAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    cron_id = f"{job.id}:talon-cron"
    await host.start()
    try:
        await host.run_scheduled_job(job)
        await host._dispatch_background_results()
        await _wait_for_request(agent, _SCHEDULED_FOLLOW_UP)
        follow_up = host._tasks[cron_id]

        await host.run_scheduled_job(job)

        assert follow_up.cancelled()
        assert agent.recoveries == [cron_id]
        # The turn was cancelled, not the workers: the results it was consuming are
        # still pending, so the dispatcher delivers them again.
        assert agent.background.pending == {cron_id}
        assert cron_id in host._background_routes
        assert channel.sent == []
    finally:
        agent.released.set()
        await host.stop()


async def test_preempting_a_turn_blocked_on_the_conversation_lock_completes(
    tmp_path: Path,
) -> None:
    """A run preempts a follow-up turn while holding the lock that turn is awaiting.

    `_run_agent_turn` takes the conversation lock only to deliver, so a follow-up
    turn can be parked on it at the moment a scheduled run acquires it and preempts.
    Cancellation has to be what releases it; anything that waited for the turn to
    make progress instead would deadlock the scheduler against its own thread.
    """
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    cron_id = "job:talon-cron"
    await host.start()
    try:
        parked = asyncio.Event()

        async def blocked_turn() -> None:
            parked.set()
            async with host._conversation_lock(cron_id):
                pass

        async with host._conversation_lock(cron_id):
            turn = asyncio.create_task(blocked_turn())
            host._tasks[cron_id] = turn
            await asyncio.wait_for(parked.wait(), 2)
            for _ in range(10):
                await asyncio.sleep(0)

            await asyncio.wait_for(host._preempt_scheduled_turn(cron_id), 2)

        assert turn.cancelled()
        assert agent.recoveries == [cron_id]
        assert cron_id not in host._blocked
    finally:
        await host.stop()


async def test_two_runs_of_one_job_never_overlap(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path, prompt="block")
    await host.start()
    try:
        first = asyncio.create_task(host.run_scheduled_job(job))
        await _wait_for_request(agent, "block")
        second = asyncio.create_task(host.run_scheduled_job(job))
        for _ in range(50):
            await asyncio.sleep(0)

        assert len(agent.requests) == 1

        agent.released.set()
        assert await asyncio.wait_for(first, 2) == "reply:block"
        assert await asyncio.wait_for(second, 2) == "reply:block"
        assert len(agent.requests) == 2
    finally:
        await host.stop()


async def test_channel_background_turn_keeps_its_operator_context(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        agent.background.pending.add("test:chat")
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="hello"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)

        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks["test:chat"], 2)

        follow_up = agent.requests[-1]
        assert follow_up.text == _BACKGROUND_FOLLOW_UP
        assert "trigger" not in follow_up.metadata
        assert follow_up.approval_handler is not None
        assert follow_up.authorization_handler is not None
    finally:
        await host.stop()


class BackgroundResultAgent(BlockingAgent):
    """Agent whose turn reports the background results it consumed."""

    def __init__(self, *, text: str | None = None) -> None:
        super().__init__()
        self.background = StubBackground()
        self.text = text

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text == "block":
            await self.released.wait()
        return AgentResult(
            text=self.text if self.text is not None else f"reply:{request.text}",
            background_results=("subagent-1",),
        )


async def _park_turn_at_delivery(agent: BlockingAgent) -> None:
    """Let a blocked turn finish its model work while the delivery lock is held."""
    agent.released.set()
    for _ in range(50):
        await asyncio.sleep(0)


async def test_superseded_turn_requeues_its_background_results(tmp_path: Path) -> None:
    """A newer turn takes the thread while the previous reply waits to go out.

    The runtime acknowledged those results when the model consumed them, but the
    reply carrying them is dropped by the generation check, so nobody was told. The
    ids go back to the queue rather than counting as delivered.
    """
    channel = RecordingChannel()
    agent = BackgroundResultAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
        await _wait_for_request(agent, "block")
        turn = host._tasks["test:chat"]

        async with host._conversation_lock("test:chat"):
            await _park_turn_at_delivery(agent)
            # Exactly what `_cancel_active` records before it cancels.
            host._generations["test:chat"] += 1

        await asyncio.wait_for(turn, 2)

        assert agent.background.requeued == ["subagent-1"]
        assert channel.sent == []
    finally:
        await host.stop()


async def test_turn_cancelled_awaiting_delivery_requeues_its_background_results(
    tmp_path: Path,
) -> None:
    """The same loss reached by cancellation instead of the generation check.

    A scheduled run preempts a follow-up turn by cancelling it, and the turn can
    already be past the model call and parked on the conversation lock the run
    holds. Re-queueing has to happen while that cancellation is in flight.
    """
    channel = RecordingChannel()
    agent = BackgroundResultAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
        await _wait_for_request(agent, "block")
        turn = host._tasks["test:chat"]

        async with host._conversation_lock("test:chat"):
            await _park_turn_at_delivery(agent)
            turn.cancel()
            for _ in range(50):
                await asyncio.sleep(0)

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(turn, 2)

        assert agent.background.requeued == ["subagent-1"]
        assert channel.sent == []
    finally:
        await host.stop()


async def test_delivered_turn_does_not_requeue_its_background_results(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BackgroundResultAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="hello"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)

        assert channel.sent == [("chat", "reply:hello")]
        assert agent.background.requeued == []
    finally:
        await host.stop()


async def test_deliberately_suppressed_turn_does_not_requeue(tmp_path: Path) -> None:
    """Suppression the host chose is not a lost result.

    A terminal authorization withholds the reply on purpose, and a silent scheduled
    run does the same. The model still consumed the results, so they stay
    acknowledged; re-queueing them would replay work the conversation has handled.
    """
    channel = RecordingChannel()
    agent = BackgroundResultAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        host._terminal_authorizations.add("test:chat")
        await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="hello"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)

        assert channel.sent == []
        assert agent.background.requeued == []
    finally:
        await host.stop()


async def test_silent_scheduled_turn_does_not_requeue(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BackgroundResultAgent(text="[SILENT]")
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    job = _cron_job(tmp_path)
    cron_id = f"{job.id}:talon-cron"
    await host.start()
    try:
        agent.background.pending.add(cron_id)
        await host.run_scheduled_job(job)
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks[cron_id], 2)

        assert channel.sent == []
        assert agent.background.requeued == []
    finally:
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

    assert [request.text for request in agent.requests] == ["run"]
    assert agent.recoveries == []
    assert channel.sent[1] == (
        "chat",
        "Only the operator who started this run can approve or deny it.",
    )
    assert channel.sent[2] == channel.sent[0]
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
    assert agent.requests[0].conversation_id == "test:chat"
    assert agent.requests[1].conversation_id == f"{job.id}:talon-cron"
    assert not host._tasks["test:chat"].done()
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


async def test_failed_turn_replies_without_leaking_the_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    channel = RecordingChannel()
    agent = ExplodingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    try:
        with caplog.at_level(logging.ERROR, logger="deepagents_talon.host"):
            await host.receive_message(channel, ChannelMessage("chat", "hello"))
            await asyncio.wait_for(host._tasks["test:chat"], 2)

        assert [conversation for conversation, _ in channel.sent] == ["chat"]
        assert channel.sent[0][1]
        assert "sensitive upstream detail" not in channel.sent[0][1]
        assert "sensitive upstream detail" in caplog.text
    finally:
        await host.stop()


async def test_start_unwinds_started_components_when_a_channel_fails(tmp_path: Path) -> None:
    first = RecordingChannel()
    second = FailingStartChannel(provider="broken")
    agent = BlockingAgent()
    scheduler = RecordingScheduler()
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[first, second],
        scheduler=scheduler,
    )

    with pytest.raises(RuntimeError, match="channel start failed"):
        await host.start()

    assert first.started is True
    assert first.stopped is True
    assert second.stopped is True
    assert agent.stopped is True
    assert scheduler.started is False
    assert host.running is False


async def test_stop_completes_when_a_component_fails_to_stop(tmp_path: Path) -> None:
    channel = RecordingChannel()
    scheduler = RecordingScheduler()
    agent = FailingStopAgent()
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[channel],
        scheduler=scheduler,
    )
    await host.start()

    await host.stop()

    assert channel.stopped is True
    assert scheduler.stopped is True
    assert host._stopped.is_set()


async def test_stop_completes_when_the_background_loop_already_died(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BackgroundAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    assert host._background_loop is not None
    host._background_loop.cancel()
    await asyncio.sleep(0)

    async def died() -> None:
        message = "dispatcher died"
        raise RuntimeError(message)

    host._background_loop = asyncio.create_task(died())
    await asyncio.sleep(0)

    await host.stop()

    assert channel.stopped is True
    assert agent.stopped is True
    assert host._stopped.is_set()


async def test_background_delivery_survives_a_failing_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=BackgroundAgent(), channels=[channel])
    ticks = 0

    async def dispatch() -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            message = "dispatch exploded"
            raise RuntimeError(message)
        host._running = False

    monkeypatch.setattr(host, "_dispatch_background_results", dispatch)
    with caplog.at_level(logging.ERROR, logger="deepagents_talon.host"):
        await host.start()
        assert host._background_loop is not None
        await asyncio.wait_for(host._background_loop, 5)

    assert ticks == 2
    assert "dispatch exploded" in caplog.text
    await host.stop()


async def test_one_locked_conversation_does_not_stall_delivery_for_others(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        for owner in ("stuck", "waiting"):
            agent.background.pending.add(owner)
            host._background_routes[owner] = _BackgroundRoute(
                channel=channel,
                message=ChannelMessage(owner, "research"),
                conversation_root=owner,
                conversation_id=owner,
                provider="test",
            )
        held = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            async with host._conversation_lock("stuck"):
                held.set()
                await release.wait()

        holder = asyncio.create_task(hold())
        await asyncio.wait_for(held.wait(), 2)

        await asyncio.wait_for(host._dispatch_background_results(), 2)

        assert "waiting" in host._tasks
        assert "stuck" not in host._tasks
    finally:
        release.set()
        await asyncio.gather(holder, return_exceptions=True)
        await host.stop()


async def test_finished_conversations_do_not_accumulate_state(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        for index in range(3):
            await host.receive_message(channel, ChannelMessage(f"chat{index}", "hello"))
            await asyncio.wait_for(host._tasks[f"test:chat{index}"], 2)
        await asyncio.sleep(0)

        assert len(channel.sent) == 3
        assert host._locks == {}
        assert host._tasks == {}
        assert dict(host._generations) == {}
    finally:
        await host.stop()


async def test_history_reset_keeps_the_archive_when_the_counter_cannot_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = RecordingChannel()
    agent = ArchiveAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])

    def refuse(*_args: object, **_kwargs: object) -> None:
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr("deepagents_talon.host._save_conversation_resets", refuse)
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("chat", "/reset-all-history"))

        assert agent.cleared == []
        assert "Could not finish clearing history" in channel.sent[-1][1]
        assert host._agent_conversation_id("test:chat") == "test:chat"
    finally:
        await host.stop()


async def test_conversation_root_is_channel_keyed_whatever_the_host_looks_like(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    lone = TalonHost(config=_config(tmp_path), agent=BlockingAgent(), channels=[channel])
    archiving = TalonHost(
        config=_config(tmp_path),
        agent=ArchiveAgent(),
        channels=[channel, RecordingChannel(provider="telegram")],
    )

    # One channel with no history used to key by the bare conversation id, so
    # adding a channel or enabling history re-keyed every thread underneath.
    assert lone._conversation_root("test", "chat") == "test:chat"
    assert archiving._conversation_root("test", "chat") == "test:chat"


async def test_channel_keyed_threads_still_reply_to_the_channel_conversation(
    tmp_path: Path,
) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    try:
        await host.receive_message(channel, ChannelMessage("chat", "hello"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)

        assert agent.requests[0].conversation_id == "test:chat"
        # The channel's own id has to travel separately now that it is not the
        # thread id: cron origins and replies address the conversation, not the thread.
        assert agent.requests[0].metadata["origin_conversation_id"] == "chat"
        assert channel.sent == [("chat", "reply:hello")]
    finally:
        await host.stop()


async def test_start_releases_the_channel_that_failed_partway_through(tmp_path: Path) -> None:
    failing = PartiallyStartingChannel()
    never_reached = RecordingChannel(provider="telegram")
    agent = BlockingAgent()
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[failing, never_reached],
        scheduler=RecordingScheduler(),
    )

    with pytest.raises(RuntimeError, match="bridge did not become ready"):
        await host.start()

    # The failing channel is the one holding a subprocess, so it is the one that
    # must be stopped; nothing else got as far as starting.
    assert failing.bridge_running is False
    assert failing.stopped is True
    assert never_reached.started is False
    assert agent.stopped is True
    assert host.running is False


async def test_failed_history_clear_leaves_the_conversation_where_it_was(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = ArchiveAgent(failures=1)
    config = _config(tmp_path)
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()

    try:
        await host.receive_message(channel, ChannelMessage("chat", "/reset-all-history"))

        # The clear failed, so nothing may imply it succeeded: the chat keeps its thread
        # id in memory and on disk, and the reply does not claim history was cleared.
        assert host._agent_conversation_id("test:chat") == "test:chat"
        assert json.loads(config.conversation_state_path.read_text()) == {}
        assert channel.sent[-1][1] == (
            "Could not finish clearing history. Some of it may already be deleted. "
            "Send /reset-all-history to finish clearing."
        )

        # The retry the user is told to send starts from that unchanged state.
        await host.receive_message(channel, ChannelMessage("chat", "/reset-all-history"))

        assert agent.cleared == [("test", "chat")]
        assert host._agent_conversation_id("test:chat") == "test:chat:talon-reset:1"
        assert json.loads(config.conversation_state_path.read_text()) == {"test:chat": 1}
        assert "Cleared all conversation history" in channel.sent[-1][1]
    finally:
        await host.stop()


async def test_reset_counter_rollback_failure_is_logged_and_still_reverted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    channel = RecordingChannel()
    agent = ArchiveAgent(failures=1)
    config = _config(tmp_path)
    host = TalonHost(config=config, agent=agent, channels=[channel])
    original = _save_conversation_resets
    writes = 0

    def fail_the_rollback(path: Path, resets: object) -> None:
        nonlocal writes
        writes += 1
        if writes > 1:
            message = "disk full"
            raise OSError(message)
        original(path, cast("dict[str, int]", resets))

    monkeypatch.setattr("deepagents_talon.host._save_conversation_resets", fail_the_rollback)
    await host.start()

    try:
        with caplog.at_level(logging.ERROR, logger="deepagents_talon.host"):
            await host.receive_message(channel, ChannelMessage("chat", "/reset-all-history"))

        # The file keeps the bumped counter, but this process does not act on it.
        assert host._agent_conversation_id("test:chat") == "test:chat"
        assert json.loads(config.conversation_state_path.read_text()) == {"test:chat": 1}
        assert "Could not roll back the conversation reset counter" in caplog.text
    finally:
        await host.stop()
