"""Operator-controlled, conversation-scoped automatic approvals."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    ChannelMessage,
    ToolApprovalRequest,
)
from deepagents_talon.tool_approvals import ToolApprovalStore
from tests.conftest import RecordingChannel
from tests.test_host import ApprovalAgent, RoutedAgent, _config, _cron_job, _wait_for_sent_count

if TYPE_CHECKING:
    from pathlib import Path


class OperatorChannel(RecordingChannel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            exposure=ChannelExposure(
                mode=ExposureMode.ALLOWLIST, operator_ids=frozenset({"alice", "bob"})
            )
        )


def message(text: str, *, sender: str = "alice", chat: str = "chat") -> ChannelMessage:
    return ChannelMessage(conversation_id=chat, text=text, sender_id=sender)


async def test_sudo_approves_pending_batch_and_later_turns(tmp_path: Path) -> None:
    channel = OperatorChannel()
    agent = ApprovalAgent(action_count=2)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, message("run"))
        await _wait_for_sent_count(channel, 1)
        await host.receive_message(channel, message("/sudo on"))
        await _wait_for_sent_count(channel, 3)
        await host.receive_message(channel, message("again"))
        await _wait_for_sent_count(channel, 4)
        assert channel.sent[-1][1] == "decision:approve"
        assert sum("Tool approval required" in text for _, text in channel.sent) == 1
        await host.receive_message(channel, message("/sudo status"))
        assert "is on for you" in channel.sent[-1][1]
        assert [request.text for request in agent.requests] == ["run", "again"]
    finally:
        await host.stop()


@pytest.mark.parametrize(("sender", "chat"), [("bob", "chat"), ("alice", "other")])
async def test_sudo_does_not_approve_other_senders_or_threads(
    tmp_path: Path, sender: str, chat: str
) -> None:
    channel = OperatorChannel()
    host = TalonHost(config=_config(tmp_path), agent=ApprovalAgent(), channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, message("/sudo on"))
        await host.receive_message(channel, message("run", sender=sender, chat=chat))
        await _wait_for_sent_count(channel, 2)
        assert "Tool approval required" in channel.sent[-1][1]
        if sender == "bob":
            await host.receive_message(channel, message("/sudo on", chat=chat))
            assert "Only the operator who started" in channel.sent[-1][1]
        await host.receive_message(channel, message("deny", sender=sender, chat=chat))
    finally:
        await host.stop()


@pytest.mark.parametrize("sender", ["visitor", None])
async def test_sudo_requires_identified_operator(tmp_path: Path, sender: str | None) -> None:
    channel = OperatorChannel()
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.receive_message(
        channel,
        ChannelMessage("chat", "/sudo on", sender_id=sender, metadata={"from_self": True}),
    )
    assert channel.sent[-1][1] == "Only an operator can change thread approvals."
    assert agent.requests == []


@pytest.mark.parametrize("reset", ["/sudo off", "/new", "/reset-all-history", "restart"])
async def test_sudo_is_revoked(tmp_path: Path, reset: str) -> None:
    channel = OperatorChannel()
    agent = ApprovalAgent()
    if reset == "/reset-all-history":
        agent.history_enabled = True

        async def clear_history(channel_key: str, chat: str) -> None:
            del channel_key, chat

        agent.clear_history = clear_history
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, message("/sudo on"))
        if reset == "restart":
            await host.stop()
            await host.start()
        else:
            await host.receive_message(channel, message(reset))
        await host.receive_message(channel, message("run"))
        await _wait_for_sent_count(channel, 2 if reset == "restart" else 3)
        assert "Tool approval required" in channel.sent[-1][1]
    finally:
        await host.stop()


class ExplicitApprovalAgent(ApprovalAgent):
    def __init__(self, tool_name: str, *, mixed: bool) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.mixed = mixed

    async def invoke(self, request: AgentRequest) -> AgentResult:
        assert request.approval_handler is not None
        decision = await request.approval_handler(
            ToolApprovalRequest(
                conversation_id=request.conversation_id,
                interrupt_id="help",
                action_requests=({"name": self.tool_name, "args": {}},)
                + (({"name": "send_message", "args": {}},) if self.mixed else ()),
            )
        )
        return AgentResult(text=f"decision:{decision}")


@pytest.mark.parametrize("scheduled", [False, True])
async def test_sudo_does_not_authorize_unattended_work(tmp_path: Path, *, scheduled: bool) -> None:
    channel = OperatorChannel()
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.receive_message(channel, message("/sudo on"))
    if scheduled:
        await host.run_scheduled_job(_cron_job(tmp_path))
        assert agent.requests[-1].approval_handler is None
        assert agent.requests[-1].metadata["tool_approval_operator"] is False
    else:
        await host.receive_message(channel, message("delegate"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        agent.background.pending.add("test:chat")
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        assert agent.requests[-1].approval_handler is None
        assert agent.requests[-1].metadata["tool_approval_operator"] is False


@pytest.mark.parametrize("tool_name", ["ask_for_help", "update_tool_approvals"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("enable_first", [False, True])
async def test_sudo_keeps_protected_tools_approval_explicit(
    tmp_path: Path, tool_name: str, *, mixed: bool, enable_first: bool
) -> None:
    channel = OperatorChannel()
    agent = ExplicitApprovalAgent(tool_name, mixed=mixed)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        if enable_first:
            await host.receive_message(channel, message("/sudo on"))
        await host.receive_message(channel, message("run"))
        await _wait_for_sent_count(channel, 2 if enable_first else 1)
        assert "Tool approval required" in channel.sent[-1][1]
        await host.receive_message(channel, message("/sudo on"))
        await host.receive_message(channel, message("deny"))
        await _wait_for_sent_count(channel, 4 if enable_first else 3)
        assert channel.sent[-1][1] == "decision:reject"
    finally:
        await host.stop()


class PolicyEditAgent(ApprovalAgent):
    def __init__(self, store: ToolApprovalStore) -> None:
        super().__init__()
        self.store = store

    async def invoke(self, request: AgentRequest) -> AgentResult:
        assert request.approval_handler is not None
        snapshot = self.store.read()
        decision = await request.approval_handler(
            ToolApprovalRequest(
                conversation_id=request.conversation_id,
                interrupt_id="policy-edit",
                action_requests=(
                    {
                        "name": "update_tool_approvals",
                        "args": {
                            "updates": {"send_message": False},
                            "expected_revision": snapshot.revision,
                        },
                    },
                ),
            )
        )
        if decision == "approve":
            self.store.update({"send_message": False}, snapshot.revision)
        return AgentResult(text=f"decision:{decision}")


async def test_sudo_cannot_silently_change_persistent_policy(tmp_path: Path) -> None:
    store = ToolApprovalStore(tmp_path / "tools.json")
    before = store.ensure()
    channel = OperatorChannel()
    host = TalonHost(config=_config(tmp_path), agent=PolicyEditAgent(store), channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, message("/sudo on"))
        await host.receive_message(channel, message("disable send_message approvals"))
        await _wait_for_sent_count(channel, 2)
        assert "Tool approval required" in channel.sent[-1][1]
        await host.receive_message(channel, message("/sudo on"))
        await host.receive_message(channel, message("deny"))
        await _wait_for_sent_count(channel, 4)
        assert channel.sent[-1][1] == "decision:reject"
        await host.receive_message(channel, message("/sudo off"))
        persisted = ToolApprovalStore(tmp_path / "tools.json").read()
        assert persisted.revision == before.revision
        assert persisted.interrupt_on["send_message"]
    finally:
        await host.stop()
