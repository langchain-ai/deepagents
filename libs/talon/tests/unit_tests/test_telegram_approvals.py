from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.telegram import TelegramChannel
from deepagents_talon.host import TalonHost
from tests.channels.test_telegram import (
    RecordingTransport,
    _make_channel_post,
    _make_config,
    _make_reaction_update,
    _make_update,
)
from tests.test_host import ApprovalAgent, _config

if TYPE_CHECKING:
    from pathlib import Path


def _identified_post(*, sender_id: int = 111, text: str = "run") -> dict[str, object]:
    update = _make_update(chat_id=-100111, chat_type="channel", sender_id=sender_id, text=text)
    update["channel_post"] = update.pop("message")
    return update


@pytest.mark.parametrize("mode", [ExposureMode.ALLOWLIST, ExposureMode.OPEN])
@pytest.mark.parametrize("action_count", [1, 2])
async def test_senderless_channel_post_cannot_create_approval(
    tmp_path: Path, mode: ExposureMode, action_count: int
) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(
            tmp_path,
            exposure=ChannelExposure(mode=mode, conversations=frozenset({"-100111"})),
        ),
        transport=transport,
    )
    agent = ApprovalAgent(action_count=action_count)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    channel.set_message_handler(lambda message: host.receive_message(channel, message))
    try:
        await channel._process_update(_make_channel_post(text="run"))
        task = host._tasks["telegram:-100111"]
        for _ in range(100):
            if task.done() or host._pending_tool_approvals:
                break
            await asyncio.sleep(0)
        assert not host._pending_tool_approvals
        await asyncio.wait_for(task, 2)
        assert len(agent.requests) == 1
        assert agent.requests[0].metadata["sender_id"] is None
        assert not host._pending_tool_approvals
        sent = [params["text"] for method, params in transport.calls if method == "sendMessage"]
        assert sent == ["decision:reject"]
    finally:
        await host.stop()


@pytest.mark.parametrize("reaction", [False, True])
async def test_identified_telegram_approval_keeps_sender_binding(
    tmp_path: Path, *, reaction: bool
) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(
            tmp_path,
            exposure=ChannelExposure(
                mode=ExposureMode.ALLOWLIST,
                conversations=frozenset({"-100111"}),
                operator_ids=frozenset({"111", "222"}),
            ),
        ),
        transport=transport,
    )
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    channel.set_message_handler(lambda message: host.receive_message(channel, message))
    channel.set_reaction_handler(lambda event: host.receive_reaction(channel, event))
    try:
        await channel._process_update(_identified_post())
        task = host._tasks["telegram:-100111"]
        for _ in range(100):
            if host._pending_tool_approvals:
                break
            await asyncio.sleep(0)
        pending = host._pending_tool_approvals["telegram:-100111"]
        await channel._process_update(_make_channel_post(text="approve"))
        await channel._process_update(_identified_post(sender_id=222, text="approve"))
        for sender in (None, 333, 222):
            await channel._process_update(
                _make_reaction_update(chat_id=-100111, message_id=42, sender_id=sender)
            )
        assert not pending.future.done()
        if reaction:
            await channel._process_update(_make_reaction_update(chat_id=-100111, message_id=42))
        else:
            await channel._process_update(_identified_post(text="approve"))
        await asyncio.wait_for(task, 2)
        assert len(agent.requests) == 1
        sent = [params["text"] for method, params in transport.calls if method == "sendMessage"]
        assert sent[-1] == "decision:approve"
        assert not host._pending_tool_approvals
    finally:
        await host.stop()
