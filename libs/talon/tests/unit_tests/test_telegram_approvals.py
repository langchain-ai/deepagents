from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.telegram import TelegramChannel
from deepagents_talon.host import TalonHost
from tests.channels.test_telegram import RecordingTransport, _make_channel_post, _make_config
from tests.test_host import ApprovalAgent, _config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode", [ExposureMode.ALLOWLIST, ExposureMode.OPEN])
async def test_senderless_channel_post_cannot_create_approval(
    tmp_path: Path, mode: ExposureMode
) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(
            tmp_path,
            exposure=ChannelExposure(mode=mode, conversations=frozenset({"-100111"})),
        ),
        transport=transport,
    )
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    channel.set_message_handler(lambda message: host.receive_message(channel, message))
    try:
        await channel._process_update(_make_channel_post(text="run"))
        await asyncio.wait_for(host._tasks["telegram:-100111"], 2)
        assert agent.requests[0].metadata["sender_id"] is None
        assert not host._pending_tool_approvals
        sent = [params["text"] for method, params in transport.calls if method == "sendMessage"]
        assert sent == ["decision:reject"]
    finally:
        await host.stop()
