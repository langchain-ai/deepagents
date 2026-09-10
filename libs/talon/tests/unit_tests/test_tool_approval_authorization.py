from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.host import TalonHost, _BackgroundRoute, _Turn
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from tests.conftest import RecordingChannel
from tests.test_host import RoutedAgent, _config, _cron_job


@pytest.mark.parametrize(
    "case",
    [
        (ExposureMode.SELF, (), "alice", True, True),
        (ExposureMode.SELF, (), "alice", "true", False),
        (ExposureMode.SELF, (), "alice", False, False),
        (ExposureMode.SELF, ("alice",), "alice", False, True),
        (ExposureMode.SELF, (), None, True, False),
        (ExposureMode.SELF, ("",), "", True, False),
        (ExposureMode.ALLOWLIST, (), "alice", True, False),
        (ExposureMode.OPEN, (), "alice", True, False),
        (ExposureMode.ALLOWLIST, ("alice",), "alice", False, True),
        (ExposureMode.OPEN, ("alice",), "alice", False, True),
        (ExposureMode.OPEN, ("alice",), "mallory", True, False),
        (ExposureMode.OPEN, ("alice",), None, True, False),
        ("unknown", ("alice",), "alice", True, False),
    ],
)
async def test_channel_operator_identity(tmp_path, case):
    mode, operators, sender, from_self, expected = case
    channel = RecordingChannel()
    channel.config = SimpleNamespace(
        exposure=ChannelExposure(mode=mode, operator_ids=frozenset(operators))
    )
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.receive_message(
        channel,
        ChannelMessage(
            "chat",
            "edit policy",
            sender_id=sender,
            metadata={
                "from_self": from_self,
                "sender_id": "alice",
                "tool_approval_operator": not expected,
                "operator_ids": ["mallory"],
            },
        ),
    )
    await asyncio.wait_for(host._tasks["test:chat"], 2)
    assert agent.requests[-1].metadata["tool_approval_operator"] is expected


@pytest.mark.parametrize("config", [None, SimpleNamespace(), SimpleNamespace(exposure={})])
async def test_missing_trusted_exposure_denies(tmp_path, config):
    channel = RecordingChannel()
    if config is not None:
        channel.config = config
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.receive_message(
        channel,
        ChannelMessage(
            "chat",
            "edit policy",
            sender_id="alice",
            metadata={"from_self": True, "tool_approval_operator": True},
        ),
    )
    await asyncio.wait_for(host._tasks["test:chat"], 2)
    assert agent.requests[-1].metadata["tool_approval_operator"] is False


@pytest.mark.parametrize("scheduled", [False, True])
async def test_background_delivery_drops_origin_authority(tmp_path, scheduled):
    channel = RecordingChannel()
    channel.config = SimpleNamespace(exposure=ChannelExposure(operator_ids=frozenset({"alice"})))
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    if scheduled:
        await host.run_scheduled_job(_cron_job(tmp_path))
    else:
        await host.receive_message(
            channel,
            ChannelMessage("chat", "delegate", sender_id="alice", metadata={"from_self": True}),
        )
        await asyncio.wait_for(host._tasks["test:chat"], 2)
    request = agent.requests[-1]
    assert request.metadata["tool_approval_operator"] is (not scheduled)
    assert (request.approval_handler is None) is scheduled
    assert (request.authorization_handler is None) is scheduled
    agent.background.pending.add(request.conversation_id)
    await host._dispatch_background_results()
    await asyncio.wait_for(host._tasks[request.conversation_id], 2)
    delivered = agent.requests[-1]
    assert delivered.metadata["background_delivery"] is True
    assert delivered.metadata["tool_approval_operator"] is False
    assert delivered.approval_handler is None
    assert delivered.authorization_handler is None


@pytest.mark.parametrize(
    "route_metadata",
    [
        {"tool_approval_operator": True, "sender_id": "alice", "from_self": True},
        {"tool_approval_operator": True, "trigger": "cron"},
        {"tool_approval_operator": True, "background_delivery": True},
    ],
)
async def test_route_metadata_cannot_grant_authority(tmp_path, route_metadata):
    channel = RecordingChannel()
    channel.config = SimpleNamespace(exposure=ChannelExposure())
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    route = _BackgroundRoute(
        channel=channel,
        message=ChannelMessage(
            "chat",
            "edit policy",
            sender_id="mallory",
            metadata={"from_self": "sender_id" not in route_metadata},
        ),
        conversation_root="test:chat",
        conversation_id="test:chat",
        provider="test",
        metadata=route_metadata,
    )
    await host._run_agent_turn(
        route, _Turn("test:chat", "test:chat", "test", 0, recovery_degraded=False)
    )
    assert agent.requests[-1].metadata["tool_approval_operator"] is False


async def test_direct_invocation_defaults_to_deny(tmp_path):
    agent = RoutedAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[])
    await host._invoke_agent(
        request=AgentRequest(
            conversation_id="chat",
            text="edit policy",
            metadata={"tool_approval_operator": True, "from_self": True, "sender_id": "alice"},
        ),
    )
    assert agent.requests[-1].metadata["tool_approval_operator"] is False
