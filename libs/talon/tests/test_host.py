from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    ChannelMedia,
    ChannelMessage,
    ChannelStatus,
    ToolApprovalRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class RecordingChannel:
    def __init__(self) -> None:
        self.handler: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, str]] = []
        self.media: list[tuple[str, ChannelMedia]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def set_message_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        self.handler = handler

    async def send_message(self, conversation_id: str, text: str) -> None:
        self.sent.append((conversation_id, text))

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> None:
        self.media.append((conversation_id, media))
        self.sent.append((conversation_id, f"{media.media_type}:{media.path}"))

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        self.sent.append((conversation_id, f"{message_id}:{text}"))

    async def status(self) -> ChannelStatus:
        return ChannelStatus(provider="test", connected=True)


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
        self.released = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text == "block":
            await self.released.wait()
        return AgentResult(text=f"reply:{request.text}")


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


async def test_host_serializes_messages_per_conversation(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()

    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="block"))
    await _wait_for_request(agent, "block")
    await host.receive_message(channel, ChannelMessage(conversation_id="chat", text="second"))
    await asyncio.sleep(0)

    assert [request.text for request in agent.requests] == ["block"]

    agent.released.set()
    await _wait_for_request(agent, "second")
    await _wait_for_sent_count(channel, 2)
    await host.stop()

    assert [request.text for request in agent.requests] == ["block", "second"]
    assert channel.sent == [("chat", "reply:block"), ("chat", "reply:second")]


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

    assert len(agent.requests) == 1
    assert channel.sent[1] == (
        "chat",
        "Only the operator who started this run can approve or deny it.",
    )
    assert channel.sent[2] == (
        "chat",
        "Reply `approve` to run the tool call or `deny` to skip it.",
    )
    assert channel.sent[3] == ("chat", "decision:reject")


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
