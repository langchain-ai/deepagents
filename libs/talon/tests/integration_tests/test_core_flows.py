from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.scheduler import PersistentCronScheduler
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    ChannelMessage,
    ChannelStatus,
    SendResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class InMemoryChannel:
    def __init__(self) -> None:
        self.handler: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self.sent: list[tuple[str, str]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def set_message_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        self.handler = handler

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        self.sent.append((conversation_id, text))
        return SendResult(success=True)

    async def send_media(self, conversation_id: str, media: object) -> SendResult:
        del conversation_id, media
        return SendResult(success=True)

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        del conversation_id, message_id, text
        return SendResult(success=True)

    async def send_typing(self, conversation_id: str) -> None:
        del conversation_id

    async def status(self) -> ChannelStatus:
        return ChannelStatus(provider="memory", connected=True)

    async def receive(self, text: str, *, conversation_id: str = "chat") -> None:
        if self.handler is None:
            msg = "channel handler was not registered"
            raise AssertionError(msg)
        await self.handler(ChannelMessage(conversation_id=conversation_id, text=text))


class ScriptedAgent:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.requests: list[AgentRequest] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(text=self.replies.get(request.text, f"reply:{request.text}"))


class SchedulingAgent(ScriptedAgent):
    def __init__(self, store: CronJobStore, now: datetime) -> None:
        super().__init__({"scheduled prompt": "cron result"})
        self.store = store
        self.now = now

    async def invoke(self, request: AgentRequest) -> AgentResult:
        if request.text == "schedule it":
            job = self.store.create_job(
                prompt="scheduled prompt",
                schedule=CronSchedule.parse("in 1m"),
                origin=CronOrigin(conversation_id=request.conversation_id),
                name="integration",
                now=self.now,
            )
            return AgentResult(text=f"scheduled:{job.id}")
        return await super().invoke(request)


def _config(tmp_path) -> TalonConfig:
    return TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant"}, base_home=tmp_path)


async def test_inbound_channel_message_produces_reply(tmp_path) -> None:
    channel = InMemoryChannel()
    agent = ScriptedAgent({"hello": "hi there"})
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])

    await host.start()
    await channel.receive("hello")
    await _wait_for_sent_count(channel, 1)
    await host.stop()

    assert [request.text for request in agent.requests] == ["hello"]
    assert channel.sent == [("chat", "hi there")]


async def test_persisted_cron_execution_delivers_to_origin_channel(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    channel = InMemoryChannel()
    agent = ScriptedAgent({"scheduled prompt": "cron result"})
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    store = CronJobStore(assistant_id="assistant", cron_dir=tmp_path / "assistant" / "cron")
    store.create_job(
        prompt="scheduled prompt",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat", channel="memory"),
        now=now,
    )
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
        now=lambda: now + timedelta(minutes=1),
    )

    await host.start()
    await scheduler.tick_once()
    await host.stop()

    assert [request.text for request in agent.requests] == ["scheduled prompt"]
    assert agent.requests[0].metadata["trigger"] == "cron"
    assert channel.sent == [("chat", "cron result")]


async def test_agent_can_create_job_that_later_runs(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    channel = InMemoryChannel()
    store = CronJobStore(assistant_id="assistant", cron_dir=tmp_path / "assistant" / "cron")
    agent = SchedulingAgent(store, now)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
        now=lambda: now + timedelta(minutes=1),
    )

    await host.start()
    await channel.receive("schedule it")
    await _wait_for_sent_count(channel, 1)
    await scheduler.tick_once()
    await host.stop()

    assert [request.text for request in agent.requests] == ["scheduled prompt"]
    assert channel.sent[0][1].startswith("scheduled:")
    assert channel.sent[1] == ("chat", "cron result")


async def _wait_for_sent_count(channel: InMemoryChannel, count: int) -> None:
    for _ in range(100):
        if len(channel.sent) >= count:
            return
        await asyncio.sleep(0)
    msg = f"channel sent {len(channel.sent)} message(s), expected {count}"
    raise AssertionError(msg)
