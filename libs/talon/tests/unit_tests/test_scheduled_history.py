from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.runtime import _current_cron_origin
from tests.archive_helpers import make_runtime, make_saver
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config
from tests.unit_tests.test_archive import OTHER, WHATSAPP, _save

if TYPE_CHECKING:
    from pathlib import Path

CRON_THREAD = "job:talon-cron"


async def test_scheduled_turn_reads_origin_chat_without_archiving_or_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = []

    def factory(**kwargs: object):
        tools = {tool.name: tool for tool in kwargs["tools"]}

        async def reply(_state):
            listed.extend(await tools["list_conversations"].ainvoke({"limit": 20}))
            entries = await tools["read_conversation"].ainvoke({"session_id": "mine"})
            with pytest.raises(RuntimeError, match="Scheduled runs cannot delete"):
                await tools["delete_conversations"].ainvoke({"session_ids": "mine"})
            return {
                "messages": [
                    ToolMessage(entries[0]["text"], tool_call_id="read"),
                    AIMessage("Reviewed the chat."),
                ]
            }

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    async with make_saver(tmp_path / "history.sqlite") as saver:
        mine = await _save(saver, "mine", "orchard")
        await _save(saver, "theirs", "secret", scope=OTHER)
        runtime = make_runtime(saver, tmp_path)
        await runtime.start()
        try:
            result = await runtime.invoke(
                AgentRequest(
                    CRON_THREAD,
                    "Review the chat",
                    metadata={
                        "trigger": "cron",
                        "history_channel": "whatsapp",
                        "history_chat": "chat",
                    },
                )
            )
            assert set(await saver.archive.sessions(WHATSAPP)) == {"mine", CRON_THREAD}
            assert await saver.archive.entries(WHATSAPP, session_id=CRON_THREAD) == []
            checkpoint = await saver.aget({"configurable": {"thread_id": CRON_THREAD}})
            assert "orchard" in str(checkpoint)
        finally:
            await runtime.stop()

        assert result.text == "Reviewed the chat."
        assert [item["session_id"] for item in listed] == ["mine"]

    async with make_saver(tmp_path / "history.sqlite") as saver:
        await make_runtime(saver, tmp_path).clear_history("whatsapp", "chat")
        assert await saver.aget(mine) is None
        assert await saver.aget({"configurable": {"thread_id": CRON_THREAD}}) is None
        assert await saver.archive.sessions(OTHER) == ["theirs"]


@pytest.mark.parametrize(("origin_channel", "expected"), [("test", True), (None, False)])
async def test_scheduled_job_receives_origin_history_scope(
    tmp_path: Path, origin_channel: str | None, *, expected: bool
) -> None:
    class HistoryAgent(BlockingAgent):
        history_enabled = True

        async def clear_history(self, channel: str, chat: str) -> None:
            del channel, chat

    agent = HistoryAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[RecordingChannel()])
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "test" / "cron")
    job = store.create_job(
        prompt="review",
        schedule=CronSchedule.parse("in 5m"),
        origin=CronOrigin(conversation_id="chat", channel=origin_channel),
    )
    await host.start()
    try:
        await host.run_scheduled_job(job)
    finally:
        await host.stop()

    metadata = agent.requests[0].metadata
    scope = {key: metadata[key] for key in ("history_channel", "history_chat") if key in metadata}
    assert scope == ({"history_channel": "test", "history_chat": "chat"} if expected else {})


async def test_scheduled_job_does_not_block_sibling_thread(tmp_path: Path) -> None:
    class SlowHistoryAgent(BlockingAgent):
        history_enabled = True

        def __init__(self) -> None:
            super().__init__()
            self.cron_started = asyncio.Event()

        async def invoke(self, request: AgentRequest) -> AgentResult:
            if request.text == "review":
                self.cron_started.set()
                await self.released.wait()
            return await super().invoke(request)

        async def clear_history(self, channel: str, chat: str) -> None:
            del channel, chat

    agent = SlowHistoryAgent()
    channel = RecordingChannel("discord")
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    job = store.create_job(
        prompt="review",
        schedule=CronSchedule.parse("in 5m"),
        origin=CronOrigin(conversation_id="thread-1", channel="discord", history_chat="100"),
    )
    await host.start()
    run = asyncio.create_task(host.run_scheduled_job(job))
    try:
        await asyncio.wait_for(agent.cron_started.wait(), timeout=2)
        await asyncio.wait_for(
            host.receive_message(
                channel, ChannelMessage("thread-2", "hello", metadata={"history_chat": "100"})
            ),
            timeout=2,
        )
        await asyncio.wait_for(asyncio.gather(*host._tasks.values()), timeout=2)
        assert agent.requests[-1].text == "hello"
        assert not run.done()
    finally:
        agent.released.set()
        await run
        await host.stop()


@pytest.mark.parametrize(
    ("provider", "conversation", "parent", "deliver_to"),
    [
        ("discord", "thread-1", "100", "channel"),
        ("discord", "thread-1", "100", "thread"),
        ("slack", "C1:1700000000.000100", "C1", "channel"),
    ],
)
async def test_scheduled_history_uses_parent_and_delivers_by_choice(  # noqa: PLR0913  # Shared provider fixture.
    tmp_path, monkeypatch, provider, conversation, parent, deliver_to
):
    origins = []
    scopes = []

    def factory(**kwargs: object):
        tools = {tool.name: tool for tool in kwargs["tools"]}

        async def reply(state):
            origins.append(_current_cron_origin())
            if state["messages"][-1].text == "recall":
                scopes.extend(await tools["list_conversations"].ainvoke({"limit": 20}))
            return {"messages": [AIMessage("noted")]}

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    channel = RecordingChannel(provider)
    async with make_saver(tmp_path / "history.sqlite") as saver:
        host = TalonHost(
            config=_config(tmp_path), agent=make_runtime(saver, tmp_path), channels=[channel]
        )
        await host.start()
        try:
            await host.receive_message(
                channel,
                ChannelMessage(
                    conversation,
                    "hello",
                    metadata={"history_chat": parent, "is_dm": False},
                ),
            )
            await asyncio.gather(*host._tasks.values())
            store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
            job = store.create_job(
                prompt="recall",
                schedule=CronSchedule.parse("in 5m"),
                origin=origins[0],
                deliver_to=deliver_to,
            )
            saved = store.get_job(job.id)
            assert saved is not None
            assert saved.origin.conversation_id == conversation
            assert saved.origin.history_chat == parent
            result = await host.run_scheduled_job(saved)
            assert [entry["preview"] for entry in scopes] == ["hello"]
            await host.deliver_scheduled_result(channel, saved, result)
            assert channel.sent[-1] == (
                parent if deliver_to == "channel" else conversation,
                "noted",
            )
            entries = await saver.archive.entries(
                {"talon_history_channel": provider, "talon_history_chat": parent},
                session_id=f"{job.id}:talon-cron",
            )
            assert [entry["text"] for entry in entries] == ["noted"]
        finally:
            await host.stop()


async def test_scheduled_history_preserves_whatsapp_archive_address(tmp_path, monkeypatch):
    origins = []

    def factory(**kwargs: object):
        tools = {tool.name: tool for tool in kwargs["tools"]}

        async def reply(state):
            origins.append(_current_cron_origin())
            if state["messages"][-1].text == "recall":
                hits = await tools["search_conversations"].ainvoke({"query": "orchard"})
                found = any("remember orchard" in hit["text"] for hit in hits["results"])
                return {"messages": [AIMessage("recalled" if found else "missing")]}
            return {"messages": [AIMessage("noted")]}

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    channel = RecordingChannel("whatsapp")
    async with make_saver(tmp_path / "history.sqlite") as saver:
        runtime = make_runtime(saver, tmp_path)
        host = TalonHost(config=_config(tmp_path), agent=runtime, channels=[channel])
        await host.start()
        try:
            await host.receive_message(
                channel,
                ChannelMessage(
                    "chat@lid",
                    "remember orchard",
                    metadata={"chat_id_from": "123@s.whatsapp.net"},
                ),
            )
            await asyncio.gather(*host._tasks.values())
            store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
            job = store.create_job(
                prompt="recall",
                schedule=CronSchedule.parse("in 5m"),
                origin=origins[0],
            )
            job = store.get_job(job.id)
            assert job is not None
            assert job.origin.conversation_id == "chat@lid"
            result = await host.run_scheduled_job(job)
            assert result == "recalled"
            assert origins[-1].conversation_id == "chat@lid"
            await host.deliver_scheduled_result(channel, job, result)
            scope = {**WHATSAPP, "talon_history_chat": "chat@lid"}
            entries = await saver.archive.entries(scope, session_id=f"{job.id}:talon-cron")
            assert [entry["text"] for entry in entries] == ["recalled"]
            assert channel.sent[-1] == ("chat@lid", "recalled")
        finally:
            await host.stop()


class _ThreadedChannel(RecordingChannel):
    def top_level_conversation_id(self, conversation_id: str) -> str:
        return conversation_id.partition(":")[0]


@pytest.mark.parametrize(
    ("channel", "deliver_to", "target"),
    [
        (_ThreadedChannel("slack"), "channel", "C1"),
        (_ThreadedChannel("slack"), "thread", "C1:1.2"),
        (RecordingChannel("whatsapp"), "channel", "C1:1.2"),
    ],
)
async def test_scheduled_result_posts_to_the_thread_parent_by_choice(
    tmp_path: Path, channel: RecordingChannel, deliver_to: str, target: str
) -> None:
    host = TalonHost(config=_config(tmp_path), agent=BlockingAgent(), channels=[channel])
    job = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron").create_job(
        prompt="report",
        schedule=CronSchedule.parse("in 5m"),
        origin=CronOrigin(conversation_id="C1:1.2", channel=channel.provider),
        deliver_to=deliver_to,
    )

    await host.deliver_scheduled_result(channel, job, "done")

    assert channel.sent == [(target, "done")]
