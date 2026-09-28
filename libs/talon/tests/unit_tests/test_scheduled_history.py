from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest
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
