from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.archive import (
    CHUNK_SIZE,
    ArchiveScope,
    ConversationSaver,
    conversation_tools,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel

WHATSAPP = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="chat")
TELEGRAM = ArchiveScope(talon_history_channel="telegram", talon_history_chat="chat")
OTHER = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="other")


async def _save(saver, session, text, *, scope=WHATSAPP, namespace=""):
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [HumanMessage(text, id="message")]}
    config = {
        "configurable": {"thread_id": session, "checkpoint_ns": namespace},
        "metadata": scope,
    }
    return await saver.aput(config, checkpoint, {}, {})


async def test_archive_persists_across_resets_and_isolates_chats(tmp_path):
    path = str(tmp_path / "history.sqlite")
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        await _save(saver, "whatsapp:chat", "Remember the orchard")
        await _save(saver, "whatsapp:chat:talon-reset:1", "Plan the harvest")
        await _save(saver, "telegram:chat", "Telegram orchard", scope=TELEGRAM)
        await _save(saver, "whatsapp:other", "Other orchard", scope=OTHER)
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        hits = await saver.entries(WHATSAPP, query="orchard")
        assert [hit["text"] for hit in hits] == ["Remember the orchard"]
        assert len(await saver.entries(WHATSAPP)) == 2
        assert await saver.entries(WHATSAPP, session_id="telegram:chat") == []
        assert await saver.entries(WHATSAPP, session_id="whatsapp:other") == []
        assert await saver.entries(WHATSAPP, query='orchard" OR "Telegram') == []


async def test_long_transcripts_are_completely_readable_with_bounded_pages(tmp_path):
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        content = "pears " * 2000
        await _save(saver, "whatsapp:chat", content)
        chunks = []
        after = 0
        while page := await saver.entries(
            WHATSAPP, session_id="whatsapp:chat", after=after, limit=1
        ):
            chunks.extend(page)
            after = page[-1]["cursor"]
        assert "".join(chunk["text"] for chunk in chunks) == content
        assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
        for limit in (0, 21):
            with pytest.raises(ValueError, match="limit"):
                await saver.entries(WHATSAPP, limit=limit)
        with pytest.raises(ValueError, match="after"):
            await saver.entries(WHATSAPP, after=-1)


async def test_compaction_preserves_original_messages_without_duplicates(tmp_path):
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        builder = StateGraph(MessagesState)
        builder.add_node("reply", lambda _: {"messages": [AIMessage("Noted", id="reply")]})
        builder.add_edge(START, "reply")
        builder.add_edge("reply", END)
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "whatsapp:chat"}, "metadata": WHATSAPP}
        await graph.ainvoke({"messages": [HumanMessage("old orchard", id="original")]}, config)
        await graph.aupdate_state(config, {"messages": [RemoveMessage(id="original")]})
        snapshot = await graph.aget_state(config)
        assert all(message.id != "original" for message in snapshot.values["messages"])
        assert len(await saver.entries(WHATSAPP, query="orchard")) == 1
        assert len(await saver.entries(WHATSAPP)) == 2


async def test_clear_removes_checkpoints_writes_and_archive_only_in_scope(tmp_path):
    path = str(tmp_path / "history.sqlite")
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        for session in ("whatsapp:chat", "whatsapp:chat:talon-reset:1"):
            config = await _save(saver, session, "old orchard")
            await saver.aput_writes(config, [("messages", [HumanMessage("pending")])], "task")
            await _save(saver, session, "subagent secret", namespace="worker")
        await _save(saver, "telegram:chat", "Telegram orchard", scope=TELEGRAM)
        await saver.clear_history(WHATSAPP)
        await saver.clear_history(WHATSAPP)
        assert await saver.entries(WHATSAPP) == []
        assert len(await saver.entries(TELEGRAM, query="orchard")) == 1
        for table in ("checkpoints", "writes"):
            async with connection.execute(f"SELECT thread_id FROM {table}") as cursor:  # noqa: S608  # Fixed table names.
                assert all(row[0] == "telegram:chat" for row in await cursor.fetchall())
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        assert await saver.entries(WHATSAPP, query="orchard") == []
        assert len(await saver.entries(TELEGRAM)) == 1


async def test_backfill_scoped_legacy_and_recover_unindexed_checkpoints(tmp_path):
    path = str(tmp_path / "history.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await _save(saver, "whatsapp:chat", "Legacy orchard", scope={})
        await _save(saver, "chat", "Unknown channel secret", scope={})
        await _save(saver, "new-session", "Recover interrupted archive", scope=TELEGRAM)
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        assert [hit["text"] for hit in await saver.entries(WHATSAPP)] == ["Legacy orchard"]
        assert len(await saver.entries(TELEGRAM, query="interrupted")) == 1
        await saver.adelete_thread("whatsapp:chat")
        assert await saver.entries(WHATSAPP) == []


async def test_tools_enforce_scope_and_paginate_search(tmp_path):
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        await _save(saver, "one", "orchard one")
        await _save(saver, "two", "orchard two")
        await _save(saver, "secret", "orchard secret", scope=TELEGRAM)
        search, read = conversation_tools(saver, lambda: WHATSAPP)
        first = await search.ainvoke({"query": "orchard", "limit": 1})
        second = await search.ainvoke({"query": "orchard", "limit": 1, "after": first[0]["cursor"]})
        assert {first[0]["session_id"], second[0]["session_id"]} == {"one", "two"}
        assert await read.ainvoke({"session_id": "secret"}) == []


def _graph_factory(**kwargs: object):
    search = next(
        tool for tool in kwargs["tools"] if getattr(tool, "name", "") == "search_conversations"
    )

    async def reply(state):
        query = state["messages"][-1].text
        if query == "recall":
            hits = await search.ainvoke({"query": "orchard"})
            return {"messages": [AIMessage(f"found:{len(hits)}")]}
        return {"messages": [AIMessage("noted")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=kwargs["checkpointer"])


async def _send(host, channel, text):
    await host.receive_message(channel, ChannelMessage("chat", text))
    await asyncio.gather(*host._tasks.values())


async def test_host_new_recall_and_reset_all_history(tmp_path, monkeypatch):
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", _graph_factory)
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    whatsapp, telegram = RecordingChannel("whatsapp"), RecordingChannel("telegram")
    async with aiosqlite.connect(str(config.checkpoint_path)) as connection:
        saver = ConversationSaver(connection)
        runtime = DeepAgentRuntime(
            model="test:model",
            checkpointer=saver,
            assistant_dir=tmp_path,
            include_web_tools=False,
            skills=(),
            memory=(),
        )
        host = TalonHost(config=config, agent=runtime, channels=[whatsapp, telegram])
        await host.start()
        try:
            await _send(host, whatsapp, "remember orchard")
            await _send(host, whatsapp, "/new")
            await _send(host, whatsapp, "recall")
            assert whatsapp.sent[-1] == ("chat", "found:1")
            await _send(host, telegram, "recall")
            assert telegram.sent[-1] == ("chat", "found:0")
            await _send(host, whatsapp, "/reset-all-history@TestBot")
            assert "Cleared all conversation history" in whatsapp.sent[-1][1]
            await _send(host, whatsapp, "recall")
            assert whatsapp.sent[-1] == ("chat", "found:0")
            assert await saver.entries(WHATSAPP, query="orchard") == []
            assert await saver.entries(TELEGRAM)
        finally:
            await host.stop()


async def test_failed_archive_write_rolls_back_checkpoint(tmp_path, monkeypatch):
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        await saver.setup()

        async def fail_index(*_args: object) -> None:
            msg = "archive unavailable"
            raise OSError(msg)

        monkeypatch.setattr(saver, "_index", fail_index)
        with pytest.raises(OSError, match="archive unavailable"):
            await _save(saver, "whatsapp:chat", "must not partially save")
        assert await saver.aget({"configurable": {"thread_id": "whatsapp:chat"}}) is None


async def test_failed_reset_rolls_back_archive_and_checkpoints(tmp_path):
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        config = await _save(saver, "whatsapp:chat", "keep orchard")
        await connection.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON conversation_chunks "
            "BEGIN SELECT RAISE(ABORT, 'delete failed'); END"
        )
        with pytest.raises(aiosqlite.IntegrityError, match="delete failed"):
            await saver.clear_history(WHATSAPP)
        assert await saver.aget(config) is not None
        assert len(await saver.entries(WHATSAPP, query="orchard")) == 1


async def test_reset_cancels_active_turn_before_deleting_history(tmp_path, monkeypatch):
    entered = asyncio.Event()

    def factory(**kwargs: object):
        async def reply(_state):
            entered.set()
            await asyncio.Event().wait()

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    channel = RecordingChannel("whatsapp")
    async with aiosqlite.connect(str(config.checkpoint_path)) as connection:
        saver = ConversationSaver(connection)
        runtime = DeepAgentRuntime(
            model="test:model",
            checkpointer=saver,
            assistant_dir=tmp_path,
            include_web_tools=False,
            skills=(),
            memory=(),
        )
        host = TalonHost(config=config, agent=runtime, channels=[channel])
        await host.start()
        try:
            await host.receive_message(channel, ChannelMessage("chat", "orchard"))
            await entered.wait()
            await host.receive_message(channel, ChannelMessage("chat", "/reset-all-history"))
            assert channel.sent == [
                (
                    "chat",
                    "Cleared all conversation history for this chat. Started a fresh conversation.",
                )
            ]
            assert await saver.entries(WHATSAPP) == []
            assert await saver.aget({"configurable": {"thread_id": "whatsapp:chat"}}) is None
        finally:
            await host.stop()


async def test_explicit_legacy_import_survives_restart_and_can_be_erased(tmp_path):
    path = str(tmp_path / "history.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await _save(saver, "chat", "old orchard", scope={})
        await _save(saver, "chat:talon-reset:1", "older orchard", scope={})
        await _save(saver, "job:talon-cron", "cron secret", scope={})
        await _save(saver, "subagent-worker", "worker secret", scope={})
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        assert await saver.entries(WHATSAPP) == []
        await saver.import_legacy_history("whatsapp")
        assert len(await saver.entries(WHATSAPP, query="orchard")) == 2
        assert await saver.entries(TELEGRAM) == []
        await saver.clear_history(WHATSAPP)
    async with aiosqlite.connect(path) as connection:
        saver = ConversationSaver(connection)
        await saver.import_legacy_history("whatsapp")
        assert await saver.entries(WHATSAPP) == []
        with pytest.raises(ValueError, match="another channel"):
            await saver.import_legacy_history("telegram")
        with pytest.raises(ValueError, match="must be"):
            await saver.import_legacy_history("unknown")


async def test_concurrent_channels_do_not_share_retrieval_scope(tmp_path, monkeypatch):
    arrived = 0
    ready = asyncio.Event()

    def factory(**kwargs: object):
        search = next(
            tool for tool in kwargs["tools"] if getattr(tool, "name", "") == "search_conversations"
        )

        async def reply(_state):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await ready.wait()
            hits = await search.ainvoke({"query": "secret"})
            return {"messages": [AIMessage(hits[0]["text"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        await _save(saver, "old-whatsapp", "whatsapp secret")
        await _save(saver, "old-telegram", "telegram secret", scope=TELEGRAM)
        runtime = DeepAgentRuntime(
            model="test:model",
            checkpointer=saver,
            assistant_dir=tmp_path,
            include_web_tools=False,
            skills=(),
            memory=(),
        )
        await runtime.start()
        try:
            results = await asyncio.gather(
                *(
                    runtime.invoke(
                        AgentRequest(
                            f"{channel}:chat",
                            "recall",
                            metadata={"history_channel": channel, "history_chat": "chat"},
                        )
                    )
                    for channel in ("whatsapp", "telegram")
                )
            )
            assert [result.text for result in results] == ["whatsapp secret", "telegram secret"]
        finally:
            await runtime.stop()


async def test_message_revisions_are_retained_without_checkpoint_duplicates(tmp_path):
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = ConversationSaver(connection)
        await _save(saver, "whatsapp:chat", "Meet on Tuesday")
        await _save(saver, "whatsapp:chat", "Meet on Wednesday")
        await _save(saver, "whatsapp:chat", "Meet on Wednesday")
        transcript = await saver.entries(WHATSAPP, session_id="whatsapp:chat")
        assert [chunk["text"] for chunk in transcript] == ["Meet on Tuesday", "Meet on Wednesday"]
        assert len(await saver.entries(WHATSAPP, query="Wednesday")) == 1
