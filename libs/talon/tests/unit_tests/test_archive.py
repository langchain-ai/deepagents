from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from deepagents.graph import DeepAgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.archive import (
    CHUNK_SIZE,
    ArchiveScope,
    conversation_tools,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from tests.archive_helpers import make_runtime, make_saver
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
    async with make_saver(path) as saver:
        await _save(saver, "whatsapp:chat", "Remember the orchard")
        await _save(saver, "whatsapp:chat:talon-reset:1", "Plan the harvest")
        await _save(saver, "telegram:chat", "Telegram orchard", scope=TELEGRAM)
        await _save(saver, "whatsapp:other", "Other orchard", scope=OTHER)
    async with make_saver(path) as saver:
        hits = await saver.archive.entries(WHATSAPP, query="orchard")
        assert [hit["text"] for hit in hits] == ["Remember the orchard"]
        assert len(await saver.archive.entries(WHATSAPP)) == 2
        assert await saver.archive.entries(WHATSAPP, session_id="telegram:chat") == []
        assert await saver.archive.entries(WHATSAPP, session_id="whatsapp:other") == []
        assert await saver.archive.entries(WHATSAPP, query='orchard" OR "Telegram') == []


async def test_long_transcripts_are_completely_readable_with_bounded_pages(tmp_path):
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        content = "pears " * 2000
        await _save(saver, "whatsapp:chat", content)
        chunks = []
        after = 0
        while page := await saver.archive.entries(
            WHATSAPP, session_id="whatsapp:chat", after=after, limit=1
        ):
            chunks.extend(page)
            after = page[-1]["cursor"]
        assert "".join(chunk["text"] for chunk in chunks) == content
        assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
        for limit in (0, 21):
            with pytest.raises(ValueError, match="limit"):
                await saver.archive.entries(WHATSAPP, limit=limit)
        with pytest.raises(ValueError, match="after"):
            await saver.archive.entries(WHATSAPP, after=-1)


@pytest.mark.parametrize("state_schema", [MessagesState, DeepAgentState])
@pytest.mark.parametrize("backend", [InMemorySaver, AsyncSqliteSaver])
async def test_compaction_preserves_original_messages_without_duplicates(
    tmp_path, state_schema, backend
):
    async with make_saver(str(tmp_path / "history.sqlite"), backend) as saver:
        builder = StateGraph(state_schema)
        builder.add_node("reply", lambda _: {"messages": [AIMessage("Noted", id="reply")]})
        builder.add_edge(START, "reply")
        builder.add_edge("reply", END)
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "whatsapp:chat"}, "metadata": WHATSAPP}
        await graph.ainvoke({"messages": [HumanMessage("old orchard", id="original")]}, config)
        await graph.aupdate_state(config, {"messages": [RemoveMessage(id="original")]})
        snapshot = await graph.aget_state(config)
        assert all(message.id != "original" for message in snapshot.values["messages"])
        assert len(await saver.archive.entries(WHATSAPP, query="orchard")) == 1
        assert len(await saver.archive.entries(WHATSAPP)) == 2


async def test_clear_removes_checkpoints_writes_and_archive_only_in_scope(tmp_path):
    path = str(tmp_path / "history.sqlite")
    async with make_saver(path) as saver:
        for session in ("whatsapp:chat", "whatsapp:chat:talon-reset:1"):
            config = await _save(saver, session, "old orchard")
            await saver.aput_writes(config, [("messages", [HumanMessage("pending")])], "task")
            await _save(saver, session, "subagent secret", namespace="worker")
        await _save(saver, "telegram:chat", "Telegram orchard", scope=TELEGRAM)
        await saver.clear_history(WHATSAPP)
        await saver.clear_history(WHATSAPP)
        assert await saver.archive.entries(WHATSAPP) == []
        assert len(await saver.archive.entries(TELEGRAM, query="orchard")) == 1
        for table in ("checkpoints", "writes"):
            async with saver.checkpointer.conn.execute(f"SELECT thread_id FROM {table}") as cursor:  # noqa: S608  # Fixed table names.
                assert all(row[0] == "telegram:chat" for row in await cursor.fetchall())
    async with make_saver(path) as saver:
        assert await saver.archive.entries(WHATSAPP, query="orchard") == []
        assert len(await saver.archive.entries(TELEGRAM)) == 1


async def test_tools_enforce_scope_and_paginate_search(tmp_path):
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        await _save(saver, "one", "orchard one")
        await _save(saver, "two", "orchard two")
        await _save(saver, "secret", "orchard secret", scope=TELEGRAM)
        tools = {tool.name: tool for tool in conversation_tools(saver.archive, lambda: WHATSAPP)}
        search, read = tools["search_conversations"], tools["read_conversation"]
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
        if query == "list":
            listing = next(tool for tool in kwargs["tools"] if tool.name == "list_conversations")
            sessions = await listing.ainvoke({})
            return {"messages": [AIMessage(f"sessions:{len(sessions)}")]}
        if query == "recall":
            hits = await search.ainvoke({"query": "orchard"})
            return {"messages": [AIMessage(f"found:{len(hits)}")]}
        return {"messages": [AIMessage("noted")]}

    builder = StateGraph(DeepAgentState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=kwargs["checkpointer"])


async def _send(host, channel, text):
    await host.receive_message(channel, ChannelMessage("chat", text))
    await asyncio.gather(*host._tasks.values())


@pytest.mark.parametrize("backend", [InMemorySaver, AsyncSqliteSaver])
async def test_host_new_recall_and_reset_all_history(tmp_path, monkeypatch, backend):
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", _graph_factory)
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    whatsapp, telegram = RecordingChannel("whatsapp"), RecordingChannel("telegram")
    async with make_saver(str(config.checkpoint_path), backend) as saver:
        runtime = make_runtime(saver, tmp_path)
        host = TalonHost(config=config, agent=runtime, channels=[whatsapp, telegram])
        await host.start()
        try:
            await _send(host, whatsapp, "remember orchard")
            await _send(host, whatsapp, "/new")
            await _send(host, whatsapp, "recall")
            assert whatsapp.sent[-1] == ("chat", "found:1")
            await _send(host, whatsapp, "list")
            assert whatsapp.sent[-1] == ("chat", "sessions:2")
            await _send(host, telegram, "recall")
            assert telegram.sent[-1] == ("chat", "found:0")
            await _send(host, whatsapp, "/reset-all-history@TestBot")
            assert "Cleared all conversation history" in whatsapp.sent[-1][1]
            await _send(host, whatsapp, "recall")
            assert whatsapp.sent[-1] == ("chat", "found:0")
            assert await saver.archive.entries(WHATSAPP, query="orchard") == []
            assert await saver.archive.entries(TELEGRAM)
        finally:
            await host.stop()


async def test_failed_archive_deletion_retains_registration_for_retry(tmp_path):
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        config = await _save(saver, "whatsapp:chat", "keep orchard")
        await saver.checkpointer.conn.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON store "
            "BEGIN SELECT RAISE(ABORT, 'delete failed'); END"
        )
        with pytest.raises(aiosqlite.IntegrityError, match="delete failed"):
            await saver.clear_history(WHATSAPP)
        assert await saver.aget(config) is None
        await saver.checkpointer.conn.execute("DROP TRIGGER fail_delete")
        assert await saver.archive.sessions(WHATSAPP) == ["whatsapp:chat"]
        await saver.clear_history(WHATSAPP)
        assert await saver.archive.sessions(WHATSAPP) == []
        assert await saver.archive.entries(WHATSAPP) == []


@pytest.mark.parametrize("backend", [InMemorySaver, AsyncSqliteSaver])
async def test_reset_cancels_active_turn_before_deleting_history(tmp_path, monkeypatch, backend):
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
    async with make_saver(str(config.checkpoint_path), backend) as saver:
        runtime = make_runtime(saver, tmp_path)
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
            assert await saver.archive.entries(WHATSAPP) == []
            assert await saver.aget({"configurable": {"thread_id": "whatsapp:chat"}}) is None
        finally:
            await host.stop()


@pytest.mark.parametrize("backend", [InMemorySaver, AsyncSqliteSaver])
async def test_concurrent_channels_do_not_share_retrieval_scope(tmp_path, monkeypatch, backend):
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
    async with make_saver(str(tmp_path / "history.sqlite"), backend) as saver:
        await _save(saver, "old-whatsapp", "whatsapp secret")
        await _save(saver, "old-telegram", "telegram secret", scope=TELEGRAM)
        runtime = make_runtime(saver, tmp_path)
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
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        await _save(saver, "whatsapp:chat", "Meet on Tuesday")
        await _save(saver, "whatsapp:chat", "Meet on Wednesday")
        await _save(saver, "whatsapp:chat", "Meet on Wednesday")
        transcript = await saver.archive.entries(WHATSAPP, session_id="whatsapp:chat")
        assert [chunk["text"] for chunk in transcript] == ["Meet on Tuesday", "Meet on Wednesday"]
        assert len(await saver.archive.entries(WHATSAPP, query="Wednesday")) == 1


async def test_list_conversations_is_scoped_paginated_and_readable(tmp_path):
    path = str(tmp_path / "history.sqlite")
    async with make_saver(path) as saver:
        await _save(saver, "old", "orchard " * 2000)
        await _save(saver, "old", "edited orchard")
        await _save(saver, "old", "edited orchard")
        await _save(saver, "new", "new harvest")
        await _save(saver, "secret", "telegram secret", scope=TELEGRAM)
        await _save(saver, "other", "other chat secret", scope=OTHER)
        await _save(saver, "empty", "")
    async with make_saver(path) as saver:
        tools = {tool.name: tool for tool in conversation_tools(saver.archive, lambda: WHATSAPP)}
        listing = tools["list_conversations"]
        first = await listing.ainvoke({"limit": 1})
        assert [item["session_id"] for item in first] == ["new"]
        # Updating a listed session must not disturb traversal to older sessions.
        await _save(saver, "new", "revised harvest")
        second = await listing.ainvoke({"limit": 1, "after": first[0]["cursor"]})
        assert [item["session_id"] for item in second] == ["old"]
        assert second[0]["message_count"] == 1
        assert second[0]["preview"] == ("orchard " * 2000)[:300]
        assert second[0]["started_at"] <= second[0]["updated_at"]
        assert await listing.ainvoke({"after": second[0]["cursor"]}) == []
        transcript = await tools["read_conversation"].ainvoke({"session_id": "old", "limit": 20})
        assert "".join(chunk["text"] for chunk in transcript) == (
            "orchard " * 2000 + "edited orchard"
        )
        await saver.clear_history(WHATSAPP)
        assert await listing.ainvoke({}) == []
        assert [item["session_id"] for item in await saver.archive.conversations(TELEGRAM)] == [
            "secret"
        ]


@pytest.mark.parametrize(("after", "limit"), [(-1, 5), (0, 0), (0, 21)])
async def test_list_conversations_rejects_invalid_pagination(tmp_path, after, limit):
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        with pytest.raises(ValueError, match="limit"):
            await saver.archive.conversations(WHATSAPP, after=after, limit=limit)


async def test_listing_counts_messages_and_excludes_its_own_tool_results(tmp_path):
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "messages": [
                HumanMessage("List my sessions", id="question"),
                ToolMessage(
                    "private listing preview", name="list_conversations", tool_call_id="call"
                ),
                AIMessage("Here are your sessions", id="answer"),
            ]
        }
        await saver.aput(
            {"configurable": {"thread_id": "session", "checkpoint_ns": ""}, "metadata": WHATSAPP},
            checkpoint,
            {},
            {},
        )
        sessions = await saver.archive.conversations(WHATSAPP)
        assert sessions[0]["message_count"] == 2
        assert sessions[0]["preview"] == "List my sessions"
        assert await saver.archive.entries(WHATSAPP, query="private") == []
