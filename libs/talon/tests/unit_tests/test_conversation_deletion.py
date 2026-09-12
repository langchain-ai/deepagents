from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.interfaces import AgentRequest
from tests.archive_helpers import make_runtime, make_saver
from tests.unit_tests.test_archive import OTHER, TELEGRAM, WHATSAPP, _save


@pytest.mark.parametrize("session_ids", ["one", ["one", "two", "one"], ["one", "active"]])
@pytest.mark.parametrize("backend", [InMemorySaver, AsyncSqliteSaver])
async def test_runtime_deletes_only_requested_conversations(
    tmp_path, monkeypatch, session_ids, backend
):
    results = []

    def factory(**kwargs: object):
        deletion = next(tool for tool in kwargs["tools"] if tool.name == "delete_conversations")

        async def reply(_state):
            if "active" in session_ids:
                with pytest.raises(ValueError, match="active conversation"):
                    await deletion.ainvoke({"session_ids": session_ids})
                return {"messages": [AIMessage("Use /new before deleting this conversation.")]}
            results.append(await deletion.ainvoke({"session_ids": session_ids}))
            return {"messages": [AIMessage("Deleted requested conversations.")]}

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    path = tmp_path / "history.sqlite"
    async with make_saver(path, backend) as saver:
        configs = {}
        for session in ("one", "two", "keep"):
            configs[session] = await _save(saver, session, "orchard")
            await saver.aput_writes(configs[session], [("test", "pending")], "task")
            await _save(saver, session, "nested", namespace="worker")
        runtime = make_runtime(saver, tmp_path)
        await runtime.start()
        try:
            await runtime.invoke(
                AgentRequest(
                    "active",
                    "Delete the requested conversations",
                    metadata={"history_channel": "whatsapp", "history_chat": "chat"},
                )
            )
        finally:
            await runtime.stop()
        deleted = ["one"] if isinstance(session_ids, str) else ["one", "two"]
        if "active" in session_ids:
            deleted = []
            assert results == []
        else:
            assert results == [{"deleted": deleted, "not_found": []}]
        for session in deleted:
            assert await saver.aget(configs[session]) is None
            assert [
                item async for item in saver.alist({"configurable": {"thread_id": session}})
            ] == []
        assert await saver.aget(configs["keep"])
        assert await saver.aget({"configurable": {"thread_id": "active"}})
    async with make_saver(path, backend) as saver:
        sessions = {item["session_id"] for item in await saver.archive.conversations(WHATSAPP)}
        assert sessions == {"active", "keep", "one", "two"} - set(deleted)
        hits = (await saver.archive.search_page(WHATSAPP, query="orchard"))["results"]
        assert {hit["session_id"] for hit in hits} == {"keep", "one", "two"} - set(deleted)


async def test_deletion_hides_foreign_sessions_and_is_idempotent(tmp_path):
    async with make_saver(tmp_path / "history.sqlite") as saver:
        await _save(saver, "owned", "orchard")
        foreign = [
            await _save(saver, session, "secret", scope=scope)
            for session, scope in (("channel", TELEGRAM), ("chat", OTHER))
        ]
        ids = ["owned", "channel", "chat", "missing"]
        assert await saver.delete_conversations(WHATSAPP, ids, current_session="active") == {
            "deleted": ["owned"],
            "not_found": ids[1:],
        }
        assert await saver.delete_conversations(WHATSAPP, ids, current_session="active") == {
            "deleted": [],
            "not_found": ids,
        }
        for config in foreign:
            assert await saver.aget(config)
        assert await saver.archive.entries(TELEGRAM)
        assert await saver.archive.entries(OTHER)


@pytest.mark.parametrize("ids", [[], [""], ["owned", " "], ["owned", "active"]])
async def test_invalid_batch_does_not_delete_anything(tmp_path, ids):
    async with make_saver(tmp_path / "history.sqlite") as saver:
        owned = await _save(saver, "owned", "orchard")
        active = await _save(saver, "active", "current")
        with pytest.raises(ValueError, match=r"nonempty session IDs|active conversation"):
            await saver.delete_conversations(WHATSAPP, ids, current_session="active")
        assert await saver.aget(owned)
        assert await saver.aget(active)
        assert set(await saver.archive.sessions(WHATSAPP)) == {"owned", "active"}


async def test_partial_deletion_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "history.sqlite"
    async with make_saver(path) as saver:
        for session in ("one", "two"):
            await _save(saver, session, "orchard")
        delete = saver.archive.delete_session

        async def fail_second(session):
            if session == "two":
                msg = "archive unavailable"
                raise OSError(msg)
            await delete(session)

        monkeypatch.setattr(saver.archive, "delete_session", fail_second)
        with pytest.raises(OSError, match="archive unavailable"):
            await saver.delete_conversations(WHATSAPP, ["one", "two"], current_session="active")
        assert await saver.archive.sessions(WHATSAPP) == ["two"]
    async with make_saver(path) as saver:
        assert await saver.delete_conversations(
            WHATSAPP, ["one", "two"], current_session="active"
        ) == {"deleted": ["two"], "not_found": ["one"]}
        assert await saver.archive.entries(WHATSAPP) == []


async def test_tool_without_host_scope_cannot_delete(tmp_path):
    async with make_saver(tmp_path / "history.sqlite") as saver:
        owned = await _save(saver, "owned", "orchard")
        runtime = make_runtime(saver, tmp_path)
        deletion = next(
            tool for tool in runtime._build_tools() if tool.name == "delete_conversations"
        )
        with pytest.raises(RuntimeError, match="supplied by the host"):
            await deletion.ainvoke({"session_ids": "owned"})
        assert await saver.aget(owned)
