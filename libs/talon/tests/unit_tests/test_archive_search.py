from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint

from deepagents_talon.archive import CHUNK_SIZE, ArchiveScope
from tests.archive_helpers import make_saver

if TYPE_CHECKING:
    from deepagents_talon.archive_saver import ConversationSaver

SCOPE = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="chat")


async def _save(saver: ConversationSaver, text: str) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [HumanMessage(text, id="message")]}
    await saver.aput(
        {"configurable": {"thread_id": "session", "checkpoint_ns": ""}, "metadata": SCOPE},
        checkpoint,
        {},
        {},
    )


@pytest.mark.parametrize(
    ("content", "query"),
    [
        ("x " * 1998 + " pineapple", "pineapple"),
        ("x" * (CHUNK_SIZE + 10), "x" * (CHUNK_SIZE + 10)),
        ("cafe\u0301 " * 799 + " pine\u0301apple", "pinéapple"),
        ("orchard " + "x " * CHUNK_SIZE + "pineapple", "orchard pineapple"),
    ],
)
async def test_search_matches_complete_revisions_with_bounded_display(tmp_path, content, query):
    async with make_saver(str(tmp_path / "history.sqlite")) as saver:
        await _save(saver, content)
        await _save(saver, content)
        hits = (await saver.archive.search_page(SCOPE, query=query))["results"]
        assert len(hits) == 1
        assert hits[0]["session_id"] == "session"
        chunks = await saver.archive.entries(SCOPE, session_id="session", limit=20)
        assert "".join(chunk["text"] for chunk in chunks) == content
        assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
        await saver.clear_history(SCOPE)
        assert (await saver.archive.search_page(SCOPE, query=query))["results"] == []
        async with saver.checkpointer.conn.execute("SELECT value FROM store") as cursor:
            assert content not in str(await cursor.fetchall())


async def test_archive_search_survives_reopening_without_checkpoints(tmp_path):
    path = str(tmp_path / "history.sqlite")
    content = "x " * 1998 + " pineapple"
    async with make_saver(path) as saver:
        await _save(saver, content)
        before = await saver.archive.entries(SCOPE, session_id="session")
        await saver.checkpointer.conn.execute("DELETE FROM checkpoints")
        await saver.checkpointer.conn.commit()
    for _ in range(2):
        async with make_saver(path) as saver:
            hits = (await saver.archive.search_page(SCOPE, query="pineapple"))["results"]
            assert len(hits) == 1
            assert hits[0]["cursor"] == before[0]["cursor"]
            assert await saver.archive.entries(SCOPE, session_id="session") == before
    async with make_saver(path) as saver:
        await saver.adelete_thread("session")
        assert (await saver.archive.search_page(SCOPE, query="pineapple"))["results"] == []
        async with saver.checkpointer.conn.execute("SELECT value FROM store") as cursor:
            assert content not in str(await cursor.fetchall())
