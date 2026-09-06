from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
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
    async with aiosqlite.connect(str(tmp_path / "history.sqlite")) as connection:
        saver = make_saver(connection)
        await _save(saver, content)
        await _save(saver, content)
        hits = await saver.archive.entries(SCOPE, query=query)
        assert len(hits) == 1
        assert hits[0]["session_id"] == "session"
        chunks = await saver.archive.entries(SCOPE, session_id="session", limit=20)
        assert "".join(chunk["text"] for chunk in chunks) == content
        assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
        await saver.clear_history(SCOPE)
        assert await saver.archive.entries(SCOPE, query=query) == []
        async with connection.execute("SELECT count(*) FROM conversation_search") as cursor:
            assert await cursor.fetchone() == (0,)


async def test_archive_search_survives_reopening_without_checkpoints(tmp_path):
    path = str(tmp_path / "history.sqlite")
    content = "x " * 1998 + " pineapple"
    async with aiosqlite.connect(path) as connection:
        saver = make_saver(connection)
        await _save(saver, content)
        before = await saver.archive.entries(SCOPE, session_id="session")
        await connection.execute("DELETE FROM checkpoints")
        await connection.commit()
    for _ in range(2):
        async with aiosqlite.connect(path) as connection:
            saver = make_saver(connection)
            hits = await saver.archive.entries(SCOPE, query="pineapple")
            assert len(hits) == 1
            assert hits[0]["cursor"] == before[0]["cursor"]
            assert await saver.archive.entries(SCOPE, session_id="session") == before
    async with aiosqlite.connect(path) as connection:
        saver = make_saver(connection)
        await saver.adelete_thread("session")
        assert await saver.archive.entries(SCOPE, query="pineapple") == []
        async with connection.execute("SELECT count(*) FROM conversation_search") as cursor:
            assert await cursor.fetchone() == (0,)
