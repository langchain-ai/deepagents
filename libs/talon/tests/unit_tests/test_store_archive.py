from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents_talon.store_archive import StoreConversationArchive
from tests.store_archive_contract import (
    OTHER,
    SCOPE,
    assert_store_archive_contract,
)


class CountingStore(InMemoryStore):
    reads = 0

    async def aget(self, namespace, key, *, refresh_ttl=True):
        self.reads += 1
        return await super().aget(namespace, key, refresh_ttl=refresh_ttl)


@pytest.mark.parametrize(
    "options",
    [
        {"query": "missing"},
        {"query": "rare"},
        {"session_id": "session", "query": "missing"},
    ],
)
async def test_retrieval_budget_bounds_reads_and_releases_lock(options):
    metadata = CountingStore()
    archive = StoreConversationArchive(metadata, namespace=("budget",))
    await archive.append(
        SCOPE,
        "session",
        "time",
        [
            HumanMessage("rare" if index == 500 else "ordinary", id=str(index))
            for index in range(501)
        ],
    )
    metadata.reads = 0
    with pytest.raises(RuntimeError, match="scan limit exceeded"):
        await archive.entries(SCOPE, **options)
    assert metadata.reads <= 1003  # Recovery, scope/session lookup, 500 chunks and owners.
    await asyncio.wait_for(
        archive.append(SCOPE, "session", "later", [HumanMessage("still writable")]),
        timeout=1,
    )
    assert (await archive.entries(SCOPE, limit=1))[0]["text"] == "still writable"


async def test_long_sessions_page_with_bounded_reads_and_scoped_cursors():
    metadata = CountingStore()
    archive = StoreConversationArchive(metadata, namespace=("paging",))
    await archive.append(
        SCOPE,
        "session",
        "time",
        [HumanMessage(str(index), id=str(index)) for index in range(500)],
    )
    assert await archive.entries(SCOPE, query="missing") == []
    await archive.append(
        SCOPE,
        "session",
        "later",
        [HumanMessage(str(index), id=str(index)) for index in range(500, 520)],
    )
    for session_id in ("", "session"):
        entries, cursor = [], 0
        while True:
            metadata.reads = 0
            page = await archive.entries(SCOPE, session_id=session_id, after=cursor, limit=20)
            assert metadata.reads <= 45
            if not page:
                break
            entries.extend(page)
            cursor = page[-1]["cursor"]
        expected = [str(index) for index in range(520)]
        assert [entry["text"] for entry in entries] == (expected if session_id else expected[::-1])
    await archive.append(OTHER, "private", "time", [HumanMessage("private")])
    foreign = (await archive.entries(OTHER))[0]["cursor"]
    assert await archive.entries(SCOPE, session_id="session", after=foreign) == []
    assert await archive.entries(OTHER, session_id="session", after=cursor) == []
    first = entries[0]["cursor"]
    item = await metadata.aget(archive.records.namespace, str(first))
    await metadata.aput(
        archive.records.namespace, str(first), {**item.value, "next_session": first}
    )
    for after in (0, first):
        with pytest.raises(RuntimeError, match="invalid ordering link"):
            await archive.entries(SCOPE, session_id="session", after=after)


@pytest.mark.parametrize("deleted", [False, True])
async def test_conversation_budget_counts_empty_and_deleted_sessions(deleted):
    metadata = CountingStore()
    archive = StoreConversationArchive(metadata, namespace=("listing",))
    await archive.append(SCOPE, "retained", "time", [HumanMessage("retained")])
    for index in range(501):
        await archive.append(SCOPE, str(index), "time", [])
        if deleted:
            await archive.delete_session(str(index))
    metadata.reads = 0
    with pytest.raises(RuntimeError, match="scan limit exceeded"):
        await archive.conversations(SCOPE)
    assert metadata.reads <= 502


async def test_transcript_budget_counts_deleted_chunks_and_reset_remains_complete():
    metadata = CountingStore()
    archive = StoreConversationArchive(metadata, namespace=("deleted",))
    await archive.append(SCOPE, "retained", "time", [HumanMessage("retained")])
    await archive.append(
        SCOPE, "deleted", "time", [HumanMessage("erase", id=str(index)) for index in range(501)]
    )
    await archive.delete_session("deleted")
    metadata.reads = 0
    with pytest.raises(RuntimeError, match="scan limit exceeded"):
        await archive.entries(SCOPE)
    assert metadata.reads <= 502
    assert await archive.sessions(SCOPE) == ["retained"]
    await archive.delete_session("retained")
    assert await archive.entries(SCOPE) == []


async def test_shared_archive_contract(tmp_path):
    await assert_store_archive_contract(InMemoryStore(), tmp_path)


async def test_remote_message_chunks_and_exclusions():
    archive = StoreConversationArchive(InMemoryStore(), namespace=("chunks",))
    messages = [
        SystemMessage("excluded"),
        HumanMessage("A" * 4500, id="human"),
        ToolMessage("excluded", name="search_conversations", tool_call_id="search"),
    ]
    await archive.append(SCOPE, "session", "time", messages)
    await archive.append(SCOPE, "session", "time", messages)
    entries = await archive.entries(SCOPE, session_id="session")
    assert [len(entry["text"]) for entry in entries] == [4000, 500]
    assert [entry["part"] for entry in entries] == [0, 1]


class InterruptedStore(InMemoryStore):
    remaining = None

    async def aput(self, namespace, key, value, index=None, *, ttl=None):
        if self.remaining is not None:
            self.remaining -= 1
            if self.remaining == 0:
                self.remaining = None
                msg = "interrupted metadata write"
                raise OSError(msg)
        await super().aput(namespace, key, value, index=index, ttl=ttl)


@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6, 7, 8])
async def test_partial_chunk_write_recovers_idempotently_on_reopen(failure):
    metadata = InterruptedStore()
    message = [HumanMessage("durable content", id="message")]
    archive = StoreConversationArchive(metadata, namespace=("recovery",))
    await archive.append(SCOPE, "session", "time", [HumanMessage("first", id="first")])
    metadata.remaining = failure
    with pytest.raises(OSError, match="interrupted metadata"):
        await archive.append(SCOPE, "session", "time", message)
    archive = StoreConversationArchive(metadata, namespace=("recovery",))
    await archive.append(SCOPE, "session", "time", message)
    entries = await archive.entries(SCOPE, session_id="session")
    assert [entry["text"] for entry in entries] == ["first", "durable content"]
    assert (await archive.conversations(SCOPE))[0]["message_count"] == 2


async def test_cancelled_metadata_write_finishes_before_releasing_archive():
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedStore(InMemoryStore):
        blocking = False

        async def aput(self, namespace, key, value, index=None, *, ttl=None):
            if self.blocking and key == "journal":
                entered.set()
                await release.wait()
            await super().aput(namespace, key, value, index=index, ttl=ttl)

    metadata = DelayedStore()
    archive = StoreConversationArchive(metadata, namespace=("cancel",))
    await archive.append(SCOPE, "session", "time", [])
    metadata.blocking = True
    task = asyncio.create_task(archive.append(SCOPE, "session", "time", [HumanMessage("retained")]))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [entry["text"] for entry in await archive.entries(SCOPE)] == ["retained"]


@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6, 7])
async def test_partial_metadata_deletion_is_restartable(failure):
    metadata = InterruptedStore()
    archive = StoreConversationArchive(metadata, namespace=("delete-recovery",))
    await archive.append(SCOPE, "session", "time", [HumanMessage("erase this")])
    metadata.remaining = failure
    with pytest.raises(OSError, match="interrupted metadata"):
        await archive.delete_session("session")
    archive = StoreConversationArchive(metadata, namespace=("delete-recovery",))
    await archive.delete_session("session")
    assert await archive.sessions(SCOPE) == []
    assert await archive.entries(SCOPE) == []


async def test_deleted_session_can_be_reused_without_stale_history():
    metadata = InMemoryStore()
    archive = StoreConversationArchive(metadata, namespace=("reuse",))
    await archive.append(SCOPE, "session", "time", [HumanMessage("old text", id="message")])
    await archive.append(SCOPE, "other", "time", [HumanMessage("other text")])
    await archive.delete_session("session")
    await archive.append(SCOPE, "session", "later", [HumanMessage("new text", id="message")])
    assert [entry["text"] for entry in await archive.entries(SCOPE)] == [
        "new text",
        "other text",
    ]
    assert [entry["text"] for entry in await archive.entries(SCOPE, session_id="session")] == [
        "new text"
    ]
    summaries = await archive.conversations(SCOPE, limit=1)
    rest = await archive.conversations(SCOPE, after=summaries[0]["cursor"], limit=1)
    assert [summaries[0]["session_id"], rest[0]["session_id"]] == ["session", "other"]
