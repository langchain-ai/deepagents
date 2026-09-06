from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

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
        {"after": 1},
        {"session_id": "session"},
        {"session_id": "session", "query": "missing"},
        {"session_id": "session", "query": "rare"},
    ],
)
async def test_retrieval_budget_bounds_reads_and_releases_lock(options):
    metadata = CountingStore()
    async with StoreConversationArchive(metadata, namespace=("budget",)).open() as archive:
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


async def test_retrieval_exact_budget_preserves_chronological_pages():
    metadata = CountingStore()
    async with StoreConversationArchive(metadata, namespace=("boundary",)).open() as archive:
        await archive.append(
            SCOPE,
            "session",
            "time",
            [HumanMessage(str(index), id=str(index)) for index in range(500)],
        )
        assert await archive.entries(SCOPE, query="missing") == []
        first = await archive.entries(SCOPE, session_id="session", limit=2)
        second = await archive.entries(
            SCOPE, session_id="session", after=first[-1]["cursor"], limit=2
        )
        assert [entry["text"] for entry in first + second] == ["0", "1", "2", "3"]
        latest = await archive.entries(SCOPE, limit=1)
        metadata.reads = 0
        assert await archive.entries(SCOPE, session_id="session", after=latest[0]["cursor"]) == []
        assert metadata.reads <= 4


@pytest.mark.parametrize("deleted", [False, True])
async def test_conversation_budget_counts_empty_and_deleted_sessions(deleted):
    metadata = CountingStore()
    async with StoreConversationArchive(metadata, namespace=("listing",)).open() as archive:
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
    async with StoreConversationArchive(metadata, namespace=("deleted",)).open() as archive:
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


@asynccontextmanager
async def stores(backend, tmp_path):
    if backend == "memory":
        yield InMemoryStore()
    else:
        async with AsyncSqliteStore.from_conn_string(str(tmp_path / "metadata.sqlite")) as metadata:
            try:
                await metadata.setup()
                yield metadata
            finally:
                metadata._task.cancel()
                await asyncio.gather(metadata._task, return_exceptions=True)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_shared_archive_contract(backend, tmp_path):
    async with stores(backend, tmp_path) as metadata:
        await assert_store_archive_contract(metadata, tmp_path)


async def test_revisions_chunking_exclusions_and_chronological_paging():
    metadata = InMemoryStore()
    async with StoreConversationArchive(metadata, namespace=("test",)).open() as archive:
        messages = [
            SystemMessage("do not archive"),
            HumanMessage("A" * 4500, id="human"),
            AIMessage(
                "summary",
                id="assistant",
                tool_calls=[{"name": "example", "args": {"x": 1}, "id": "call"}],
            ),
            ToolMessage("tool result", name="example", tool_call_id="call", id="tool"),
            ToolMessage(
                "do not recurse", name="search_conversations", tool_call_id="search", id="search"
            ),
        ]
        await archive.append(SCOPE, "session", "2026-09-05T00:00:00Z", messages)
        await archive.append(SCOPE, "session", "2026-09-05T00:00:00Z", messages)
        await archive.append(
            SCOPE, "session", "2026-09-05T00:01:00Z", [HumanMessage("revision", id="human")]
        )
        entries = await archive.entries(SCOPE, session_id="session", limit=20)
        assert [entry["part"] for entry in entries[:2]] == [0, 1]
        assert [len(entry["text"]) for entry in entries[:2]] == [4000, 500]
        assert len(entries) == 5
        assert "Tool calls:" in entries[2]["text"]
        assert entries[-1]["text"] == "revision"
        summary = (await archive.conversations(SCOPE))[0]
        assert summary["message_count"] == 3
        pages, cursor = [], 0
        while page := await archive.entries(SCOPE, session_id="session", after=cursor, limit=1):
            pages.extend(page)
            cursor = page[-1]["cursor"]
        assert pages == entries
        assert await archive.entries(OTHER, session_id="session") == []
        await archive.delete_session("session")
        assert await archive.entries(SCOPE) == []
        assert await archive.conversations(SCOPE) == []
        records = await metadata.asearch(archive.records.namespace, limit=100)
        assert all(item.value.get("kind") not in {"session", "chunk"} for item in records)
        assert not any(
            item.key.startswith(("scope:", "message:", "dedup:", "session:")) for item in records
        )


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


@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6, 7])
async def test_partial_chunk_write_recovers_idempotently_on_reopen(failure):
    metadata = InterruptedStore()
    message = [HumanMessage("durable content", id="message")]
    async with StoreConversationArchive(metadata, namespace=("recovery",)).open() as archive:
        await archive.append(SCOPE, "session", "time", [])
        metadata.remaining = failure
        with pytest.raises(OSError, match="interrupted metadata"):
            await archive.append(SCOPE, "session", "time", message)
    async with StoreConversationArchive(metadata, namespace=("recovery",)).open() as archive:
        await archive.append(SCOPE, "session", "time", message)
        assert [entry["text"] for entry in await archive.entries(SCOPE)] == ["durable content"]
        assert (await archive.conversations(SCOPE))[0]["message_count"] == 1


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
    async with StoreConversationArchive(metadata, namespace=("cancel",)).open() as archive:
        await archive.append(SCOPE, "session", "time", [])
        metadata.blocking = True
        task = asyncio.create_task(
            archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
        )
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


@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6, 7, 8])
async def test_partial_metadata_deletion_is_restartable(failure):
    metadata = InterruptedStore()
    async with StoreConversationArchive(metadata, namespace=("delete-recovery",)).open() as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("erase this")])
        metadata.remaining = failure
        with pytest.raises(OSError, match="interrupted metadata"):
            await archive.delete_session("session")
    async with StoreConversationArchive(metadata, namespace=("delete-recovery",)).open() as archive:
        await archive.delete_session("session")
        assert await archive.sessions(SCOPE) == []
        assert await archive.entries(SCOPE) == []


async def test_deleted_session_can_be_reused_without_stale_history():
    metadata = InMemoryStore()
    async with StoreConversationArchive(metadata, namespace=("reuse",)).open() as archive:
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
