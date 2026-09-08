from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.store_archive import StoreConversationArchive
from tests.store_archive_contract import (
    OTHER,
    SCOPE,
    StaticEmbeddings,
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


@asynccontextmanager
async def stores(backend, tmp_path):
    index = {"dims": 2, "embed": StaticEmbeddings(), "fields": ["text"]}
    if backend == "memory":
        yield InMemoryStore(), InMemoryStore(index=index)
    else:
        async with (
            AsyncSqliteStore.from_conn_string(str(tmp_path / "metadata.sqlite")) as metadata,
            AsyncSqliteStore.from_conn_string(
                str(tmp_path / "vectors.sqlite"), index=index
            ) as vectors,
        ):
            try:
                for store in (metadata, vectors):
                    await store.conn.execute("PRAGMA foreign_keys=ON")
                    await store.setup()
                yield metadata, vectors
            finally:
                for store in (metadata, vectors):
                    store._task.cancel()
                    await asyncio.gather(store._task, return_exceptions=True)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_shared_archive_contract(backend, tmp_path):
    async with stores(backend, tmp_path) as (metadata, vectors):
        await assert_store_archive_contract(metadata, vectors, tmp_path)


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


async def test_separate_assistant_namespaces_and_missing_vector_store_reset():
    metadata = InMemoryStore()
    vectors = InMemoryStore(index={"dims": 2, "embed": StaticEmbeddings(), "fields": ["text"]})
    async with (
        StoreConversationArchive(metadata, namespace=("one",), vector_store=vectors).open() as one,
        StoreConversationArchive(metadata, namespace=("two",)).open() as two,
    ):
        await one.append(SCOPE, "session", "time", [HumanMessage("car secret")])
        assert await two.entries(SCOPE) == []
    async with StoreConversationArchive(metadata, namespace=("one",)).open() as archive:
        with pytest.raises(RuntimeError, match="vector Store"):
            await archive.delete_session("session")
        assert await archive.sessions(SCOPE) == ["session"]


async def test_vector_deletion_failure_recovers_without_resurrecting_history():
    class FailingDeleteStore(InMemoryStore):
        failing = True

        async def abatch(self, ops):
            operations = list(ops)
            if self.failing and any(
                isinstance(op, PutOp) and op.value is None for op in operations
            ):
                msg = "vector deletion failed"
                raise OSError(msg)
            return await super().abatch(operations)

    metadata = InMemoryStore()
    vectors = FailingDeleteStore(index={"dims": 2, "embed": StaticEmbeddings(), "fields": ["text"]})
    async with StoreConversationArchive(
        metadata, namespace=("delete",), vector_store=vectors
    ).open() as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("car secret")])
        async with asyncio.timeout(2):
            while True:
                if not (await archive.search_page(SCOPE, query="automobile"))["indexing_pending"]:
                    break
                await asyncio.sleep(0)
        with pytest.raises(OSError, match="vector deletion failed"):
            await archive.delete_session("session")
        assert await archive.sessions(SCOPE) == ["session"]
        assert await archive.entries(SCOPE) == []
        with pytest.raises(ValueError, match="being deleted"):
            await archive.append(SCOPE, "session", "later", [HumanMessage("must not resurrect")])
    vectors.failing = False
    async with StoreConversationArchive(
        metadata, namespace=("delete",), vector_store=vectors
    ).open() as archive:
        async with asyncio.timeout(2):
            while True:
                if not await archive.sessions(SCOPE):
                    break
                await asyncio.sleep(0)
        assert await archive.entries(SCOPE) == []
        assert not await vectors.asearch(archive.vectors.namespace("test", "one"))
        values = [
            item.value for item in await metadata.asearch(archive.records.namespace, limit=100)
        ]
        assert "car secret" not in str(values)


@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6, 7, 8])
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


async def test_append_does_not_wait_for_vector_inference():
    started, release = asyncio.Event(), asyncio.Event()

    class BlockedStore(InMemoryStore):
        async def abatch(self, ops):
            operations = list(ops)
            if any(isinstance(op, PutOp) and op.value is not None for op in operations):
                started.set()
                await release.wait()
            return await super().abatch(operations)

    metadata = InMemoryStore()
    vectors = BlockedStore(index={"dims": 2, "embed": StaticEmbeddings(), "fields": ["text"]})
    async with StoreConversationArchive(
        metadata, namespace=("nonblocking",), vector_store=vectors
    ).open() as archive:
        try:
            await archive.append(SCOPE, "session", "time", [HumanMessage("car first")])
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.wait_for(
                archive.append(SCOPE, "session", "later", [HumanMessage("car second")]), 0.2
            )
            assert len(await archive.entries(SCOPE, session_id="session")) == 2
        finally:
            release.set()


async def test_backfill_skips_empty_and_deleted_sequence_ranges():
    metadata = InMemoryStore()
    vectors = InMemoryStore(index={"dims": 2, "embed": StaticEmbeddings(), "fields": ["text"]})
    async with StoreConversationArchive(metadata, namespace=("gaps",)).open() as archive:
        for index in range(12):
            await archive.append(SCOPE, f"empty-{index}", "time", [])
        await archive.append(SCOPE, "removed", "time", [HumanMessage("removed")])
        await archive.delete_session("removed")
        await archive.append(SCOPE, "retained", "time", [HumanMessage("car survives")])
    async with StoreConversationArchive(
        metadata, namespace=("gaps",), vector_store=vectors
    ).open() as archive:
        async with asyncio.timeout(2):
            while True:
                page = await archive.search_page(SCOPE, query="automobile")
                if not page["indexing_pending"]:
                    break
                await asyncio.sleep(0)
        assert [entry["text"] for entry in page["results"]] == ["car survives"]
        assert [item["session_id"] for item in await archive.conversations(SCOPE)] == ["retained"]


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


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("query", ["car", ""])
async def test_keyword_tokens_bind_query_scope_and_archive(tmp_path, backend, query):
    async with stores(backend, tmp_path) as (metadata, _vectors):
        async with StoreConversationArchive(metadata, namespace=("tokens",)).open() as archive:
            await archive.append(
                SCOPE, "session", "time", [HumanMessage(f"car {i}", id=str(i)) for i in range(3)]
            )
            first = await archive.search_page(SCOPE, query=query, limit=1)
            token = first["next_after"]
            assert token
            assert token != str(first["results"][0]["cursor"])
            for scope, text, after in (
                (SCOPE, "different", token),
                (OTHER, query, token),
                ({**SCOPE, "talon_history_channel": "other"}, query, token),
                (SCOPE, query, "malformed:cursor"),
                (SCOPE, query, str(first["results"][0]["cursor"])),
            ):
                page = await archive.search_page(scope, query=text, after=after)
                assert page["pagination_status"] == "expired"
                assert not page["results"]
                assert page["next_after"] is None
            await archive.append(SCOPE, "new", "later", [HumanMessage("car new")])
            rest = await archive.search_page(SCOPE, query=query, after=token)
            assert [row["text"] for row in first["results"] + rest["results"]] == [
                "car 2",
                "car 1",
                "car 0",
            ]
            assert not rest["has_more"]
        async with StoreConversationArchive(metadata, namespace=("tokens",)).open() as reopened:
            page = await reopened.search_page(SCOPE, query=query, after=token)
            assert page["pagination_status"] == "expired"


async def test_keyword_tokens_expire_on_eviction_and_reset():
    async with StoreConversationArchive(InMemoryStore(), namespace=("expiry",)).open() as archive:
        await archive.append(
            SCOPE, "session", "time", [HumanMessage("car", id=str(i)) for i in range(3)]
        )
        first = await archive.search_page(SCOPE, query="car", limit=1)
        for _ in range(32):
            latest = await archive.search_page(SCOPE, query="car", limit=1)
        expired = await archive.search_page(SCOPE, query="car", after=first["next_after"])
        assert expired["pagination_status"] == "expired"
        assert not expired["results"]
        await archive.delete_session("session")
        reset = await archive.search_page(SCOPE, query="car", after=latest["next_after"])
        assert reset["pagination_status"] == "expired"
        assert not reset["results"]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_tokens_do_not_cross_keyword_and_vector_modes(tmp_path, backend):
    async with (
        stores(backend, tmp_path) as (metadata, vectors),
        StoreConversationArchive(
            metadata, namespace=("modes",), vector_store=vectors
        ).open() as archive,
    ):
        await archive.append(
            SCOPE, "session", "time", [HumanMessage("car", id=str(i)) for i in range(3)]
        )
        vector = await archive.search_page(SCOPE, query="car", limit=1)
        assert vector["next_after"]
        archive.vector_search = False
        rejected = await archive.search_page(SCOPE, query="car", after=vector["next_after"])
        assert rejected["pagination_status"] == "expired"
        assert not rejected["results"]
        keyword = await archive.search_page(SCOPE, query="car", limit=1)
        assert keyword["next_after"]
        archive.vector_search = True
        rejected = await archive.search_page(SCOPE, query="car", after=keyword["next_after"])
        assert rejected["pagination_status"] == "expired"
        assert not rejected["results"]


async def test_pending_vector_deletions_recover_in_any_session_order():
    class FailingDeletes(InMemoryStore):
        failing = True

        async def abatch(self, ops):
            operations = list(ops)
            if self.failing and any(
                isinstance(op, PutOp) and op.value is None for op in operations
            ):
                msg = "deletion unavailable"
                raise OSError(msg)
            return await super().abatch(operations)

    metadata = InMemoryStore()
    vectors = FailingDeletes()
    async with StoreConversationArchive(metadata, namespace=("recover",)).open() as archive:
        await archive.append(SCOPE, "older", "time", [HumanMessage("car older")])
        await archive.append(SCOPE, "newer", "time", [HumanMessage("car newer")])
    async with StoreConversationArchive(
        metadata, namespace=("recover",), vector_store=vectors, vector_search=False
    ).open() as archive:
        for session in ("newer", "older"):
            with pytest.raises(OSError, match="deletion unavailable"):
                await archive.delete_session(session)
        assert not await archive.entries(SCOPE)
        assert set(await archive.sessions(SCOPE)) == {"older", "newer"}
    vectors.failing = False
    async with StoreConversationArchive(
        metadata, namespace=("recover",), vector_store=vectors, vector_search=False
    ).open() as archive:
        async with asyncio.timeout(2):
            while True:
                if not await archive.sessions(SCOPE):
                    break
                await asyncio.sleep(0)
        assert not await archive.entries(SCOPE)
        assert not await vectors.asearch(archive.vectors.namespace("test", "one"))
