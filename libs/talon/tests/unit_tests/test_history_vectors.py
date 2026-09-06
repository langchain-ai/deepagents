from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

import aiosqlite
import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.base import PutOp, SearchOp
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.archive import ArchiveScope, conversation_tools
from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.history_adapters import BoundedEmbeddings
from deepagents_talon.history_backends import open_history
from deepagents_talon.history_embeddings import QUERY_PROMPT, HistoryEmbeddings
from deepagents_talon.history_profiles import EmbeddingProfile
from deepagents_talon.sqlite_history import _HistorySqliteStore, sqlite_store
from tests.archive_helpers import open_vector_archive
from tests.store_archive_contract import StaticEmbeddings as Embedding

SCOPE = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="one")
OTHER = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="two")


@asynccontextmanager
async def vector_store(backend, path, embeddings=None):
    index = {"dims": 2, "embed": embeddings or Embedding(), "fields": ["text"]}
    if backend == "memory":
        yield InMemoryStore(index=index)
    else:
        async with AsyncSqliteStore.from_conn_string(str(path), index=index) as store:
            await store.conn.execute("PRAGMA foreign_keys=ON")
            await store.setup()
            try:
                yield store
            finally:
                store._task.cancel()
                await asyncio.gather(store._task, return_exceptions=True)


async def settled(archive):
    async with asyncio.timeout(3):
        while True:
            if not any(
                [
                    await archive.vectors.archive.pending(SCOPE),
                    await archive.vectors.archive.pending(OTHER),
                ]
            ):
                break
            await asyncio.sleep(0.01)
    # Wait until the worker releases the indexing lock too.
    async with archive.vectors.lock:
        pass


async def append(archive, text, session="session", scope=SCOPE):
    await archive.append(scope, session, "2026-09-05T00:00:00Z", [HumanMessage(text)])


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_hybrid_backfill_scope_pagination_and_reset(tmp_path, backend):
    path = str(tmp_path / "archive.sqlite")
    async with open_vector_archive(path) as archive:
        await append(archive, "my automobile needs repairs")
        await append(archive, "car insurance", session="second")
        await append(archive, "car secret", session="private", scope=OTHER)
    async with (
        vector_store(backend, tmp_path / "vectors.sqlite") as store,
        open_vector_archive(path, store=store) as archive,
    ):
        await settled(archive)
        hits = (await archive.search_page(SCOPE, query="car"))["results"]
        assert [hit["text"] for hit in hits] == ["car insurance", "my automobile needs repairs"]
        first = await archive.search_page(SCOPE, query="car", limit=1)
        second = await archive.search_page(SCOPE, query="car", after=first["next_after"], limit=1)
        assert first["results"] + second["results"] == hits
        assert second["next_after"] is None
        await archive.delete_session("session")
        await archive.delete_session("second")
        assert (await archive.search_page(SCOPE, query="car"))["results"] == []
        assert await store.asearch(archive.vectors.namespace("whatsapp", "one")) == []
        assert len((await archive.search_page(OTHER, query="car"))["results"]) == 1


class BlockingEmbedding(Embedding):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def aembed_documents(self, texts):
        self.started.set()
        await self.release.wait()
        return self.embed_documents(texts)


async def test_writes_and_keyword_search_continue_while_embedding_and_reset_waits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("deepagents_talon.history_vectors._SEARCH_TIMEOUT_SECONDS", 0.05)
    embed = BlockingEmbedding()
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        try:
            await append(archive, "car")
            await asyncio.wait_for(embed.started.wait(), 2)
            await asyncio.wait_for(append(archive, "car fast", session="new"), 0.5)
            assert (await asyncio.wait_for(archive.search_page(SCOPE, query="car"), 0.5))["results"]
            reset = asyncio.create_task(archive.delete_session("session"))
            await asyncio.sleep(0)
            assert not reset.done()
        finally:
            embed.release.set()
        await reset
        await settled(archive)
        hits = (await archive.search_page(SCOPE, query="car"))["results"]
        assert [hit["session_id"] for hit in hits] == ["new"]


class FailingEmbedding(Embedding):
    def __init__(self) -> None:
        self.failed = asyncio.Event()

    async def aembed_documents(self, _texts):
        self.failed.set()
        msg = "embedding unavailable"
        raise RuntimeError(msg)


async def test_failed_indexing_is_retried_after_restart(tmp_path):
    path = str(tmp_path / "archive.sqlite")
    failing = FailingEmbedding()
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", failing) as store,
        open_vector_archive(path, store=store) as archive,
    ):
        await append(archive, "car")
        await asyncio.wait_for(failing.failed.wait(), 2)
        assert ((await archive.search_page(SCOPE, query="car"))["results"])[0]["text"] == "car"
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite") as store,
        open_vector_archive(path, store=store) as archive,
    ):
        await settled(archive)
        assert (
            ((await archive.search_page(SCOPE, query="automobile"))["results"])[0]["text"] == "car"
        )


async def test_existing_vectors_survive_restart_and_do_not_reembed(tmp_path):
    path = str(tmp_path / "archive.sqlite")
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite") as store,
        open_vector_archive(path, store=store) as archive,
    ):
        await append(archive, "car")
        await settled(archive)
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", FailingEmbedding()) as store,
        open_vector_archive(path, store=store) as archive,
    ):
        await settled(archive)
        assert len(await store.asearch(archive.vectors.namespace("whatsapp", "one"))) == 1


@pytest.mark.parametrize("restart", [False, True])
async def test_partial_vector_write_is_retried(tmp_path, restart, monkeypatch):
    monkeypatch.setattr("deepagents_talon.history_vectors._RETRY_SECONDS", 0.01)
    monkeypatch.setattr("deepagents_talon.history_vectors.secrets.randbelow", lambda _bound: 0)
    path = str(tmp_path / "archive.sqlite")
    async with vector_store("sqlite", tmp_path / "vectors.sqlite") as store:
        await store.conn.execute(
            "CREATE TRIGGER reject_vector BEFORE INSERT ON store_vectors "
            "BEGIN SELECT RAISE(ABORT, 'vector insert failed'); END"
        )
        await store.conn.commit()
        async with open_vector_archive(path, store=store) as archive:
            await append(archive, "car")
            async with asyncio.timeout(3):
                # Poll persisted state; the worker exposes no failure notification.
                while not await store.asearch(archive.vectors.namespace("whatsapp", "one")):  # noqa: ASYNC110
                    await asyncio.sleep(0.01)
            async with archive.vectors.lock:
                await store.conn.execute("DROP TRIGGER reject_vector")
                await store.conn.commit()
            if not restart:
                await settled(archive)
                page = await archive.search_page(SCOPE, query="automobile")
                assert [entry["text"] for entry in page["results"]] == ["car"]
        async with open_vector_archive(path, store=store) as archive:
            await settled(archive)
            page = await archive.search_page(SCOPE, query="automobile")
            assert [entry["text"] for entry in page["results"]] == ["car"]
            assert not page["indexing_pending"]


async def test_disabled_mode_can_delete_existing_vector_rows(tmp_path):
    config = TalonConfig.from_env({}, base_home=tmp_path)
    config.ensure_home()
    async with (
        vector_store("sqlite", config.history_vector_path) as store,
        open_vector_archive(str(config.checkpoint_path), store=store) as archive,
    ):
        await append(archive, "car")
        await settled(archive)
    async with (
        sqlite_store(config) as store,
        open_vector_archive(
            str(config.checkpoint_path), store=store, vector_search=False
        ) as archive,
    ):
        await archive.delete_session("session")
    async with (
        aiosqlite.connect(config.history_vector_path) as conn,
        conn.execute("SELECT count(*) FROM store_vectors") as cur,
    ):
        assert (await cur.fetchone())[0] == 0


async def test_store_results_cannot_cross_archive_or_chat_boundaries(tmp_path):
    store = InMemoryStore(index={"dims": 2, "embed": Embedding(), "fields": ["text"]})
    async with (
        open_vector_archive(str(tmp_path / "a.sqlite"), store=store) as first,
        open_vector_archive(str(tmp_path / "b.sqlite"), store=store) as second,
    ):
        await append(first, "car private", scope=OTHER)
        await append(second, "car second")
        await settled(first)
        await settled(second)
        assert (await first.search_page(SCOPE, query="car"))["results"] == []
        assert (
            ((await second.search_page(SCOPE, query="car"))["results"])[0]["text"] == "car second"
        )
        # A forged result in the right namespace still cannot authorize another chat's row.
        private = (await first.entries(OTHER))[0]
        await store.aput(
            first.vectors.namespace("whatsapp", "one"), str(private["cursor"]), {"text": "car"}
        )
        assert (await first.search_page(SCOPE, query="car"))["results"] == []


async def test_default_store_adds_qwen_instruction_only_to_queries(tmp_path):
    class RecordingEmbedding(Embedding):
        def __init__(self) -> None:
            self.texts = []

        async def aembed_documents(self, texts):
            self.texts.extend(texts)
            return self.embed_documents(texts)

        async def aembed_query(self, text):
            self.texts.append(text)
            return self.embed_query(text)

    embed = RecordingEmbedding()
    async with _HistorySqliteStore.from_conn_string(
        str(tmp_path / "vectors.sqlite"),
        index={
            "dims": 2,
            "embed": BoundedEmbeddings(embed, EmbeddingProfile(dims=2)),
            "fields": ["text"],
        },
    ) as store:
        try:
            await store.setup()
            await store.abatch([PutOp(("test",), "1", {"text": "car"}, index=["text"])])
            await store.abatch([SearchOp(("test",), query="automobile")])
            assert embed.texts == ["car", QUERY_PROMPT + "automobile"]
        finally:
            store._task.cancel()
            await asyncio.gather(store._task, return_exceptions=True)


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_vector_environment_opt_in(tmp_path, value):
    config = TalonConfig.from_env(
        {"DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": value}, base_home=tmp_path
    )
    assert config.history_vector_search
    assert not TalonConfig.from_env({}, base_home=tmp_path).history_vector_search


async def test_disabled_default_creates_no_vector_file(tmp_path):
    config = TalonConfig.from_env({}, base_home=tmp_path)
    config.ensure_home()
    async with open_history(config) as archive:
        assert archive.vectors is None
    assert not config.history_vector_path.exists()


def test_invalid_vector_environment(tmp_path):
    with pytest.raises(TalonConfigError, match="must be a boolean"):
        TalonConfig.from_env(
            {"DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": "maybe"}, base_home=tmp_path
        )


async def test_pagination_survives_indexing_and_a_fresh_search(tmp_path, monkeypatch):
    monkeypatch.setattr("deepagents_talon.history_vectors._SEARCH_TIMEOUT_SECONDS", 0.01)
    async with (
        vector_store("memory", tmp_path / "unused") as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        for text in ("car first", "car second", "car third"):
            await append(archive, text)
        await settled(archive)
        async with archive.vectors.lock:
            first = await archive.search_page(SCOPE, query="car", limit=1)
        await append(archive, "car new", session="new")
        await settled(archive)
        refreshed = await archive.search_page(SCOPE, query="car")
        rest = await archive.search_page(SCOPE, query="car", after=first["next_after"])
        assert [entry["text"] for entry in first["results"] + rest["results"]] == [
            "car third",
            "car second",
            "car first",
        ]
        assert rest["semantic_status"] == "timeout"
        assert len(refreshed["results"]) == 4
        assert refreshed["semantic_status"] == "completed"
        foreign = await archive.search_page(OTHER, query="car", after=first["next_after"])
        assert foreign["pagination_status"] == "expired"
        assert foreign["results"] == []


async def test_vector_delete_failure_retains_history_for_retry(tmp_path):
    class FailingDeleteStore(InMemoryStore):
        failing = True

        async def abatch(self, ops):
            operations = list(ops)
            if self.failing and any(
                isinstance(op, PutOp) and op.value is None for op in operations
            ):
                msg = "delete unavailable"
                raise RuntimeError(msg)
            return await super().abatch(operations)

    store = FailingDeleteStore(index={"dims": 2, "embed": Embedding(), "fields": ["text"]})
    async with open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive:
        await append(archive, "car")
        await settled(archive)
        with pytest.raises(RuntimeError, match="delete unavailable"):
            await archive.delete_session("session")
        assert await archive.sessions(SCOPE) == ["session"]
        assert await archive.entries(SCOPE, session_id="session") == []
        store.failing = False
        await archive.delete_session("session")
        assert await archive.sessions(SCOPE) == []
        assert await store.asearch(archive.vectors.namespace("whatsapp", "one")) == []


async def test_embedding_cancellation_does_not_accumulate_inference_threads(monkeypatch):
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def embed(_self, texts):
        calls.append(texts)
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=3)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(HistoryEmbeddings, "embed_documents", embed)
    embeddings = HistoryEmbeddings()
    try:
        first = asyncio.create_task(embeddings.aembed_documents(["first"]))
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(embeddings.aembed_documents(["second"]))
        await asyncio.sleep(0)
        assert calls == [["first"]]
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
    finally:
        release.set()
        await embeddings.aclose()
    assert await embeddings.aembed_documents(["third"]) == [[1.0, 0.0]]
    await embeddings.aclose()
    assert calls == [["first"], ["third"]]


async def test_search_waits_for_indexing_and_retrieves_semantic_history(tmp_path):
    embed = BlockingEmbedding()
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        try:
            await append(archive, "car from a past conversation")
            await asyncio.wait_for(embed.started.wait(), 2)
            query = asyncio.create_task(archive.search_page(SCOPE, query="automobile"))
            await asyncio.sleep(0.02)
            assert not query.done()
        finally:
            embed.release.set()
        page = await asyncio.wait_for(query, 2)
        assert page["semantic_status"] == "completed"
        assert page["results"][0]["text"] == "car from a past conversation"
        assert not page["has_more"]


async def test_timed_out_search_reports_fallback_and_can_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("deepagents_talon.history_vectors._SEARCH_TIMEOUT_SECONDS", 0.05)
    embed = BlockingEmbedding()
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        try:
            await append(archive, "car")
            await asyncio.wait_for(embed.started.wait(), 2)
            page = await asyncio.wait_for(archive.search_page(SCOPE, query="car"), 0.5)
            assert page["semantic_status"] == "timeout"
            assert page["indexing_pending"]
            assert page["results"][0]["text"] == "car"
            # A semantic-only term must report incomplete retrieval, not absence.
            empty = await archive.search_page(SCOPE, query="automobile")
            assert empty["results"] == []
            assert empty["semantic_status"] == "timeout"
        finally:
            embed.release.set()
        await settled(archive)
        retry = await archive.search_page(SCOPE, query="automobile")
        assert retry["semantic_status"] == "completed"
        assert not retry["indexing_pending"]
        assert retry["results"][0]["text"] == "car"


async def test_search_tool_pages_preserve_status_and_report_expired_cursors(tmp_path):
    async with (
        vector_store("memory", tmp_path / "unused") as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        await append(archive, "car first")
        await append(archive, "car second", session="second")
        await settled(archive)
        search = next(
            t
            for t in conversation_tools(archive, lambda: SCOPE)
            if t.name == "search_conversations"
        )
        first = await search.ainvoke({"query": "automobile", "limit": 1})
        assert first["semantic_status"] == "completed"
        assert first["has_more"]
        second = await search.ainvoke(
            {"query": "automobile", "limit": 1, "after": first["next_after"]}
        )
        assert second["semantic_status"] == "completed"
        assert second["results"] != first["results"]
        assert not second["has_more"]
        assert second["next_after"] is None
        # A cursor from another query is not evidence of an empty result set.
        expired = await search.ainvoke({"query": "vehicle", "after": first["next_after"]})
        assert expired["pagination_status"] == "expired"
        assert expired["semantic_status"] == "not_requested"
        assert expired["results"] == []


async def test_search_backend_failure_reports_error(tmp_path):
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", FailingEmbedding()) as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        # No indexing in flight: this fails inside query execution itself.
        page = await archive.search_page(SCOPE, query="automobile")
        assert page["semantic_status"] == "error"
        assert page["results"] == []


async def test_search_reports_store_without_semantic_support(tmp_path):
    store = InMemoryStore()
    async with open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive:
        await append(archive, "car")
        await settled(archive)
        page = await archive.search_page(SCOPE, query="car")
        assert page["semantic_status"] == "unavailable"
        assert page["results"][0]["text"] == "car"


async def test_cached_fallback_does_not_claim_semantic_success_after_recovery(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("deepagents_talon.history_vectors._SEARCH_TIMEOUT_SECONDS", 0.05)
    embed = BlockingEmbedding()
    async with (
        vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store,
        open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive,
    ):
        try:
            await append(archive, "car first")
            await asyncio.wait_for(embed.started.wait(), 2)
            await append(archive, "car second", session="second")
            first = await archive.search_page(SCOPE, query="car", limit=1)
            assert first["has_more"]
            assert first["semantic_status"] == "timeout"
        finally:
            embed.release.set()
        await settled(archive)
        second = await archive.search_page(SCOPE, query="car", after=first["next_after"])
        assert second["semantic_status"] == "timeout"
        assert second["indexing_pending"]
        assert second["results"][0]["cursor"] != first["results"][0]["cursor"]
        refreshed = await archive.search_page(SCOPE, query="car")
        assert refreshed["semantic_status"] == "completed"
        assert not refreshed["indexing_pending"]


@pytest.mark.parametrize("visibility", ["immediate", "unknown"])
async def test_search_reports_backend_visibility_separately_from_queue(tmp_path, visibility):
    store = InMemoryStore(index={"dims": 2, "embed": Embedding(), "fields": ["text"]})
    async with open_vector_archive(
        str(tmp_path / "archive.sqlite"), store=store, search_visibility=visibility
    ) as archive:
        await append(archive, "car")
        await settled(archive)
        page = await archive.search_page(SCOPE, query="automobile")
        assert not page["indexing_pending"]
        assert page["indexing_status"] == ("ready" if visibility == "immediate" else "unknown")


async def test_eventual_store_does_not_claim_complete_index_when_results_are_empty(tmp_path):
    class DelayedStore(InMemoryStore):
        visible = False

        async def abatch(self, ops):
            operations = list(ops)
            if not self.visible and all(isinstance(op, SearchOp) for op in operations):
                return [[] for _ in operations]
            return await super().abatch(operations)

    store = DelayedStore(index={"dims": 2, "embed": Embedding(), "fields": ["text"]})
    async with open_vector_archive(str(tmp_path / "archive.sqlite"), store=store) as archive:
        await append(archive, "car")
        await settled(archive)
        first = await archive.search_page(SCOPE, query="automobile")
        assert first["results"] == []
        assert not first["indexing_pending"]
        assert first["indexing_status"] == "unknown"
        store.visible = True
        refreshed = await archive.search_page(SCOPE, query="automobile")
        assert refreshed["results"][0]["text"] == "car"
        assert refreshed["indexing_status"] == "unknown"
