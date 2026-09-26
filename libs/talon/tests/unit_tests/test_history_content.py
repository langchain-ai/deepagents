from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore

from deepagents_talon.history_backends import _sqlite_store
from deepagents_talon.history_vectors import HistoryVectorIndex
from deepagents_talon.store_archive import StoreConversationArchive
from deepagents_talon.store_archive_index import StoreVectorArchive
from tests.store_archive_contract import StaticEmbeddings
from tests.unit_tests.test_history_vectors import OTHER, SCOPE, settled, vector_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable
    from pathlib import Path

    from langgraph.store.base import BaseStore, Op, Result


class CountingEmbeddings(StaticEmbeddings):
    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return super().embed_documents(texts)


class ServerEmbeddingStore(InMemoryStore):
    """Count server-side indexing requests without a client Embeddings adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.documents: list[str] = []

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        operations = list(ops)
        self.documents.extend(
            op.value["text"]
            for op in operations
            if isinstance(op, PutOp) and op.value is not None and op.index is not False
        )
        return await super().abatch(operations)


class InterruptedCatalogStore(InMemoryStore):
    boundary: str | None = None

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp) and self.boundary:
                boundaries = {
                    "mapping": op.key.startswith("vector-content:"),
                    "deletion-reference": bool(op.value and "vector_content" in op.value),
                    "journal-clear": op.key == "journal" and op.value is None,
                }
                if boundaries[self.boundary]:
                    self.boundary = None
                    msg = "interrupted catalog"
                    raise OSError(msg)
            results.extend(await super().abatch([op]))
        return results


@asynccontextmanager
async def reopen(path: Path, store: BaseStore) -> AsyncIterator[StoreConversationArchive]:
    async with (
        _sqlite_store(path.as_uri()) as metadata,
        StoreConversationArchive(
            metadata, namespace=("content",), vector_store=store
        ).open() as archive,
    ):
        yield archive


@pytest.mark.parametrize("backend", ["memory", "sqlite", "server"])
async def test_revisions_reuse_vectors_after_restart_and_delete_cleanly(
    tmp_path: Path, backend: str
) -> None:
    embed = CountingEmbeddings()
    server = ServerEmbeddingStore()
    metadata_path = tmp_path / "metadata.sqlite"
    first = "a" * 4000 + "b" * 4000 + "original"
    edited = "a" * 4000 + "b" * 4000 + "edited"
    kind = "memory" if backend == "server" else backend
    async with vector_store(kind, tmp_path / "vectors.sqlite", embed) as backend_store:
        store = server if backend == "server" else backend_store
        async with reopen(metadata_path, store) as archive:
            await archive.append(SCOPE, "session", "first", [HumanMessage(first, id="message")])
            await settled(archive)
        async with reopen(metadata_path, store) as archive:
            await archive.append(SCOPE, "session", "edit", [HumanMessage(edited, id="message")])
            await settled(archive)
            documents = server.documents if backend == "server" else embed.documents
            assert documents == ["a" * 4000, "b" * 4000, "original", "edited"]
            entries = await archive.entries(SCOPE, session_id="session", limit=20)
            assert "".join(entry["text"] for entry in entries) == first + edited
            assert len(await archive.entries(SCOPE, query="original")) == 1
            assert len(await archive.entries(SCOPE, query="edited")) == 1
            namespace = archive.vectors.namespace("whatsapp", "one")
            assert len(await store.asearch(namespace)) == 4
            await archive.append(OTHER, "other", "other", [HumanMessage("original")])
            await settled(archive)
            await archive.delete_session("session")
            assert not await store.asearch(namespace)
            assert len(await archive.entries(OTHER)) == 1
            await archive.append(SCOPE, "session", "recreated", [HumanMessage("original")])
            await settled(archive)
            assert documents.count("original") == 3


async def test_sqlite_vectors_survive_connection_reopen(tmp_path: Path) -> None:
    embed = CountingEmbeddings()
    for index in range(2):
        async with (
            vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store,
            reopen(tmp_path / "metadata.sqlite", store) as archive,
        ):
            await archive.append(
                SCOPE, "session", str(index), [HumanMessage("same", id=str(index))]
            )
            await settled(archive)
    assert embed.documents == ["same"]


async def test_legacy_index_backfills_without_reembedding(tmp_path: Path) -> None:
    embed = CountingEmbeddings()
    async with vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store:
        async with _sqlite_store((tmp_path / "metadata.sqlite").as_uri()) as metadata:
            archive = StoreConversationArchive(metadata, namespace=("content",))
            await archive.append(SCOPE, "session", "old", [HumanMessage("same", id="old")])
            root = await archive.records.root()
            index = HistoryVectorIndex(StoreVectorArchive(archive), store)
            index.identity = str(root["identity"])
            entry = (await archive.entries(SCOPE))[0]
            await store.aput(
                index.namespace("whatsapp", "one"),
                str(entry["cursor"]),
                {"text": "same", "session_id": "session"},
                index=["text"],
            )
            await archive.records.commit(
                [("root", {**root, "vectors": True, "indexed": root["last"]})]
            )
        async with reopen(tmp_path / "metadata.sqlite", store) as archive:
            await archive.append(SCOPE, "session", "new", [HumanMessage("same", id="new")])
            await settled(archive)
            assert embed.documents == ["same"]
            assert len(await archive.entries(SCOPE)) == 2
            page = await archive.search_page(SCOPE, query="same")
            assert page["results"]


async def test_duplicate_only_batches_persist_progress_and_reindex_unique_content(
    tmp_path: Path,
) -> None:
    embed = CountingEmbeddings()
    async with vector_store("memory", tmp_path / "unused", embed) as store:
        async with reopen(tmp_path / "metadata.sqlite", store) as archive:
            await archive.append(
                SCOPE, "session", "time", [HumanMessage("same", id=str(i)) for i in range(12)]
            )
            await settled(archive)
            root = await archive.records.root()
            assert root["indexed"] == root["last"]
        async with reopen(tmp_path / "metadata.sqlite", store) as archive:
            await settled(archive)
            assert embed.documents == ["same"]
            root = await archive.records.root()
            await archive.records.commit([("root", {**root, "indexed": 0})])
    fresh = CountingEmbeddings()
    async with (
        vector_store("memory", tmp_path / "unused", fresh) as store,
        reopen(tmp_path / "metadata.sqlite", store) as archive,
    ):
        await settled(archive)
        assert fresh.documents == ["same"]


@pytest.mark.parametrize("boundary", ["mapping", "deletion-reference", "journal-clear"])
async def test_partial_content_catalog_write_recovers_before_indexing(
    tmp_path: Path, boundary: str
) -> None:
    metadata = InterruptedCatalogStore()
    embed = CountingEmbeddings()
    archive = StoreConversationArchive(metadata, namespace=("interrupted-content",))
    await archive.append(
        SCOPE, "session", "time", [HumanMessage("same", id="one"), HumanMessage("same", id="two")]
    )
    index = StoreVectorArchive(archive)
    await index.prepare()
    metadata.boundary = boundary
    with pytest.raises(OSError, match="interrupted catalog"):
        await index.rows("", indexing=True, limit=4)
    async with (
        vector_store("memory", tmp_path / "unused", embed) as store,
        StoreConversationArchive(
            metadata, namespace=("interrupted-content",), vector_store=store
        ).open() as archive,
    ):
        await settled(archive)
        assert embed.documents == ["same"]
        assert len(await archive.entries(SCOPE)) == 2
        await archive.delete_session("session")
        await archive.append(SCOPE, "session", "later", [HumanMessage("same", id="new")])
        await settled(archive)
        assert embed.documents == ["same", "same"]
