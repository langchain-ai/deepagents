from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents_talon import history_prepared_store
from deepagents_talon.history_adapters import BoundedEmbeddings
from deepagents_talon.history_prepared_store import PreparedVectorStore
from deepagents_talon.store_archive import StoreConversationArchive
from tests.unit_tests.test_history_profiles import RecordingEmbeddings, configuration
from tests.unit_tests.test_history_vectors import SCOPE, settled

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from langgraph.store.base import Op, Result


def prepared(
    tmp_path: Path, raw: RecordingEmbeddings, backend: type[InMemoryStore] = InMemoryStore
) -> PreparedVectorStore:
    profile = configuration(tmp_path, MAX_INPUT_TOKENS="8192").history_embedding_profile
    embed = BoundedEmbeddings(raw, profile)
    return PreparedVectorStore(
        backend(index={"dims": 2, "embed": embed, "fields": ["text"]}), embed
    )


async def test_message_revision_reuses_unchanged_embeddings_and_preserves_transcript(
    tmp_path: Path,
) -> None:
    raw = RecordingEmbeddings()
    store = prepared(tmp_path, raw)
    first = "a" * 4000 + "b" * 4000 + "original"
    edited = "a" * 4000 + "b" * 4000 + "edited"
    async with StoreConversationArchive(
        InMemoryStore(), namespace=("revision-cache",), vector_store=store
    ).open() as archive:
        await archive.append(SCOPE, "session", "first", [HumanMessage(first, id="message")])
        await settled(archive)
        await archive.append(SCOPE, "session", "second", [HumanMessage(edited, id="message")])
        await settled(archive)
        assert raw.documents == ["a" * 4000, "b" * 4000, "original", "edited"]
        entries = await archive.entries(SCOPE, session_id="session", limit=20)
        assert "".join(entry["text"] for entry in entries) == first + edited
        assert len(await archive.entries(SCOPE, query="original")) == 1
        assert len(await archive.entries(SCOPE, query="edited")) == 1
        stored = await store.asearch(
            archive.vectors.namespace(SCOPE["talon_history_channel"], SCOPE["talon_history_chat"])
        )
        assert len(stored) == 6


@pytest.mark.parametrize("bound", ["_CACHE_ENTRIES", "_CACHE_BYTES"])
async def test_embedding_cache_evicts_least_recently_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    monkeypatch.setattr(history_prepared_store, bound, 2 if bound == "_CACHE_ENTRIES" else 32)
    raw = RecordingEmbeddings()
    store = prepared(tmp_path, raw)
    for index, text in enumerate(["one", "two", "one", "three", "one", "two"]):
        await store.aput(("test",), str(index), {"text": text})
    assert raw.documents == ["one", "two", "three", "two"]


async def test_embedding_cache_is_instance_local_and_invalidated_on_deletion(
    tmp_path: Path,
) -> None:
    raw = RecordingEmbeddings()
    store = prepared(tmp_path, raw)
    await store.aput(("test",), "one", {"text": "document"})
    await store.adelete(("test",), "one")
    await store.aput(("test",), "two", {"text": "document"})
    await prepared(tmp_path, raw).aput(("test",), "three", {"text": "document"})
    assert raw.documents == ["document"] * 3


async def test_database_retry_reuses_successful_embeddings(tmp_path: Path) -> None:
    class FailingStore(InMemoryStore):
        failed = False

        async def abatch(self, ops: Iterable[Op]) -> list[Result]:
            if not self.failed:
                self.failed = True
                msg = "database unavailable"
                raise OSError(msg)
            return await super().abatch(ops)

    raw = RecordingEmbeddings()
    store = prepared(tmp_path, raw, FailingStore)
    with pytest.raises(OSError, match="database unavailable"):
        await store.aput(("test",), "one", {"text": "document"})
    await store.aput(("test",), "one", {"text": "document"})
    assert raw.documents == ["document"]
    assert (await store.aget(("test",), "one")).value == {"text": "document"}
