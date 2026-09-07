from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from deepagents_talon.archive import ArchiveScope
from deepagents_talon.config import TalonConfig
from deepagents_talon.history_backends import open_history
from deepagents_talon.store_archive import StoreConversationArchive

SCOPE = ArchiveScope(talon_history_channel="test", talon_history_chat="one")
OTHER = ArchiveScope(talon_history_channel="test", talon_history_chat="two")


class StaticEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if any(word in text for word in ("car", "automobile")) else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aclose(self):
        pass


def _archive_factory(metadata, vectors=None):
    @asynccontextmanager
    async def archives(config):
        async with StoreConversationArchive(
            metadata,
            namespace=("talon", config.assistant_id),
            vector_store=vectors,
            vector_search=config.history_vector_search,
        ).open() as archive:
            yield archive

    return archives


async def assert_store_archive_contract(metadata, vectors, tmp_path, *, history_uri=None):
    """Exercise identical archive behavior on every real metadata/vector backend."""
    env = {
        "DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": "1",
        "DEEPAGENTS_TALON_HISTORY_EMBED_DIMS": "2",
    }
    if history_uri:
        env["DEEPAGENTS_TALON_HISTORY_URI"] = history_uri
    config = TalonConfig.from_env(env, base_home=tmp_path)
    plain = _archive_factory(metadata)
    hybrid = open_history if history_uri else _archive_factory(metadata, vectors)
    async with plain(config) as archive:
        first = [HumanMessage("car repairs", id="message")]
        await archive.append(SCOPE, "first", "2026-09-05T00:00:00Z", first)
        await archive.append(SCOPE, "first", "2026-09-05T00:00:00Z", first)
        assert len(await archive.entries(SCOPE, session_id="first")) == 1
        await archive.append(
            SCOPE, "second", "2026-09-05T00:01:00Z", [HumanMessage("car insurance")]
        )
        await archive.append(
            OTHER, "private", "2026-09-05T00:02:00Z", [HumanMessage("private car")]
        )
        with pytest.raises(ValueError, match="another scope"):
            await archive.append(OTHER, "first", "2026-09-05T00:03:00Z", [])
        assert await archive.entries(OTHER, session_id="first") == []
        assert {row["session_id"] for row in await archive.conversations(SCOPE)} == {
            "first",
            "second",
        }

    async with hybrid(config) as archive:
        async with asyncio.timeout(60):
            while True:
                page = await archive.search_page(SCOPE, query="automobile", limit=1)
                other = await archive.search_page(OTHER, query="automobile")
                if page["has_more"] and other["results"] and not page["indexing_pending"]:
                    break
                await asyncio.sleep(0.02)
        assert page["semantic_status"] == "completed"
        assert page["indexing_status"] == (
            "ready" if history_uri and history_uri.startswith("postgres") else "unknown"
        )
        second = await archive.search_page(
            SCOPE, query="automobile", limit=1, after=page["next_after"]
        )
        assert not second["has_more"]
        assert {page["results"][0]["session_id"], second["results"][0]["session_id"]} == {
            "first",
            "second",
        }
        assert [row["session_id"] for row in other["results"]] == ["private"]
        await archive.delete_session("first")
        assert "first" not in await archive.sessions(SCOPE)
        assert await archive.entries(SCOPE, session_id="first") == []
        refreshed = await archive.search_page(SCOPE, query="automobile")
        assert all(row["session_id"] != "first" for row in refreshed["results"])
        vector_namespace = archive.vectors.namespace("test", "one")
        assert all(
            item.value["session_id"] != "first"
            for item in await archive.vectors.store.asearch(vector_namespace)
        )
        records = await metadata.asearch(archive.records.namespace, limit=100)
        assert "car repairs" not in str([item.value for item in records])

    disabled = TalonConfig.from_env(
        {"DEEPAGENTS_TALON_HISTORY_URI": history_uri} if history_uri else {}, base_home=tmp_path
    )
    async with hybrid(disabled) as archive:
        await archive.delete_session("second")
        assert not await archive.sessions(SCOPE)
        assert not await archive.vectors.store.asearch(vector_namespace)
        assert [entry["text"] for entry in await archive.entries(OTHER)] == ["private car"]
    assert not config.checkpoint_path.exists()
    assert not config.history_vector_path.exists()
