from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import HumanMessage

from deepagents_talon.config import TalonConfig
from deepagents_talon.history import ArchiveScope
from deepagents_talon.history_store import HistoryStorage
from deepagents_talon.store_archive import StoreConversationArchive

SCOPE = ArchiveScope(talon_history_channel="test", talon_history_chat="one")
OTHER = ArchiveScope(talon_history_channel="test", talon_history_chat="two")


def _archive_factory(metadata):
    @asynccontextmanager
    async def archives(config):
        async with StoreConversationArchive(
            metadata,
            namespace=("talon", config.assistant_id),
        ).open() as archive:
            yield archive

    return archives


async def assert_store_archive_contract(metadata, tmp_path):
    """Exercise persistence, isolation, keyword pagination, and reset on each backend."""
    config = TalonConfig.from_env({}, base_home=tmp_path)
    plain = HistoryStorage(archive_factory=_archive_factory(metadata))
    async with plain.open(config) as archive:
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

    async with plain.open(config) as archive:
        page = await archive.entries(SCOPE, query="car", limit=1)
        second = await archive.entries(SCOPE, query="car", limit=1, after=page[0]["cursor"])
        assert {page[0]["session_id"], second[0]["session_id"]} == {"first", "second"}
        assert await archive.entries(SCOPE, query="car", after=second[0]["cursor"]) == []
        assert [row["session_id"] for row in await archive.entries(OTHER, query="car")] == [
            "private"
        ]
        await archive.delete_session("first")
        assert "first" not in await archive.sessions(SCOPE)
        assert await archive.entries(SCOPE, session_id="first") == []
        records = await metadata.asearch(archive.records.namespace, limit=100)
        assert "car repairs" not in str([item.value for item in records])

    async with plain.open(config) as archive:
        await archive.delete_session("second")
        assert not await archive.sessions(SCOPE)
        assert [entry["text"] for entry in await archive.entries(OTHER)] == ["private car"]
    assert not config.checkpoint_path.exists()
