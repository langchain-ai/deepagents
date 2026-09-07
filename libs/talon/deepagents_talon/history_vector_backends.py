"""Generation-scoped vector Stores and explicit, restartable index rebuilding."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from langgraph.store.base import PutOp

from deepagents_talon.config import TalonConfigError
from deepagents_talon.history_vectors import HistoryVectorIndex
from deepagents_talon.store_archive import number
from deepagents_talon.store_archive_index import StoreVectorArchive
from deepagents_talon.store_records import finish

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from langgraph.store.base import BaseStore, IndexConfig

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.history_profiles import EmbeddingProfile
    from deepagents_talon.store_archive import StoreConversationArchive


async def prepare_generation(config: TalonConfig, archive: StoreConversationArchive) -> str | None:
    """Reject incompatible indexes before opening a model or starting any indexing."""
    async with asyncio.timeout(15), archive.records.access():
        root = await archive.records.root()
        await archive.records.commit([("root", root)])
    generation = str(root.get("embedding_generation", ""))
    if generation and re.fullmatch(r"[a-f0-9]{64}", generation) is None:
        msg = "Invalid history vector generation"
        raise TalonConfigError(msg)
    if not config.history_vector_search:
        return generation if root["vectors"] else None
    fingerprint = config.history_embedding_profile.fingerprint
    if root["vectors"] and (
        root.get("embedding_fingerprint") != fingerprint or root.get("reindexing")
    ):
        if not config.history_reindex:
            msg = (
                "History embedding profile changed or is missing; restore the original profile "
                "or set DEEPAGENTS_TALON_HISTORY_REINDEX=1 to rebuild vectors "
                "without deleting transcripts"
            )
            raise TalonConfigError(msg)
        async with asyncio.timeout(15), archive.records.access():
            await archive.records.commit([("root", {**root, "reindexing": True})])
        async with vector_backend(config, None, generation) as old:
            await _erase_vectors(archive, old)
        root = await archive.records.root()
        root = {**root, "indexed": 0, "reindex_cursor": 0, "reindexing": False}
    async with asyncio.timeout(15), archive.records.access():
        await archive.records.commit(
            [
                (
                    "root",
                    {
                        **root,
                        "embedding_fingerprint": fingerprint,
                        "embedding_generation": fingerprint,
                    },
                )
            ]
        )
    return fingerprint


async def _erase_vectors(archive: StoreConversationArchive, store: BaseStore) -> None:
    records = archive.records
    root = await records.root()
    index = HistoryVectorIndex(StoreVectorArchive(archive), store, indexing=False)
    index.identity = str(root["identity"])
    cursor, last = number(root, "reindex_cursor"), number(root, "last")
    while cursor < last:
        stop = min(cursor + 96, last)
        operations: list[PutOp] = []
        for identifier in range(cursor + 1, stop + 1):
            record = await records.get(str(identifier))
            if record is not None and record.get("kind") == "chunk":
                owner = await records.get(str(number(record, "owner")))
                if owner is not None:
                    scope = cast("dict[str, str]", owner["scope"])
                    namespace = index.namespace(
                        scope["talon_history_channel"], scope["talon_history_chat"]
                    )
                    operations.append(PutOp(namespace, str(identifier), None))
        if operations:
            await finish(store.abatch(operations))
        cursor = stop
        async with records.access():
            root = await records.root()
            await records.commit([("root", {**root, "reindex_cursor": cursor})])


@asynccontextmanager
async def vector_backend(
    config: TalonConfig, profile: EmbeddingProfile | None, generation: str
) -> AsyncIterator[BaseStore]:
    """Open vectors separately from metadata, preserving the metadata plugin contract."""
    from deepagents_talon.history_backends import _mongodb_store, _postgres_store  # noqa: PLC0415
    from deepagents_talon.sqlite_history import sqlite_store  # noqa: PLC0415

    uri = config.history_uri or config.checkpoint_path.as_uri()
    scheme = urlsplit(uri).scheme
    if profile is not None and not profile.client_side and scheme not in {"mongodb", "mongodb+srv"}:
        msg = "Atlas server-side embeddings require a MongoDB history backend"
        raise TalonConfigError(msg)
    index = profile.index if profile is not None else None
    if scheme in {"sqlite", "file"}:
        manager = sqlite_store(config, profile=profile, generation=generation)
    elif scheme in {"mongodb", "mongodb+srv"}:
        suffix = f"_{generation}" if generation else ""
        manager = _mongodb_store(
            uri, index=index, collection=f"talon_history_vectors{suffix}", profile=profile
        )
    elif scheme in {"postgres", "postgresql"}:
        manager = _postgres_store(uri, index=index, generation=generation)
    else:
        manager = _plugin_vectors(uri, index, generation)
    async with manager as store:
        if profile is not None and profile.client_side and profile.embed is not None:
            from deepagents_talon.history_prepared_store import PreparedVectorStore  # noqa: PLC0415

            yield PreparedVectorStore(store, profile.embed)
        else:
            yield store


@asynccontextmanager
async def _plugin_vectors(
    uri: str, index: IndexConfig | None, generation: str
) -> AsyncIterator[BaseStore]:
    plugins = entry_points(
        group="deepagents_talon.history_vector_backends", name=urlsplit(uri).scheme
    )
    if len(plugins) != 1:
        msg = (
            "Vector history requires exactly one installed "
            "deepagents_talon.history_vector_backends entry point"
        )
        raise TalonConfigError(msg)
    from contextlib import AsyncExitStack  # noqa: PLC0415

    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(15):
                factory = cast(
                    "Callable[..., AbstractAsyncContextManager[BaseStore]]",
                    next(iter(plugins)).load(),
                )
                store = await stack.enter_async_context(
                    factory(uri, index=index, generation=generation)
                )
        except Exception:  # noqa: BLE001  # Plugin errors may contain URI credentials.
            msg = (
                "Could not initialize history vector plugin; "
                "check its installation and configuration"
            )
            raise TalonConfigError(msg) from None
        yield store
