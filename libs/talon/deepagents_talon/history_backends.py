"""Environment-selected history Stores with owned, bounded startup lifecycles."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from deepagents_talon.config import TalonConfigError
from deepagents_talon.history_embeddings import DIMS, HistoryEmbeddings
from deepagents_talon.sqlite_archive import SQLiteConversationArchive
from deepagents_talon.store_archive import StoreConversationArchive
from deepagents_talon.store_records import finish

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import ModuleType

    from langgraph.store.base import BaseStore, IndexConfig

    from deepagents_talon.config import TalonConfig

_STARTUP_TIMEOUT = 15


@asynccontextmanager
async def open_history(config: TalonConfig) -> AsyncIterator[StoreConversationArchive]:
    """Open URI-selected history, defaulting to local SQLite.

    Args:
        config: Host configuration containing the optional history URI.
    """
    if config.history_uri is None:
        from deepagents_talon.sqlite_history import sqlite_store  # noqa: PLC0415

        async with (
            sqlite_store(config) as vectors,
            SQLiteConversationArchive.from_conn_string(
                str(config.checkpoint_path),
                store=vectors,
                vector_search=config.history_vector_search,
                search_visibility="immediate",
            ) as archive,
        ):
            yield archive
    else:
        async with remote_archive(config) as archive:
            yield archive


@asynccontextmanager
async def remote_archive(config: TalonConfig) -> AsyncIterator[StoreConversationArchive]:
    """Open the configured remote archive, keeping assistant namespaces separate.

    Args:
        config: Trusted configuration containing a validated remote history URI.

    Yields:
        An initialized archive closed before its metadata connection.
    """
    uri = config.history_uri
    if uri is None:
        msg = "Remote history requires DEEPAGENTS_TALON_HISTORY_URI"
        raise TalonConfigError(msg)
    async with (
        _remote_store(uri) as metadata,
        _vector_store(uri, enabled=config.history_vector_search) as vectors,
        AsyncExitStack() as stack,
    ):
        archive = StoreConversationArchive(
            metadata,
            namespace=("talon", config.assistant_id),
            vector_store=vectors,
            vector_search=config.history_vector_search,
            search_visibility="unknown"
            if urlsplit(uri).scheme.startswith("mongodb")
            else "immediate",
        )
        try:
            await stack.enter_async_context(archive.open())
        except Exception:  # noqa: BLE001  # Archive setup can surface credential-bearing driver errors.
            msg = "Could not initialize history archive; check permissions and storage format"
            raise TalonConfigError(msg) from None
        yield archive


@asynccontextmanager
async def _remote_store(
    uri: str, *, index: IndexConfig | None = None, vectors: bool = False
) -> AsyncIterator[BaseStore]:
    if urlsplit(uri).scheme.startswith("mongodb"):
        collection = "talon_history_vectors" if vectors else "talon_history"
        async with _mongodb_store(uri, index=index, collection=collection) as store:
            yield store
    else:
        async with _postgres_store(uri, index=index) as store:
            yield store


@asynccontextmanager
async def _vector_store(uri: str, *, enabled: bool) -> AsyncIterator[BaseStore]:
    embeddings, index = _embedding_index(enabled=enabled)
    async with _remote_store(uri, index=index, vectors=True) as store:
        try:
            yield store
        finally:
            await embeddings.aclose()


def _embedding_index(*, enabled: bool) -> tuple[HistoryEmbeddings, IndexConfig | None]:
    if enabled and importlib.util.find_spec("sentence_transformers") is None:
        msg = "Vector history requires deepagents-talon[history]: uv sync --extra history"
        raise ImportError(msg)
    embeddings = HistoryEmbeddings()
    index: IndexConfig | None = (
        {"dims": DIMS, "embed": embeddings, "fields": ["text"]} if enabled else None
    )
    return embeddings, index


def _driver(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError:
        msg = f"History backend requires deepagents-talon[{extra}]: uv sync --extra {extra}"
        raise ImportError(msg) from None


@asynccontextmanager
async def _postgres_store(
    uri: str, *, index: IndexConfig | None = None
) -> AsyncIterator[BaseStore]:
    driver = _driver("langgraph.store.postgres.aio", "postgres")
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT):
                store = await stack.enter_async_context(
                    driver.AsyncPostgresStore.from_conn_string(uri, index=index)
                )
                stack.push_async_callback(_stop_dispatcher, store)
                await store.setup()
        except Exception:  # noqa: BLE001  # Driver startup errors may contain URI credentials.
            msg = "Could not initialize PostgreSQL history; check the URI, server, and permissions"
            raise TalonConfigError(msg) from None
        yield cast("BaseStore", store)


async def _stop_dispatcher(store: BaseStore) -> None:
    # AsyncBatchedBaseStore has no public dispatcher shutdown API.
    task = getattr(store, "_task", None)
    if isinstance(task, asyncio.Task):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def _mongodb_store(
    uri: str, *, index: IndexConfig | None = None, collection: str = "talon_history"
) -> AsyncIterator[BaseStore]:
    driver = _driver("langgraph.store.mongodb", "mongodb")
    pymongo = _driver("pymongo", "mongodb")
    async with AsyncExitStack() as stack:
        try:
            client = pymongo.MongoClient(
                uri,
                connect=False,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                readPreference="primary",
                w="majority",
            )
            stack.push_async_callback(asyncio.to_thread, client.close)
            target = client.get_default_database()[collection]
            config = (
                driver.create_vector_index_config(
                    embed=index["embed"], dims=index["dims"], fields=["text"]
                )
                if index is not None
                else None
            )
            store = await finish(
                asyncio.to_thread(
                    driver.MongoDBStore, target, index_config=config, auto_index_timeout=60
                )
            )
        except Exception:  # noqa: BLE001  # Driver startup errors may contain URI credentials.
            msg = "Could not initialize MongoDB history; check the URI, server, and permissions"
            raise TalonConfigError(msg) from None
        yield cast("BaseStore", store)
