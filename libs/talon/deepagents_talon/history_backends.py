"""Environment-selected history Stores with owned, bounded startup lifecycles."""

from __future__ import annotations

import asyncio
import importlib
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from functools import partial
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

import aiosqlite
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.config import TalonConfigError
from deepagents_talon.store_archive import StoreConversationArchive
from deepagents_talon.store_records import finish

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from types import ModuleType

    from langgraph.store.base import BaseStore, IndexConfig

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.history_profiles import EmbeddingProfile

_STARTUP_TIMEOUT = 15
_MAX_HNSW_DIMS = 2000
type StoreFactory = Callable[[str], AbstractAsyncContextManager[BaseStore]]


@asynccontextmanager
async def open_history(config: TalonConfig) -> AsyncIterator[StoreConversationArchive]:
    """Open URI-selected history, defaulting to local SQLite.

    Args:
        config: Host configuration containing the optional history URI.
    """
    uri = config.history_uri or config.checkpoint_path.as_uri()
    scheme = urlsplit(uri).scheme
    if (
        config.history_vector_search
        and config.history_embedding_profile.adapter == "atlas"
        and scheme not in {"mongodb", "mongodb+srv"}
    ):
        msg = "Atlas server-side embeddings require a MongoDB history backend"
        raise TalonConfigError(msg)
    factory = _store_factory(scheme)
    from deepagents_talon.history_adapters import open_profile  # noqa: PLC0415
    from deepagents_talon.history_vector_backends import (  # noqa: PLC0415
        prepare_generation,
        vector_backend,
    )

    async with factory(uri) as metadata, AsyncExitStack() as stack:
        archive = StoreConversationArchive(metadata, namespace=("talon", config.assistant_id))
        try:
            generation = await prepare_generation(config, archive)
        except TalonConfigError:
            raise
        except Exception:  # noqa: BLE001  # Metadata errors may include URI credentials.
            msg = "Could not initialize history archive; check permissions and storage format"
            raise TalonConfigError(msg) from None
        profile = await stack.enter_async_context(open_profile(config))
        vectors = (
            await stack.enter_async_context(vector_backend(config, profile, generation))
            if generation is not None
            else None
        )
        archive = StoreConversationArchive(
            metadata,
            namespace=("talon", config.assistant_id),
            vector_store=vectors,
            vector_search=config.history_vector_search,
            embedding_profile=profile,
            search_visibility="unknown"
            if urlsplit(uri).scheme.startswith("mongodb")
            else "immediate",
        )
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT):
                await stack.enter_async_context(archive.open())
        except Exception:  # noqa: BLE001  # Driver errors may contain credentials.
            msg = "Could not initialize history archive; check permissions and storage format"
            raise TalonConfigError(msg) from None
        yield archive


def _store_factory(scheme: str) -> StoreFactory:
    if factory := _BUILTIN_STORES.get(scheme):
        return factory
    plugins = entry_points(group="deepagents_talon.history_backends", name=scheme)
    if len(plugins) == 1:
        return partial(_plugin_store, next(iter(plugins)))
    msg = (
        "DEEPAGENTS_TALON_HISTORY_URI requires a built-in backend or exactly one "
        "installed deepagents_talon.history_backends entry point for its scheme"
    )
    raise TalonConfigError(msg)


@asynccontextmanager
async def _plugin_store(plugin: EntryPoint, uri: str) -> AsyncIterator[BaseStore]:
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT):
                factory = cast("StoreFactory", plugin.load())
                store = await stack.enter_async_context(factory(uri))
        except Exception:  # noqa: BLE001  # Plugin errors may contain URI credentials.
            msg = "Could not initialize history plugin; check its installation and configuration"
            raise TalonConfigError(msg) from None
        yield store


def _validate_remote_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    if not parsed.hostname or not parsed.path.strip("/"):
        msg = "DEEPAGENTS_TALON_HISTORY_URI requires a host and database name for this backend"
        raise TalonConfigError(msg)


@asynccontextmanager
async def _sqlite_store(uri: str) -> AsyncIterator[BaseStore]:
    parsed = urlsplit(uri)
    if parsed.netloc or not parsed.path or parsed.path == "/":
        msg = "DEEPAGENTS_TALON_HISTORY_URI requires a SQLite file path without a host"
        raise TalonConfigError(msg)
    connection = "file:" + uri.split(":", 1)[1]
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT):
                conn = await stack.enter_async_context(
                    aiosqlite.connect(connection, uri=True, isolation_level=None)
                )
                store = AsyncSqliteStore(conn)
                stack.push_async_callback(_stop_dispatcher, store)
                await store.setup()
        except Exception:  # noqa: BLE001  # SQLite errors may contain sensitive connection details.
            msg = "Could not initialize SQLite history; check the URI, path, and permissions"
            raise TalonConfigError(msg) from None
        yield store


def _driver(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError:
        msg = f"History backend requires deepagents-talon[{extra}]: uv sync --extra {extra}"
        raise ImportError(msg) from None


@asynccontextmanager
async def _postgres_store(
    uri: str, *, index: IndexConfig | None = None, generation: str = ""
) -> AsyncIterator[BaseStore]:
    _validate_remote_uri(uri)
    driver = _driver("langgraph.store.postgres.aio", "postgres")
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT):
                store = await stack.enter_async_context(
                    driver.AsyncPostgresStore.from_conn_string(uri, index=index)
                )
                stack.push_async_callback(_stop_dispatcher, store)
                if index is not None and index["dims"] > _MAX_HNSW_DIMS:
                    store.index_config["ann_index_config"] = {"kind": "flat"}
                if generation:
                    await _driver("deepagents_talon.history_postgres", "postgres").setup_generation(
                        store, generation
                    )
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
    uri: str,
    *,
    index: IndexConfig | None = None,
    collection: str = "talon_history",
    profile: EmbeddingProfile | None = None,
) -> AsyncIterator[BaseStore]:
    _validate_remote_uri(uri)
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
                    embed=index["embed"],
                    dims=-1 if profile and not profile.client_side else index["dims"],
                    fields=["text"],
                    relevance_score_fn=None if profile and not profile.client_side else "cosine",
                    embedding_key="embedding",
                )
                if index is not None
                else None
            )
            store = await finish(
                asyncio.to_thread(
                    driver.MongoDBStore,
                    target,
                    index_config=config,
                    auto_index_timeout=60,
                    query_model=profile.query_model or None if profile else None,
                )
            )
        except Exception:  # noqa: BLE001  # Driver startup errors may contain URI credentials.
            msg = "Could not initialize MongoDB history; check the URI, server, and permissions"
            raise TalonConfigError(msg) from None
        yield cast("BaseStore", store)


_BUILTIN_STORES: dict[str, StoreFactory] = {
    "sqlite": _sqlite_store,
    "file": _sqlite_store,
    "mongodb": _mongodb_store,
    "mongodb+srv": _mongodb_store,
    "postgres": _postgres_store,
    "postgresql": _postgres_store,
}
