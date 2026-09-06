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

    from langgraph.store.base import BaseStore

    from deepagents_talon.config import TalonConfig

_STARTUP_TIMEOUT = 15
type StoreFactory = Callable[[str], AbstractAsyncContextManager[BaseStore]]


@asynccontextmanager
async def open_history(config: TalonConfig) -> AsyncIterator[StoreConversationArchive]:
    """Open URI-selected history, defaulting to local SQLite.

    Args:
        config: Host configuration containing the optional history URI.
    """
    uri = config.history_uri or config.checkpoint_path.as_uri()
    factory = _store_factory(urlsplit(uri).scheme)
    async with factory(uri) as metadata:
        archive = StoreConversationArchive(metadata, namespace=("talon", config.assistant_id))
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT), archive.records.access():
                root = await archive.records.root()
                await archive.records.commit([("root", root)])
        except Exception:  # noqa: BLE001  # Archive setup can surface credential-bearing driver errors.
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
async def _postgres_store(uri: str) -> AsyncIterator[BaseStore]:
    _validate_remote_uri(uri)
    driver = _driver("langgraph.store.postgres.aio", "postgres")
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(_STARTUP_TIMEOUT):
                store = await stack.enter_async_context(
                    driver.AsyncPostgresStore.from_conn_string(uri)
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
async def _mongodb_store(uri: str) -> AsyncIterator[BaseStore]:
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
            collection = client.get_default_database()["talon_history"]
            store = await finish(asyncio.to_thread(driver.MongoDBStore, collection))
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
