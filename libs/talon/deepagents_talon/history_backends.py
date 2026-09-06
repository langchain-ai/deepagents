"""Environment-selected history Stores with owned, bounded startup lifecycles."""

from __future__ import annotations

import asyncio
import importlib
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from deepagents_talon.config import TalonConfigError
from deepagents_talon.store_archive import StoreConversationArchive
from deepagents_talon.store_records import finish

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import ModuleType

    from langgraph.store.base import BaseStore

    from deepagents_talon.config import TalonConfig

_STARTUP_TIMEOUT = 15


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
    factory = _mongodb_store if urlsplit(uri).scheme.startswith("mongodb") else _postgres_store
    async with factory(uri) as metadata, AsyncExitStack() as stack:
        archive = StoreConversationArchive(metadata, namespace=("talon", config.assistant_id))
        try:
            await stack.enter_async_context(archive.open())
        except Exception:  # noqa: BLE001  # Archive setup can surface credential-bearing driver errors.
            msg = "Could not initialize history archive; check permissions and storage format"
            raise TalonConfigError(msg) from None
        yield archive


def _driver(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError:
        msg = f"History backend requires deepagents-talon[{extra}]: uv sync --extra {extra}"
        raise ImportError(msg) from None


@asynccontextmanager
async def _postgres_store(uri: str) -> AsyncIterator[BaseStore]:
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
