"""URI-selected LangGraph checkpointers with owned connection lifecycles."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast
from urllib.parse import unquote, urlsplit

from deepagents_talon.config import TalonConfigError
from deepagents_talon.history_drivers import load_driver
from deepagents_talon.store_records import finish

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from langgraph.checkpoint.base import BaseCheckpointSaver

    from deepagents_talon.config import TalonConfig

type CheckpointFactory = Callable[[str], AbstractAsyncContextManager[BaseCheckpointSaver]]


@asynccontextmanager
async def open_checkpointer(config: TalonConfig) -> AsyncIterator[BaseCheckpointSaver]:
    """Open the configured checkpointer, defaulting to the local SQLite database.

    Args:
        config: Host configuration containing the optional checkpoint URI.
    """
    uri = config.checkpoint_uri or config.checkpoint_path.as_uri()
    async with AsyncExitStack() as stack:
        try:
            factory = _checkpoint_factory(urlsplit(uri).scheme)
            saver = await stack.enter_async_context(factory(uri))
        except (TalonConfigError, ImportError):
            raise
        except Exception:  # noqa: BLE001  # Driver errors can contain URI credentials.
            msg = "Could not initialize checkpointer; check installation, URI, and permissions"
            raise TalonConfigError(msg) from None
        yield saver


def _checkpoint_factory(scheme: str) -> CheckpointFactory:
    if factory := _BUILTIN_CHECKPOINTERS.get(scheme):
        return factory
    plugins = entry_points(group="deepagents_talon.checkpoint_backends", name=scheme)
    if len(plugins) == 1:
        return cast("CheckpointFactory", next(iter(plugins)).load())
    msg = (
        "DEEPAGENTS_TALON_CHECKPOINT_URI requires a built-in backend or exactly one "
        "installed deepagents_talon.checkpoint_backends entry point for its scheme"
    )
    raise TalonConfigError(msg)


@asynccontextmanager
async def _sqlite(uri: str) -> AsyncIterator[BaseCheckpointSaver]:
    parsed = urlsplit(uri)
    if parsed.netloc or not parsed.path or parsed.path == "/":
        msg = "DEEPAGENTS_TALON_CHECKPOINT_URI requires a SQLite file path without a host"
        raise TalonConfigError(msg)
    driver = load_driver("langgraph.checkpoint.sqlite.aio", "sqlite", "Checkpointer requires")
    async with driver.AsyncSqliteSaver.from_conn_string(unquote(parsed.path)) as saver:
        await saver.setup()
        yield saver


def _remote_database(uri: str) -> str:
    parsed = urlsplit(uri)
    if not parsed.hostname or not parsed.path.strip("/"):
        msg = "DEEPAGENTS_TALON_CHECKPOINT_URI requires a host and database name"
        raise TalonConfigError(msg)
    return unquote(parsed.path.lstrip("/"))


@asynccontextmanager
async def _postgres(uri: str) -> AsyncIterator[BaseCheckpointSaver]:
    _remote_database(uri)
    driver = load_driver("langgraph.checkpoint.postgres.aio", "postgres", "Checkpointer requires")
    async with AsyncExitStack() as stack:
        async with asyncio.timeout(15):
            saver = await stack.enter_async_context(driver.AsyncPostgresSaver.from_conn_string(uri))
            await saver.setup()
        yield saver


@asynccontextmanager
async def _mongodb(uri: str) -> AsyncIterator[BaseCheckpointSaver]:
    database = _remote_database(uri)
    driver = load_driver("langgraph.checkpoint.mongodb", "mongodb", "Checkpointer requires")
    pymongo = load_driver("pymongo", "mongodb", "Checkpointer requires")
    client = pymongo.MongoClient(
        uri,
        connect=False,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
    )
    try:
        saver = await finish(asyncio.to_thread(driver.MongoDBSaver, client, db_name=database))
        yield saver
    finally:
        await finish(asyncio.to_thread(client.close))


_BUILTIN_CHECKPOINTERS: dict[str, CheckpointFactory] = {
    "sqlite": _sqlite,
    "file": _sqlite,
    "postgres": _postgres,
    "postgresql": _postgres,
    "mongodb": _mongodb,
    "mongodb+srv": _mongodb,
}
