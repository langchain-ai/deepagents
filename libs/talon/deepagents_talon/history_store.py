"""Backend-independent history factories and resource ownership.

Warning:
    Experimental API; subject to change with the Talon runtime.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver

from deepagents_talon.archive_saver import ConversationSaver

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from langgraph.types import Checkpointer

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.history import ConversationArchive

    ArchiveFactory = Callable[[TalonConfig], AbstractAsyncContextManager[ConversationArchive]]


@dataclass(frozen=True, slots=True)
class HistoryStorage:
    """Open history through one factory that owns the archive and its resources.

    Args:
        archive_factory: Open an initialized archive, closing the archive before
            its metadata store. Defaults to local SQLite history.
    """

    archive_factory: ArchiveFactory | None = None

    @asynccontextmanager
    async def open(self, config: TalonConfig) -> AsyncIterator[ConversationArchive]:
        """Open initialized history resources for the lifetime of a host.

        Args:
            config: Trusted host configuration passed to the archive factory.

        Yields:
            An archive ready for checkpoint writes, retrieval, and reset.
        """
        async with (self.archive_factory or _local_history)(config) as archive:
            yield archive


@asynccontextmanager
async def _local_history(config: TalonConfig) -> AsyncIterator[ConversationArchive]:
    from deepagents_talon.sqlite_history import sqlite_archive  # noqa: PLC0415

    async with sqlite_archive(config) as archive:
        yield archive


@asynccontextmanager
async def history_checkpointer(
    config: TalonConfig,
    *,
    checkpointer: Checkpointer | None = None,
    storage: HistoryStorage | None = None,
) -> AsyncIterator[Checkpointer]:
    """Compose checkpoint and history backends without taking caller ownership.

    Args:
        config: Host configuration used by default backends and factories.
        checkpointer: Caller-owned checkpoint backend. Without explicit storage,
            a supplied backend passes through unchanged, preserving opt-in history.
        storage: History factories to compose with the checkpoint backend.
            When no backend is supplied, defaults provide local persistence.

    Yields:
        The configured backend, optionally wrapped with conversation history.

    Raises:
        TypeError: Explicit storage is paired with a non-saver or an already
            wrapped saver, which would otherwise duplicate archive ownership.
    """
    if checkpointer is not None and storage is None:
        yield checkpointer
        return
    async with AsyncExitStack() as stack:
        if checkpointer is None:
            from deepagents_talon.sqlite_history import sqlite_checkpointer  # noqa: PLC0415

            checkpointer = await stack.enter_async_context(sqlite_checkpointer(config))
        if not isinstance(checkpointer, BaseCheckpointSaver) or isinstance(
            checkpointer, ConversationSaver
        ):
            msg = "Explicit history storage requires an unwrapped checkpoint saver"
            raise TypeError(msg)
        archive = await stack.enter_async_context((storage or HistoryStorage()).open(config))
        yield ConversationSaver(checkpointer, archive=archive)
