"""Default SQLite history and checkpoint resource lifecycles."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from deepagents_talon.archive import SQLiteConversationArchive
    from deepagents_talon.config import TalonConfig


@asynccontextmanager
async def sqlite_archive(config: TalonConfig) -> AsyncIterator[SQLiteConversationArchive]:
    """Open the default SQLite transcript archive.

    Args:
        config: Host configuration providing the local checkpoint path.

    Yields:
        Initialized archive owned by this context.
    """
    from deepagents_talon.archive import SQLiteConversationArchive  # noqa: PLC0415

    async with SQLiteConversationArchive.from_conn_string(str(config.checkpoint_path)) as archive:
        yield archive


@asynccontextmanager
async def sqlite_checkpointer(config: TalonConfig) -> AsyncIterator[AsyncSqliteSaver]:
    """Open the default checkpoint backend.

    Args:
        config: Host configuration providing the local checkpoint path.

    Yields:
        Initialized SQLite checkpointer owned by this context.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415

    async with AsyncSqliteSaver.from_conn_string(str(config.checkpoint_path)) as saver:
        await saver.setup()
        yield saver
