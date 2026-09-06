"""SQLite metadata adapter for the shared conversation archive.

Warning:
    Experimental API; subject to change with the Talon runtime.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

import aiosqlite
from langgraph.store.base import BaseStore, GetOp, Item, Result
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.store_archive import StoreConversationArchive

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _MetadataStore(AsyncSqliteStore):
    # Direct operations avoid a dispatcher outliving caller-owned connections.
    aput = BaseStore.aput
    adelete = BaseStore.adelete

    async def aget(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        # Reads may share the connection with an in-flight checkpoint transaction.
        results: list[Result] = [None]
        async with self._cursor(transaction=False) as cursor:
            await self._batch_get_ops(
                [(0, GetOp(namespace, key, refresh_ttl=bool(refresh_ttl)))], results, cursor
            )
        return cast("Item | None", results[0])


class SQLiteConversationArchive(StoreConversationArchive):
    """Use the shared archive with SQLite metadata.

    Args:
        conn: Caller-owned metadata connection.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
    ) -> None:
        """Keep connection ownership with the caller."""
        self.conn = conn
        self.metadata = _MetadataStore(conn)
        if self.metadata._task is not None:  # noqa: SLF001  # Upstream has no dispatcher close API; this adapter uses direct operations.
            self.metadata._task.cancel()  # noqa: SLF001
        self._sqlite_setup_lock = asyncio.Lock()
        super().__init__(
            self.metadata,
            namespace=("talon",),
        )

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
    ) -> AsyncIterator[SQLiteConversationArchive]:
        """Open local metadata for the shared archive.

        Args:
            conn_string: SQLite path or `:memory:`.

        Yields:
            Initialized archive with a connection owned by this context.
        """
        async with aiosqlite.connect(conn_string) as conn:
            archive = cls(conn)
            async with archive.open():
                yield archive

    async def setup(self) -> None:
        """Initialize SQLite metadata before exposing the shared archive."""
        async with self._sqlite_setup_lock:
            if self._ready:
                await super().setup()
                return
            await self.metadata.setup()
            await self.conn.commit()
            await super().setup()
