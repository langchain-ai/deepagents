"""SQLite adapter for the shared conversation archive."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

import aiosqlite
from langgraph.store.base import BaseStore, GetOp, Item, Result
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.store_archive import StoreConversationArchive

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from deepagents_talon.archive import SearchVisibility


class _MetadataStore(AsyncSqliteStore):
    aput = BaseStore.aput
    adelete = BaseStore.adelete

    async def setup(self) -> None:
        await super().setup()
        await self.conn.commit()

    async def aget(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        # A read must not commit a checkpoint transaction on a shared connection.
        results: list[Result] = [None]
        async with self._cursor(transaction=False) as cursor:
            await self._batch_get_ops(
                [(0, GetOp(namespace, key, refresh_ttl=bool(refresh_ttl)))], results, cursor
            )
        return cast("Item | None", results[0])


class SQLiteConversationArchive(StoreConversationArchive):
    """Use a caller-owned SQLite connection for the shared archive.

    Args:
        conn: SQLite metadata connection.
        store: Separate optional vector Store, including for deletion-only use.
        vector_search: Enable indexing and semantic search.
        search_visibility: Whether vector writes are immediately searchable.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        store: BaseStore | None = None,
        vector_search: bool = True,
        search_visibility: SearchVisibility = "unknown",
    ) -> None:
        """Use direct store operations without a background dispatcher."""
        metadata = _MetadataStore(conn)
        if metadata._task is not None:  # noqa: SLF001  # Upstream has no public dispatcher shutdown API.
            metadata._task.cancel()  # noqa: SLF001
        super().__init__(
            metadata,
            namespace=("talon",),
            vector_store=store,
            vector_search=vector_search,
            search_visibility=search_visibility,
        )

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        store: BaseStore | None = None,
        vector_search: bool = True,
        search_visibility: SearchVisibility = "unknown",
    ) -> AsyncIterator[SQLiteConversationArchive]:
        """Open and close an archive connection.

        Args:
            conn_string: SQLite path or `:memory:`.
            store: Caller-owned vector Store.
            vector_search: Enable indexing and semantic search.
            search_visibility: Whether vector writes are immediately searchable.
        """
        async with aiosqlite.connect(conn_string) as conn:
            archive = cls(
                conn, store=store, vector_search=vector_search, search_visibility=search_visibility
            )
            async with archive.open():
                yield archive
