"""SQLite vector storage with explicit query embedding semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.history_adapters import QUERY_EMBEDDING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    import aiosqlite
    from langgraph.store.base import Result, SearchOp

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.history_profiles import EmbeddingProfile


class _HistorySqliteStore(AsyncSqliteStore):
    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        # Upstream uses aembed_documents for search; preserve provider input_type=query.
        token = QUERY_EMBEDDING.set(True)
        try:
            await super()._batch_search_ops(search_ops, results, cur)
        finally:
            QUERY_EMBEDDING.reset(token)


@asynccontextmanager
async def sqlite_store(
    config: TalonConfig, *, profile: EmbeddingProfile | None = None, generation: str = ""
) -> AsyncIterator[AsyncSqliteStore]:
    """Open one vector generation, including deletion without an embedding client."""
    path = config.history_generation_path(generation)
    index = profile.index if profile else None
    async with _HistorySqliteStore.from_conn_string(str(path), index=index) as store:
        try:
            await store.conn.execute("PRAGMA foreign_keys=ON")
            await store.setup()
            yield store
        finally:
            if store._task is not None:  # noqa: SLF001  # Upstream exposes no dispatcher close API.
                store._task.cancel()  # noqa: SLF001
                await asyncio.gather(store._task, return_exceptions=True)  # noqa: SLF001
