"""Optional default SQLite vector Store and its runtime lifecycle."""

from __future__ import annotations

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langgraph.store.base import Op, Result, SearchOp
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.history_embeddings import DIMS, QUERY_PROMPT, HistoryEmbeddings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from langgraph.store.sqlite.base import SqliteIndexConfig

    from deepagents_talon.config import TalonConfig


class _HistorySqliteStore(AsyncSqliteStore):
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # SQLite embeds searches through aembed_documents, unlike other Stores.
        # Add Qwen's retrieval instruction only to query operations.
        return await super().abatch(
            op._replace(query=QUERY_PROMPT + op.query)
            if isinstance(op, SearchOp) and op.query
            else op
            for op in ops
        )


@asynccontextmanager
async def sqlite_store(config: TalonConfig) -> AsyncIterator[AsyncSqliteStore | None]:
    """Open local vectors when enabled or needed to delete previously indexed history.

    Args:
        config: Assistant configuration controlling opt-in semantic search.

    Yields:
        Caller-scoped Store, or None when no vector data exists.

    Raises:
        ImportError: If vector search is enabled without the optional dependencies.
    """
    enabled = config.history_vector_search
    path = config.history_vector_path
    if not enabled and not path.exists():
        yield None
        return
    if enabled and importlib.util.find_spec("sentence_transformers") is None:
        msg = "Vector history requires deepagents-talon[history]: uv sync --extra history"
        raise ImportError(msg)
    embeddings = HistoryEmbeddings()
    index: SqliteIndexConfig | None = (
        {"dims": DIMS, "embed": embeddings, "fields": ["text"]} if enabled else None
    )
    async with _HistorySqliteStore.from_conn_string(str(path), index=index) as store:
        try:
            await store.conn.execute("PRAGMA foreign_keys=ON")
            await store.setup()
            yield store
        finally:
            await embeddings.aclose()
            # AsyncBatchedBaseStore creates a dispatcher even when only abatch is used.
            if store._task is not None:  # noqa: SLF001  # Upstream exposes no dispatcher close API.
                store._task.cancel()  # noqa: SLF001
                await asyncio.gather(store._task, return_exceptions=True)  # noqa: SLF001
