"""Compute client embeddings before acquiring a database connection lock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.store.base import BaseStore, PutOp, SearchOp

from deepagents_talon.history_adapters import EMBEDDING_CACHE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.embeddings import Embeddings
    from langgraph.store.base import Op, Result


class PreparedVectorStore(BaseStore):
    """Keep slow provider calls outside database locks while retaining Store batching."""

    def __init__(self, store: BaseStore, embed: Embeddings) -> None:
        """Wrap a caller-owned Store and its configured embedding client."""
        self.store = store
        self.embed = embed

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Delegate synchronous operations to the underlying Store."""
        return self.store.batch(ops)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Prepare complete document and query vectors before Store I/O."""
        operations = list(ops)
        documents = list(
            dict.fromkeys(
                str(op.value["text"])
                for op in operations
                if isinstance(op, PutOp) and op.value is not None and op.index is not False
            )
        )
        cache = {
            (False, text): vector
            for text, vector in zip(
                documents, await self.embed.aembed_documents(documents), strict=True
            )
        }
        for op in operations:
            if isinstance(op, SearchOp) and op.query is not None:
                cache[True, op.query] = await self.embed.aembed_query(op.query)
        token = EMBEDDING_CACHE.set(cache)
        try:
            return await self.store.abatch(operations)
        finally:
            EMBEDDING_CACHE.reset(token)
