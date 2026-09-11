"""Compute client embeddings before acquiring a database connection lock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.store.base import BaseStore, PutOp, SearchOp

from deepagents_talon.history_adapters import EMBEDDING_CACHE

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from langchain_core.embeddings import Embeddings
    from langgraph.store.base import Op, Result


class PreparedVectorStore(BaseStore):
    """Keep slow provider calls outside database locks while retaining Store batching."""

    def __init__(self, store: BaseStore, embed: Embeddings) -> None:
        """Wrap a caller-owned Store and its configured embedding client."""
        self.store = store
        self.embed = embed

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Refuse synchronous operations, which would bypass vector preparation.

        Delegating would reach the inner Store with an empty cache, so embedding
        would run from its worker thread inside the database lock - the exact
        failure this wrapper exists to prevent.

        Raises:
            NotImplementedError: Always; use the async Store API.
        """
        msg = "History vectors require the async Store API; a synchronous batch cannot prepare them"
        raise NotImplementedError(msg)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Prepare complete document and query vectors before Store I/O."""
        operations = list(ops)
        documents = list(
            dict.fromkeys(
                text
                for op in operations
                if isinstance(op, PutOp) and op.value is not None and op.index is not False
                for text in _indexed(op)
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
                vector = await self.embed.aembed_query(op.query)
                cache[True, op.query] = vector
                # Only some Stores embed a search through aembed_query; SQLite and
                # PostgreSQL route it through aembed_documents instead. Publishing the
                # query vector under the document key too keeps query prompts and
                # provider input types backend-neutral, without a second round trip.
                # A document in this same batch keeps its own vector: that one is
                # stored durably, while a query vector is used once and discarded.
                cache.setdefault((False, op.query), vector)
        token = EMBEDDING_CACHE.set(cache)
        try:
            return await self.store.abatch(operations)
        finally:
            EMBEDDING_CACHE.reset(token)


def _indexed(op: PutOp) -> Iterator[str]:
    """Yield the fields this write will embed, without assuming they are named `text`."""
    # `index=None` defers to the Store's own configuration, which is not visible from
    # here; `text` is what the archive configures. A structured path is left to the
    # Store, which then embeds it itself instead of reading a prepared vector.
    fields = op.index if isinstance(op.index, (list, tuple)) else ("text",)
    for field in fields:
        value = op.value.get(field) if op.value is not None else None
        if value is not None and not any(token in field for token in ".[$"):
            yield str(value)
