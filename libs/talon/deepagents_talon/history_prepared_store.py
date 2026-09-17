"""Compute client embeddings before acquiring a database connection lock."""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from typing import TYPE_CHECKING

from langgraph.store.base import BaseStore, PutOp, SearchOp

from deepagents_talon.history_adapters import EMBEDDING_CACHE

_CACHE_BYTES = 8 * 1024 * 1024
_CACHE_ENTRIES = 512

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from langchain_core.embeddings import Embeddings
    from langgraph.store.base import Op, Result


class PreparedVectorStore(BaseStore):
    """Prepare vectors outside database locks and reuse recent document embeddings.

    The instance caches at most 512 vectors with 8 MiB of packed vector data.
    Keys are content digests; deletes invalidate the cache. Reopening the Store
    starts cold, preserving model/profile isolation without changing archive records.
    """

    def __init__(self, store: BaseStore, embed: Embeddings) -> None:
        """Wrap a caller-owned Store and its configured embedding client."""
        self.store = store
        self.embed = embed
        self._vectors: OrderedDict[bytes, bytes] = OrderedDict()
        self._bytes = 0
        self._generation = 0

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
        if any(isinstance(op, PutOp) and op.value is None for op in operations):
            self._vectors.clear()
            self._bytes = 0
            self._generation += 1
        documents = list(
            dict.fromkeys(
                text
                for op in operations
                if isinstance(op, PutOp) and op.value is not None and op.index is not False
                for text in _indexed(op)
            )
        )
        cache = await self._documents(documents)
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

    async def _documents(self, documents: list[str]) -> dict[tuple[bool, str], list[float]]:
        generation = self._generation
        cached: dict[tuple[bool, str], list[float]] = {}
        missing: dict[str, bytes] = {}
        for text in documents:
            key = hashlib.sha256(text.encode()).digest()
            if (packed := self._vectors.get(key)) is not None:
                self._vectors.move_to_end(key)
                cached[False, text] = array("d", packed).tolist()
            else:
                missing[text] = key
        if missing:
            vectors = await self.embed.aembed_documents(list(missing))
            for (text, key), vector in zip(missing.items(), vectors, strict=True):
                cached[False, text] = vector
                if generation == self._generation:
                    self._remember(key, vector)
        return cached

    def _remember(self, key: bytes, vector: list[float]) -> None:
        packed = array("d", vector).tobytes()
        if len(packed) > _CACHE_BYTES:
            return
        self._bytes -= len(self._vectors.pop(key, b""))
        self._vectors[key] = packed
        self._bytes += len(packed)
        while self._bytes > _CACHE_BYTES or len(self._vectors) > _CACHE_ENTRIES:
            _, evicted = self._vectors.popitem(last=False)
            self._bytes -= len(evicted)


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
