"""Lazy, in-process embeddings for optional conversation search."""

from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Protocol, cast

from langchain_core.embeddings import Embeddings


class _Vectors(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _Encoder(Protocol):
    max_seq_length: int

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> _Vectors: ...


MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIMS = 1024
QUERY_PROMPT = "Instruct: Retrieve past conversation passages relevant to the query.\nQuery: "


class HistoryEmbeddings(Embeddings):
    """Load Qwen on first use; serialize inference to bound memory consumption.

    Warning:
        Experimental API; subject to change with the Talon runtime.
    """

    def __init__(self) -> None:
        """Defer optional imports and model downloads until embedding is requested."""
        self._model: _Encoder | None = None
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._pending: asyncio.Task[list[list[float]]] | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed archive chunks without a query instruction.

        Args:
            texts: Bounded transcript chunks to embed.
        """
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                module = importlib.import_module("sentence_transformers")
                self._model = cast(
                    "_Encoder",
                    module.SentenceTransformer(MODEL, trust_remote_code=False, device="cpu"),
                )
                self._model.max_seq_length = 8192
            return self._model.encode(
                texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query using Qwen's instruction format.

        Args:
            text: Natural-language history query.
        """
        return self.embed_documents([QUERY_PROMPT + text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Run inference off-loop without accumulating threads after query timeouts.

        Args:
            texts: Transcript chunks or prepared queries.
        """
        async with self._async_lock:
            if self._pending is not None:
                await asyncio.shield(asyncio.gather(self._pending, return_exceptions=True))
            self._pending = asyncio.create_task(asyncio.to_thread(self.embed_documents, texts))
            self._pending.add_done_callback(_consume_exception)
            return await asyncio.shield(self._pending)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query off-loop with the retrieval instruction.

        Args:
            text: Natural-language history query.
        """
        return (await self.aembed_documents([QUERY_PROMPT + text]))[0]

    async def aclose(self) -> None:
        """Wait for inference still running after a cancelled search."""
        if self._pending is not None:
            await asyncio.shield(asyncio.gather(self._pending, return_exceptions=True))


def _consume_exception(task: asyncio.Task[list[list[float]]]) -> None:
    if not task.cancelled():
        task.exception()
