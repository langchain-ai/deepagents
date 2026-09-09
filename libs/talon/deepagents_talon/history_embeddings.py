"""Lazy, in-process embeddings for optional conversation search."""

from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Protocol, cast

from langchain_core.embeddings import Embeddings

from deepagents_talon.config import TalonConfig
from deepagents_talon.history_profiles import LOCAL_MODEL, LOCAL_PROMPT


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


MODEL = LOCAL_MODEL
QUERY_PROMPT = LOCAL_PROMPT


class HistoryEmbeddings(Embeddings):
    """Load Qwen on first use; serialize inference to bound memory consumption.

    Warning:
        Experimental API; subject to change with the Talon runtime.
    """

    def __init__(
        self,
        *,
        model: str = MODEL,
        max_input_tokens: int = 8192,
        batch_size: int = 4,
        query_prompt: str = "",
        config: TalonConfig | None = None,
    ) -> None:
        """Defer optional imports and model downloads until embedding is requested.

        `query_prompt` defaults to none because the caller that owns the profile
        applies the model's instruction format before delegating here. Defaulting
        to Qwen's prefix instead would embed it twice whenever a caller forgot to
        pass an empty string.

        Args:
            model: Hugging Face model identifier or local model path.
            max_input_tokens: Maximum sequence length for the encoder.
            batch_size: Number of texts encoded together.
            query_prompt: Optional prefix applied to retrieval queries.
            config: Talon home configuration; defaults to the process environment.
        """
        self.config = config if config is not None else TalonConfig.from_env()
        self.model = model
        self.max_input_tokens = max_input_tokens
        self.batch_size = batch_size
        self.query_prompt = query_prompt
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
                    module.SentenceTransformer(
                        self.model,
                        trust_remote_code=False,
                        device="cpu",
                        cache_folder=str(self.config.huggingface_cache_dir),
                    ),
                )
                self._model.max_seq_length = self.max_input_tokens
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query using Qwen's instruction format.

        Args:
            text: Natural-language history query.
        """
        return self.embed_documents([self.query_prompt + text])[0]

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
        return (await self.aembed_documents([self.query_prompt + text]))[0]

    async def aclose(self) -> None:
        """Wait for inference still running after a cancelled search."""
        if self._pending is not None:
            await asyncio.shield(asyncio.gather(self._pending, return_exceptions=True))


def _consume_exception(task: asyncio.Task[list[list[float]]]) -> None:
    if not task.cancelled():
        task.exception()
