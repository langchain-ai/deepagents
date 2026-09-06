"""Owned embedding clients and bounded, provider-neutral preprocessing."""

from __future__ import annotations

import asyncio
import importlib
import math
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from deepagents_talon.config import TalonConfigError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine
    from types import ModuleType

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.history_profiles import EmbeddingProfile

QUERY_EMBEDDING: ContextVar[bool] = ContextVar("talon_history_query", default=False)
EMBEDDING_CACHE: ContextVar[dict[tuple[bool, str], list[float]] | None] = ContextVar(
    "talon_history_embeddings", default=None
)
_REQUEST_TIMEOUT = 30


@asynccontextmanager
async def open_profile(config: TalonConfig) -> AsyncIterator[EmbeddingProfile | None]:
    """Resolve one adapter, releasing only resources created by this context."""
    if not config.history_vector_search:
        yield None
        return
    profile = config.history_embedding_profile
    async with AsyncExitStack() as stack:
        try:
            embed = await _adapter(config, profile, stack)
        except (ImportError, TalonConfigError):
            raise
        except Exception:  # noqa: BLE001  # Provider validation can include API keys.
            msg = "Could not initialize history embeddings; check the selected adapter and settings"
            raise TalonConfigError(msg) from None
        if profile.client_side:
            embed = BoundedEmbeddings(embed, profile)
        yield replace(profile, embed=embed)


def _driver(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError:
        msg = f"History embeddings require deepagents-talon[{extra}]: uv sync --extra {extra}"
        raise ImportError(msg) from None


async def _adapter(
    config: TalonConfig, profile: EmbeddingProfile, stack: AsyncExitStack
) -> Embeddings:
    if profile.adapter == "local":
        _driver("sentence_transformers", "history-local")
        from deepagents_talon.history_embeddings import HistoryEmbeddings  # noqa: PLC0415

        embed = HistoryEmbeddings(
            model=profile.model,
            max_input_tokens=profile.max_input_tokens,
            batch_size=profile.batch_size,
            query_prompt="",
        )
        stack.push_async_callback(embed.aclose)
        return embed
    if profile.adapter == "atlas":
        driver = _driver("langchain_mongodb.embeddings", "mongodb")
        return driver.AutoEmbeddings(model=profile.model)
    if profile.adapter == "voyage":
        _driver("langchain_voyageai", "history-voyage")
        from deepagents_talon.history_voyage import HistoryVoyageEmbeddings  # noqa: PLC0415

        return HistoryVoyageEmbeddings(
            model=profile.model,
            api_key=_key(config, "VOYAGE_API_KEY"),
            output_dimension=cast("Literal[256, 512, 1024, 2048]", profile.dims),
            batch_size=profile.batch_size,
            truncation=False,
            base_url=profile.base_url or "https://api.voyageai.com/v1",
        )
    return await _openai(config, profile, stack)


def _key(config: TalonConfig, name: str) -> SecretStr:
    value = config.env.get("DEEPAGENTS_TALON_HISTORY_EMBED_API_KEY") or config.env.get(name)
    if not value:
        msg = f"History embeddings require {name} or DEEPAGENTS_TALON_HISTORY_EMBED_API_KEY"
        raise TalonConfigError(msg)
    return SecretStr(value)


async def _openai(
    config: TalonConfig, profile: EmbeddingProfile, stack: AsyncExitStack
) -> Embeddings:
    driver = _driver("langchain_openai", "history-openai")
    import httpx  # noqa: PLC0415  # Optional provider transport.

    key = (
        "OPENROUTER_API_KEY"
        if profile.base_url == "https://openrouter.ai/api/v1"
        else "OPENAI_API_KEY"
    )
    sync = stack.enter_context(httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=False))
    asynchronous = await stack.enter_async_context(
        httpx.AsyncClient(timeout=_REQUEST_TIMEOUT, follow_redirects=False)
    )
    return driver.OpenAIEmbeddings(
        model=profile.model,
        dimensions=profile.dims,
        base_url=profile.base_url,
        api_key=_key(config, key),
        max_retries=0,
        request_timeout=_REQUEST_TIMEOUT,
        check_embedding_ctx_length=False,
        chunk_size=profile.batch_size,
        http_client=sync,
        http_async_client=asynchronous,
    )


class BoundedEmbeddings(Embeddings):
    """Bound HTTP concurrency and pool complete document chunks without truncation."""

    def __init__(self, embed: Embeddings, profile: EmbeddingProfile) -> None:
        """Keep native async provider calls on their owning event loop."""
        self.embed = embed
        self.profile = profile
        self.loop = asyncio.get_running_loop()
        self.slots = asyncio.Semaphore(profile.concurrency)

    def _bridge[T](self, operation: Coroutine[object, object, T]) -> T:
        # MongoDBStore invokes sync embeddings from its worker thread.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is self.loop:
            operation.close()
            msg = "Use async history embeddings on the event loop"
            raise RuntimeError(msg)
        return asyncio.run_coroutine_threadsafe(operation, self.loop).result()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Bridge Store worker threads to the owned async client."""
        return self._bridge(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Preserve provider query semantics when a Store uses a worker thread."""
        return self._bridge(self.aembed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        """Embed one complete query, refusing inputs above the conservative budget."""
        if (cached := (EMBEDDING_CACHE.get() or {}).get((True, text))) is not None:
            return cached
        query = self.profile.query_prompt + text
        if len(query.encode()) > self.profile.max_input_tokens - 128:
            msg = "History embedding query exceeds the configured input budget"
            raise ValueError(msg)
        async with self.slots, asyncio.timeout(_REQUEST_TIMEOUT):
            vector = await self.embed.aembed_query(query)
        self._validate([vector], 1)
        return vector

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Split by a conservative UTF-8 token bound, then pool every source chunk."""
        if QUERY_EMBEDDING.get():
            return [await self.aembed_query(text) for text in texts]
        cache = EMBEDDING_CACHE.get() or {}
        if all((False, text) in cache for text in texts):
            return [cache[False, text] for text in texts]
        chunks = [_chunks(text, self.profile.max_input_tokens - 128) for text in texts]
        flattened = [chunk for document in chunks for chunk in document]
        vectors = await self._documents(flattened)
        output: list[list[float]] = []
        cursor = 0
        for document in chunks:
            group = vectors[cursor : cursor + len(document)]
            output.append(_pool(group, [len(chunk.encode()) for chunk in document]))
            cursor += len(document)
        return output

    async def _documents(self, texts: list[str]) -> list[list[float]]:
        size = self.profile.batch_size
        output: list[list[float]] = []
        concurrency = max(1, self.profile.concurrency - 1)
        for start in range(0, len(texts), size * concurrency):
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._batch(texts[offset : offset + size]))
                    for offset in range(start, min(len(texts), start + size * concurrency), size)
                ]
            output.extend(vector for task in tasks for vector in task.result())
        return output

    async def _batch(self, texts: list[str]) -> list[list[float]]:
        async with self.slots, asyncio.timeout(_REQUEST_TIMEOUT):
            vectors = await self.embed.aembed_documents(texts)
        self._validate(vectors, len(texts))
        return vectors

    def _validate(self, vectors: list[list[float]], count: int) -> None:
        if len(vectors) != count or any(
            len(vector) != self.profile.dims or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            msg = "History embedding response has invalid dimensions, values, or count"
            raise ValueError(msg)


def _chunks(text: str, budget: int) -> list[str]:
    # UTF-8 bytes conservatively bound byte/subword tokens; reserve provider instructions.
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for char in text:
        width = len(char.encode())
        if size + width > budget:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(char)
        size += width
    return [*chunks, "".join(current)]


def _pool(vectors: list[list[float]], weights: list[int]) -> list[float]:
    if len(vectors) == 1:
        return vectors[0]
    pooled = [
        sum(value * max(weight, 1) for value, weight in zip(column, weights, strict=True))
        for column in zip(*vectors, strict=True)
    ]
    norm = math.sqrt(sum(value * value for value in pooled)) or 1
    return [value / norm for value in pooled]
