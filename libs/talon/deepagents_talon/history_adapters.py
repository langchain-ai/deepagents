"""Owned embedding clients and bounded, provider-neutral preprocessing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from deepagents_talon.config import TalonConfigError
from deepagents_talon.history_drivers import load_driver

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine
    from types import ModuleType

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.history_profiles import EmbeddingProfile

_PREFIX = "DEEPAGENTS_TALON_HISTORY_EMBED_"
QUERY_EMBEDDING: ContextVar[bool] = ContextVar("talon_history_query", default=False)
EMBEDDING_CACHE: ContextVar[dict[tuple[bool, str], list[float]] | None] = ContextVar(
    "talon_history_embeddings", default=None
)
_REQUEST_TIMEOUT = 30
# A Store worker thread must never wait on the event loop indefinitely, so the
# bridge polls for a loop that has stopped and enforces an overall deadline.
_BRIDGE_TIMEOUT_SECONDS = 300
_BRIDGE_POLL_SECONDS = 1

logger = logging.getLogger(__name__)


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
        # Registered here rather than per adapter: `atlas` and `voyage` previously
        # had no cleanup at all, and a reindex reopens the archive, so a client that
        # keeps its pool past close leaks sockets on every cycle.
        _register_close(stack, embed)
        if profile.client_side:
            embed = BoundedEmbeddings(embed, profile)
        yield replace(profile, embed=embed)


async def _close_client(closer: Callable[[], object]) -> None:
    # A synchronous close tears down sockets, so it runs off the event loop; an
    # async one only builds its coroutine there and is awaited here.
    result = await asyncio.to_thread(closer)
    if inspect.isawaitable(result):
        await result


def _register_close(stack: AsyncExitStack, embed: object) -> None:
    for name in ("aclose", "close"):
        closer = getattr(embed, name, None)
        if callable(closer):
            stack.push_async_callback(_close_client, closer)
            return


def _driver(module: str, extra: str) -> ModuleType:
    return load_driver(module, extra, "History embeddings require")


async def _adapter(
    config: TalonConfig, profile: EmbeddingProfile, stack: AsyncExitStack
) -> Embeddings:
    if profile.adapter == "local":
        _driver("sentence_transformers", "history-local")
        from deepagents_talon.history_embeddings import HistoryEmbeddings  # noqa: PLC0415

        return HistoryEmbeddings(
            model=profile.model,
            max_input_tokens=profile.max_input_tokens,
            batch_size=profile.batch_size,
            query_prompt="",
        )
    if profile.adapter == "atlas":
        driver = _driver("langchain_mongodb.embeddings", "mongodb")
        return driver.AutoEmbeddings(model=profile.model)
    if profile.adapter == "voyage":
        _driver("langchain_voyageai", "history-voyage")
        from deepagents_talon.history_voyage import HistoryVoyageEmbeddings  # noqa: PLC0415

        return HistoryVoyageEmbeddings(
            model=profile.model,
            api_key=_key(config, "VOYAGE_API_KEY"),
            # Omitting the width leaves the model's native output, which is what
            # SEND_DIMENSIONS=0 asks for when a model cannot resize.
            output_dimension=cast("Literal[256, 512, 1024, 2048] | None", profile.dims)
            if profile.send_dimensions
            else None,
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

    host = urlsplit(profile.base_url).hostname or ""
    key = "OPENROUTER_API_KEY" if host.lower() == "openrouter.ai" else "OPENAI_API_KEY"
    sync = stack.enter_context(httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=False))
    asynchronous = await stack.enter_async_context(
        httpx.AsyncClient(timeout=_REQUEST_TIMEOUT, follow_redirects=False)
    )
    return driver.OpenAIEmbeddings(
        model=profile.model,
        dimensions=profile.dims if profile.send_dimensions else None,
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
        self.query_slot = asyncio.Semaphore(1)
        self.reported_pooling = False

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
        future = asyncio.run_coroutine_threadsafe(operation, self.loop)
        deadline = time.monotonic() + _BRIDGE_TIMEOUT_SECONDS
        while True:
            try:
                return future.result(timeout=_BRIDGE_POLL_SECONDS)
            except TimeoutError:
                # A loop that has stopped will never run the coroutine, and waiting on
                # it holds the Store's worker thread open through its own shutdown.
                if self.loop.is_closed() or not self.loop.is_running():
                    future.cancel()
                    msg = "History embeddings stopped; the event loop is no longer running"
                    raise RuntimeError(msg) from None
                if time.monotonic() >= deadline:
                    future.cancel()
                    msg = "History embedding exceeded its deadline waiting for the event loop"
                    raise RuntimeError(msg) from None

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
        if len(query.encode()) > self.profile.input_budget:
            msg = "History embedding query exceeds the configured input budget"
            raise ValueError(msg)
        async with self.query_slot, asyncio.timeout(_REQUEST_TIMEOUT):
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
        chunks = [_chunks(text, self.profile.input_budget) for text in texts]
        self._report_pooling(chunks)
        flattened = [chunk for document in chunks for chunk in document]
        vectors = await self._documents(flattened)
        output: list[list[float]] = []
        cursor = 0
        for document in chunks:
            group = vectors[cursor : cursor + len(document)]
            output.append(_pool(group, [len(chunk.encode()) for chunk in document]))
            cursor += len(document)
        return output

    def _report_pooling(self, chunks: list[list[str]]) -> None:
        """Announce, once, that documents no longer fit one request and are averaged.

        Pooled documents are compared against unpooled queries, so retrieval quality
        degrades quietly. Raising the token budget or the byte/token ratio removes it.
        """
        widest = max((len(document) for document in chunks), default=1)
        if widest > 1 and not self.reported_pooling:
            self.reported_pooling = True
            logger.warning(
                "History documents exceed the %d-byte embedding budget and are averaged "
                "across up to %d requests; raise %sMAX_INPUT_TOKENS or %sBYTES_PER_TOKEN "
                "to embed them whole",
                self.profile.input_budget,
                widest,
                _PREFIX,
                _PREFIX,
            )

    async def _documents(self, texts: list[str]) -> list[list[float]]:
        size = self.profile.batch_size
        output: list[list[float]] = []
        concurrency = self.profile.concurrency
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
        if len(vectors) != count or not all(
            all(math.isfinite(value) for value in vector) for vector in vectors
        ):
            msg = "History embedding response has an invalid count or values"
            raise ValueError(msg)
        widths = {len(vector) for vector in vectors}
        if widths - {self.profile.dims}:
            msg = (
                f"History embedding provider returned {sorted(widths)}-dimensional vectors "
                f"but {_PREFIX}DIMS is {self.profile.dims}; set DIMS to the model's native "
                f"width, or {_PREFIX}SEND_DIMENSIONS=0 if the model cannot resize output"
            )
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
