"""Durable background indexing and chat-scoped hybrid archive retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langgraph.store.base import PutOp, SearchItem, SearchOp

from deepagents_talon.archive import (
    ArchiveScope,
    SearchPage,
    SearchVisibility,
    SemanticStatus,
    _indexing_status,
    _search_page,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.store.base import BaseStore, Op, Result

    from deepagents_talon.history_index import Row, VectorArchive
    from deepagents_talon.history_profiles import EmbeddingProfile

logger = logging.getLogger(__name__)
_CANDIDATES = 100
_MAX_SEARCH_PAGES = 32
_BATCH_SIZE = 4
_RETRY_SECONDS = 30
_SEARCH_TIMEOUT_SECONDS = 2
# One indexing batch embeds up to 500 chunks in sequential provider requests, each
# already bounded by the adapter's own request timeout, so this only has to catch a
# Store write that never returns at all.
_BATCH_TIMEOUT_SECONDS = 300
# Unacknowledged work is retried from durable state on the next start, so abandoning
# a wedged batch at shutdown costs a repeat, never data.
_CLOSE_TIMEOUT_SECONDS = 10


@dataclass
class _SearchSnapshot:
    query: tuple[str, str, str]
    keys: list[str]
    status: SemanticStatus
    pending: bool


class HistoryVectorIndex:
    """Coordinate a caller-owned Store with the authoritative transcript archive.

    Args:
        archive: Transcript archive owning the durable indexing queue.
        store: Initialized Store configured for semantic indexing of `text`.
        search_visibility: Whether acknowledged Store writes are immediately searchable.
        indexing: Whether to embed chunks or only process pending deletions.
        profile: Embedding batch and concurrency limits when configured.
    """

    def __init__(
        self,
        archive: VectorArchive,
        store: BaseStore,
        *,
        indexing: bool = True,
        profile: EmbeddingProfile | None = None,
        search_visibility: SearchVisibility = "unknown",
    ) -> None:
        """Keep indexing separate from checkpoint persistence."""
        self.profile = profile
        # Indexing needs no permit of its own: every non-query caller holds `self.lock`
        # for the whole batch, so at most one is ever outstanding. Provider concurrency
        # is bounded by `BoundedEmbeddings.slots` instead.
        self._query_slots = asyncio.Semaphore(1)
        self._pending: set[asyncio.Task[list[Result]]] = set()
        self.search_visibility = search_visibility
        self.indexing = indexing
        self.archive = archive
        self.store = store
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.identity = ""
        self.stopping = False
        self._pages: OrderedDict[str, _SearchSnapshot] = OrderedDict()

    async def start(self) -> None:
        """Recover pending deletions and reconcile existing chunks in bounded batches."""
        self.identity = await self.archive.prepare()
        self.task = asyncio.create_task(self._run(), name="talon-history-index")

    async def close(self) -> None:
        """Finish the active batch before the caller closes database connections.

        A Store write that never returns must not hold host shutdown open forever, so
        the batch is abandoned once the deadline passes and retried on the next start.
        """
        self.stopping = True
        self.wake.set()
        tasks = [task for task in (self.task, *self._pending) if task is not None]
        if not tasks:
            return
        _, unfinished = await asyncio.wait(tasks, timeout=_CLOSE_TIMEOUT_SECONDS)
        if unfinished:
            logger.warning(
                "History vector indexing did not stop within %ss; abandoning the active batch "
                "for the next start to retry",
                _CLOSE_TIMEOUT_SECONDS,
            )
            for task in unfinished:
                task.cancel()
            await asyncio.gather(*unfinished, return_exceptions=True)
        if self.task is not None and not self.task.cancelled() and (error := self.task.exception()):
            raise error

    def namespace(self, channel: str, chat: str) -> tuple[str, ...]:
        """Build a collision-resistant namespace without Store-specific escaping.

        Args:
            channel: Trusted channel identifier.
            chat: Trusted chat identifier.
        """
        return ("talon_history", self.identity, _digest(channel), _digest(chat))

    async def _batch(self, operations: Sequence[Op], *, query: bool = False) -> list[Result]:
        slots = self._query_slots if query else None
        if slots is not None:
            await slots.acquire()
        task = asyncio.create_task(self._call_store(operations, slots))
        self._pending.add(task)
        task.add_done_callback(self._finished)
        if query:
            # Indexing writes are shielded so a cancelled caller cannot leave a
            # partial batch, but a search that timed out has no result worth
            # keeping. Awaiting it directly cancels the request and frees the
            # single query permit, instead of holding it until the provider's own
            # timeout and stalling every later search behind it.
            return await task
        return await asyncio.shield(task)

    async def _call_store(
        self, operations: Sequence[Op], slots: asyncio.Semaphore | None
    ) -> list[Result]:
        try:
            async with asyncio.timeout(_BATCH_TIMEOUT_SECONDS):
                return await self.store.abatch(operations)
        finally:
            if slots is not None:
                slots.release()

    def _finished(self, task: asyncio.Task[list[Result]]) -> None:
        self._pending.discard(task)
        if not task.cancelled():
            task.exception()

    async def _rows(self, session: str = "") -> list[Row]:
        size = (
            min(500, self.profile.batch_size * self.profile.concurrency)
            if self.profile
            else _BATCH_SIZE
        )
        return await self.archive.rows(session, indexing=self.indexing, limit=size)

    async def _process(self, rows: list[Row]) -> None:
        operations = self._operations(rows)
        if operations:
            await self._batch(operations)
        await self.archive.acknowledge(rows)

    def _operations(self, rows: list[Row]) -> list[PutOp]:
        return [
            PutOp(
                self.namespace(channel, chat),
                str(identifier),
                None if deleted else {"text": text, "session_id": session},
                index=["text"],
            )
            for identifier, channel, chat, session, deleted, text in rows
        ]

    async def _run(self) -> None:
        failures = 0
        while not self.stopping:
            self.wake.clear()
            try:
                async with self.lock:
                    rows = await self._rows()
                    await self._process(rows)
                    failures = 0
                if rows:
                    await asyncio.sleep(0)
                    continue
            except Exception as error:  # noqa: BLE001  # Optional indexing must not stop conversation writes.
                failures += 1
                delay = _retry_delay(error, failures)
                # Without the cause a permanently broken backend repeats an identical
                # line forever; the adapter's dimension and credential errors name the
                # exact settings that fix them.
                logger.warning(
                    "History vector indexing failed; retrying in %.0fs", delay, exc_info=error
                )
            if failures:
                await self._backoff(delay)
            else:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.wake.wait(), timeout=_RETRY_SECONDS)

    async def _backoff(self, delay: float) -> None:
        deadline = asyncio.get_running_loop().time() + delay
        while not self.stopping:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=remaining)
            except TimeoutError:
                return

    async def delete_session(self, session: str) -> None:
        """Remove vectors before deleting transcript ownership, allowing retries.

        Args:
            session: Trusted session identifier to erase.
        """
        async with self.lock:
            await self.archive.mark_deleted(session)
            while rows := await self._rows(session):
                await self._process(rows)
            await self.archive.delete_text(session)

    async def search_page(
        self, scope: ArchiveScope, query: str, after: str, limit: int
    ) -> SearchPage:
        """Return stable ranked pages with explicit fallback and coverage metadata.

        Args:
            scope: Trusted chat scope.
            query: Search text.
            after: Opaque continuation token from the previous page.
            limit: Maximum number of results.
        """
        cache_key = (scope["talon_history_channel"], scope["talon_history_chat"], query)
        pending = await self.archive.pending(scope)
        identifier, _, cursor = after.partition(":")
        snapshot = self._pages.get(identifier) if after else None
        if after and (
            snapshot is None or snapshot.query != cache_key or cursor not in snapshot.keys
        ):
            return _search_page(
                [],
                limit,
                "not_requested",
                pending=pending,
                expired=True,
            )
        if snapshot is None:
            semantic, status = await self._semantic(scope, query)
            snapshot = _SearchSnapshot(
                cache_key,
                _fuse(await self.archive.lexical(scope, query, _CANDIDATES), semantic),
                status,
                pending,
            )
            identifier = uuid4().hex
            self._pages[identifier] = snapshot
        self._pages.move_to_end(identifier)
        if len(self._pages) > _MAX_SEARCH_PAGES:
            self._pages.popitem(last=False)
        hits = await self.archive.ranked(scope, snapshot.keys, int(cursor or 0), limit + 1)
        page = _search_page(
            hits,
            limit,
            snapshot.status,
            pending=pending or snapshot.pending,
        )
        page["indexing_status"] = _indexing_status(
            snapshot.status, pending=page["indexing_pending"], visibility=self.search_visibility
        )
        if page["next_after"] is not None:
            page["next_after"] = f"{identifier}:{page['next_after']}"
        return page

    async def _semantic(self, scope: ArchiveScope, query: str) -> tuple[list[str], SemanticStatus]:
        try:
            if (
                self.profile
                and not self.profile.client_side
                and len(query.encode()) > self.profile.input_budget
            ):
                return [], "error"
            # Remote adapters hold a dedicated query slot, so a search never waits on
            # indexing whatever the configured concurrency. Local inference is
            # single-flight, so the fair lock orders a waiting query ahead of the
            # next indexing batch. The deadline covers waiting and execution alike.
            guard = nullcontext() if self.profile and self.profile.adapter != "local" else self.lock
            async with asyncio.timeout(_SEARCH_TIMEOUT_SECONDS), guard:
                results = await self._batch(
                    [
                        SearchOp(
                            self.namespace(
                                scope["talon_history_channel"], scope["talon_history_chat"]
                            ),
                            query=query,
                            limit=_CANDIDATES,
                        )
                    ],
                    query=True,
                )
            items = cast("list[SearchItem]", results[0])
            keys = [item.key for item in items if item.score is not None]
        except TimeoutError:
            logger.warning(
                "History vector search timed out after %ss; using keyword search",
                _SEARCH_TIMEOUT_SECONDS,
            )
            return [], "timeout"
        except Exception:  # noqa: BLE001  # Search remains usable without the optional backend.
            logger.warning("History vector search unavailable; using keyword search", exc_info=True)
            return [], "error"
        else:
            return keys, "unavailable" if items and not keys else "completed"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fuse(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def _retry_delay(error: Exception, failures: int) -> float:
    delay = min(300, _RETRY_SECONDS * 2 ** min(failures - 1, 4))
    errors = error.exceptions if isinstance(error, ExceptionGroup) else [error]
    for failure in errors:
        if isinstance(failure, ExceptionGroup):
            delay = max(delay, _retry_delay(failure, failures))
            continue
        headers = getattr(failure, "headers", None) or getattr(
            getattr(failure, "response", None), "headers", None
        )
        if isinstance(headers, Mapping):
            with suppress(TypeError, ValueError):
                delay = max(delay, min(3600, float(headers.get("retry-after", 0))))
    return delay + secrets.randbelow(1001) / 1000
