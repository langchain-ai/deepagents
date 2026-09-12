"""Ordered, recoverable records using the portable LangGraph Store contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine

    from langgraph.store.base import BaseStore

    from deepagents_talon.archive import ArchiveScope

Record = dict[str, object]
Write = tuple[str, Record | None]
_MAX_WRITES = 12


def digest(*values: str) -> str:
    """Hash identifiers without relying on backend namespace escaping."""
    return hashlib.sha256(json.dumps(values, ensure_ascii=True).encode()).hexdigest()


def number(record: Record, key: str) -> int:
    """Read a numeric ordering field from a versioned archive record."""
    return cast("int", record.get(key, 0))


def scope_key(scope: ArchiveScope) -> str:
    """Encode a trusted chat scope as a backend-independent lookup key."""
    return "scope:" + digest(scope["talon_history_channel"], scope["talon_history_chat"])


async def finish[T](operation: Coroutine[object, object, T]) -> T:
    """Finish storage mutations before propagating cancellation to the caller."""
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class StoreRecords:
    """Bounded redo journal for one active archive writer per namespace.

    The Store must provide read-after-write consistency and no automatic TTL.
    A journal makes partial batches restartable; it is not a distributed lock.
    """

    def __init__(self, store: BaseStore, namespace: tuple[str, ...]) -> None:
        """Keep caller ownership of the unindexed metadata Store."""
        if not namespace or any(not part for part in namespace):
            msg = "Archive namespace must contain nonempty components"
            raise ValueError(msg)
        self.store = store
        self.namespace = ("talon_archive_v1", digest(*namespace))
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def access(self) -> AsyncIterator[None]:
        """Serialize access and recover interrupted writes before exposing records."""
        async with self.lock:
            await finish(self.recover())
            yield

    async def get(self, key: str) -> Record | None:
        """Read an authoritative record without refreshing any TTL."""
        item = await self.store.aget(self.namespace, key, refresh_ttl=False)
        return None if item is None else cast("Record", item.value)

    async def root(self) -> Record:
        """Read the versioned sequence and session registry."""
        root = await self.get("root")
        if root is None:
            root = {"version": 1, "last": 0}
        if root.get("version") != 1:
            msg = "Unsupported conversation archive format"
            raise ValueError(msg)
        return {"identity": uuid4().hex, "vectors": False, "deletions": 0, **root}

    async def commit(self, writes: list[Write]) -> None:
        """Durably record a bounded mutation before applying its idempotent writes."""
        if len(writes) > _MAX_WRITES:
            msg = "Archive mutation exceeds the bounded journal size"
            raise ValueError(msg)
        await finish(self._commit(writes))

    async def _commit(self, writes: list[Write]) -> None:
        await self.store.aput(self.namespace, "journal", {"writes": writes}, index=False, ttl=None)
        await self.recover()

    async def recover(self) -> None:
        """Replay a partially applied mutation; clear its journal only on success."""
        journal = await self.get("journal")
        if journal is None:
            return
        for key, value in cast("list[Write]", journal["writes"]):
            if value is None:
                await self.store.adelete(self.namespace, key)
            else:
                await self.store.aput(self.namespace, key, value, index=False, ttl=None)
        await self.store.adelete(self.namespace, "journal")

    async def chain(self, cursor: int, link: str) -> AsyncIterator[tuple[int, Record]]:
        """Follow explicit ordering links without relying on backend search order."""
        while cursor:
            record = await self.get(str(cursor))
            if record is None:
                msg = "Conversation archive contains a missing ordering record"
                raise RuntimeError(msg)
            yield cursor, record
            previous = int(cast("int", record.get(link, 0)))
            if (
                previous < 0
                or previous == cursor
                or (previous and (previous > cursor) != (link == "next_session"))
            ):
                msg = "Conversation archive contains an invalid ordering link"
                raise RuntimeError(msg)
            cursor = previous
