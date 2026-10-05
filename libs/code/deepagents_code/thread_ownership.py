"""Local thread reservations and fencing for independent checkpoint writers."""

from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, override
from uuid import uuid4

from filelock import FileLock, SoftFileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Sequence
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
    )
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class ThreadOwnershipError(RuntimeError):
    """A checkpoint writer no longer owns its thread reservation."""


OWNER_KEY = "x-dcode-thread-owner"
_registry: dict[tuple[int, str], ThreadLease] = {}
_registry_lock = RLock()


def _paths(thread_id: str, db_path: Path) -> Path:
    directory = db_path.resolve().with_name(f"{db_path.name}.owners")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(thread_id.encode()).hexdigest()
    return directory / name


def _lock(path: Path, suffix: str) -> FileLock:
    return FileLock(f"{path}.{suffix}", timeout=0, thread_local=False, mode=0o600)


def _acquire(lock: FileLock) -> None:
    lock.acquire(timeout=0)
    if isinstance(lock, SoftFileLock):
        lock.release()
        msg = "Thread ownership requires operating-system file locks."
        raise ThreadOwnershipError(msg)


@dataclass
class ThreadLease:
    """Keep a local thread reserved until its client releases it or exits."""

    thread_id: str
    token: str
    _reservation: FileLock
    _key: tuple[int, str]

    async def rotate(self) -> None:
        """Invalidate previous writers without releasing the client reservation."""
        from pathlib import Path

        path = Path(self._key[1])
        async with _mutation_gate(path):
            await _finish_mutation(asyncio.to_thread(self._rotate, path))

    def _rotate(self, path: Path) -> None:
        with _registry_lock:
            if _registry.get(self._key) is not self:
                msg = "Cannot rotate a released thread reservation."
                raise ThreadOwnershipError(msg)
            token = uuid4().hex
            path.with_suffix(".owner").write_text(token, encoding="utf-8")
            self.token = token

    def release(self) -> None:
        """Release this reservation without deleting its lock path."""
        with _registry_lock:
            if _registry.get(self._key) is self:
                del _registry[self._key]
            self._reservation.release()


def try_acquire(thread_id: str, *, db_path: Path | None = None) -> ThreadLease | None:
    """Reserve an unowned thread atomically.

    Returns:
        A lifetime lease, or None when the thread is already occupied.
    """
    if db_path is None:
        from deepagents_code.sessions import get_db_path

        db_path = get_db_path()
    path = _paths(thread_id, db_path)
    gate = _lock(path, "writer")
    reservation = _lock(path, "client")
    try:
        _acquire(gate)
    except Timeout:
        return None
    try:
        with _registry_lock:
            lease = _reserve(thread_id, path, reservation)
            if lease is not None:
                _registry[lease._key] = lease
            return lease
    finally:
        gate.release()


def _reserve(thread_id: str, path: Path, reservation: FileLock) -> ThreadLease | None:
    try:
        _acquire(reservation)
    except Timeout:
        return None
    try:
        token = uuid4().hex
        path.with_suffix(".owner").write_text(token, encoding="utf-8")
        return ThreadLease(thread_id, token, reservation, (os.getpid(), str(path)))
    except BaseException:
        reservation.release()
        raise


def ensure_owned(thread_id: str, *, db_path: Path | None = None) -> ThreadLease:
    """Return a reservation, raising ThreadOwnershipError if already occupied.

    Raises:
        ThreadOwnershipError: Another client owns the thread.
    """
    if db_path is None:
        from deepagents_code.sessions import get_db_path

        db_path = get_db_path()
    key = (os.getpid(), str(_paths(thread_id, db_path)))
    with _registry_lock:
        lease = _registry.get(key) or try_acquire(thread_id, db_path=db_path)
        if lease is None:
            msg = f"Thread {thread_id} is open elsewhere. Close it there to resume."
            raise ThreadOwnershipError(msg)
        return lease


def held_lease(thread_id: str, *, db_path: Path | None = None) -> ThreadLease | None:
    """Look up a live local reservation without acquiring or renewing it.

    Returns:
        The held reservation, or None if this process does not own the thread.
    """
    if db_path is None:
        from deepagents_code.sessions import get_db_path

        db_path = get_db_path()
    key = (os.getpid(), str(_paths(thread_id, db_path)))
    with _registry_lock:
        return _registry.get(key)


def release_all() -> None:
    """Release this process's reservations after its writable sessions stop."""
    with _registry_lock:
        for key, lease in tuple(_registry.items()):
            if key[0] == os.getpid():
                lease.release()


def _validate(path: Path, token: str) -> None:
    try:
        current = path.with_suffix(".owner").read_text(encoding="utf-8")
    except FileNotFoundError:
        current = None
    if not token or current != token:
        msg = "Thread ownership changed; this server cannot write to it."
        raise ThreadOwnershipError(msg)
    reservation = _lock(path, "client")
    try:
        _acquire(reservation)
    except Timeout:
        return
    reservation.release()
    msg = "The thread's client exited or released ownership; refusing a stale write."
    raise ThreadOwnershipError(msg)


@asynccontextmanager
async def writer_guard(
    thread_id: str, *, db_path: Path, token: str
) -> AsyncIterator[None]:
    """Fence a database mutation against both live reservations and new owners."""
    path = await asyncio.to_thread(_paths, thread_id, db_path)
    async with _mutation_gate(path):
        await _finish_mutation(asyncio.to_thread(_validate, path, token))
        yield


@asynccontextmanager
async def _mutation_gate(path: Path) -> AsyncIterator[None]:
    gate = _lock(path, "writer")
    try:
        while True:
            if await _finish_mutation(asyncio.to_thread(_try_acquire_gate, gate)):
                break
            await asyncio.sleep(0.01)
        yield
    finally:
        await _finish_mutation(asyncio.to_thread(gate.release))


def _try_acquire_gate(gate: FileLock) -> bool:
    try:
        _acquire(gate)
    except Timeout:
        return False
    return True


async def _finish_mutation[T](operation: Awaitable[T]) -> T:
    task = asyncio.ensure_future(operation)
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


def owned_saver_class(*, db_path: Path) -> type[AsyncSqliteSaver]:
    """Build an ownership-fenced SQLite saver.

    Returns:
        A saver class that rejects mutations from stale client processes.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    class OwnedSqliteSaver(AsyncSqliteSaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            async with writer_guard(
                str(config["configurable"]["thread_id"]),
                db_path=db_path,
                token=str(config.get("configurable", {}).get(OWNER_KEY, "")),
            ):
                saved = await _finish_mutation(
                    super().aput(config, checkpoint, metadata, new_versions)
                )
                saved["configurable"][OWNER_KEY] = config["configurable"][OWNER_KEY]
                return saved

        async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, object]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            async with writer_guard(
                str(config["configurable"]["thread_id"]),
                db_path=db_path,
                token=str(config.get("configurable", {}).get(OWNER_KEY, "")),
            ):
                await _finish_mutation(
                    super().aput_writes(config, writes, task_id, task_path)
                )

        @override
        async def adelete_thread(self, thread_id: str) -> None:
            lease: ThreadLease | None = None

            def reserve() -> None:
                nonlocal lease
                lease = try_acquire(thread_id, db_path=db_path)

            try:
                await _finish_mutation(asyncio.to_thread(reserve))
                if lease is None:
                    msg = (
                        f"Thread {thread_id} is open elsewhere. "
                        "Close it before deleting."
                    )
                    raise ThreadOwnershipError(msg)
                async with writer_guard(thread_id, db_path=db_path, token=lease.token):
                    await _finish_mutation(super().adelete_thread(thread_id))
            finally:
                if lease is not None:
                    await _finish_mutation(asyncio.to_thread(lease.release))

    return OwnedSqliteSaver
