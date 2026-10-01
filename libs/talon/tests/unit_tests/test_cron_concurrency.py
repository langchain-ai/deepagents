"""Concurrent callers must preserve committed cron records."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier, Event, local
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.cron.errors import CronJobError
from deepagents_talon.cron.jobs import CronJob, CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.tools import CronTools

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from threading import RLock
    from types import TracebackType

NOW = datetime(2026, 1, 1, tzinfo=UTC)
ORIGIN = CronOrigin(conversation_id="chat", channel="test")


class _ObservedLock:
    def __init__(self, lock: RLock, attempted: Event, contender: local) -> None:
        self.lock = lock
        self.attempted = attempted
        self.contender = contender

    def __enter__(self) -> None:
        if getattr(self.contender, "active", False):
            self.contender.active = False
            if self.lock.acquire(blocking=False):
                self.lock.release()
                msg = "The paused writer must still hold the store lock"
                raise AssertionError(msg)
            self.attempted.set()
        assert self.lock.acquire(timeout=3)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lock.release()


def _store(path: Path) -> CronJobStore:
    return CronJobStore(assistant_id="agent", cron_dir=path)


def _create(store: CronJobStore, prompt: str = "job") -> CronJob:
    return store.create_job(
        prompt=prompt,
        schedule=CronSchedule.parse("every 1m"),
        repeat_times=2,
        origin=ORIGIN,
        now=NOW,
    )


def _overlap[First, Second](
    monkeypatch: pytest.MonkeyPatch,
    store: CronJobStore,
    first: Callable[[], First],
    second: Callable[[], Second],
    *,
    competitor: CronJobStore | None = None,
) -> tuple[First, Second]:
    """Hold a writer until its competitor encounters the held store lock."""
    writing, release, attempted = Event(), Event(), Event()
    contender = local()
    original = store._write_jobs

    def paused_write(jobs: list[CronJob]) -> None:
        if not writing.is_set():
            writing.set()
            assert release.wait(3)
        original(jobs)

    def competing() -> Second:
        contender.active = True
        return second()

    for current in {store, competitor or store}:
        monkeypatch.setattr(current, "_lock", _ObservedLock(current._lock, attempted, contender))
    monkeypatch.setattr(store, "_write_jobs", paused_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(first)
        try:
            assert writing.wait(3)
            other = pool.submit(competing)
            # The wrapper signals only after a failed nonblocking acquisition;
            # elapsed time never decides whether the callers actually overlap.
            assert attempted.wait(3)
        finally:
            release.set()
        return pending.result(timeout=3), other.result(timeout=3)


@pytest.mark.parametrize("separate_store", [False, True])
@pytest.mark.parametrize(
    "operation", ["create", "edit", "remove", "claim", "result", "discard", "prune"]
)
def test_mutations_preserve_concurrent_creation(tmp_path, monkeypatch, operation, separate_store):
    store = _store(tmp_path)
    other = _store(tmp_path) if separate_store else store
    job = _create(store)
    due = NOW + timedelta(minutes=1)
    if operation in {"result", "discard", "prune"}:
        store.advance_next_run(job.id, now=due)
    if operation in {"discard", "prune"}:
        store.advance_next_run(job.id, now=due + timedelta(minutes=1))
        store.mark_job_run(job.id, status="ok", now=due)
    other.list_jobs()  # Prime each instance's cache before the contested write.
    actions = {
        "create": lambda: _create(store, "first"),
        "edit": lambda: store.edit_job(job.id, origin=ORIGIN, enabled=False, prompt="edited"),
        "remove": lambda: store.remove_job(job.id, origin=ORIGIN),
        "claim": lambda: store.advance_next_run(job.id, now=due),
        "result": lambda: store.mark_job_run(job.id, status="ok", now=due),
        "discard": lambda: store.discard_finished(now=due),
        "prune": lambda: store.prune_completed(retain_for=timedelta(0), now=due),
    }
    result, created = _overlap(
        monkeypatch,
        store,
        actions[operation],
        lambda: _create(other, "second"),
        competitor=other,
    )
    fresh = _store(tmp_path).list_jobs()
    assert created in fresh
    if operation in {"remove", "discard", "prune"}:
        assert job.id not in {entry.id for entry in fresh}
    else:
        assert result in fresh
    assert store.list_jobs() == other.list_jobs() == fresh


@pytest.mark.parametrize("separate_store", [False, True])
def test_claim_is_exclusive_and_repeat_count_survives(tmp_path, monkeypatch, separate_store):
    store = _store(tmp_path)
    other = _store(tmp_path) if separate_store else store
    job = _create(store)
    due = NOW + timedelta(minutes=1)
    claimed, duplicate = _overlap(
        monkeypatch,
        store,
        lambda: store.advance_next_run(job.id, now=due),
        lambda: other.advance_next_run(job.id, now=due),
        competitor=other,
    )
    assert claimed is not None
    assert duplicate is None
    saved = _store(tmp_path).get_job(job.id)
    assert saved == claimed
    assert saved.repeat.completed == 1
    assert saved.next_run_at == due + timedelta(minutes=1)


def test_reader_observes_completed_write(tmp_path, monkeypatch):
    store = _store(tmp_path)
    job, observed = _overlap(monkeypatch, store, lambda: _create(store), store.list_jobs)
    assert observed == [job]
    assert observed == _store(tmp_path).list_jobs()


@pytest.mark.parametrize("failure", ["validation", "write"])
def test_failure_releases_lock_for_another_thread(tmp_path, monkeypatch, failure):
    store = _store(tmp_path)
    job = _create(store)
    with monkeypatch.context() as patch:
        if failure == "write":

            def fail_write(_jobs):
                msg = "disk failure"
                raise OSError(msg)

            patch.setattr(store, "_write_jobs", fail_write)
            with pytest.raises(OSError, match="disk failure"):
                _create(store)
        else:
            with pytest.raises(CronJobError):
                store.remove_job(job.id, origin=CronOrigin(conversation_id="wrong"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        created = pool.submit(_create, store).result(timeout=3)
    assert _store(tmp_path).list_jobs() == store.list_jobs() == [job, created]


async def test_async_langchain_creations_all_survive(tmp_path, monkeypatch):
    store = _store(tmp_path)
    cron = CronTools(store=store, origin=lambda: ORIGIN)
    tool = next(tool for tool in cron.as_langchain_tools() if tool.name == "create_job")
    barrier = Barrier(4, timeout=3)
    original = store.create_job

    def concurrent_create(**kwargs: object):
        barrier.wait()  # Synchronize executor callers before taking the store lock.
        return original(**kwargs)

    monkeypatch.setattr(store, "create_job", concurrent_create)
    results = await asyncio.wait_for(
        asyncio.gather(
            *(tool.ainvoke({"prompt": str(i), "schedule": "every 1m"}) for i in range(4))
        ),
        timeout=5,
    )
    assert {result["id"] for result in results} == {job.id for job in _store(tmp_path).list_jobs()}
    assert len(store.list_jobs()) == 4
    assert store.list_jobs() == _store(tmp_path).list_jobs()
