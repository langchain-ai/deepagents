from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

from deepagents_talon.cron.scheduler import PersistentCronScheduler
from tests.cron.test_until import NOON, _hourly_until, _Runner, _store

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.cron.jobs import CronJob


async def test_restarted_worker_coalesces_overlapping_checks(tmp_path: Path) -> None:
    job = _hourly_until(_store(tmp_path), NOON + timedelta(days=1))
    now = NOON + timedelta(hours=6)
    entered = asyncio.Event()
    release = asyncio.Event()
    runner = _Runner()

    async def run(claimed: CronJob) -> str:
        entered.set()
        await release.wait()
        return await runner.run(claimed)

    scheduler = PersistentCronScheduler(
        store=_store(tmp_path), run_job=run, deliver_result=runner.deliver, now=lambda: now
    )
    first = asyncio.create_task(scheduler.tick_once())
    await asyncio.wait_for(entered.wait(), 2)
    now += timedelta(hours=1)
    try:
        await asyncio.wait_for(scheduler.tick_once(), 2)
    finally:
        release.set()
        await first
    assert runner.ran == [job.id]
    await scheduler.tick_once()
    assert runner.ran == [job.id, job.id]
