from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule

if TYPE_CHECKING:
    from pathlib import Path

ORIGIN = CronOrigin(channel="slack", conversation_id="C1:thread", sender_id="alice")


@pytest.mark.parametrize(
    ("prompt", "handoff"), [("Check all builds", None), (None, "Signing is pending.")]
)
def test_stale_completion_preserves_revised_objective(
    tmp_path: Path, prompt: str | None, handoff: str | None
) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(
        prompt="Check builds",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
    )
    if handoff:
        store.send_job_context(job.id, handoff, origin=ORIGIN)
    else:
        store.edit_job(job.id, origin=ORIGIN, prompt=prompt)
    revised = store.get_job(job.id)
    assert not store.apply_objective_result(job, wake=datetime.now(UTC) + timedelta(days=1))
    assert not store.apply_objective_result(job, complete=True)
    assert store.get_job(job.id) == revised
    assert store.apply_objective_result(revised, complete=True)
    assert store.get_job(job.id).next_run_at is None
