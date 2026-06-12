from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.data_lifecycle import cleanup_sensitive_state


def test_cleanup_sensitive_state_prunes_cron_and_inbound_media(tmp_path) -> None:
    now = datetime(2026, 1, 31, 12, tzinfo=UTC)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_CRON_RETENTION_DAYS": "30",
            "DEEPAGENTS_TALON_INBOUND_MEDIA_RETENTION_HOURS": "24",
        },
        base_home=tmp_path,
    )
    config.ensure_home()
    store = CronJobStore(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    job = store.create_job(
        prompt="expired",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now - timedelta(days=40),
    )
    store.advance_next_run(job.id, now=now - timedelta(days=39))
    store.mark_job_run(job.id, status="ok", now=now - timedelta(days=39))

    expired = config.inbound_media_dir / "old.ogg"
    fresh = config.inbound_media_dir / "new.ogg"
    expired.write_bytes(b"old")
    fresh.write_bytes(b"new")
    expired_time = (now - timedelta(hours=25)).timestamp()
    fresh_time = (now - timedelta(hours=1)).timestamp()
    os.utime(expired, (expired_time, expired_time))
    os.utime(fresh, (fresh_time, fresh_time))

    report = cleanup_sensitive_state(config=config, cron_store=store, now=now)

    assert [removed.id for removed in report.removed_cron_jobs] == [job.id]
    assert report.removed_media_files == (expired,)
    assert store.list_jobs() == []
    assert not expired.exists()
    assert fresh.exists()
