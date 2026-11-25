from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.journal import JobJournal, ObjectiveReport

if TYPE_CHECKING:
    from pathlib import Path

ORIGIN = CronOrigin(channel="slack", conversation_id="C1:thread", sender_id="alice")


def test_unchanged_polls_back_off_until_progress_or_timed_event() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    proposed = now + timedelta(minutes=15)
    previous = {"status": "checked", "started": (now - timedelta(minutes=20)).isoformat()}
    report = ObjectiveReport(
        status="checked", summary="Same head", next_wake_at=proposed.isoformat()
    )
    assert report.next_wake(previous, now, None) == now + timedelta(minutes=40)
    deadline = now + timedelta(minutes=30)
    assert report.next_wake(previous, now, deadline) == deadline
    for reason in ("owner_timing", "expected_event"):
        assert (
            report.model_copy(update={"wake_reason": reason}).next_wake(previous, now, None)
            == proposed
        )
    assert (
        report.model_copy(update={"status": "partial"}).next_wake(previous, now, None) == proposed
    )
    assert report.next_wake(dict(previous, status="partial"), now, None) == proposed


def test_malformed_reference_preserves_valid_progress_across_checks(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(
        prompt="Verify build and release",
        mode="objective",
        origin=ORIGIN,
        completion_criteria=["build", "release"],
    )
    journal = JobJournal(tmp_path)
    journal.start(job, "first")
    report = ObjectiveReport.from_reply(
        json.dumps(
            {
                "status": "complete",
                "summary": {"result": "Inspected both runs."},
                "wake_reason": None,
                "evidence": {"build": "run 29", "release": {"notes": "needs verification"}},
            }
        )
    )
    assert journal.finish_check(job, "first", report, set()).status == "partial"
    assert JobJournal(tmp_path).verification(job)[1] == {
        "build": {"reference": "run 29", "verified": False}
    }
    assert "Supply text references for release" in journal.history(job.id, ORIGIN)[0]["summary"]


def test_blocked_run_saves_verified_milestones_after_required_reads(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(
        prompt="Verify the build and release",
        mode="objective",
        origin=ORIGIN,
        completion_criteria=["build", "release"],
        required_sources=["path:/builds"],
    )
    journal = JobJournal(tmp_path)
    report = ObjectiveReport(
        status="blocked", summary="Release awaits signing.", evidence={"build": "build 29"}
    )
    for run, reads in [("first", set()), ("second", {"path:/builds"})]:
        journal.start(job, run)
        assert journal.finish_check(job, run, report, reads).status == "blocked"
        assert JobJournal(tmp_path).verification(job)[1] == (
            {"build": {"reference": "build 29", "verified": False}} if reads else {}
        )


def test_ongoing_milestones_and_legacy_criteria_are_durable(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(mode="objective", origin=ORIGIN, prompt="Maintain detections")
    journal = JobJournal(tmp_path)
    for run, criteria in [("first", []), ("second", ["build"])]:
        journal.start(job, run)
        report = ObjectiveReport(
            status="complete",
            summary="Passed",
            completion_criteria=criteria,
            evidence={"build": "build 27"},
        )
        assert journal.finish_check(job, run, report, set()).status == "partial"
    assert JobJournal(tmp_path).verification(job) == (
        (),
        {"build": {"reference": "build 27", "verified": False}},
    )
    edited = store.edit_job(job.id, origin=ORIGIN, completion_criteria=["ship"])
    assert journal.verification(edited) == (("ship",), {})


def test_partial_and_unmatched_references_cannot_complete_objective(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(
        mode="objective", origin=ORIGIN, prompt="Test login", completion_criteria=["Real login"]
    )
    journal = JobJournal(tmp_path)
    journal.start(job, "mock")
    report = ObjectiveReport(status="partial", summary="Mock passed", evidence={"C1": "mock log"})
    journal.finish_check(job, "mock", report, set())
    journal.start(job, "retry")
    report = ObjectiveReport(status="complete", summary="Done", evidence={"login test": "run 23"})
    outcome = journal.finish_check(job, "retry", report, set())
    assert outcome.status == "partial"
    assert "login test: run 23" in outcome.summary
    journal.start(job, "quiet")
    journal.append_work("quiet", "Checked issuer logs. Real login is pending.")
    report = ObjectiveReport(status="complete", summary="[SILENT]")
    outcome = journal.finish_check(job, "quiet", report, set())
    assert outcome.status == "partial"
    assert "Checked issuer logs" in journal.history(job.id, ORIGIN)[0]["summary"]
    assert "[SILENT]" not in outcome.summary


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
