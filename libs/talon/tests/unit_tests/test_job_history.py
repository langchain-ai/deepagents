from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.journal import JobJournal, ObjectiveReport, record_job_work
from deepagents_talon.cron.scheduler import PersistentCronScheduler
from deepagents_talon.cron.scope import SCHEDULER_FINALIZING, JobScopeMiddleware
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult
from deepagents_talon.mcp_auth import _no_authorization_channel_error
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config
from tests.unit_tests.test_background import _request

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ORIGIN = CronOrigin(channel="slack", conversation_id="C1:thread", sender_id="alice")


def _attach_scheduler(
    host: TalonHost, store: CronJobStore, *, now: Callable[[], datetime] | None = None
) -> PersistentCronScheduler:
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(host.channels[0], job, text),
        now=now,
    )
    host.scheduler = scheduler
    return scheduler


class _WorkingGoalAgent(BlockingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.check_started = asyncio.Event()
        self.report = {
            "status": "partial",
            "summary": "Checked build 29. Signing is pending.",
            "notice": "Build 29 is unsigned.",
        }

    async def invoke(self, request: AgentRequest) -> AgentResult:
        if request.metadata.get("job_mode") != "objective":
            return await super().invoke(request)
        self.requests.append(request)
        record_job_work.invoke({"note": "Inspected build 29."})
        self.check_started.set()
        assert request.message_handler is None
        await self.released.wait()
        return AgentResult(text=json.dumps(self.report))


async def test_pausing_an_active_goal_preserves_work_and_stops_the_check(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    agent = _WorkingGoalAgent()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    goal = store.create_job(
        prompt="Check builds",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1m"),
        now=now,
    )
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    now += timedelta(minutes=1)
    scheduler = _attach_scheduler(host, store, now=lambda: now)
    await host.start()
    try:
        await scheduler.tick_once(wait=False)
        await asyncio.wait_for(agent.check_started.wait(), 2)
        run = scheduler._running[goal.id]
        store.edit_job(goal.id, origin=ORIGIN, enabled=False)
        await scheduler.tick_once(wait=False)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 2)
        latest = JobJournal(store.cron_dir).history(goal.id, ORIGIN)[0]
        assert latest["status"] == "interrupted"
        assert "Inspected build 29" in latest["work"]
        assert channel.sent == []
    finally:
        await host.stop()


@pytest.mark.parametrize("wake", [None, "invalid", "2020-01-01T00:00:00Z", "2099-01-01"])
def test_unusable_objective_wake_preserves_retry_and_findings(
    tmp_path: Path, wake: str | None
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(prompt="Check builds", mode="objective", origin=ORIGIN, now=now)
    claimed = store.advance_next_run(job.id, now=now)
    assert claimed is not None
    report = ObjectiveReport.from_reply(
        json.dumps({"status": "partial", "summary": "Build queued.", "next_wake_at": wake})
    )
    if report.next_wake_at is not None:
        store.apply_objective_result(claimed, wake=report.next_wake_at)
    assert report.status == "partial"
    assert store.get_job(job.id).next_run_at == now + timedelta(hours=1)


def test_previous_runs_exclude_later_starts(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(prompt="Check builds", mode="objective", origin=ORIGIN)
    journal = JobJournal(tmp_path)
    for run in ("first", "current", "later"):
        journal.start(job, run)
        journal.finish(run, "checked", run)
    rows = journal.history(job.id, ORIGIN, before_run="current", limit=1)
    assert [row["summary"] for row in rows] == ["first"]


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


@pytest.mark.parametrize("stall_finalizer", [False, True])
async def test_deadline_handoff_survives_a_stalled_tool_or_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stall_finalizer: bool
) -> None:
    monkeypatch.setattr("deepagents_talon.host._SCHEDULED_RUN_TIMEOUT_SECONDS", 0.5)
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(prompt="Verify build", mode="objective", origin=ORIGIN)
    wake = datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=5)
    middleware = JobScopeMiddleware()
    tool = StructuredTool.from_function(lambda: "", name="poll", description="Poll the build")
    call = _request("owner", tool)
    check = _request(
        "owner", StructuredTool.from_function(lambda: "", name="check", description="Check")
    )

    async def stall(_request):
        await asyncio.Event().wait()

    class DeadlineAgent(BlockingAgent):
        async def invoke(self, request: AgentRequest) -> AgentResult:
            self.requests.append(request)
            if not SCHEDULER_FINALIZING.get():
                label = record_job_work.invoke({"note": "Build 29 " + "x" * 2100 + " next: sign"})
                assert "truncated" in label
                await middleware.awrap_tool_call(check, _successful_check)
                await middleware.awrap_tool_call(call, stall)
            assert (await middleware.awrap_tool_call(call, stall)).status == "error"
            if stall_finalizer:
                await asyncio.Event().wait()
            return AgentResult(
                text=json.dumps(
                    {
                        "status": "partial",
                        "summary": "Build 29; verify signing.",
                        "next_wake_at": wake.isoformat(),
                    }
                )
            )

    agent = DeadlineAgent()
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    _attach_scheduler(host, store)
    if stall_finalizer:
        with pytest.raises(TimeoutError):
            await host.run_scheduled_job(job)
    else:
        assert await host.run_scheduled_job(job) == "[SILENT]"
        assert store.get_job(job.id).next_run_at == wake
    first, final = agent.requests
    assert first.conversation_id == final.conversation_id
    assert "Final report deadline" in first.text
    row = JobJournal(tmp_path).history(job.id, ORIGIN)[0]
    assert row["status"] == ("failed" if stall_finalizer else "partial")
    assert all(
        s in row["work"] for s in ("Build 29", "[truncated]", "next: sign", first.conversation_id)
    )
    assert "Last tool receipt: check (call), status success, exit code 0" in row["work"]
    assert "Calls without a receipt: poll (call)" in row["work"]


async def _successful_check(request):
    return ToolMessage(
        content="Complete.", tool_call_id=request.tool_call["id"], artifact={"exit_code": 0}
    )


def test_unacknowledged_and_changed_blockers_are_deliverable(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    goal = store.create_job(
        prompt="Ship",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
    )
    journal = JobJournal(tmp_path)
    report = ObjectiveReport(
        status="blocked", summary="Signing pending.", blocker_key="sign", notice="Who signs?"
    )
    assert journal.notice(goal, report) == report.notice
    journal = JobJournal(tmp_path)
    assert journal.notice(goal, report) == report.notice
    journal.mark_delivered(goal.id, report.notice)
    assert journal.notice(goal, report) == ""
    changed = report.model_copy(update={"blocker_key": "publish", "notice": "Where to publish?"})
    assert journal.notice(goal, changed) == changed.notice
    journal.mark_delivered(goal.id, changed.notice)
    assert journal.notice(goal, ObjectiveReport(status="checked", summary="Published.")) == ""
    assert journal.notice(goal, changed) == changed.notice


class _VerificationAgent(_WorkingGoalAgent):
    paths: tuple[str, ...] = ()
    needs_login = False

    async def invoke(self, request: AgentRequest) -> AgentResult:
        if self.needs_login:
            # The MCP transport can replace the original error.
            msg = "Client failed to connect"
            raise RuntimeError(msg) from _no_authorization_channel_error("records")
        for path in self.paths:
            tool = StructuredTool.from_function(
                lambda file_path: file_path, name="read_file", description="Read"
            )
            await JobScopeMiddleware().awrap_tool_call(
                _request("owner", tool, file_path=path),
                _file_response,
            )
        return await super().invoke(request)


async def _file_response(request):
    text = (
        "Error: permission denied"
        if request.tool_call["args"]["file_path"].endswith("denied")
        else "contents"
    )
    return ToolMessage(content=text, tool_call_id="call")


async def test_background_login_notice_is_deduplicated(tmp_path: Path) -> None:
    agent = _VerificationAgent()
    agent.needs_login = True
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(mode="objective", origin=ORIGIN, prompt="Check records")
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    _attach_scheduler(host, store)
    notice = await host.run_scheduled_job(job)
    assert "Authorize" in notice
    assert "records" in notice
    await host.deliver_scheduled_result(channel, job, notice)
    assert await host.run_scheduled_job(job) == "[SILENT]"
    assert JobJournal(tmp_path).history(job.id, ORIGIN)[0]["status"] == "blocked"


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
