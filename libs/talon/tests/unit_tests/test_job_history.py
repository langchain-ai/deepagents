from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule, CronTools
from deepagents_talon.cron.journal import JobJournal, ObjectiveReport, record_job_work
from deepagents_talon.cron.scheduler import PersistentCronScheduler
from deepagents_talon.cron.scope import JobScopeMiddleware
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.mcp_auth import _no_authorization_channel_error
from deepagents_talon.questions import ask_user, question_scope
from deepagents_talon.runtime import _approval_decision, _question_handler_from_request
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_sent_count
from tests.unit_tests.test_background import _request
from tests.unit_tests.test_questions import DATABASE

if TYPE_CHECKING:
    from pathlib import Path

ORIGIN = CronOrigin(channel="slack", conversation_id="C1:thread", sender_id="alice")


def test_goals_are_managed_with_job_tools_and_are_private_to_the_owner_and_thread(
    tmp_path: Path,
) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = ORIGIN
    tools = {
        tool.name: tool
        for tool in CronTools(store=store, origin=lambda: origin).as_langchain_tools()
    }
    outcome = "Keep the release on track. The build should finish in ten minutes."
    assert "schedule" in tools["create_job"].invoke({"prompt": outcome})["error"]
    job = tools["create_job"].invoke({"mode": "objective", "prompt": outcome})
    assert job["upcoming"] == [job["next_run_at"]]
    assert store.due_jobs()[0].id == job["id"]
    assert job["mode"] == "objective"
    for other in (replace(ORIGIN, sender_id="bob"), replace(ORIGIN, conversation_id="C1:sibling")):
        origin = other
        assert tools["list_jobs"].invoke({}) == []
        assert "error" in tools["remove_job"].invoke({"job_id": job["id"]})
    origin = ORIGIN
    assert tools["edit_job"].invoke({"job_id": job["id"], "enabled": False})["enabled"] is False
    assert tools["remove_job"].invoke({"job_id": job["id"]})["id"] == job["id"]


class _GoalAgent(BlockingAgent):
    wake: datetime

    async def invoke(self, request: AgentRequest) -> AgentResult:
        if request.metadata.get("job_mode") != "objective":
            return await super().invoke(request)
        self.requests.append(request)
        assert request.metadata["tool_approval_operator"] is False
        with question_scope(_question_handler_from_request(request)):
            answer = await ask_user.ainvoke({"questions": [DATABASE]})
        decision, _, _ = await _approval_decision(
            request, "gate", ({"name": "publish", "args": {}},)
        )
        assert decision == "reject"
        assert "none were asked" in answer
        return AgentResult(
            text=json.dumps(
                {
                    "status": "blocked",
                    "summary": "Publishing needs approval.",
                    "notice": "Please review the pending publication.",
                    "blocker_key": "publish",
                    "next_wake_at": self.wake.isoformat(),
                }
            )
        )


async def test_objective_sets_next_wake_and_does_not_claim_chat_replies(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    goal = store.create_job(
        prompt="Prepare the release",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
        now=now,
    )
    channel = RecordingChannel("slack")
    agent = _GoalAgent()
    agent.wake = now + timedelta(minutes=73)
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
        now=lambda: now,
    )
    host.scheduler = scheduler
    await host.start()
    try:
        now += timedelta(hours=1)
        await scheduler.tick_once(wait=False)
        await asyncio.wait_for(scheduler._running[goal.id], 2)
        assert channel.sent == [(ORIGIN.conversation_id, "Please review the pending publication.")]
        reopened = CronJobStore(assistant_id="test", cron_dir=store.cron_dir)
        assert reopened.get_job(goal.id).next_run_at == agent.wake
        agent.wake += timedelta(minutes=19)
        CronTools(store=store, origin=lambda: ORIGIN).edit_job(goal.id, run_now=True)
        await scheduler.tick_once(wait=False)
        await asyncio.wait_for(scheduler._running[goal.id], 2)
        assert reopened.get_job(goal.id).next_run_at == agent.wake
        assert not reopened.due_jobs(now=now)
        await host.receive_message(
            channel, ChannelMessage(ORIGIN.conversation_id, "unrelated", sender_id="alice")
        )
        await _wait_for_sent_count(channel, 2)
        assert channel.sent[-1][1] == "reply:unrelated"
    finally:
        await host.stop()


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


async def test_background_goal_isolation_history_and_notification_policy(tmp_path: Path) -> None:
    agent = _WorkingGoalAgent()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    goal = store.create_job(
        prompt="Check builds",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
    )
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    host.scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
    )
    await host.start()
    try:
        check = asyncio.create_task(host.run_scheduled_job(goal))
        await asyncio.wait_for(agent.check_started.wait(), 2)
        await host.receive_message(
            channel, ChannelMessage(ORIGIN.conversation_id, "unrelated chat", sender_id="alice")
        )
        await _wait_for_sent_count(channel, 1)
        assert not check.done()
        assert (
            "Inspected build 29" in JobJournal(store.cron_dir).history(goal.id, ORIGIN)[0]["work"]
        )
        agent.released.set()
        assert await asyncio.wait_for(check, 2) == "[SILENT]"
        first, chat = agent.requests
        assert first.conversation_id != chat.conversation_id
        assert len(channel.sent) == 1

        tools = {
            tool.name: tool
            for tool in CronTools(store=store, origin=lambda: ORIGIN).as_langchain_tools()
        }
        assert await host.run_scheduled_job(goal) == "[SILENT]"
        assert "Checked build 29" in agent.requests[-1].text
        assert "unrelated chat" not in agent.requests[-1].text
        assert agent.requests[-1].conversation_id != first.conversation_id

        agent.report = {
            "status": "blocked",
            "summary": "Signing requires a release owner.",
            "notice": "Please name the release owner.",
            "blocker_key": "release-owner",
        }
        notice = await host.run_scheduled_job(goal)
        assert notice == "Please name the release owner."
        await host.deliver_scheduled_result(channel, goal, notice)
        agent.report.update(summary="Still missing a signer.", notice="Who owns signing?")
        assert await host.run_scheduled_job(goal) == "[SILENT]"
        tools["remove_job"].invoke({"job_id": goal.id})
        assert [row["status"] for row in tools["get_job_history"].invoke({})] == [
            "blocked",
            "blocked",
            "partial",
            "partial",
        ]
        journal = JobJournal(store.cron_dir)
        for other in (replace(ORIGIN, sender_id="bob"), replace(ORIGIN, conversation_id="sibling")):
            assert journal.history(goal.id, other) == []
    finally:
        agent.released.set()
        await host.stop()


async def test_invalid_read_scope_pauses_until_owner_repairs_scope(tmp_path: Path) -> None:
    agent = _WorkingGoalAgent()
    agent.released.set()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    job = store.create_job(
        prompt="Audit", mode="objective", origin=ORIGIN, schedule=CronSchedule.parse("every 1h")
    )
    store.edit_job(job.id, origin=ORIGIN, read_sources=["the audit directory"])
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    host.scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
    )
    assert "Job paused" in await host.run_scheduled_job(job)
    assert not store.get_job(job.id).enabled
    assert not agent.requests
    assert JobJournal(store.cron_dir).history(job.id, ORIGIN)[0]["status"] == "blocked"
    tools = CronTools(store=store, origin=lambda: ORIGIN)
    agent.report = {"status": "checked", "summary": "Claimed a check without a read."}
    tools.edit_job(job.id, read_sources=["path:/audit"], enabled=True)
    assert await host.run_scheduled_job(job) == "[SILENT]"
    assert agent.requests[-1].metadata["read_sources"] == ["path:/audit"]
    assert tools.list_jobs()[0]["last_result"]["status"] == "unverified"


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
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
        now=lambda: now,
    )
    host.scheduler = scheduler
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
        store.set_objective_wake(claimed, report.next_wake_at)
    assert report.status == "partial"
    assert store.get_job(job.id).next_run_at == now + timedelta(hours=1)


def test_stale_completion_preserves_revised_objective(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(
        prompt="Check builds",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
    )
    revised = store.edit_job(job.id, origin=ORIGIN, prompt="Check all builds")
    store.set_objective_wake(job, datetime.now(UTC) + timedelta(days=1))
    store.finish_objective(job)
    assert store.get_job(job.id) == revised
    store.finish_objective(revised)
    assert store.get_job(job.id).next_run_at is None


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


async def test_both_modes_share_listing_and_routine_history_survives_removal(
    tmp_path: Path,
) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    tools = {
        tool.name: tool
        for tool in CronTools(store=store, origin=lambda: ORIGIN).as_langchain_tools()
    }
    for mode in ("routine", "objective"):
        tools["create_job"].invoke({"prompt": "check status", "schedule": "every 5m", "mode": mode})
    routine, objective = tools["list_jobs"].invoke({})
    assert [routine["mode"], objective["mode"]] == ["routine", "objective"]
    host = TalonHost(
        config=_config(tmp_path), channels=[RecordingChannel("slack")], agent=BlockingAgent()
    )
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(host.channels[0], job, text),
    )
    host.scheduler = scheduler
    assert tools["edit_job"].invoke({"job_id": routine["id"], "run_now": True})["run_requested"]
    await scheduler.tick_once()
    routine, objective = tools["list_jobs"].invoke({})
    assert routine["last_result"]["summary"] == "reply:check status"
    assert objective["last_result"] is None
    tools["remove_job"].invoke({"job_id": routine["id"]})
    assert tools["get_job_history"].invoke({"job_id": routine["id"]})[0]["status"] == "checked"


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


async def test_required_reads_and_evidence_survive_separate_checks(tmp_path: Path) -> None:
    agent = _VerificationAgent()
    agent.released.set()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    tools = {t.name: t for t in CronTools(store=store, origin=lambda: ORIGIN).as_langchain_tools()}
    created = tools["create_job"].invoke(
        {
            "prompt": "Verify release",
            "mode": "objective",
            "required_sources": ["path:/audit", "path:/logs"],
            "completion_criteria": ["publish", "rescan"],
        }
    )
    job = store.get_job(created["id"])
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    host.scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
    )
    journal = JobJournal(tmp_path)
    agent.paths = ("/audit/image", "/logs-neighbor/file", "/logs/denied")
    agent.report = {"status": "partial", "summary": "Inspected image."}
    await host.run_scheduled_job(job)
    row = journal.history(job.id, ORIGIN)[0]
    assert row["status"] == "unverified"
    assert "Missing required reads: path:/logs" in row["summary"]
    assert row["read_tools"] == "path:/audit"
    agent.paths = ("/audit/image", "/logs/release")
    agent.report.update(status="complete", evidence={"publish": "registry digest abcd"})
    await host.run_scheduled_job(job)
    assert store.get_job(job.id).enabled
    assert "Missing evidence for: rescan" in journal.history(job.id, ORIGIN)[0]["summary"]
    soon = datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=1)
    agent.report = {"status": "checked", "summary": "No change", "next_wake_at": soon.isoformat()}
    for _ in range(4):
        await host.run_scheduled_job(job)
    assert store.get_job(job.id).next_run_at >= soon + timedelta(minutes=9)
    agent.report.update(wake_reason="owner_timing")
    await host.run_scheduled_job(job)
    assert store.get_job(job.id).next_run_at == soon
    agent.report.update(status="complete", evidence={"rescan": "scan 123"})
    await host.run_scheduled_job(job)
    assert "registry digest abcd" in agent.requests[-1].text
    assert not store.get_job(job.id).enabled
    assert journal.history(job.id, ORIGIN)[0]["status"] == "complete"


async def test_background_login_notice_is_deduplicated(tmp_path: Path) -> None:
    agent = _VerificationAgent()
    agent.needs_login = True
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    job = store.create_job(mode="objective", origin=ORIGIN, prompt="Check records")
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    host.scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
    )
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
    assert JobJournal(tmp_path).verification(job) == ((), {"build": "build 27"})
    edited = store.edit_job(job.id, origin=ORIGIN, completion_criteria=["ship"])
    assert journal.verification(edited) == (("ship",), {})
