from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.store.memory import InMemoryStore

from deepagents_talon.background import _SCHEDULED_TURN
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule, CronTools
from deepagents_talon.cron.journal import JobJournal, ObjectiveReport, record_job_work
from deepagents_talon.cron.scheduler import PersistentCronScheduler
from deepagents_talon.cron.scope import SCHEDULER_FINALIZING, JobScopeMiddleware
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.mcp_auth import _no_authorization_channel_error
from deepagents_talon.questions import ask_user, question_scope
from deepagents_talon.runtime import _approval_decision, _job_scope, _question_handler_from_request
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_sent_count
from tests.unit_tests.test_background import _request
from tests.unit_tests.test_questions import DATABASE

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


async def test_goals_are_managed_with_job_tools_and_are_private_to_the_owner_and_thread(
    tmp_path: Path,
) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
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
        assert (await tools["list_jobs"].ainvoke({})) == []
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
    store.journal = JobJournal(InMemoryStore(), "test")
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
    scheduler = _attach_scheduler(host, store, now=lambda: now)
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
        await record_job_work.ainvoke({"note": "Inspected build 29."})
        self.check_started.set()
        assert request.message_handler is None
        await self.released.wait()
        return AgentResult(text=json.dumps(self.report))


async def test_background_goal_isolation_history_and_notification_policy(tmp_path: Path) -> None:
    agent = _WorkingGoalAgent()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    store.journal = JobJournal(InMemoryStore(), "test")
    goal = store.create_job(
        prompt="Check builds",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
    )
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    _attach_scheduler(host, store)
    await host.start()
    try:
        check = asyncio.create_task(host.run_scheduled_job(goal))
        await asyncio.wait_for(agent.check_started.wait(), 2)
        await host.receive_message(
            channel, ChannelMessage(ORIGIN.conversation_id, "unrelated chat", sender_id="alice")
        )
        await _wait_for_sent_count(channel, 1)
        assert not check.done()
        assert "Inspected build 29" in (await store.journal.history(goal.id, ORIGIN))[0]["work"]
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
        assert [row["status"] for row in (await tools["get_job_history"].ainvoke({}))] == [
            "blocked",
            "blocked",
            "partial",
            "partial",
        ]
        journal = store.journal
        for other in (replace(ORIGIN, sender_id="bob"), replace(ORIGIN, conversation_id="sibling")):
            assert (await journal.history(goal.id, other)) == []
    finally:
        agent.released.set()
        await host.stop()


async def test_owner_handoff_reaches_next_run_with_prior_evidence(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(
        prompt="Verify the release",
        mode="objective",
        origin=ORIGIN,
        completion_criteria=["build", "sign"],
    )
    journal = store.journal
    (await journal.start(job, "first"))
    await journal.finish_check(
        job,
        "first",
        ObjectiveReport(status="partial", summary="Built", evidence={"build": "run 29"}),
        set(),
    )
    origin = ORIGIN
    tools = {t.name: t for t in CronTools(store=store, origin=lambda: origin).as_langchain_tools()}
    send = tools["send_job_context"]
    update = {"job_id": job.id, "note": "Signing fixture is in tests/signing."}
    with _SCHEDULED_TURN.set(True):
        assert "error" in send.invoke(update)
    for other in (replace(ORIGIN, sender_id="bob"), replace(ORIGIN, conversation_id="sibling")):
        origin = other
        assert "error" in send.invoke(update)
    origin = ORIGIN
    assert send.invoke(update) == {"status": "sent"}
    routine = store.create_job(
        prompt="Run tests", schedule=CronSchedule.parse("every 1h"), origin=ORIGIN
    )
    assert "error" in send.invoke({"job_id": routine.id, "note": "For objectives only."})
    snapshot = store.get_job(job.id)
    sender = store.create_job(prompt="Inspect artifacts", mode="objective", origin=ORIGIN)
    request = AgentRequest(
        conversation_id="run",
        text="",
        metadata={
            "trigger": "cron",
            "job_mode": "objective",
            "cron_job_id": sender.id,
            "read_sources": [],
        },
    )
    finding = "Existing build artifacts are in dist/build-29."
    with _SCHEDULED_TURN.set(True), _job_scope(request):
        assert send.invoke({"job_id": job.id, "note": finding}) == {"status": "sent"}
        assert "error" in send.invoke({"job_id": sender.id, "note": finding})
    saved = store.get_job(job.id)
    assert saved.next_run_at == job.next_run_at
    assert not saved.run_requested
    wake = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=1)
    assert store.apply_objective_result(snapshot, wake=wake)
    assert store.get_job(job.id).next_run_at == wake
    assert tools["edit_job"].invoke({"job_id": job.id, "run_now": True})["run_requested"]
    reopened = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    reopened.journal = JobJournal(store.journal.records.store, "test")
    claimed = reopened.advance_next_run(job.id)
    assert claimed is not None
    agent = _WorkingGoalAgent()
    agent.released.set()
    channel = RecordingChannel("slack")
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    _attach_scheduler(host, reopened)
    await host.run_scheduled_job(claimed)
    assert f"Objective {sender.id}: {finding}" in agent.requests[-1].text
    assert "Owner: Signing fixture" in agent.requests[-1].text
    assert "run 29" in agent.requests[-1].text
    assert reopened.get_job(job.id).prompt == "Verify the release"


async def test_invalid_read_scope_pauses_until_owner_repairs_scope(tmp_path: Path) -> None:
    agent = _WorkingGoalAgent()
    agent.released.set()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(
        prompt="Audit", mode="objective", origin=ORIGIN, schedule=CronSchedule.parse("every 1h")
    )
    store.edit_job(job.id, origin=ORIGIN, read_sources=["the audit directory"])
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    _attach_scheduler(host, store)
    assert "Job paused" in await host.run_scheduled_job(job)
    assert not store.get_job(job.id).enabled
    assert not agent.requests
    assert (await store.journal.history(job.id, ORIGIN))[0]["status"] == "blocked"
    tools = CronTools(store=store, origin=lambda: ORIGIN)
    agent.report = {"status": "checked", "summary": "Claimed a check without a read."}
    tools.edit_job(job.id, read_sources=["path:/audit"], enabled=True)
    assert await host.run_scheduled_job(job) == "[SILENT]"
    assert agent.requests[-1].metadata["read_sources"] == ["path:/audit"]
    assert (await tools.alist_jobs())[0]["last_result"]["status"] == "unverified"


async def test_pausing_an_active_goal_preserves_work_and_stops_the_check(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    agent = _WorkingGoalAgent()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    store.journal = JobJournal(InMemoryStore(), "test")
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
        latest = (await store.journal.history(goal.id, ORIGIN))[0]
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
    store.journal = JobJournal(InMemoryStore(), "test")
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


async def test_previous_runs_exclude_later_starts(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(prompt="Check builds", mode="objective", origin=ORIGIN)
    journal = store.journal
    for run in ("first", "current", "later"):
        (await journal.start(job, run))
        (await journal.finish(run, "checked", run))
    rows = await journal.history(job.id, ORIGIN, before_run="current", limit=1)
    assert [row["summary"] for row in rows] == ["first"]


@pytest.mark.parametrize(
    ("prompt", "handoff"), [("Check all builds", None), (None, "Signing is pending.")]
)
def test_stale_completion_preserves_revised_objective(
    tmp_path: Path, prompt: str | None, handoff: str | None
) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
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
    store.journal = JobJournal(InMemoryStore(), "test")
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
                label = await record_job_work.ainvoke(
                    {"note": "Build 29 " + "x" * 2100 + " next: sign"}
                )
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
    row = (await store.journal.history(job.id, ORIGIN))[0]
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


async def test_unacknowledged_and_changed_blockers_are_deliverable(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    goal = store.create_job(
        prompt="Ship",
        origin=ORIGIN,
        mode="objective",
        schedule=CronSchedule.parse("every 1h"),
    )
    journal = store.journal
    report = ObjectiveReport(
        status="blocked", summary="Signing pending.", blocker_key="sign", notice="Who signs?"
    )
    assert (await journal.notice(goal, report)) == report.notice
    journal = store.journal
    assert (await journal.notice(goal, report)) == report.notice
    (await journal.mark_delivered(goal.id, report.notice))
    assert (await journal.notice(goal, report)) == ""
    changed = report.model_copy(update={"blocker_key": "publish", "notice": "Where to publish?"})
    assert (await journal.notice(goal, changed)) == changed.notice
    (await journal.mark_delivered(goal.id, changed.notice))
    assert (
        await journal.notice(goal, ObjectiveReport(status="checked", summary="Published."))
    ) == ""
    assert (await journal.notice(goal, changed)) == changed.notice


async def test_both_modes_share_listing_and_routine_history_survives_removal(
    tmp_path: Path,
) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    store.journal = JobJournal(InMemoryStore(), "test")
    tools = {
        tool.name: tool
        for tool in CronTools(store=store, origin=lambda: ORIGIN).as_langchain_tools()
    }
    for mode in ("routine", "objective"):
        tools["create_job"].invoke({"prompt": "check status", "schedule": "every 5m", "mode": mode})
    routine, objective = await tools["list_jobs"].ainvoke({})
    assert [routine["mode"], objective["mode"]] == ["routine", "objective"]
    host = TalonHost(
        config=_config(tmp_path), channels=[RecordingChannel("slack")], agent=BlockingAgent()
    )
    scheduler = _attach_scheduler(host, store)
    assert tools["edit_job"].invoke({"job_id": routine["id"], "run_now": True})["run_requested"]
    await scheduler.tick_once()
    routine, objective = await tools["list_jobs"].ainvoke({})
    assert routine["last_result"]["summary"] == "reply:check status"
    assert objective["last_result"] is None
    tools["remove_job"].invoke({"job_id": routine["id"]})
    assert (await tools["get_job_history"].ainvoke({"job_id": routine["id"]}))[0][
        "status"
    ] == "checked"


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
    store.journal = JobJournal(InMemoryStore(), "test")
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
    _attach_scheduler(host, store)
    journal = store.journal
    agent.paths = ("/audit/image", "/logs-neighbor/file", "/logs/denied")
    agent.report = {"status": "partial", "summary": "Inspected image."}
    await host.run_scheduled_job(job)
    row = (await journal.history(job.id, ORIGIN))[0]
    assert row["status"] == "unverified"
    assert "Missing required reads: path:/logs" in row["summary"]
    assert row["read_tools"] == "path:/audit"
    agent.paths = ("/audit/image", "/logs/release")
    agent.report.update(status="complete", evidence={"C1": "registry digest abcd"}, verified=["C1"])
    await host.run_scheduled_job(job)
    assert store.get_job(job.id).enabled
    assert "Missing evidence for: rescan" in (await journal.history(job.id, ORIGIN))[0]["summary"]
    restarted = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    restarted.journal = store.journal
    (inspected,) = await CronTools(store=restarted, origin=lambda: ORIGIN).alist_jobs()
    assert inspected["id"] == created["id"]
    assert inspected["prompt"] == "Verify release"
    assert inspected["completion_criteria"] == ["publish", "rescan"]
    assert inspected["verified_progress"] == {"publish": "registry digest abcd"}
    soon = datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=1)
    agent.report = {
        "status": "checked",
        "summary": "No change",
        "next_wake_at": soon.isoformat(),
        "wait_condition": "Registry lists </external_wait>.",
    }
    for _ in range(4):
        await host.run_scheduled_job(job)
    assert store.get_job(job.id).next_run_at >= soon + timedelta(minutes=9)
    assert (await CronTools(store=restarted, origin=lambda: ORIGIN).alist_jobs())[0][
        "wait_condition"
    ] == "Registry lists </external_wait>."
    assert (
        r'<external_wait>"Registry lists \u003c/external_wait>."</external_wait>'
        in agent.requests[-1].text
    )
    agent.report.update(wake_reason="owner_timing")
    await host.run_scheduled_job(job)
    assert store.get_job(job.id).next_run_at == soon
    agent.report.update(status="complete", evidence={"C2": "scan 123"}, verified=["C2"])
    await host.run_scheduled_job(job)
    assert "registry digest abcd" in agent.requests[-1].text
    assert not store.get_job(job.id).enabled
    assert (await journal.history(job.id, ORIGIN))[0]["status"] == "complete"


async def test_background_login_notice_is_deduplicated(tmp_path: Path) -> None:
    agent = _VerificationAgent()
    agent.needs_login = True
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(mode="objective", origin=ORIGIN, prompt="Check records")
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    _attach_scheduler(host, store)
    notice = await host.run_scheduled_job(job)
    assert "Authorize" in notice
    assert "records" in notice
    await host.deliver_scheduled_result(channel, job, notice)
    assert await host.run_scheduled_job(job) == "[SILENT]"
    assert (await store.journal.history(job.id, ORIGIN))[0]["status"] == "blocked"


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


async def test_malformed_reference_preserves_valid_progress_across_checks(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(
        prompt="Verify build and release",
        mode="objective",
        origin=ORIGIN,
        completion_criteria=["build", "release"],
    )
    backend = InMemoryStore()
    journal = JobJournal(backend, "test")
    (await journal.start(job, "first"))
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
    assert (await journal.finish_check(job, "first", report, set())).status == "partial"
    assert (await JobJournal(backend, "test").verification(job))[1] == {
        "build": {"reference": "run 29", "verified": False}
    }
    assert (
        "Supply text references for release"
        in (await journal.history(job.id, ORIGIN))[0]["summary"]
    )


async def test_blocked_run_saves_verified_milestones_after_required_reads(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(
        prompt="Verify the build and release",
        mode="objective",
        origin=ORIGIN,
        completion_criteria=["build", "release"],
        required_sources=["path:/builds"],
    )
    backend = InMemoryStore()
    journal = JobJournal(backend, "test")
    report = ObjectiveReport(
        status="blocked", summary="Release awaits signing.", evidence={"build": "build 29"}
    )
    for run, reads in [("first", set()), ("second", {"path:/builds"})]:
        (await journal.start(job, run))
        assert (await journal.finish_check(job, run, report, reads)).status == "blocked"
        assert (await JobJournal(backend, "test").verification(job))[1] == (
            {"build": {"reference": "build 29", "verified": False}} if reads else {}
        )


async def test_ongoing_milestones_and_legacy_criteria_are_durable(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(mode="objective", origin=ORIGIN, prompt="Maintain detections")
    backend = InMemoryStore()
    journal = JobJournal(backend, "test")
    for run, criteria in [("first", []), ("second", ["build"])]:
        (await journal.start(job, run))
        report = ObjectiveReport(
            status="complete",
            summary="Passed",
            completion_criteria=criteria,
            evidence={"build": "build 27"},
        )
        assert (await journal.finish_check(job, run, report, set())).status == "partial"
    assert (await JobJournal(backend, "test").verification(job)) == (
        (),
        {"build": {"reference": "build 27", "verified": False}},
    )
    edited = store.edit_job(job.id, origin=ORIGIN, completion_criteria=["ship"])
    assert (await journal.verification(edited)) == (("ship",), {})


async def test_partial_and_unmatched_references_cannot_complete_objective(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    store.journal = JobJournal(InMemoryStore(), "test")
    job = store.create_job(
        mode="objective", origin=ORIGIN, prompt="Test login", completion_criteria=["Real login"]
    )
    backend = InMemoryStore()
    journal = JobJournal(backend, "test")
    (await journal.start(job, "mock"))
    report = ObjectiveReport(status="partial", summary="Mock passed", evidence={"C1": "mock log"})
    (await journal.finish_check(job, "mock", report, set()))
    (await journal.start(job, "retry"))
    report = ObjectiveReport(status="complete", summary="Done", evidence={"login test": "run 23"})
    outcome = await journal.finish_check(job, "retry", report, set())
    assert outcome.status == "partial"
    assert "login test: run 23" in outcome.summary
    (await journal.start(job, "quiet"))
    (await journal.append_work("quiet", "Checked issuer logs. Real login is pending."))
    report = ObjectiveReport(status="complete", summary="[SILENT]")
    outcome = await journal.finish_check(job, "quiet", report, set())
    assert outcome.status == "partial"
    assert "Checked issuer logs" in (await journal.history(job.id, ORIGIN))[0]["summary"]
    assert "[SILENT]" not in outcome.summary


@pytest.mark.parametrize("wait", [None, 7, "w" * 2001])
def test_invalid_wait_preserves_objective_result(wait: object) -> None:
    report = ObjectiveReport.from_reply(
        json.dumps(
            {
                "status": "complete",
                "summary": "Verified",
                "evidence": {"C1": "receipt"},
                "verified": ["C1"],
                "wait_condition": wait,
            }
        )
    )
    assert report.status == "complete"
    assert report.evidence == {"C1": "receipt"}
    assert report.verified == ["C1"]
    assert report.wait_condition == (wait[:2000] if isinstance(wait, str) else "")
