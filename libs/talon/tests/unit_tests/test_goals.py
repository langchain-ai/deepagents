from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule, CronTools
from deepagents_talon.cron.scheduler import PersistentCronScheduler
from deepagents_talon.goals.journal import GoalJournal, record_goal_work
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.questions import ask_user, question_scope
from deepagents_talon.runtime import _approval_decision, _question_handler_from_request
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_sent_count
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
    with pytest.raises(ValidationError, match="schedule"):
        tools["create_goal"].invoke({"outcome": outcome})
    job = tools["create_goal"].invoke({"outcome": outcome, "schedule": "every 10m"})
    first, second, *_ = map(datetime.fromisoformat, job["upcoming"])
    assert second - first == timedelta(minutes=10)
    for other in (replace(ORIGIN, sender_id="bob"), replace(ORIGIN, conversation_id="C1:sibling")):
        origin = other
        assert tools["list_jobs"].invoke({}) == []
        assert "error" in tools["remove_job"].invoke({"job_id": job["id"]})
    origin = ORIGIN
    assert tools["edit_job"].invoke({"job_id": job["id"], "enabled": False})["enabled"] is False
    assert tools["remove_job"].invoke({"job_id": job["id"]})["id"] == job["id"]


class _GoalAgent(BlockingAgent):
    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        assert request.metadata["tool_approval_operator"] is False
        if request.metadata.get("trigger") == "cron":
            return AgentResult(text="scheduled reminder")
        with question_scope(_question_handler_from_request(request)):
            answer = await ask_user.ainvoke({"questions": [DATABASE]})
        decision, _, _ = await _approval_decision(
            request, "gate", ({"name": "publish", "args": {}},)
        )
        summary = f"{decision} after {answer.splitlines()[2].strip()}"
        return AgentResult(text=json.dumps({"status": "checked", "summary": summary}))


async def test_scheduled_goal_waits_for_owner_and_separate_tool_approval(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    goal = store.create_job(
        prompt="Prepare the release",
        origin=ORIGIN,
        continuing=True,
        schedule=CronSchedule.parse("every 1m"),
        now=now,
    )
    channel = RecordingChannel("slack")
    agent = _GoalAgent()
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
        now += timedelta(minutes=1)
        await scheduler.tick_once()
        await _wait_for_sent_count(channel, 1)
        store.create_job(
            prompt="Remind me", origin=ORIGIN, schedule=CronSchedule.parse("in 1m"), now=now
        )
        now += timedelta(minutes=1)
        await scheduler.tick_once()
        assert channel.sent[-1][1] == "scheduled reminder"
        await host.receive_message(
            channel, ChannelMessage(ORIGIN.conversation_id, "1", sender_id="bob")
        )
        assert "Only the person" in channel.sent[-1][1]
        await host.receive_message(
            channel, ChannelMessage(ORIGIN.conversation_id, "2", sender_id="alice")
        )
        await _wait_for_sent_count(channel, 5)
        assert "publish" in channel.sent[-1][1]
        await host.receive_message(
            channel, ChannelMessage(ORIGIN.conversation_id, "approve", sender_id="alice")
        )
        await asyncio.wait_for(scheduler._goals[goal.id], 2)
        assert len(channel.sent) == 5
        latest = GoalJournal(store.cron_dir).history(goal.id, ORIGIN)[0]
        assert latest["summary"] == "approve after Answer: SQLite"
        assert "SQLite" in latest["decisions"]
        assert "publish: approve" in latest["decisions"]
        assert [r.metadata["trigger"] for r in agent.requests] == ["goal", "cron"]
        assert store.get_job(goal.id) is not None
    finally:
        await host.stop()


def test_upgrade_keeps_confirmed_scope_progress_and_paused_drafts(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    job = store.create_job(
        prompt="old placeholder",
        goal_id="old",
        origin=ORIGIN,
        schedule=CronSchedule.parse("every 6h"),
    )
    charter = {"outcome": "Resolve blockers", "sources": ["path:/release"]}
    goal = {
        "id": "old",
        "goal_job_id": job.id,
        "state": "active",
        "charter": charter,
        "confirmed_charter": charter,
        "created_at": job.created_at.isoformat(),
        "scope": {
            "channel": ORIGIN.channel,
            "conversation_id": ORIGIN.conversation_id,
            "owner_id": ORIGIN.sender_id,
        },
    }
    draft = {
        **goal,
        "id": "draft",
        "state": "draft",
        "goal_job_id": None,
        "confirmed_charter": None,
    }
    raw = json.dumps(
        {
            "version": 1,
            "goals": [goal, draft],
            "items": [{"goal_id": "old", "status": "done", "artifact": "ticket-27"}],
        }
    )
    path = tmp_path / "ledger.json"
    path.write_text(raw)
    store.import_goal_ledger(tmp_path)
    upgraded = CronJobStore(assistant_id="test", cron_dir=store.cron_dir)
    active, paused = upgraded.list_jobs(origin=ORIGIN)
    assert active.enabled
    assert (active.id, active.next_run_at) == (job.id, job.next_run_at)
    assert "ticket-27" in active.prompt
    assert active.goal_sources == ("path:/release",)
    assert not paused.enabled
    assert (tmp_path / "ledger.imported.json").read_text() == raw
    upgraded.remove_job(active.id, origin=ORIGIN)
    upgraded.import_goal_ledger(tmp_path)
    assert upgraded.list_jobs() == [paused]


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
        if request.metadata.get("trigger") != "goal":
            return await super().invoke(request)
        self.requests.append(request)
        record_goal_work.invoke({"note": "Inspected build 29."})
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
        continuing=True,
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
            "Inspected build 29" in GoalJournal(store.cron_dir).history(goal.id, ORIGIN)[0]["work"]
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
        assert tools["edit_job"].invoke({"job_id": goal.id, "goal_updates": True})["goal_updates"]
        assert await host.run_scheduled_job(goal) == "Build 29 is unsigned."
        assert "Checked build 29" in agent.requests[-1].text
        assert "unrelated chat" not in agent.requests[-1].text
        assert agent.requests[-1].conversation_id != first.conversation_id

        tools["edit_job"].invoke({"job_id": goal.id, "goal_updates": False})
        agent.report = {
            "status": "blocked",
            "summary": "Signing requires a release owner.",
            "notice": "Please name the release owner.",
        }
        assert await host.run_scheduled_job(goal) == "Please name the release owner."
        assert await host.run_scheduled_job(goal) == "[SILENT]"
        tools["remove_job"].invoke({"job_id": goal.id})
        assert [row["status"] for row in tools["get_goal_history"].invoke({})] == [
            "blocked",
            "blocked",
            "partial",
            "partial",
        ]
        journal = GoalJournal(store.cron_dir)
        for other in (replace(ORIGIN, sender_id="bob"), replace(ORIGIN, conversation_id="sibling")):
            assert journal.history(goal.id, other) == []
    finally:
        agent.released.set()
        await host.stop()


async def test_blind_import_pauses_until_owner_repairs_scope(tmp_path: Path) -> None:
    agent = _WorkingGoalAgent()
    agent.released.set()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    job = store.create_job(
        prompt="Audit", goal_id="legacy", origin=ORIGIN, schedule=CronSchedule.parse("every 1h")
    )
    store.edit_job(job.id, origin=ORIGIN, goal_sources=["the audit directory"])
    host = TalonHost(config=_config(tmp_path), channels=[channel], agent=agent)
    host.scheduler = PersistentCronScheduler(
        store=store,
        run_job=host.run_scheduled_job,
        deliver_result=lambda job, text: host.deliver_scheduled_result(channel, job, text),
    )
    assert "Goal paused" in await host.run_scheduled_job(job)
    assert not store.get_job(job.id).enabled
    assert not agent.requests
    assert GoalJournal(store.cron_dir).history(job.id, ORIGIN)[0]["status"] == "blocked"
    tools = CronTools(store=store, origin=lambda: ORIGIN)
    agent.report = {"status": "checked", "summary": "Claimed a check without a read."}
    tools.edit_job(job.id, goal_sources=["path:/audit"], enabled=True)
    assert await host.run_scheduled_job(job) == "[SILENT]"
    assert agent.requests[-1].metadata["goal_sources"] == ["path:/audit"]
    assert tools.list_jobs()[0]["last_check"]["status"] == "unverified"


async def test_pausing_an_active_goal_preserves_work_and_stops_the_check(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    agent = _WorkingGoalAgent()
    channel = RecordingChannel("slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    goal = store.create_job(
        prompt="Check builds",
        origin=ORIGIN,
        continuing=True,
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
        await scheduler.tick_once()
        await asyncio.wait_for(agent.check_started.wait(), 2)
        run = scheduler._goals[goal.id]
        store.edit_job(goal.id, origin=ORIGIN, enabled=False)
        await scheduler.tick_once()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 2)
        latest = GoalJournal(store.cron_dir).history(goal.id, ORIGIN)[0]
        assert latest["status"] == "interrupted"
        assert "Inspected build 29" in latest["work"]
        assert channel.sent == []
    finally:
        await host.stop()
