from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule, CronTools
from deepagents_talon.cron.scheduler import PersistentCronScheduler
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
    job = tools["create_goal"].invoke({"outcome": "Keep the release on track"})
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
        return AgentResult(text=f"{decision} after {answer.splitlines()[2].strip()}")


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
        await _wait_for_sent_count(channel, 6)
        assert channel.sent[-1] == (ORIGIN.conversation_id, "approve after Answer: SQLite")
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
