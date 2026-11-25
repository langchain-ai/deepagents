"""Owner corrections take effect before subsequent destination writes."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.journal import JobJournal, ObjectiveReport
from deepagents_talon.cron.scope import JobScopeMiddleware
from tests.unit_tests.test_action_recovery import request

if TYPE_CHECKING:
    from pathlib import Path


async def test_corrected_goal_refuses_queued_write(tmp_path: Path) -> None:

    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="test", conversation_id="chat", sender_id="owner")
    goal = store.create_job(
        prompt="release A", mode="objective", origin=origin, schedule=CronSchedule.parse("every 1m")
    )
    published = []

    async def publish(_request):
        published.append("release A")
        return ToolMessage(content="published", tool_call_id="queued")

    with JobJournal(tmp_path).capture_actions(
        goal.id, "run", current=lambda: store.get_job(goal.id) == goal
    ):
        store.edit_job(goal.id, origin=origin, prompt="release B")
        result = await JobScopeMiddleware().awrap_tool_call(request("publish", "queued"), publish)
    assert result.status == "error"
    assert published == []


def test_release_correction_retains_verified_merge_and_retires_wait(tmp_path: Path) -> None:

    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="test", conversation_id="chat", sender_id="owner")
    goal = store.create_job(
        prompt="ship A",
        mode="objective",
        origin=origin,
        schedule=CronSchedule.parse("every 1m"),
        completion_criteria=["PR 42 merged", "A published"],
    )
    journal = JobJournal(tmp_path)
    journal.start(goal, "first")
    journal.finish_check(
        goal,
        "first",
        ObjectiveReport(
            status="partial",
            summary="merge confirmed",
            evidence={"C1": "merge/42", "C2": "release/A"},
            verified=["C1", "C2"],
        ),
        set(),
    )
    now = datetime.now(UTC)
    assert store.apply_objective_result(
        goal, wake=now + timedelta(days=2), wait_condition="Release A is published"
    )
    corrected = store.edit_job(
        goal.id,
        origin=origin,
        prompt="ship B",
        now=now,
        completion_criteria=["PR 42 merged", "B published"],
    )
    assert not store.get_job(goal.id).wait_condition
    assert journal.verification(corrected)[1] == {
        "PR 42 merged": {"reference": "merge/42", "verified": True}
    }
    assert store.due_jobs(now=now) == [corrected]


async def test_pause_resume_and_cancel_preserve_inflight_receipt(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="test", conversation_id="chat", sender_id="owner")
    goal = store.create_job(prompt="publish", mode="objective", origin=origin)
    journal = JobJournal(tmp_path)
    journal.start(goal, "run")
    middleware = JobScopeMiddleware()
    calls = []

    async def publish(_request):
        calls.append("publish")
        store.edit_job(goal.id, origin=origin, enabled=False)
        return ToolMessage(content="applied", tool_call_id="first")

    with journal.capture_actions(goal.id, "run", current=lambda: store.get_job(goal.id) == goal):
        assert (
            await middleware.awrap_tool_call(request("publish", "first"), publish)
        ).status == "success"
        assert store.due_jobs(now=datetime.now(UTC) + timedelta(days=2)) == []
        store.edit_job(goal.id, origin=origin, enabled=True)
        assert (
            await middleware.awrap_tool_call(request("publish", "queued"), publish)
        ).status == "error"
    assert store.apply_objective_result(
        store.get_job(goal.id), wait_condition="Publication visible"
    )
    store.remove_job(goal.id, origin=origin)
    assert not store.list_jobs(origin=origin)
    assert calls == ["publish"]
    assert store.due_jobs(now=datetime.now(UTC) + timedelta(days=2)) == []
    assert (
        "Tool receipt: publish (first), status success."
        in journal.history(goal.id, origin)[0]["work"]
    )
