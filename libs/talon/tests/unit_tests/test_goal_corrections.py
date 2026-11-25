"""Owner corrections take effect before subsequent destination writes."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.decisions import decision_reply
from deepagents_talon.cron.journal import JobJournal, ObjectiveReport
from deepagents_talon.cron.scope import JobScopeMiddleware
from deepagents_talon.interfaces import ChannelMessage
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

    with JobJournal(InMemoryStore(), "test").capture_actions(
        goal.id, "run", current=lambda: store.get_job(goal.id) == goal
    ):
        store.edit_job(goal.id, origin=origin, prompt="release B")
        result = await JobScopeMiddleware().awrap_tool_call(request("publish", "queued"), publish)
    assert result.status == "error"
    assert published == []


async def test_release_correction_retains_verified_merge_and_retires_wait(tmp_path: Path) -> None:

    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="test", conversation_id="chat", sender_id="owner")
    goal = store.create_job(
        prompt="ship A",
        mode="objective",
        origin=origin,
        schedule=CronSchedule.parse("every 1m"),
        completion_criteria=["PR 42 merged", "A published"],
    )
    store.journal = journal = JobJournal(InMemoryStore(), "test")
    await journal.start(goal, "first")
    await journal.finish_check(
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
    assert (await journal.verification(corrected))[1] == {
        "PR 42 merged": {"reference": "merge/42", "verified": True}
    }
    assert store.due_jobs(now=now) == [corrected]


async def test_pause_resume_and_cancel_preserve_inflight_receipt(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="test", conversation_id="chat", sender_id="owner")
    goal = store.create_job(prompt="publish", mode="objective", origin=origin)
    store.journal = journal = JobJournal(InMemoryStore(), "test")
    await journal.start(goal, "run")
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
        in (await journal.history(goal.id, origin))[0]["work"]
    )


def test_owner_controls_wake_indefinite_wait(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="test", conversation_id="chat", sender_id="owner")
    goal = store.create_job(prompt="ship", mode="objective", origin=origin)
    assert store.apply_objective_result(goal, waiting_for_owner=True)
    corrected = store.edit_job(goal.id, origin=origin, prompt="ship beta")
    assert store.due_jobs() == [corrected]
    assert store.apply_objective_result(corrected, waiting_for_owner=True)
    requested = store.edit_job(goal.id, origin=origin, run_now=True)
    assert store.due_jobs() == [requested]


async def test_unrelated_thread_cannot_answer_owner_question(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="slack", conversation_id="chat", sender_id="owner")
    goal = store.create_job(prompt="ship", mode="objective", origin=origin)
    store.journal = journal = JobJournal(InMemoryStore(), "test")
    report = ObjectiveReport(status="blocked", summary="pending", decision_question="Ship?")
    notice = await journal.decision_notice(goal, report)
    await journal.mark_delivered(goal.id, notice, "question")
    reply = ChannelMessage(
        conversation_id="chat",
        sender_id="owner",
        text="yes",
        metadata={"thread_root_message_id": "elsewhere"},
    )
    assert await decision_reply(store, reply, "slack") is None
    assert await journal.open_decision(goal, "question")


async def test_pending_question_survives_progress_until_completion(tmp_path: Path) -> None:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path)
    origin = CronOrigin(channel="slack", conversation_id="chat", sender_id="owner")
    goal = store.create_job(prompt="ship", mode="objective", origin=origin)
    store.journal = journal = JobJournal(InMemoryStore(), "test")
    report = ObjectiveReport(status="blocked", summary="pending", decision_question="Ship?")
    notice = await journal.decision_notice(goal, report)
    await journal.mark_delivered(goal.id, notice, "question")
    await journal.decision_notice(goal, ObjectiveReport(status="partial", summary="progress"))
    assert await journal.open_decision(goal, "question")
    await journal.decision_notice(goal, ObjectiveReport(status="complete", summary="shipped"))
    assert not await journal.open_decision(goal, "question")
