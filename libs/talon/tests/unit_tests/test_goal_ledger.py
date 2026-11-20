from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from deepagents_talon.goals import (
    MAX_ITEM_ATTEMPTS,
    GoalCharter,
    GoalLedger,
    GoalLedgerCorruptError,
    GoalLedgerError,
    GoalScope,
    ItemChange,
)

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.goals import Goal, WorkItem

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
OWNER = "U-owner"
SCOPE = GoalScope(channel="slack", conversation_id="C1", owner_id=OWNER)
SETTLED = GoalCharter(
    outcome="drive security outcomes",
    sources=("linear:SEC",),
    progress_test="weekly count of closed SEC issues",
)
LEASE = timedelta(minutes=10)


@pytest.fixture
def ledger(tmp_path: Path) -> GoalLedger:
    return GoalLedger(goal_dir=tmp_path / "goals")


def _active(ledger: GoalLedger) -> Goal:
    goal = ledger.create_draft(scope=SCOPE, charter=SETTLED, now=NOW)
    ledger.submit_for_confirmation(goal.id, owner_id=OWNER, expected_revision=1, now=NOW)
    ledger.confirm(goal.id, owner_id=OWNER, revision=1, now=NOW)
    return ledger.activate(goal.id, goal_job_id="job-goal", now=NOW)


def _planned(ledger: GoalLedger, goal: Goal, key: str = "linear:SEC-1") -> WorkItem:
    item, _ = ledger.add_candidate(goal.id, source_key=key, source_version="v1", now=NOW)
    return ledger.update_item(item.id, ItemChange(status="planned"), now=NOW)


def _awaiting_decision(ledger: GoalLedger, goal: Goal) -> WorkItem:
    item = _planned(ledger, goal)
    run = ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=NOW)
    ledger.finish_run(run.id, result="ok", change=ItemChange(status="awaiting_decision"), now=NOW)
    return item


def test_ledger_persists_privately_across_instances(tmp_path: Path) -> None:
    goal = _active(GoalLedger(goal_dir=tmp_path / "goals"))

    reopened = GoalLedger(goal_dir=tmp_path / "goals")

    assert reopened.get_goal(goal.id) == goal
    assert stat.S_IMODE((tmp_path / "goals").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "goals" / "ledger.json").stat().st_mode) == 0o600


def test_unreadable_ledger_fails_closed_without_overwrite(ledger: GoalLedger) -> None:
    ledger.goal_dir.mkdir(parents=True)
    ledger.path.write_text(json.dumps({"version": 99, "goals": []}))

    with pytest.raises(GoalLedgerCorruptError):
        ledger.create_draft(scope=SCOPE, charter=SETTLED, now=NOW)

    assert json.loads(ledger.path.read_text())["version"] == 99


def test_rejected_run_update_preserves_lease(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _planned(ledger, goal)
    run = ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=NOW)
    before = ledger.path.read_bytes()

    with pytest.raises(ValidationError, match="detail"):
        ledger.finish_run(run.id, result="error", detail="x" * 5000, now=NOW)
    with pytest.raises(GoalLedgerError, match="artifact"):
        ledger.finish_run(
            run.id,
            result="ok",
            change=ItemChange(status="done", outcome="verified", artifact="   "),
            now=NOW,
        )

    assert ledger.path.read_bytes() == before
    finished = ledger.finish_run(run.id, result="error", detail="  retry later  ", now=NOW)
    assert finished.detail == "retry later"
    assert ledger.list_runs(goal.id) == [finished]


def test_activation_requires_settled_charter_and_owner_confirmation(ledger: GoalLedger) -> None:
    goal = ledger.create_draft(scope=SCOPE, charter=GoalCharter(outcome="ship it"), now=NOW)

    with pytest.raises(GoalLedgerError, match="sources, progress_test"):
        ledger.submit_for_confirmation(goal.id, owner_id=OWNER, expected_revision=1)
    ledger.revise_charter(goal.id, owner_id=OWNER, charter=SETTLED, expected_revision=1)
    ledger.submit_for_confirmation(goal.id, owner_id=OWNER, expected_revision=2)
    with pytest.raises(GoalLedgerError, match="owner"):
        ledger.confirm(goal.id, owner_id="U-other", revision=2)
    with pytest.raises(GoalLedgerError, match="revision"):
        ledger.confirm(goal.id, owner_id=OWNER, revision=1)
    with pytest.raises(GoalLedgerError, match="no confirmed activation"):
        ledger.activate(goal.id, goal_job_id="job")

    confirmed = ledger.confirm(goal.id, owner_id=OWNER, revision=2)

    assert confirmed.state == "awaiting_confirmation"
    assert confirmed.activation_pending
    active = ledger.activate(goal.id, goal_job_id="job")
    assert (active.state, active.confirmed_revision, active.activation_pending) == (
        "active",
        2,
        False,
    )


def test_one_open_goal_per_conversation(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    other_owner = SCOPE.model_copy(update={"owner_id": "U-other"})

    with pytest.raises(GoalLedgerError, match="already has a goal"):
        ledger.create_draft(scope=other_owner, charter=SETTLED)
    ledger.remove(goal.id, owner_id=OWNER)

    assert ledger.create_draft(scope=SCOPE, charter=SETTLED).state == "draft"


def test_editing_active_goal_keeps_confirmed_scope_until_reconfirmed(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    broader = SETTLED.model_copy(update={"sources": ("linear:SEC", "github:repo")})

    edited = ledger.revise_charter(goal.id, owner_id=OWNER, charter=broader, expected_revision=1)

    assert edited.state == "active"
    assert edited.confirmed_charter == SETTLED
    reconfirmed = ledger.confirm(goal.id, owner_id=OWNER, revision=2)
    assert reconfirmed.confirmed_charter == broader


def test_candidates_are_deduplicated_by_source_change(ledger: GoalLedger) -> None:
    goal = _active(ledger)

    first, created = ledger.add_candidate(goal.id, source_key=" linear:SEC-1 ", source_version="v1")
    again, created_again = ledger.add_candidate(
        goal.id, source_key="linear:SEC-1", source_version="v1"
    )
    newer, created_newer = ledger.add_candidate(
        goal.id, source_key="linear:SEC-1", source_version="v2"
    )

    assert (created, created_again, created_newer) == (True, False, True)
    assert again.id == first.id
    assert newer.id != first.id


def test_lease_is_exclusive_and_fenced(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _planned(ledger, goal)
    run = ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=NOW)

    with pytest.raises(GoalLedgerError, match="running lease"):
        ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=NOW)
    with pytest.raises(GoalLedgerError, match="lost its lease"):
        ledger.finish_run(
            run.id,
            result="ok",
            change=ItemChange(status="done", outcome="unverified"),
            now=NOW + LEASE,
        )

    expired = ledger.reconcile_leases(now=NOW + LEASE)

    assert [r.result for r in expired] == ["unknown"]
    assert ledger.get_item(item.id).status == "needs_attention"
    with pytest.raises(GoalLedgerError, match="already finished"):
        ledger.finish_run(run.id, result="ok", now=NOW)


def test_lease_requires_confirmed_revision_and_active_goal(ledger: GoalLedger) -> None:
    goal = _active(ledger)

    with pytest.raises(GoalLedgerError, match="confirmed revision"):
        ledger.lease_run(goal.id, goal_revision=2, ttl=LEASE)
    ledger.set_state(goal.id, "paused")
    with pytest.raises(GoalLedgerError, match="not active"):
        ledger.lease_run(goal.id, goal_revision=1, ttl=LEASE)


def test_failed_runs_back_off_then_fail(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _planned(ledger, goal)
    current = NOW

    for _ in range(MAX_ITEM_ATTEMPTS):
        run = ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=current)
        ledger.finish_run(run.id, result="error", now=current)
        retry_at = ledger.get_item(item.id).retry_at
        if retry_at is not None:
            with pytest.raises(GoalLedgerError, match="backing off"):
                ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=current)
            current = retry_at

    assert ledger.get_item(item.id).status == "failed"


def test_done_requires_recorded_outcome(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _planned(ledger, goal)
    run = ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=NOW)

    with pytest.raises(GoalLedgerError, match="outcome"):
        ledger.finish_run(run.id, result="ok", change=ItemChange(status="done"), now=NOW)
    with pytest.raises(GoalLedgerError, match="artifact"):
        ledger.finish_run(
            run.id, result="ok", change=ItemChange(status="done", outcome="verified"), now=NOW
        )
    ledger.finish_run(
        run.id,
        result="ok",
        change=ItemChange(status="done", outcome="verified", artifact="https://x/1"),
        now=NOW,
    )

    assert ledger.get_item(item.id).outcome == "verified"


def test_approved_decision_reserves_item_for_interactive_turn(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _awaiting_decision(ledger, goal)
    decision = ledger.request_decision(
        item.id, proposal="create SEC ticket", expires_at=NOW + timedelta(days=1), now=NOW
    )

    with pytest.raises(GoalLedgerError, match="owner"):
        ledger.decide(decision.id, owner_id="U-other", approve=True, proposal="create SEC ticket")
    with pytest.raises(GoalLedgerError, match="proposal changed"):
        ledger.decide(decision.id, owner_id=OWNER, approve=True, proposal="delete SEC project")
    decided = ledger.decide(
        decision.id, owner_id=OWNER, approve=True, proposal="create SEC ticket", now=NOW
    )

    assert decided.state == "approved"
    assert ledger.get_item(item.id).status == "approved_for_interactive_turn"
    with pytest.raises(GoalLedgerError, match="planned"):
        ledger.lease_run(goal.id, goal_revision=1, item_id=item.id, ttl=LEASE, now=NOW)
    running = ledger.update_item(item.id, ItemChange(status="running_interactive"), now=NOW)
    assert running.status == "running_interactive"


@pytest.mark.parametrize(
    ("answer", "decided_at", "decision_state"),
    [
        ("reject", NOW, "rejected"),
        ("approve", NOW + timedelta(days=2), "expired"),
    ],
)
def test_rejected_or_expired_decision_abandons_item(
    ledger: GoalLedger, answer: str, decided_at: datetime, decision_state: str
) -> None:
    goal = _active(ledger)
    item = _awaiting_decision(ledger, goal)
    decision = ledger.request_decision(
        item.id, proposal="create SEC ticket", expires_at=NOW + timedelta(days=1), now=NOW
    )

    decided = ledger.decide(
        decision.id,
        owner_id=OWNER,
        approve=answer == "approve",
        proposal="create SEC ticket",
        now=decided_at,
    )

    assert decided.state == decision_state
    assert ledger.get_item(item.id).status == "abandoned"


def test_scope_edit_invalidates_pending_decisions(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _awaiting_decision(ledger, goal)
    decision = ledger.request_decision(
        item.id, proposal="create SEC ticket", expires_at=NOW + timedelta(days=1), now=NOW
    )
    ledger.revise_charter(goal.id, owner_id=OWNER, charter=SETTLED, expected_revision=1)
    ledger.confirm(goal.id, owner_id=OWNER, revision=2)

    with pytest.raises(GoalLedgerError, match="already 'invalidated'"):
        ledger.decide(decision.id, owner_id=OWNER, approve=True, proposal="create SEC ticket")
    assert ledger.get_item(item.id).status == "abandoned"


def test_remove_invalidates_work_and_reports_owned_jobs(ledger: GoalLedger) -> None:
    goal = _active(ledger)
    item = _planned(ledger, goal)
    ledger.record_item_job(item.id, wake_key="recheck", job_id="job-item")
    ledger.record_item_job(item.id, wake_key="recheck", job_id="job-item")

    with pytest.raises(GoalLedgerError, match="already has job"):
        ledger.record_item_job(item.id, wake_key="recheck", job_id="job-other")
    removed = ledger.remove(goal.id, owner_id=OWNER)

    assert removed.state == "removed"
    assert ledger.get_item(item.id).status == "abandoned"
    assert ledger.owned_job_ids(goal.id) == ("job-goal", "job-item")
    with pytest.raises(GoalLedgerError, match="cannot move"):
        ledger.set_state(goal.id, "active")


def test_outbox_is_idempotent_and_tracks_uncertain_delivery(ledger: GoalLedger) -> None:
    goal = _active(ledger)

    event, created = ledger.enqueue(goal.id, event_id="digest-1", payload="weekly status")
    _, created_again = ledger.enqueue(goal.id, event_id="digest-1", payload="weekly status")
    with pytest.raises(GoalLedgerError, match="different content"):
        ledger.enqueue(goal.id, event_id="digest-1", payload="other status")
    unknown = ledger.record_delivery("digest-1", state="unknown")
    sent = ledger.record_delivery("digest-1", state="sent", receipt="ts-1")

    assert (created, created_again) == (True, False)
    assert event.destination == SCOPE
    assert (unknown.state, sent.state, sent.attempts) == ("unknown", "sent", 2)
    with pytest.raises(GoalLedgerError, match="already sent"):
        ledger.record_delivery("digest-1", state="sent")


def test_charter_rejects_one_shot_cadence() -> None:
    with pytest.raises(ValidationError, match="recurring"):
        GoalCharter(outcome="x", cadence="in 30m")
