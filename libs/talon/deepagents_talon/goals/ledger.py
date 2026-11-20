"""Crash-safe, file-backed ledger for goals and their work.

Talon is an experimental runtime and is subject to change or removal at any time.

The ledger is the durable source of truth for `/goal`: charters, work items,
leased runs, human decisions, KPI observations, and outbox notifications. Each
public mutation reads the file, validates the change against the state
machines in `deepagents_talon.goals.models`, and atomically replaces the file,
so a crash leaves either the previous or the next state on disk.

Like the cron store, a ledger file must have a single owning process. Unlike
the cron store, an unreadable ledger fails closed: it raises and is never
overwritten, because silently starting empty would forget confirmed scopes and
pending decisions.
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

from deepagents_talon.goals.errors import GoalLedgerCorruptError, GoalLedgerError
from deepagents_talon.goals.models import (
    GOAL_TRANSITIONS,
    TERMINAL_WORK,
    WORK_TRANSITIONS,
    DecisionState,
    DeliveryState,
    Goal,
    GoalCharter,
    GoalScope,
    GoalState,
    HumanDecision,
    KpiObservation,
    LedgerSnapshot,
    OutboxEvent,
    Outcome,
    Run,
    RunResult,
    WorkItem,
    WorkStatus,
    digest,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Generator, Mapping

    from pydantic import BaseModel

MAX_ITEM_ATTEMPTS = 3
"""Failed scheduled attempts before a work item is marked `failed`."""

_RETRY_BASE = timedelta(minutes=15)
_RETRY_CAP = timedelta(hours=24)
_OPEN_GOAL_STATES: frozenset[GoalState] = frozenset(GOAL_TRANSITIONS) - {"removed"}
_EDITABLE_GOAL_STATES: frozenset[GoalState] = frozenset({"active", "paused", "needs_attention"})
_LEASE_ONLY: frozenset[WorkStatus] = frozenset({"running", "approved_for_interactive_turn"})
_IN_FLIGHT: frozenset[WorkStatus] = frozenset({"running", "running_interactive"})
_DELIVERY_RESULTS: frozenset[DeliveryState] = frozenset({"sent", "unknown", "withheld"})

_LOCKS: weakref.WeakValueDictionary[Path, threading.RLock] = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()


class _Keyed(Protocol):
    @property
    def id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ItemChange:
    """Requested work item update.

    Args:
        status: Target status.
        next_step: Replacement next step, if any.
        evidence: Replacement evidence, if any.
        artifact: Link to the produced artifact or result.
        outcome: Whether completion was checked against the success criteria.
    """

    status: WorkStatus
    next_step: str | None = None
    evidence: str | None = None
    artifact: str | None = None
    outcome: Outcome | None = None


@dataclass(slots=True)
class _Records:
    """Mutable working copy of one ledger snapshot."""

    goals: list[Goal] = field(default_factory=list)
    items: list[WorkItem] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)
    decisions: list[HumanDecision] = field(default_factory=list)
    observations: list[KpiObservation] = field(default_factory=list)
    outbox: list[OutboxEvent] = field(default_factory=list)

    @classmethod
    def load(cls, snapshot: LedgerSnapshot) -> _Records:
        return cls(
            goals=list(snapshot.goals),
            items=list(snapshot.items),
            runs=list(snapshot.runs),
            decisions=list(snapshot.decisions),
            observations=list(snapshot.observations),
            outbox=list(snapshot.outbox),
        )

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            goals=tuple(self.goals),
            items=tuple(self.items),
            runs=tuple(self.runs),
            decisions=tuple(self.decisions),
            observations=tuple(self.observations),
            outbox=tuple(self.outbox),
        )

    def goal(self, goal_id: str) -> Goal:
        return _find(self.goals, goal_id, "goal")

    def item(self, item_id: str) -> WorkItem:
        return _find(self.items, item_id, "work item")


class GoalLedger:
    """JSON-backed ledger for one assistant's goals.

    Stores sharing a resolved path serialize access within this process; other
    processes are not coordinated.
    """

    def __init__(self, *, goal_dir: Path) -> None:
        """Initialize the ledger.

        Args:
            goal_dir: Private directory that holds `ledger.json`.
        """
        self.goal_dir = goal_dir.resolve()
        self.path = self.goal_dir / "ledger.json"
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(self.path, threading.RLock())

    # Reads

    def get_goal(self, goal_id: str) -> Goal | None:
        """Return a goal by ID, or `None` when absent."""
        return next((goal for goal in self._read().goals if goal.id == goal_id), None)

    def goal_for_conversation(self, scope: GoalScope) -> Goal | None:
        """Return the conversation's goal that has not been removed, if any.

        Args:
            scope: Conversation to look up; the owner is not compared.

        Returns:
            The open goal, or `None`.
        """
        goals = self.list_goals(states=_OPEN_GOAL_STATES)
        return next((goal for goal in goals if goal.scope.same_conversation(scope)), None)

    def list_goals(self, *, states: Collection[GoalState] | None = None) -> list[Goal]:
        """List goals, optionally filtered by state."""
        return [goal for goal in self._read().goals if states is None or goal.state in states]

    def get_item(self, item_id: str) -> WorkItem | None:
        """Return a work item by ID, or `None` when absent."""
        return next((item for item in self._read().items if item.id == item_id), None)

    def list_items(
        self, goal_id: str, *, statuses: Collection[WorkStatus] | None = None
    ) -> list[WorkItem]:
        """List a goal's work items, optionally filtered by status."""
        return [
            item
            for item in self._read().items
            if item.goal_id == goal_id and (statuses is None or item.status in statuses)
        ]

    def list_runs(self, goal_id: str) -> list[Run]:
        """List a goal's runs in lease order."""
        return [run for run in self._read().runs if run.goal_id == goal_id]

    def list_decisions(
        self, goal_id: str, *, states: Collection[DecisionState] | None = None
    ) -> list[HumanDecision]:
        """List a goal's human decisions, optionally filtered by state."""
        return [
            decision
            for decision in self._read().decisions
            if decision.goal_id == goal_id and (states is None or decision.state in states)
        ]

    def list_observations(self, goal_id: str) -> list[KpiObservation]:
        """List a goal's KPI observations in recorded order."""
        return [obs for obs in self._read().observations if obs.goal_id == goal_id]

    def list_outbox(
        self, *, goal_id: str | None = None, states: Collection[DeliveryState] | None = None
    ) -> list[OutboxEvent]:
        """List outbox events, optionally filtered by goal and delivery state."""
        return [
            event
            for event in self._read().outbox
            if (goal_id is None or event.goal_id == goal_id)
            and (states is None or event.state in states)
        ]

    def owned_job_ids(self, goal_id: str) -> tuple[str, ...]:
        """Return every cron job ID the goal and its work items own.

        Args:
            goal_id: Goal to inspect.

        Returns:
            The recurring goal job, if any, followed by work item jobs.
        """
        records = self._read()
        goal = records.goal(goal_id)
        jobs = [goal.goal_job_id] if goal.goal_job_id else []
        for item in records.items:
            if item.goal_id == goal_id:
                jobs.extend(item.wake_jobs.values())
        return tuple(dict.fromkeys(jobs))

    # Goal lifecycle

    def create_draft(
        self, *, scope: GoalScope, charter: GoalCharter, now: datetime | None = None
    ) -> Goal:
        """Persist a new draft goal for a conversation.

        Args:
            scope: Trusted conversation and owner.
            charter: Initial charter, usually just the outcome.
            now: Time override for deterministic tests.

        Returns:
            The draft goal at revision 1.

        Raises:
            GoalLedgerError: If the conversation already has an open goal.
        """
        current = _utc(now)
        with self._editing() as records:
            if any(
                g.state != "removed" and g.scope.same_conversation(scope) for g in records.goals
            ):
                msg = "this conversation already has a goal; edit or remove it first"
                raise GoalLedgerError(msg)
            goal = Goal(
                id=_new_id(),
                scope=scope,
                state="draft",
                revision=1,
                charter=charter,
                created_at=current,
                updated_at=current,
            )
            records.goals.append(goal)
            return goal

    def revise_charter(
        self,
        goal_id: str,
        *,
        owner_id: str,
        charter: GoalCharter,
        expected_revision: int,
        now: datetime | None = None,
    ) -> Goal:
        """Store a new charter revision; an active goal keeps its confirmed one.

        Args:
            goal_id: Goal to revise.
            owner_id: Trusted sender making the change.
            charter: Replacement charter.
            expected_revision: Revision the caller last read.
            now: Time override for deterministic tests.

        Returns:
            The goal at its next revision.
        """
        with self._editing() as records:
            goal = _owned(records.goal(goal_id), owner_id)
            _check_revision(goal, expected_revision)
            if goal.state == "removed":
                msg = "a removed goal cannot be revised"
                raise GoalLedgerError(msg)
            update = {"charter": charter, "revision": goal.revision + 1}
            return _put_goal(records, goal, now, **update)

    def submit_for_confirmation(
        self, goal_id: str, *, owner_id: str, expected_revision: int, now: datetime | None = None
    ) -> Goal:
        """Move a settled draft to `awaiting_confirmation`.

        Raises:
            GoalLedgerError: If the charter still has unresolved decisions.
        """
        with self._editing() as records:
            goal = _owned(records.goal(goal_id), owner_id)
            _check_revision(goal, expected_revision)
            _require_settled(goal.charter)
            return _transition_goal(records, goal, "awaiting_confirmation", now)

    def confirm(
        self, goal_id: str, *, owner_id: str, revision: int, now: datetime | None = None
    ) -> Goal:
        """Record the owner's explicit confirmation of one charter revision.

        A first confirmation records activation intent and leaves the goal in
        `awaiting_confirmation` until `activate` stores its cron job. Confirming
        an edit of a running goal replaces the confirmed charter and invalidates
        pending decisions made under the old scope.

        Args:
            goal_id: Goal to confirm.
            owner_id: Trusted sender confirming.
            revision: Revision the owner was shown.
            now: Time override for deterministic tests.

        Returns:
            The goal with its confirmed charter.
        """
        with self._editing() as records:
            goal = _owned(records.goal(goal_id), owner_id)
            _check_revision(goal, revision)
            _require_settled(goal.charter)
            if goal.state in _EDITABLE_GOAL_STATES and revision != goal.confirmed_revision:
                _invalidate_decisions(records, goal.id, now)
                return _put_goal(records, goal, now, **_confirmed(goal))
            if goal.state != "awaiting_confirmation":
                msg = f"a goal in state {goal.state!r} has nothing to confirm"
                raise GoalLedgerError(msg)
            return _put_goal(records, goal, now, activation_pending=True, **_confirmed(goal))

    def activate(self, goal_id: str, *, goal_job_id: str, now: datetime | None = None) -> Goal:
        """Mark a confirmed goal active once its recurring cron job exists.

        Raises:
            GoalLedgerError: If no confirmation is pending activation.
        """
        with self._editing() as records:
            goal = records.goal(goal_id)
            if goal.state != "awaiting_confirmation" or not goal.activation_pending:
                msg = "goal has no confirmed activation pending"
                raise GoalLedgerError(msg)
            goal = _put_goal(records, goal, now, activation_pending=False, goal_job_id=goal_job_id)
            return _transition_goal(records, goal, "active", now)

    def set_state(
        self,
        goal_id: str,
        state: GoalState,
        *,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> Goal:
        """Pause, resume, or flag a goal for attention.

        Activation and removal have dedicated methods because they carry extra
        checks and side effects.

        Raises:
            GoalLedgerError: If the transition is not allowed.
        """
        if state == "removed":
            msg = "use remove() to remove a goal"
            raise GoalLedgerError(msg)
        with self._editing() as records:
            goal = records.goal(goal_id)
            if state == "active" and goal.state == "awaiting_confirmation":
                msg = "use activate() to activate a confirmed goal"
                raise GoalLedgerError(msg)
            goal = _put_goal(records, goal, now, attention_reason=reason)
            return _transition_goal(records, goal, state, now)

    def remove(self, goal_id: str, *, owner_id: str, now: datetime | None = None) -> Goal:
        """Remove a goal, invalidate its decisions, and abandon idle work.

        In-flight items keep their status so their runs can still report an
        outcome. Callers disable the jobs listed by `owned_job_ids`.

        Returns:
            The removed goal; its record is retained.
        """
        with self._editing() as records:
            goal = _owned(records.goal(goal_id), owner_id)
            _invalidate_decisions(records, goal.id, now)
            for item in list(records.items):
                if item.goal_id == goal.id and item.status not in TERMINAL_WORK | _IN_FLIGHT:
                    _put_item(records, item, now, status="abandoned")
            goal = _put_goal(records, goal, now, activation_pending=False)
            return _transition_goal(records, goal, "removed", now)

    # Work items

    def add_candidate(
        self,
        goal_id: str,
        *,
        source_key: str,
        source_version: str = "",
        evidence: str = "",
        now: datetime | None = None,
    ) -> tuple[WorkItem, bool]:
        """Add a candidate unless the same source change was already seen.

        Args:
            goal_id: Active goal that discovered the candidate.
            source_key: Stable source identifier, such as a link.
            source_version: Change or version marker at that source.
            evidence: Observed change that motivates the item.
            now: Time override for deterministic tests.

        Returns:
            The item and whether it was newly created.
        """
        current = _utc(now)
        key, version = source_key.strip(), source_version.strip()
        with self._editing() as records:
            goal = _require_active(records.goal(goal_id))
            for item in records.items:
                if (item.goal_id, item.source_key, item.source_version) == (goal.id, key, version):
                    return item, False
            item = WorkItem(
                id=_new_id(),
                goal_id=goal.id,
                status="candidate",
                source_key=key,
                source_version=version,
                evidence=evidence,
                created_at=current,
                updated_at=current,
            )
            records.items.append(item)
            return item, True

    def update_item(
        self, item_id: str, change: ItemChange, *, now: datetime | None = None
    ) -> WorkItem:
        """Apply a validated status change to a work item.

        `running` is entered only through `lease_run`, and
        `approved_for_interactive_turn` only through `decide`.

        Raises:
            GoalLedgerError: If the change is not allowed.
        """
        if change.status in _LEASE_ONLY:
            msg = f"{change.status!r} is set by the ledger, not requested directly"
            raise GoalLedgerError(msg)
        with self._editing() as records:
            item = records.item(item_id)
            if item.status == "running" and _running_run(records, item.goal_id, item.id):
                msg = "finish the item's run before changing its status"
                raise GoalLedgerError(msg)
            if change.status == "running_interactive":
                _require_approval(records, item, _utc(now))
            return _apply_change(records, item, change, now)

    def record_item_job(
        self, item_id: str, *, wake_key: str, job_id: str, now: datetime | None = None
    ) -> WorkItem:
        """Record the cron job created for a work item wake, idempotently.

        Raises:
            GoalLedgerError: If the wake key already maps to a different job.
        """
        with self._editing() as records:
            item = records.item(item_id)
            existing = item.wake_jobs.get(wake_key)
            if existing == job_id:
                return item
            if existing is not None:
                msg = f"wake {wake_key!r} already has job {existing!r}"
                raise GoalLedgerError(msg)
            return _put_item(records, item, now, wake_jobs={**item.wake_jobs, wake_key: job_id})

    # Runs

    def lease_run(
        self,
        goal_id: str,
        *,
        goal_revision: int,
        ttl: timedelta,
        item_id: str | None = None,
        now: datetime | None = None,
    ) -> Run:
        """Lease a goal discovery turn or one planned work item.

        Args:
            goal_id: Active goal.
            goal_revision: Confirmed revision the caller read.
            ttl: How long the lease is valid.
            item_id: Planned item to run, or `None` for discovery.
            now: Time override for deterministic tests.

        Returns:
            The running run; its ID fences `finish_run`.

        Raises:
            GoalLedgerError: If the goal is inactive or stale, or the target is
                already leased, backing off, or not planned.
        """
        current = _utc(now)
        if ttl <= timedelta(0):
            msg = "lease ttl must be positive"
            raise GoalLedgerError(msg)
        with self._editing() as records:
            goal = _require_active(records.goal(goal_id))
            if goal_revision != goal.confirmed_revision:
                msg = f"goal revision {goal_revision} is not the confirmed revision"
                raise GoalLedgerError(msg)
            if _running_run(records, goal.id, item_id) is not None:
                msg = "target already has a running lease; reconcile it first"
                raise GoalLedgerError(msg)
            attempt = _start_item(records, goal.id, item_id, current) if item_id else 1
            run = Run(
                id=_new_id(),
                goal_id=goal.id,
                goal_revision=goal_revision,
                item_id=item_id,
                attempt=attempt,
                result="running",
                started_at=current,
                lease_expires_at=current + ttl,
            )
            records.runs.append(run)
            return run

    def finish_run(  # noqa: PLR0913  # the outcome, its detail, and the item change commit together
        self,
        run_id: str,
        *,
        result: RunResult,
        change: ItemChange | None = None,
        detail: str | None = None,
        usage: dict[str, int] | None = None,
        now: datetime | None = None,
    ) -> Run:
        """Record a run outcome and move its work item in the same write.

        A successful item run must say where the item goes next. An error
        retries with backoff until `MAX_ITEM_ATTEMPTS`; a timeout or unknown
        outcome flags the item for attention instead of retrying blindly.

        Raises:
            GoalLedgerError: If the run is not the live lease holder.
        """
        current = _utc(now)
        with self._editing() as records:
            run = _find(records.runs, run_id, "run")
            _check_finish(run, result, change, current)
            run = _put(
                records.runs,
                _updated(
                    run,
                    {
                        "result": result,
                        "detail": detail,
                        "usage": usage or {},
                        "ended_at": current,
                    },
                ),
            )
            if run.item_id is not None:
                _settle_item(records, records.item(run.item_id), result, change, current)
            return run

    def reconcile_leases(self, *, now: datetime | None = None) -> list[Run]:
        """Mark expired leases unknown and flag their items for attention.

        Also flags items left `running` with no live run, which a crash between
        writes cannot produce today but an older ledger might contain.

        Returns:
            Runs whose outcome became unknown.
        """
        current = _utc(now)
        with self._editing() as records:
            expired = [
                r for r in records.runs if r.result == "running" and r.lease_expires_at <= current
            ]
            for run in expired:
                detail = "lease expired before the run reported an outcome"
                _put(records.runs, _updated(run, {"result": "unknown", "detail": detail}))
            for item in list(records.items):
                if (
                    item.status == "running"
                    and _running_run(records, item.goal_id, item.id) is None
                ):
                    _put_item(records, item, current, status="needs_attention")
            return [_find(records.runs, run.id, "run") for run in expired]

    # Human decisions

    def request_decision(
        self,
        item_id: str,
        *,
        proposal: str,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> HumanDecision:
        """Ask the owner to approve one exact proposed action.

        Re-requesting the same proposal returns the pending decision; a changed
        proposal invalidates the old one.

        Args:
            item_id: Item in `awaiting_decision`.
            proposal: Canonical text of the operation, target, and arguments.
            expires_at: When the decision lapses.
            now: Time override for deterministic tests.

        Returns:
            The pending decision.
        """
        current = _utc(now)
        proposal_hash = digest(proposal)
        with self._editing() as records:
            item = records.item(item_id)
            goal = _require_active(records.goal(item.goal_id))
            if item.status != "awaiting_decision" or _utc(expires_at) <= current:
                msg = "decisions need an item awaiting a decision and a future expiry"
                raise GoalLedgerError(msg)
            for pending in _pending_decisions(records, goal.id, item.id):
                if pending.proposal_hash == proposal_hash:
                    return pending
                _put(records.decisions, _updated(pending, {"state": "invalidated"}))
            decision = HumanDecision(
                id=_new_id(),
                goal_id=goal.id,
                item_id=item.id,
                owner_id=goal.scope.owner_id,
                proposal_hash=proposal_hash,
                proposal=proposal,
                goal_revision=goal.confirmed_revision or goal.revision,
                state="pending",
                expires_at=_utc(expires_at),
            )
            records.decisions.append(decision)
            return decision

    def decide(  # noqa: PLR0913  # the decision binds owner, proposal, and origin together
        self,
        decision_id: str,
        *,
        owner_id: str,
        approve: bool,
        proposal: str,
        origin_message: str | None = None,
        now: datetime | None = None,
    ) -> HumanDecision:
        """Record the owner's decision on the exact proposal they were shown.

        A stale decision (expired, or made under an older goal revision or an
        inactive goal) is closed and its item abandoned rather than approved.

        Returns:
            The decision in its final state.

        Raises:
            GoalLedgerError: If the sender is not the owner, the decision is no
                longer pending, or the proposal changed.
        """
        current = _utc(now)
        with self._editing() as records:
            decision = _find(records.decisions, decision_id, "decision")
            _check_decider(decision, owner_id, proposal)
            item = records.item(decision.item_id)
            if item.status != "awaiting_decision":
                msg = f"work item is {item.status!r}, not awaiting a decision"
                raise GoalLedgerError(msg)
            goal = records.goal(decision.goal_id)
            state = _decision_outcome(goal, decision, approve=approve, now=current)
            decided = _updated(
                decision, {"state": state, "origin_message": origin_message, "decided_at": current}
            )
            _put(records.decisions, decided)
            status: WorkStatus = (
                "approved_for_interactive_turn" if state == "approved" else "abandoned"
            )
            _put_item(records, item, current, status=status)
            return decided

    # KPI observations and outbox

    def record_observation(  # an observation is only meaningful with its provenance
        self,
        goal_id: str,
        *,
        definition: str,
        value: float | str | None,
        source: str,
        measured_at: datetime,
    ) -> KpiObservation:
        """Record an observed KPI value; `None` marks it unmeasured."""
        with self._editing() as records:
            goal = records.goal(goal_id)
            observation = KpiObservation(
                goal_id=goal.id,
                definition=definition,
                value=value,
                source=source,
                measured_at=_utc(measured_at),
            )
            records.observations.append(observation)
            return observation

    def enqueue(
        self, goal_id: str, *, event_id: str, payload: str, now: datetime | None = None
    ) -> tuple[OutboxEvent, bool]:
        """Persist a notification for the goal's starting conversation.

        Args:
            goal_id: Goal the notification belongs to.
            event_id: Idempotency key for this notification.
            payload: Full text to deliver.
            now: Time override for deterministic tests.

        Returns:
            The event and whether it was newly created.

        Raises:
            GoalLedgerError: If the event ID was reused for different content.
        """
        current = _utc(now)
        with self._editing() as records:
            goal = records.goal(goal_id)
            existing = next((e for e in records.outbox if e.event_id == event_id), None)
            if existing is not None:
                if existing.payload_digest != digest(payload) or existing.goal_id != goal.id:
                    msg = f"outbox event {event_id!r} already holds different content"
                    raise GoalLedgerError(msg)
                return existing, False
            event = OutboxEvent(
                event_id=event_id,
                goal_id=goal.id,
                destination=goal.scope,
                payload=payload,
                payload_digest=digest(payload),
                state="pending",
                created_at=current,
                updated_at=current,
            )
            records.outbox.append(event)
            return event, True

    def claim_delivery(self, event_id: str) -> bool:
        """Mark a pending notification as unknown before sending it.

        Args:
            event_id: Stable notification identifier.

        Returns:
            Whether this caller claimed the notification.
        """
        with self._editing() as records:
            for index, event in enumerate(records.outbox):
                if event.event_id == event_id and event.state == "pending":
                    records.outbox[index] = _updated(
                        event,
                        {
                            "state": "unknown",
                            "attempts": event.attempts + 1,
                            "updated_at": _utc(None),
                        },
                    )
                    return True
            return False

    def record_delivery(
        self,
        event_id: str,
        *,
        state: DeliveryState,
        receipt: str | None = None,
        now: datetime | None = None,
    ) -> OutboxEvent:
        """Record a delivery attempt's result.

        Raises:
            GoalLedgerError: If the state is not a result or the event was
                already confirmed sent.
        """
        if state not in _DELIVERY_RESULTS:
            msg = f"delivery state must be one of {sorted(_DELIVERY_RESULTS)}"
            raise GoalLedgerError(msg)
        with self._editing() as records:
            event = next((e for e in records.outbox if e.event_id == event_id), None)
            if event is None or event.state == "sent":
                msg = f"outbox event {event_id!r} is missing or already sent"
                raise GoalLedgerError(msg)
            update = {
                "state": state,
                "receipt": receipt,
                "attempts": event.attempts + 1,
                "updated_at": _utc(now),
            }
            updated = _updated(event, update)
            records.outbox[records.outbox.index(event)] = updated
            return updated

    # Storage

    @contextmanager
    def _editing(self) -> Generator[_Records]:
        with self._lock:
            records = _Records.load(self._load())
            yield records
            self._write(records.snapshot())

    def _read(self) -> _Records:
        with self._lock:
            return _Records.load(self._load())

    def _load(self) -> LedgerSnapshot:
        self._ensure_dir()
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return LedgerSnapshot()
        except OSError as exc:
            msg = f"could not read goal ledger {self.path}"
            raise GoalLedgerCorruptError(msg) from exc
        try:
            return LedgerSnapshot.model_validate_json(raw)
        except ValidationError as exc:
            msg = f"goal ledger {self.path} is unreadable; it was left untouched"
            raise GoalLedgerCorruptError(msg) from exc

    def _write(self, snapshot: LedgerSnapshot) -> None:
        fd, name = tempfile.mkstemp(dir=self.goal_dir, prefix=".ledger.", suffix=".tmp")
        tmp_path = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(snapshot.model_dump_json(indent=2))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            tmp_path.chmod(0o600)
            tmp_path.replace(self.path)
            _fsync_dir(self.goal_dir)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _ensure_dir(self) -> None:
        self.goal_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.goal_dir.chmod(0o700)
        if self.path.exists():
            self.path.chmod(0o600)


def _find[T: _Keyed](records: list[T], record_id: str, kind: str) -> T:
    for record in records:
        if record.id == record_id:
            return record
    msg = f"{kind} {record_id!r} not found"
    raise GoalLedgerError(msg)


def _updated[T: BaseModel](record: T, update: Mapping[str, object]) -> T:
    return type(record).model_validate({**record.model_dump(), **update})


def _put[T: _Keyed](records: list[T], record: T) -> T:
    index = next(i for i, existing in enumerate(records) if existing.id == record.id)
    records[index] = record
    return record


def _put_goal(records: _Records, goal: Goal, now: datetime | None, **update: object) -> Goal:
    return _put(records.goals, _updated(goal, {**update, "updated_at": _utc(now)}))


def _put_item(
    records: _Records, item: WorkItem, now: datetime | None, **update: object
) -> WorkItem:
    return _put(records.items, _updated(item, {**update, "updated_at": _utc(now)}))


def _transition_goal(records: _Records, goal: Goal, state: GoalState, now: datetime | None) -> Goal:
    if state not in GOAL_TRANSITIONS[goal.state]:
        msg = f"goal cannot move from {goal.state!r} to {state!r}"
        raise GoalLedgerError(msg)
    reason = goal.attention_reason if state == "needs_attention" else None
    return _put_goal(records, goal, now, state=state, attention_reason=reason)


def _apply_change(
    records: _Records, item: WorkItem, change: ItemChange, now: datetime | None
) -> WorkItem:
    if change.status not in WORK_TRANSITIONS[item.status]:
        msg = f"work item cannot move from {item.status!r} to {change.status!r}"
        raise GoalLedgerError(msg)
    if change.status == "done":
        _require_outcome(change)
    fields = ("next_step", "evidence", "artifact", "outcome")
    update = {name: getattr(change, name) for name in fields if getattr(change, name) is not None}
    return _put_item(records, item, now, status=change.status, **update)


def _require_outcome(change: ItemChange) -> None:
    if change.outcome is None:
        msg = "completed work must record a verified or unverified outcome"
        raise GoalLedgerError(msg)
    if change.outcome == "verified" and not (change.artifact or "").strip():
        msg = "a verified outcome needs an artifact or result link"
        raise GoalLedgerError(msg)


def _start_item(records: _Records, goal_id: str, item_id: str, now: datetime) -> int:
    item = records.item(item_id)
    if item.goal_id != goal_id or item.status != "planned":
        msg = "only a planned item of this goal can be leased"
        raise GoalLedgerError(msg)
    if item.retry_at is not None and item.retry_at > now:
        msg = f"item is backing off until {item.retry_at.isoformat()}"
        raise GoalLedgerError(msg)
    item = _put_item(
        records, item, now, status="running", attempts=item.attempts + 1, retry_at=None
    )
    return item.attempts


def _check_finish(run: Run, result: RunResult, change: ItemChange | None, now: datetime) -> None:
    if run.result != "running":
        msg = f"run {run.id!r} already finished as {run.result!r}"
        raise GoalLedgerError(msg)
    if run.lease_expires_at <= now:
        msg = f"run {run.id!r} lost its lease; reconcile it instead"
        raise GoalLedgerError(msg)
    if result == "running":
        msg = "a finished run needs a final result"
        raise GoalLedgerError(msg)
    if change is not None and (result != "ok" or run.item_id is None):
        msg = "only a successful item run can change its item"
        raise GoalLedgerError(msg)
    if result == "ok" and run.item_id is not None and change is None:
        msg = "a successful item run must say where the item goes next"
        raise GoalLedgerError(msg)


def _settle_item(
    records: _Records, item: WorkItem, result: RunResult, change: ItemChange | None, now: datetime
) -> None:
    if change is not None:
        _apply_change(records, item, change, now)
    elif result == "error" and item.attempts >= MAX_ITEM_ATTEMPTS:
        _put_item(records, item, now, status="failed")
    elif result == "error":
        delay = min(_RETRY_BASE * 2 ** (item.attempts - 1), _RETRY_CAP)
        _put_item(records, item, now, status="planned", retry_at=now + delay)
    else:
        _put_item(records, item, now, status="needs_attention")


def _running_run(records: _Records, goal_id: str, item_id: str | None) -> Run | None:
    for run in records.runs:
        if run.result == "running" and (run.goal_id, run.item_id) == (goal_id, item_id):
            return run
    return None


def _pending_decisions(records: _Records, goal_id: str, item_id: str) -> list[HumanDecision]:
    return [
        d
        for d in records.decisions
        if d.state == "pending" and (d.goal_id, d.item_id) == (goal_id, item_id)
    ]


def _invalidate_decisions(records: _Records, goal_id: str, now: datetime | None) -> None:
    for decision in list(records.decisions):
        if decision.goal_id == goal_id and decision.state == "pending":
            update = {"state": "invalidated", "decided_at": _utc(now)}
            _put(records.decisions, _updated(decision, update))
            item = records.item(decision.item_id)
            if item.status == "awaiting_decision":
                _put_item(records, item, now, status="abandoned")


def _check_decider(decision: HumanDecision, owner_id: str, proposal: str) -> None:
    if decision.owner_id != owner_id:
        msg = "only the goal owner can decide"
        raise GoalLedgerError(msg)
    if decision.state != "pending":
        msg = f"decision is already {decision.state!r}"
        raise GoalLedgerError(msg)
    if decision.proposal_hash != digest(proposal):
        msg = "the proposal changed after the decision was requested"
        raise GoalLedgerError(msg)


def _decision_outcome(
    goal: Goal, decision: HumanDecision, *, approve: bool, now: datetime
) -> DecisionState:
    if decision.expires_at <= now:
        return "expired"
    if goal.state != "active" or goal.confirmed_revision != decision.goal_revision:
        return "invalidated"
    return "approved" if approve else "rejected"


def _require_approval(records: _Records, item: WorkItem, now: datetime) -> None:
    goal = records.goal(item.goal_id)
    for decision in records.decisions:
        if (
            decision.item_id == item.id
            and decision.state == "approved"
            and decision.expires_at > now
            and decision.goal_revision == goal.confirmed_revision
        ):
            return
    msg = "interactive work needs a current, approved decision"
    raise GoalLedgerError(msg)


def _owned(goal: Goal, owner_id: str) -> Goal:
    if goal.scope.owner_id != owner_id:
        msg = "only the goal owner can change this goal"
        raise GoalLedgerError(msg)
    return goal


def _require_active(goal: Goal) -> Goal:
    if goal.state != "active":
        msg = f"goal is {goal.state!r}, not active"
        raise GoalLedgerError(msg)
    return goal


def _require_settled(charter: GoalCharter) -> None:
    missing = charter.unresolved()
    if missing:
        msg = f"charter still needs: {', '.join(missing)}"
        raise GoalLedgerError(msg)


def _check_revision(goal: Goal, revision: int) -> None:
    if revision != goal.revision:
        msg = f"goal is at revision {goal.revision}, not {revision}"
        raise GoalLedgerError(msg)


def _confirmed(goal: Goal) -> dict[str, object]:
    return {"confirmed_revision": goal.revision, "confirmed_charter": goal.charter}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        msg = "ledger timestamps must be timezone-aware"
        raise GoalLedgerError(msg)
    return value.astimezone(UTC)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
