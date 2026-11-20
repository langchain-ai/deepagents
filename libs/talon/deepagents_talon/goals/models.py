"""Records persisted by the goal ledger.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import hashlib
from datetime import date  # noqa: TC003  # pydantic resolves field types at runtime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from deepagents_talon.cron.errors import CronJobError
from deepagents_talon.cron.jobs import CronSchedule

GOAL_LEDGER_VERSION = 1
"""Schema version of `ledger.json`."""

DEFAULT_CADENCE = "every 24h"
"""Default goal wake cadence: one goal cron fire per day."""

MAX_TEXT = 4000
MAX_SHORT_TEXT = 500
MAX_SCOPE_ENTRIES = 32

GoalState = Literal[
    "draft",
    "awaiting_confirmation",
    "active",
    "paused",
    "needs_attention",
    "removed",
]
WorkStatus = Literal[
    "candidate",
    "planned",
    "running",
    "waiting_for_source",
    "awaiting_decision",
    "approved_for_interactive_turn",
    "running_interactive",
    "needs_attention",
    "done",
    "failed",
    "abandoned",
]
RunResult = Literal["running", "ok", "error", "timeout", "unknown"]
DecisionState = Literal["pending", "approved", "rejected", "expired", "invalidated"]
DeliveryState = Literal["pending", "sent", "unknown", "withheld"]
Outcome = Literal["verified", "unverified"]

GOAL_TRANSITIONS: dict[GoalState, frozenset[GoalState]] = {
    "draft": frozenset({"awaiting_confirmation", "removed"}),
    "awaiting_confirmation": frozenset({"draft", "active", "removed"}),
    "active": frozenset({"paused", "needs_attention", "removed"}),
    "paused": frozenset({"active", "needs_attention", "removed"}),
    "needs_attention": frozenset({"active", "removed"}),
    "removed": frozenset(),
}
"""Allowed goal state changes; `removed` is terminal."""


def _to(*states: WorkStatus) -> frozenset[WorkStatus]:
    return frozenset(states)


WORK_TRANSITIONS: dict[WorkStatus, frozenset[WorkStatus]] = {
    "candidate": _to("planned", "abandoned"),
    "planned": _to("running", "abandoned"),
    "running": _to(
        "done", "waiting_for_source", "awaiting_decision", "failed", "planned", "needs_attention"
    ),
    "waiting_for_source": _to("planned", "abandoned"),
    "awaiting_decision": _to("approved_for_interactive_turn", "abandoned"),
    "approved_for_interactive_turn": _to("running_interactive", "abandoned"),
    "running_interactive": _to("done", "failed", "needs_attention"),
    "needs_attention": _to("done", "abandoned"),
    "failed": _to("abandoned"),
    "done": _to(),
    "abandoned": _to(),
}
"""Allowed work item status changes.

Running items cannot be abandoned directly: their lease must finish or expire
first, so an in-flight turn is never silently orphaned.
"""

TERMINAL_WORK: frozenset[WorkStatus] = frozenset({"done", "abandoned"})

_RECORD_CONFIG = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

ShortText = Annotated[str, Field(min_length=1, max_length=MAX_SHORT_TEXT)]


class GoalScope(BaseModel):
    """Trusted conversation and owner a goal is bound to.

    Every field comes from the host's view of the channel message, never from
    prompt text, so a goal cannot be steered to another chat or owner.
    """

    model_config = _RECORD_CONFIG

    conversation_id: ShortText
    owner_id: ShortText
    channel: str | None = Field(default=None, max_length=MAX_SHORT_TEXT)

    def same_conversation(self, other: GoalScope) -> bool:
        """Whether both scopes address the same channel conversation.

        Args:
            other: Scope to compare against.

        Returns:
            `True` when channel and conversation match, regardless of owner.
        """
        return (self.channel, self.conversation_id) == (other.channel, other.conversation_id)


class GoalCharter(BaseModel):
    """Owner-negotiated outcome, scope, and success test for a goal."""

    model_config = _RECORD_CONFIG

    outcome: str = Field(min_length=1, max_length=MAX_TEXT)
    sources: tuple[str, ...] = Field(default=(), max_length=MAX_SCOPE_ENTRIES)
    exclusions: tuple[str, ...] = Field(default=(), max_length=MAX_SCOPE_ENTRIES)
    progress_test: str | None = Field(default=None, max_length=MAX_TEXT)
    kpi: str | None = Field(default=None, max_length=MAX_TEXT)
    baseline: str | None = Field(default=None, max_length=MAX_TEXT)
    horizon: str | None = Field(default=None, max_length=MAX_SHORT_TEXT)
    review_on: date | None = None
    cadence: str = Field(default=DEFAULT_CADENCE, max_length=MAX_SHORT_TEXT)
    decisions_required: tuple[str, ...] = Field(default=(), max_length=MAX_SCOPE_ENTRIES)

    @field_validator("sources", "exclusions", "decisions_required")
    @classmethod
    def _check_entries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        entries = tuple(dict.fromkeys(entry.strip() for entry in value))
        if any(not entry or len(entry) > MAX_SHORT_TEXT for entry in entries):
            msg = f"scope entries must be 1-{MAX_SHORT_TEXT} characters"
            raise ValueError(msg)
        return entries

    @field_validator("cadence")
    @classmethod
    def _check_cadence(cls, value: str) -> str:
        try:
            schedule = CronSchedule.parse(value)
        except CronJobError as exc:
            raise ValueError(str(exc)) from exc
        if schedule.kind != "recurring":
            msg = "goal cadence must be a recurring schedule"
            raise ValueError(msg)
        return value

    def unresolved(self) -> tuple[str, ...]:
        """Name the decisions the owner must still settle before confirming.

        Returns:
            Missing charter fields; empty when the charter can be confirmed.
        """
        missing: list[str] = []
        if not self.sources:
            missing.append("sources")
        if not self.progress_test:
            missing.append("progress_test")
        return tuple(missing)


class Goal(BaseModel):
    """A continuing, owner-confirmed outcome and its scheduling state.

    `charter` is the latest proposed revision; `confirmed_charter` is the one
    the owner explicitly confirmed and the only one work turns may act on.
    """

    model_config = _RECORD_CONFIG

    id: str
    scope: GoalScope
    state: GoalState
    revision: int = Field(ge=1)
    charter: GoalCharter
    confirmed_revision: int | None = None
    confirmed_charter: GoalCharter | None = None
    activation_pending: bool = False
    goal_job_id: str | None = None
    attention_reason: str | None = Field(default=None, max_length=MAX_TEXT)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class WorkItem(BaseModel):
    """One bounded candidate action with its evidence and progress."""

    model_config = _RECORD_CONFIG

    id: str
    goal_id: str
    status: WorkStatus
    source_key: ShortText
    source_version: str = Field(default="", max_length=MAX_SHORT_TEXT)
    evidence: str = Field(default="", max_length=MAX_TEXT)
    next_step: str = Field(default="", max_length=MAX_TEXT)
    wake_jobs: dict[str, str] = Field(default_factory=dict)
    artifact: str | None = Field(default=None, max_length=MAX_SHORT_TEXT)
    outcome: Outcome | None = None
    attempts: int = Field(default=0, ge=0)
    retry_at: AwareDatetime | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class Run(BaseModel):
    """One leased goal or work item turn, fenced by its run ID."""

    model_config = _RECORD_CONFIG

    id: str
    goal_id: str
    goal_revision: int
    item_id: str | None = None
    attempt: int = Field(ge=1)
    result: RunResult
    detail: str | None = Field(default=None, max_length=MAX_TEXT)
    usage: dict[str, int] = Field(default_factory=dict)
    started_at: AwareDatetime
    lease_expires_at: AwareDatetime
    ended_at: AwareDatetime | None = None


class HumanDecision(BaseModel):
    """Owner decision bound to one exact proposed action."""

    model_config = _RECORD_CONFIG

    id: str
    goal_id: str
    item_id: str
    owner_id: str
    proposal_hash: str
    goal_revision: int
    state: DecisionState
    expires_at: AwareDatetime
    origin_message: str | None = None
    decided_at: AwareDatetime | None = None


class KpiObservation(BaseModel):
    """An observed KPI value, or an explicit unmeasured marker."""

    model_config = _RECORD_CONFIG

    goal_id: str
    definition: str = Field(min_length=1, max_length=MAX_TEXT)
    value: float | str | None = None
    source: ShortText
    measured_at: AwareDatetime


class OutboxEvent(BaseModel):
    """A notification persisted before any send is attempted."""

    model_config = _RECORD_CONFIG

    event_id: ShortText
    goal_id: str
    destination: GoalScope
    payload: str = Field(min_length=1, max_length=MAX_TEXT)
    payload_digest: str
    state: DeliveryState
    attempts: int = Field(default=0, ge=0)
    receipt: str | None = Field(default=None, max_length=MAX_SHORT_TEXT)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class LedgerSnapshot(BaseModel):
    """Versioned envelope for every ledger record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = GOAL_LEDGER_VERSION
    goals: tuple[Goal, ...] = ()
    items: tuple[WorkItem, ...] = ()
    runs: tuple[Run, ...] = ()
    decisions: tuple[HumanDecision, ...] = ()
    observations: tuple[KpiObservation, ...] = ()
    outbox: tuple[OutboxEvent, ...] = ()


def digest(text: str) -> str:
    """Hash text for proposal binding and notification idempotency.

    Args:
        text: Text to hash.

    Returns:
        Hex SHA-256 digest.
    """
    return hashlib.sha256(text.encode()).hexdigest()
