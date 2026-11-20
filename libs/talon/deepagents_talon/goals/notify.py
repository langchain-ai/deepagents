"""Notification text for goal updates and owner decisions.

Talon is an experimental runtime and is subject to change or removal at any time.

Each notification has a stable event ID so the outbox sends it at most once.
Evidence read from sources is quoted only in direct messages: in a shared chat
an update names the item and what is needed, never the private excerpt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_talon.goals.models import Goal, HumanDecision, OutboxEvent, WorkItem

DIGEST_INTERVAL = timedelta(days=7)
"""How often non-urgent progress is bundled into one update."""

DIGEST_PREFIX = "digest:"
_MAX_EVIDENCE = 600


@dataclass(frozen=True, slots=True)
class Notice:
    """One notification ready for the outbox.

    Args:
        event_id: Idempotency key.
        payload: Text to post in the goal's starting chat.
    """

    event_id: str
    payload: str


def decision_notice(goal: Goal, decision: HumanDecision, item: WorkItem) -> Notice:
    """Ask the owner to approve or reject one exact proposed action."""
    lines = [
        f"Goal decision {decision.id} needs your approval.",
        f"Proposed action: {decision.proposal}",
        *_evidence(goal, item),
        f"Expires {_when(decision.expires_at)}.",
        f"Reply /goal {goal.id} approve {decision.id} or /goal {goal.id} reject {decision.id}.",
    ]
    return Notice(f"decision:{decision.id}", "\n".join(lines))


def result_notice(goal: Goal, item: WorkItem) -> Notice:
    """Report a work item completed with a verified result."""
    lines = [f"Goal {goal.id} item {item.id} is done: {item.artifact}", *_evidence(goal, item)]
    return Notice(f"result:{item.id}", "\n".join(lines))


def attention_notice(goal: Goal, item: WorkItem) -> Notice:
    """Report a work item that is blocked or failed and needs the owner."""
    reason = "failed after repeated attempts" if item.status == "failed" else "needs attention"
    lines = [
        f"Goal item {item.id} ({item.source_key}) {reason}.",
        *_evidence(goal, item),
        f"Reply /goal {goal.id} done {item.id} <link> or /goal {goal.id} close {item.id}.",
    ]
    return Notice(f"attention:{item.id}:{item.status}:{item.attempts}", "\n".join(lines))


def digest_due(goal: Goal, outbox: Sequence[OutboxEvent], now: datetime) -> bool:
    """Whether a weekly digest is due for an active goal.

    Args:
        goal: Goal to check.
        outbox: The goal's outbox events.
        now: Current time.

    Returns:
        `True` once `DIGEST_INTERVAL` has passed since the last digest, or since
        the goal was created when it has none.
    """
    digests = [event.created_at for event in outbox if event.event_id.startswith(DIGEST_PREFIX)]
    return now - max(digests, default=goal.created_at) >= DIGEST_INTERVAL


def digest_notice(goal: Goal, items: Sequence[WorkItem], now: datetime) -> Notice:
    """Bundle non-urgent progress into one weekly update."""
    since = now - DIGEST_INTERVAL
    counts = Counter(item.status for item in items)
    recent = sum(item.status == "done" and item.updated_at >= since for item in items)
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    lines = [
        "Weekly goal update.",
        f"Completed this week: {recent}.",
        f"All work: {summary or 'none yet'}.",
        f"Send /goal {goal.id} status for details.",
    ]
    week = now.astimezone(UTC).strftime("%G-W%V")
    return Notice(f"{DIGEST_PREFIX}{goal.id}:{week}", "\n".join(lines))


def _evidence(goal: Goal, item: WorkItem) -> list[str]:
    if not goal.scope.direct or not item.evidence:
        return []
    return [f"Evidence: {item.evidence[:_MAX_EVIDENCE]}"]


def _when(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
