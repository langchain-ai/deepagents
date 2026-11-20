"""Owner decisions on proposed actions, and closing out goal work.

Talon is an experimental runtime and is subject to change or removal at any time.

Approving a decision does not run anything unattended. The approved action is
handed to an ordinary interactive turn from the owner's own message, so every
existing tool approval gate still applies. Its outcome is never assumed: the
item waits for the owner to confirm a result with `/goal done` or `/goal close`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents_talon.goals.errors import GoalLedgerError
from deepagents_talon.goals.ledger import ItemChange

if TYPE_CHECKING:
    from datetime import datetime

    from deepagents_talon.goals.ledger import GoalLedger
    from deepagents_talon.goals.models import Goal, HumanDecision, WorkItem

_CHECK_RESULT = "Check the result, then reply /goal done {item} <link> or /goal close {item}."


@dataclass(frozen=True, slots=True)
class ApprovedAction:
    """An owner-approved action to run in an interactive turn.

    Args:
        goal_id: Goal the action belongs to.
        item_id: Work item now `running_interactive`.
        decision_id: Decision that approved it.
        proposal: Exact action text the owner approved.
    """

    goal_id: str
    item_id: str
    decision_id: str
    proposal: str

    @property
    def prompt(self) -> str:
        """Turn text asking the agent to perform exactly the approved action."""
        return (
            f"The goal owner approved decision {self.decision_id} for exactly this action:\n"
            f"{self.proposal}\n\n"
            "Do only this action. If it cannot be done exactly as written, stop and explain "
            "why. Then report what happened, with a link to the result if there is one."
        )


def approve(
    ledger: GoalLedger,
    goal: Goal,
    decision_id: str,
    *,
    origin_message: str | None,
    now: datetime,
) -> tuple[str, ApprovedAction | None]:
    """Approve one pending decision and reserve its item for an interactive turn.

    Args:
        ledger: Goal ledger.
        goal: Owner-checked goal.
        decision_id: Decision named by the owner.
        origin_message: Channel message that carried the approval.
        now: Current time.

    Returns:
        Reply text, and the action to run when the approval took effect.
    """
    decision = _pending(ledger, goal, decision_id)
    decided = ledger.decide(
        decision.id,
        owner_id=goal.scope.owner_id,
        approve=True,
        proposal=decision.proposal,
        origin_message=origin_message,
        now=now,
    )
    if decided.state != "approved":
        return f"Decision {decision.id} is {decided.state}, so nothing will run.", None
    item = ledger.update_item(decision.item_id, ItemChange(status="running_interactive"), now=now)
    action = ApprovedAction(goal.id, item.id, decision.id, decision.proposal)
    return (
        f"Approved decision {decision.id}. Running it now; tool approvals still apply.",
        action,
    )


def reject(
    ledger: GoalLedger, goal: Goal, decision_id: str, *, origin_message: str | None, now: datetime
) -> str:
    """Reject one pending decision; its item is abandoned and nothing runs."""
    decision = _pending(ledger, goal, decision_id)
    decided = ledger.decide(
        decision.id,
        owner_id=goal.scope.owner_id,
        approve=False,
        proposal=decision.proposal,
        origin_message=origin_message,
        now=now,
    )
    return f"Decision {decision.id} is {decided.state}. Nothing will run."


def interactive_finished(
    ledger: GoalLedger, action: ApprovedAction, *, cancelled: bool, now: datetime
) -> None:
    """Park an approved action's item until the owner confirms its result.

    Args:
        ledger: Goal ledger.
        action: Action whose interactive turn ended.
        cancelled: Whether the turn was stopped before it finished.
        now: Current time.
    """
    item = ledger.get_item(action.item_id)
    if item is None or item.status != "running_interactive":
        return
    prefix = "The approved action was stopped before it finished. " if cancelled else ""
    step = prefix + _CHECK_RESULT.format(item=item.id)
    ledger.update_item(item.id, ItemChange(status="needs_attention", next_step=step), now=now)


def mark_done(ledger: GoalLedger, goal: Goal, text: str, *, now: datetime) -> str:
    """Record the owner's confirmation that an item is done.

    Args:
        ledger: Goal ledger.
        goal: Owner-checked goal.
        text: `<item id> [result link]`.
        now: Current time.

    Returns:
        Reply text.
    """
    item_id, _, link = text.partition(" ")
    item = _goal_item(ledger, goal, item_id)
    outcome = "verified" if link.strip() else "unverified"
    change = ItemChange(status="done", outcome=outcome, artifact=link.strip() or None)
    ledger.update_item(item.id, change, now=now)
    return f"Marked item {item.id} done ({outcome})."


def close(ledger: GoalLedger, goal: Goal, item_id: str, *, now: datetime) -> str:
    """Abandon an item without retrying it."""
    item = _goal_item(ledger, goal, item_id.strip())
    ledger.update_item(item.id, ItemChange(status="abandoned"), now=now)
    return f"Closed item {item.id}. It will not be retried."


def _pending(ledger: GoalLedger, goal: Goal, decision_id: str) -> HumanDecision:
    wanted = decision_id.strip()
    for decision in ledger.list_decisions(goal.id):
        if decision.id == wanted:
            return decision
    msg = f"This goal has no decision {wanted or '<id>'}. Send /goal status to see pending ones."
    raise GoalLedgerError(msg)


def _goal_item(ledger: GoalLedger, goal: Goal, item_id: str) -> WorkItem:
    item = ledger.get_item(item_id)
    if item is None or item.goal_id != goal.id:
        msg = f"This goal has no item {item_id or '<id>'}. Send /goal status to see its items."
        raise GoalLedgerError(msg)
    return item
