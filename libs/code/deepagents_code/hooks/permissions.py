"""Translation between `PermissionRequest` decisions and HITL review payloads.

Shared by the Textual and headless approval paths so both surfaces resolve
hook-driven permission decisions identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        EditDecision,
        RejectDecision,
    )

    from deepagents_code.hooks.models.domain import PermissionRequestDecision

    HITLDecision = ApproveDecision | EditDecision | RejectDecision
    """Element type of `HITLResponse["decisions"]`."""


class PermissionReviewDecision(TypedDict):
    """Client approval decision compatible with HITL resume payloads."""

    type: Literal["approve", "reject"]
    message: NotRequired[str]


@dataclass(frozen=True, slots=True)
class PermissionHookOutcome:
    """Normalized result shared by TUI and headless permission handling."""

    decision: PermissionReviewDecision | None
    interrupt: bool = False

    @property
    def resolved(self) -> bool:
        """Whether a hook decided this call, leaving no human review."""
        return self.decision is not None


_INTERRUPTED = PermissionHookOutcome(
    {"type": "reject", "message": "Permission interrupted by hook"},
    interrupt=True,
)


def permission_hook_outcome(
    decision: PermissionRequestDecision,
) -> PermissionHookOutcome:
    """Translate a hook permission decision into a client review outcome.

    Args:
        decision: Aggregated permission hook decision.

    Returns:
        Shared approval, rejection, or unresolved result.
    """
    if not decision.continue_processing:
        return PermissionHookOutcome(
            {
                "type": "reject",
                "message": decision.stop_reason or "Permission stopped by hook",
            },
            interrupt=True,
        )
    permission = decision.permission
    if permission.behavior == "allow":
        return PermissionHookOutcome({"type": "approve"})
    if permission.behavior == "deny":
        denied = PermissionReviewDecision(type="reject")
        if permission.reason:
            denied["message"] = permission.reason
        return PermissionHookOutcome(denied, interrupt=permission.interrupt)
    return PermissionHookOutcome(None)


@dataclass(frozen=True, slots=True)
class PermissionPlan:
    """How one batch of gated tool calls was resolved by hooks.

    Indices refer to positions in the originating `action_requests` list, so a
    caller can line outcomes back up with its own UI rows.
    """

    outcomes: tuple[PermissionHookOutcome, ...]

    @property
    def interrupted(self) -> bool:
        """Whether any hook demanded the whole batch be abandoned."""
        return any(outcome.interrupt for outcome in self.outcomes)

    @property
    def unresolved_indices(self) -> tuple[int, ...]:
        """Positions still needing human review, in request order."""
        return tuple(
            index for index, outcome in enumerate(self.outcomes) if not outcome.resolved
        )

    @property
    def fully_resolved(self) -> bool:
        """Whether hooks decided every call in the batch."""
        return not self.unresolved_indices

    def as_interrupted(self) -> PermissionPlan:
        """Return a plan whose every call carries a rejection.

        A hook that interrupts abandons the batch, so calls it never decided
        must still be answered to satisfy the HITL contract.

        Returns:
            A plan with no unresolved entries.
        """
        return PermissionPlan(
            tuple(
                outcome if outcome.resolved else _INTERRUPTED
                for outcome in self.outcomes
            )
        )


def merge_permission_decisions(
    plan: PermissionPlan,
    reviewed: Sequence[HITLDecision],
) -> list[HITLDecision]:
    """Interleave hook decisions with human decisions in request order.

    Args:
        plan: Hook resolution for the whole batch.
        reviewed: Decisions for `plan.unresolved_indices`, in the same order.

    Returns:
        One HITL decision per original action request.
    """
    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        RejectDecision,
    )

    reviewed_iter = iter(reviewed)
    merged: list[HITLDecision] = []
    for outcome in plan.outcomes:
        decision = outcome.decision
        if decision is None:
            merged.append(next(reviewed_iter))
        elif decision["type"] == "approve":
            merged.append(ApproveDecision(type="approve"))
        else:
            message = decision.get("message")
            merged.append(
                RejectDecision(type="reject", message=message)
                if message
                else RejectDecision(type="reject")
            )
    return merged
