"""Shared size limits for model-visible goal and rubric state."""

from __future__ import annotations

from typing import Final

GOAL_OBJECTIVE_CHAR_LIMIT: Final = 8_000
"""Maximum characters in one persisted goal objective."""

RUBRIC_CHAR_LIMIT: Final = 12_000
"""Maximum characters in one rubric or goal acceptance-criteria value."""

GOAL_APPLICATION_CHAR_LIMIT: Final = 12_000
"""Maximum combined characters in an accepted objective and its criteria."""

GOAL_STATUS_NOTE_CHAR_LIMIT: Final = 4_000
"""Maximum characters in a completion-evidence or blocker note."""

GOAL_NOTICE_TEXT_CHAR_LIMIT: Final = 16_000
"""Maximum combined raw text embedded in one goal-state notice.

Equal to `GOAL_APPLICATION_CHAR_LIMIT + GOAL_STATUS_NOTE_CHAR_LIMIT`, so the
per-field limits already sum to it. The aggregate check is therefore only
reachable via `prior_blocker`, the one section not covered by the application
budget; on the primary path it is defense in depth rather than a live bound.
Raise this deliberately if a new section is ever embedded in a notice.
"""


class GoalStateSizeError(ValueError):
    """Goal/rubric text exceeds a model-visible context budget."""

    def __init__(self, label: str, actual: int, limit: int) -> None:
        """Initialize a consistent user- and model-facing size error.

        Args:
            label: Human-readable name of the oversized value.
            actual: Actual character count.
            limit: Maximum accepted character count.
        """
        self.label = label
        self.actual = actual
        self.limit = limit
        excess = actual - limit
        msg = (
            f"{label} is {actual:,} characters; maximum is {limit:,}. "
            f"Remove at least {excess:,} characters."
        )
        super().__init__(msg)


def validate_goal_objective(objective: str) -> None:
    """Reject a goal objective that cannot fit its persistent context budget.

    Args:
        objective: Goal objective proposed by the user or criteria model.

    Raises:
        GoalStateSizeError: If `objective` exceeds `GOAL_OBJECTIVE_CHAR_LIMIT`.
    """
    if len(objective) > GOAL_OBJECTIVE_CHAR_LIMIT:
        label = "Goal objective"
        raise GoalStateSizeError(
            label=label,
            actual=len(objective),
            limit=GOAL_OBJECTIVE_CHAR_LIMIT,
        )


def validate_rubric(criteria: str) -> None:
    """Reject criteria that cannot fit their persistent context budget.

    Args:
        criteria: Standalone rubric or goal acceptance criteria.

    Raises:
        GoalStateSizeError: If `criteria` exceeds `RUBRIC_CHAR_LIMIT`.
    """
    if len(criteria) > RUBRIC_CHAR_LIMIT:
        label = "Rubric"
        raise GoalStateSizeError(
            label=label,
            actual=len(criteria),
            limit=RUBRIC_CHAR_LIMIT,
        )


def validate_goal_application(objective: str, criteria: str) -> None:
    """Validate an objective and its generated criteria as one application.

    Args:
        objective: Goal objective that will be activated.
        criteria: Acceptance criteria that will be activated with the goal.

    Raises:
        GoalStateSizeError: If a field or their combined text exceeds its limit.
    """
    validate_goal_objective(objective)
    validate_rubric(criteria)
    total = len(objective) + len(criteria)
    if total > GOAL_APPLICATION_CHAR_LIMIT:
        label = "Goal objective and criteria combined"
        raise GoalStateSizeError(
            label=label,
            actual=total,
            limit=GOAL_APPLICATION_CHAR_LIMIT,
        )


def validate_goal_status_note(note: str, *, label: str = "Goal status note") -> None:
    """Reject a goal status note that cannot fit its persistent context budget.

    Args:
        note: Completion evidence or blocker explanation.
        label: Name of the value in the raised message. Override it when the
            note is not the live status note, because the message reaches the
            model verbatim and would otherwise name a field the notice does not
            carry.

    Raises:
        GoalStateSizeError: If `note` exceeds `GOAL_STATUS_NOTE_CHAR_LIMIT`.
    """
    if len(note) > GOAL_STATUS_NOTE_CHAR_LIMIT:
        raise GoalStateSizeError(
            label=label,
            actual=len(note),
            limit=GOAL_STATUS_NOTE_CHAR_LIMIT,
        )


def validate_goal_notice_text(
    *,
    objective: str | None,
    criteria: str | None,
    status_note: str | None,
    prior_blocker: str | None = None,
) -> None:
    """Validate every user-controlled section of a goal-state notice.

    Args:
        objective: Actionable goal objective, when present.
        criteria: Active acceptance criteria, when present.
        status_note: Current completion evidence or blocker note.
        prior_blocker: Blocker copied into the one-time resume notice.

    Raises:
        GoalStateSizeError: If one field or all embedded text exceeds its limit.
    """
    if objective is not None and criteria is not None:
        validate_goal_application(objective, criteria)
    elif objective is not None:
        validate_goal_objective(objective)
    elif criteria is not None:
        validate_rubric(criteria)
    if status_note is not None:
        validate_goal_status_note(status_note)
    if prior_blocker is not None:
        validate_goal_status_note(prior_blocker, label="Prior blocker")
    total = sum(
        len(value)
        for value in (objective, criteria, status_note, prior_blocker)
        if value is not None
    )
    if total > GOAL_NOTICE_TEXT_CHAR_LIMIT:
        label = "Goal-state notice text"
        raise GoalStateSizeError(
            label=label,
            actual=total,
            limit=GOAL_NOTICE_TEXT_CHAR_LIMIT,
        )
