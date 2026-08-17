"""Tests for model-visible goal and rubric size budgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_code.goal_state_limits import (
    GOAL_APPLICATION_CHAR_LIMIT,
    GOAL_NOTICE_TEXT_CHAR_LIMIT,
    GOAL_OBJECTIVE_CHAR_LIMIT,
    GOAL_STATUS_NOTE_CHAR_LIMIT,
    RUBRIC_CHAR_LIMIT,
    GoalStateSizeError,
    validate_goal_application,
    validate_goal_notice_text,
    validate_goal_objective,
    validate_goal_status_note,
    validate_rubric,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("validator", "limit"),
    [
        (validate_goal_objective, GOAL_OBJECTIVE_CHAR_LIMIT),
        (validate_rubric, RUBRIC_CHAR_LIMIT),
        (validate_goal_status_note, GOAL_STATUS_NOTE_CHAR_LIMIT),
    ],
)
def test_individual_limits_accept_boundary_and_reject_one_over(
    validator: Callable[[str], None],
    limit: int,
) -> None:
    """Every field accepts its exact limit and reports actionable excess."""
    validator("x" * limit)

    with pytest.raises(GoalStateSizeError) as raised:
        validator("x" * (limit + 1))

    assert raised.value.actual == limit + 1
    assert raised.value.limit == limit
    # Anchored: a bare substring check also passes against "1 characters",
    # so it would not catch the singular case regressing.
    assert str(raised.value).endswith("Remove at least 1 characters.")


def test_goal_application_accepts_pair_at_exactly_the_combined_limit() -> None:
    """The combined budget is inclusive, matching the individual validators."""
    objective = "o" * GOAL_OBJECTIVE_CHAR_LIMIT
    criteria = "c" * (GOAL_APPLICATION_CHAR_LIMIT - len(objective))

    validate_goal_application(objective, criteria)


def test_goal_application_rejects_pair_that_only_exceeds_combined_limit() -> None:
    """Individually valid objective and criteria still share one notice budget."""
    objective = "o" * GOAL_OBJECTIVE_CHAR_LIMIT
    criteria = "c" * (GOAL_APPLICATION_CHAR_LIMIT - len(objective) + 1)
    assert len(criteria) <= RUBRIC_CHAR_LIMIT

    with pytest.raises(GoalStateSizeError, match="combined") as raised:
        validate_goal_application(objective, criteria)

    assert raised.value.actual == GOAL_APPLICATION_CHAR_LIMIT + 1
    assert raised.value.limit == GOAL_APPLICATION_CHAR_LIMIT


def test_notice_total_limit_is_reachable_only_with_a_prior_blocker() -> None:
    """The aggregate notice budget guards the resume-only `prior_blocker` field.

    Objective-plus-criteria caps at `GOAL_APPLICATION_CHAR_LIMIT` and the note at
    `GOAL_STATUS_NOTE_CHAR_LIMIT`, which sum to exactly
    `GOAL_NOTICE_TEXT_CHAR_LIMIT`. Without `prior_blocker` the aggregate branch is
    therefore unreachable, so this pins both halves: the maximal non-blocker
    notice is accepted, and adding any blocker to it trips the aggregate rather
    than an individual limit.
    """
    objective = "o" * GOAL_OBJECTIVE_CHAR_LIMIT
    criteria = "c" * (GOAL_APPLICATION_CHAR_LIMIT - len(objective))
    status_note = "n" * GOAL_STATUS_NOTE_CHAR_LIMIT
    assert (
        len(objective) + len(criteria) + len(status_note) == GOAL_NOTICE_TEXT_CHAR_LIMIT
    )

    validate_goal_notice_text(
        objective=objective,
        criteria=criteria,
        status_note=status_note,
    )

    with pytest.raises(GoalStateSizeError, match="notice text") as raised:
        validate_goal_notice_text(
            objective=objective,
            criteria=criteria,
            status_note=status_note,
            prior_blocker="b",
        )

    assert raised.value.actual == GOAL_NOTICE_TEXT_CHAR_LIMIT + 1
    assert raised.value.limit == GOAL_NOTICE_TEXT_CHAR_LIMIT
