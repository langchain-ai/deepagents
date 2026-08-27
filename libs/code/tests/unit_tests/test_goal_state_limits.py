"""Tests for model-visible goal and rubric size budgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest

from deepagents_code.goal_state_limits import (
    GOAL_APPLICATION_CHAR_LIMIT,
    GOAL_NOTICE_TEXT_CHAR_LIMIT,
    GOAL_OBJECTIVE_CHAR_LIMIT,
    GOAL_STATUS_NOTE_CHAR_LIMIT,
    RUBRIC_CHAR_LIMIT,
    GoalStateSizeError,
    validate_goal_application,
    validate_goal_application_rendered_total,
    validate_goal_notice_text,
    validate_goal_objective,
    validate_goal_objective_rendered,
    validate_goal_status_note,
    validate_rubric,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_goal_application_accepts_max_raw_pair_with_escape_headroom() -> None:
    """The rendered budget keeps the widest ordinary accepted pair valid.

    `GOAL_APPLICATION_CHAR_LIMIT` is shared with the criteria model, so the
    rendered check must accept everything the raw check does when no escaping
    applies — otherwise a proposal the schema calls valid could be rejected at
    acceptance.
    """
    objective = "o" * GOAL_OBJECTIVE_CHAR_LIMIT
    criteria = "c" * (GOAL_APPLICATION_CHAR_LIMIT - len(objective))

    validate_goal_application(objective, criteria)


def test_rendered_total_leaves_the_status_note_budget_in_reserve() -> None:
    """An accepted pair at the raw limit still fits a full status note.

    The rendered application budget is the whole notice aggregate, so a pair
    that passes it unescaped consumes only `GOAL_APPLICATION_CHAR_LIMIT` of it.
    A maximally long status note then fits without tripping the aggregate, so
    accepting a goal can never make its next status update unwritable.
    """
    objective = "o" * GOAL_OBJECTIVE_CHAR_LIMIT
    criteria = "c" * (GOAL_APPLICATION_CHAR_LIMIT - len(objective))
    validate_goal_application_rendered_total(objective, criteria)

    validate_goal_notice_text(
        objective=objective,
        criteria=criteria,
        status_note="n" * GOAL_STATUS_NOTE_CHAR_LIMIT,
    )


def test_objective_rendered_rejects_objective_with_no_room_for_criteria() -> None:
    """An objective whose escaped text fills the notice cannot become a goal.

    Checked before criteria generation: the criteria model counts raw
    characters, so no proposal it could return would survive acceptance, and
    the failure should reach the user without spending the request.
    """
    validate_goal_objective_rendered("&" * (GOAL_NOTICE_TEXT_CHAR_LIMIT // 5 - 1))

    with pytest.raises(GoalStateSizeError, match="Goal objective") as raised:
        validate_goal_objective_rendered("&" * (GOAL_NOTICE_TEXT_CHAR_LIMIT // 5))

    # The reported limit sits one below the aggregate so the message's
    # "remove at least 1" wording stays truthful at the exact boundary.
    assert raised.value.limit == GOAL_NOTICE_TEXT_CHAR_LIMIT - 1


def test_notice_total_limit_accepts_maximum_ordinary_text() -> None:
    """The aggregate notice budget accepts maximum ordinary embedded text.

    Objective-plus-criteria caps at `GOAL_APPLICATION_CHAR_LIMIT` and the note at
    `GOAL_STATUS_NOTE_CHAR_LIMIT`, which sum to exactly
    `GOAL_NOTICE_TEXT_CHAR_LIMIT`. This pins both halves: the maximal ordinary
    non-blocker notice is accepted, and adding any blocker trips the aggregate
    rather than an individual limit.
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


def test_notice_total_limit_counts_html_escaped_text() -> None:
    """Escaping user text cannot grow a valid notice past its context budget."""
    objective = "&" * GOAL_OBJECTIVE_CHAR_LIMIT

    with pytest.raises(GoalStateSizeError, match="notice text") as raised:
        validate_goal_notice_text(
            objective=objective,
            criteria=None,
            status_note=None,
        )

    assert raised.value.actual == len("&amp;") * GOAL_OBJECTIVE_CHAR_LIMIT
    assert raised.value.limit == GOAL_NOTICE_TEXT_CHAR_LIMIT


def test_status_note_label_is_narrower_than_the_full_size_label_set() -> None:
    """A note overflow must not be reportable under an unrelated budget.

    The label reaches the model verbatim, so naming a budget the notice does not
    carry would tell it to shorten a field that is not the problem.
    """
    from deepagents_code.goal_state_limits import (
        GoalStateSizeLabel,
        GoalStatusNoteLabel,
    )

    note_labels = frozenset(get_args(GoalStatusNoteLabel))
    assert note_labels == {"Goal status note", "Prior blocker"}
    assert note_labels < frozenset(get_args(GoalStateSizeLabel))


def test_every_goal_status_is_recognized_by_both_normalizers() -> None:
    """One vocabulary, so a new member cannot be actionable in only one place.

    `resume_state.coerce_goal_status` discards an unknown status and
    `goal_state_notice.project_goal_state` degrades one to `paused`. The two
    spellings differ because only the notice path has an objective to keep on
    record, but both fail closed, so neither can present an unrecognized status as
    a goal to work toward.
    """
    from deepagents_code.goal_state_limits import GOAL_STATUS_VALUES, GoalStatus
    from deepagents_code.goal_state_notice import project_goal_state
    from deepagents_code.resume_state import coerce_goal_status

    assert frozenset(get_args(GoalStatus)) == GOAL_STATUS_VALUES
    for status in GOAL_STATUS_VALUES:
        assert coerce_goal_status(status) == status
        projected = project_goal_state(
            {"_goal_objective": "ship it", "_goal_status": status},
        )
        assert projected["goal_status"] == status
        assert projected["goal_actionable"] is (status in {"active", "blocked"})

    # An unrecognized status is not silently carried through either path, and
    # neither path lets it drive work.
    assert coerce_goal_status("archived") is None
    projected = project_goal_state(
        {"_goal_objective": "ship it", "_goal_status": "archived"},
    )
    assert projected["goal_status"] == "paused"
    assert projected["goal_actionable"] is False
