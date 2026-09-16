"""Shared size limits and status vocabulary for model-visible goal state."""

from __future__ import annotations

import html
from typing import Final, Literal, get_args

GoalStatus = Literal["active", "paused", "blocked", "complete"]
"""Lifecycle status of a TUI-owned goal.

`active` and `blocked` are unfinished working states, `paused` preserves the goal
without driving work, and `complete` is terminal. A blocked goal is still
considered actionable by the goal-state notice, whereas a paused goal is
unfinished but not actionable.

Declared in this leaf module, not beside the state schema, so both normalizers
can share one vocabulary: `resume_state.coerce_goal_status` and
`goal_state_notice.project_goal_state`. The notice path deliberately avoids
`resume_state`'s heavy `deepagents` import, and this module depends only on
`typing`, so it is reachable from either side at no startup cost.
"""

GOAL_STATUS_VALUES: Final[frozenset[str]] = frozenset(get_args(GoalStatus))
"""Every recognized `GoalStatus` value, derived so it cannot drift."""

GOAL_OBJECTIVE_CHAR_LIMIT: Final = 8_000
"""Maximum characters in one persisted goal objective."""

RUBRIC_CHAR_LIMIT: Final = 12_000
"""Maximum characters in one rubric or goal acceptance-criteria value."""

GOAL_APPLICATION_CHAR_LIMIT: Final = 12_000
"""Maximum combined raw characters in an accepted objective and its criteria.

Published to the criteria model and enforced per field on raw text, so it stays
comparable to the per-field `max_length` values the schema publishes. The
rendered budget — what an accepted pair is allowed to occupy after HTML
escaping — is `GOAL_NOTICE_TEXT_CHAR_LIMIT`, which is wider.
"""

GOAL_STATUS_NOTE_CHAR_LIMIT: Final = 4_000
"""Maximum characters in a completion-evidence or blocker note."""

GOAL_NOTICE_TEXT_CHAR_LIMIT: Final = (
    GOAL_APPLICATION_CHAR_LIMIT + GOAL_STATUS_NOTE_CHAR_LIMIT
)
"""Maximum combined rendered text embedded in one goal-state notice.

Derived rather than written out, because the acceptance checks share it: an
accepted objective-and-criteria pair is validated against this exact value, so
a hand-written literal could drift apart from what `_accept_goal_rubric`
enforces and let an accepted goal fail its own notice. The acceptance checks
reuse it so a newly accepted goal always keeps the whole status-note budget in
reserve: accepting an application can never produce a goal whose next status
update is already unwritable, no matter how the note is escaped.

The application budget covers objective and criteria, so for ordinary text the
aggregate check here is only reachable via `prior_blocker` or an oversized
note. HTML escaping is what makes it live for any section: it expands embedded
text up to fivefold, and nothing limits a status note's escaped length, so the
aggregate is the only check that bounds what `update_goal` can leave embedded.
Raise this deliberately if a new section is ever embedded in a notice.
"""


GoalStateSizeLabel = Literal[
    "Goal objective",
    "Rubric",
    "Goal status note",
    "Prior blocker",
    "Goal objective and criteria combined",
    "Goal-state notice text",
]
"""Every value a `GoalStateSizeError` can name.

A closed set rather than free text: the label reaches both the user and the
model verbatim, so a consumer that wants to branch on which budget was
exceeded can do so exhaustively instead of matching strings.
"""

GoalStatusNoteLabel = Literal["Goal status note", "Prior blocker"]
"""The labels that can name the status-note budget.

Narrower than `GoalStateSizeLabel` so `validate_goal_status_note` cannot be asked
to report a note overflow under a budget that has nothing to do with notes — the
message reaches the model verbatim, naming a field the notice does not carry.
"""


class GoalStateSizeError(ValueError):
    """Goal/rubric text exceeds a model-visible context budget."""

    def __init__(
        self,
        label: GoalStateSizeLabel,
        actual: int,
        limit: int,
    ) -> None:
        """Initialize a consistent user- and model-facing size error.

        Args:
            label: Name of the oversized value, as the user and model see it.
            actual: Actual character count.
            limit: Maximum accepted character count.

        Raises:
            ValueError: If `actual` does not exceed `limit`. Constructing the
                error for text that fits would render a negative excess
                ("Remove at least -998 characters"), so the caller has a bug.
        """
        if actual <= limit:
            msg = (
                f"GoalStateSizeError needs actual > limit; got "
                f"actual={actual}, limit={limit} for {label!r}."
            )
            raise ValueError(msg)
        self.label = label
        self.actual = actual
        self.limit = limit
        self.excess = actual - limit
        msg = (
            f"{label} is {actual:,} characters; maximum is {limit:,}. "
            f"Remove at least {self.excess:,} characters."
        )
        super().__init__(msg)


def validate_goal_objective(objective: str) -> None:
    """Reject a goal objective that cannot fit its persistent context budget.

    This checks raw length only. HTML escaping can still expand the text
    fivefold, so a caller about to run criteria generation should also call
    `validate_goal_objective_rendered`: an objective that passes here can still
    leave no room for any criteria in the rendered notice.

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
    """  # noqa: DOC502 - propagates from the per-field and total validators
    validate_goal_objective(objective)
    validate_rubric(criteria)
    validate_goal_application_total(objective, criteria)
    # Raw lengths passing says nothing about the rendered notice: HTML escaping
    # expands text up to fivefold, so a pair that fits its raw budget can still
    # overflow the notice and come back as "unavailable — do not work toward
    # it" the moment it is accepted. Validate the escaped text the notice will
    # actually embed, against the wider aggregate budget so the goal keeps room
    # for a status note.
    validate_goal_application_rendered_total(objective, criteria)


def validate_goal_application_total(objective: str, criteria: str) -> None:
    """Validate only the combined budget, not either field's own limit.

    Split out because the two kinds of overshoot need opposite handling in the
    criteria agent's structured-output loop: a per-field overshoot is a model
    mistake against a limit the schema publishes, so it is worth retrying, while
    the combined budget is not in the schema at all and half of it is the user's
    own objective. See `_raise_terminal_goal_state_size_error`.

    Args:
        objective: Goal objective that will be activated.
        criteria: Acceptance criteria that will be activated with the goal.

    Raises:
        GoalStateSizeError: If the combined text exceeds
            `GOAL_APPLICATION_CHAR_LIMIT`.
    """
    total = len(objective) + len(criteria)
    if total > GOAL_APPLICATION_CHAR_LIMIT:
        label = "Goal objective and criteria combined"
        raise GoalStateSizeError(
            label=label,
            actual=total,
            limit=GOAL_APPLICATION_CHAR_LIMIT,
        )


def validate_goal_application_rendered_total(objective: str, criteria: str) -> None:
    """Validate the escaped size an accepted pair will occupy in the notice.

    Split from `validate_goal_application_total` because the two budgets answer
    different questions. The raw combined check shares its budget with the
    criteria model: the schema's per-field `max_length` values and the system
    prompt's combined limit are all raw counts the model can count against
    while drafting. This rendered check is acceptance-side only — the model has
    no way to predict HTML expansion, so a rendered overshoot is reported to
    the user, not retried.

    The budget is the aggregate notice limit rather than the raw application
    limit. Escaping has no meaningful worst case below fivefold expansion, and
    any budget tighter than `GOAL_NOTICE_TEXT_CHAR_LIMIT` would reject pairs
    whose notices fit. The wider budget is safe because acceptance clears the
    status note on `create`, and the note budget stays in reserve otherwise.

    Args:
        objective: Goal objective that will be activated.
        criteria: Acceptance criteria that will be activated with the goal.

    Raises:
        GoalStateSizeError: If the escaped combined text exceeds
            `GOAL_NOTICE_TEXT_CHAR_LIMIT`.
    """
    total = len(html.escape(objective, quote=False)) + len(
        html.escape(criteria, quote=False)
    )
    if total > GOAL_NOTICE_TEXT_CHAR_LIMIT:
        label = "Goal objective and criteria combined"
        raise GoalStateSizeError(
            label=label,
            actual=total,
            limit=GOAL_NOTICE_TEXT_CHAR_LIMIT,
        )


def validate_goal_objective_rendered(objective: str) -> None:
    """Reject an objective whose escaped text alone cannot fit a notice.

    Args:
        objective: Goal objective proposed by the user, before criteria exist.

    Raises:
        GoalStateSizeError: If the escaped objective leaves no room for any
            criteria within `GOAL_NOTICE_TEXT_CHAR_LIMIT`. The limit is reported
            one lower so the invariant that `actual` exceeds `limit` holds at
            the exact boundary, where escaping has consumed the entire budget.
    """
    rendered = len(html.escape(objective, quote=False))
    if rendered >= GOAL_NOTICE_TEXT_CHAR_LIMIT:
        label = "Goal objective"
        raise GoalStateSizeError(
            label=label,
            actual=rendered,
            limit=GOAL_NOTICE_TEXT_CHAR_LIMIT - 1,
        )


def validate_goal_status_note(
    note: str,
    *,
    label: GoalStatusNoteLabel = "Goal status note",
) -> None:
    """Reject a goal status note that cannot fit its persistent context budget.

    Args:
        note: Completion evidence or blocker explanation.
        label: Name of the value in the raised message. Override it when the
            note is not the live status note, because the message reaches the
            model verbatim and would otherwise name a field the notice does not
            carry. Restricted to the two note budgets, so it cannot name an
            unrelated one.

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
        len(html.escape(value, quote=False))
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
