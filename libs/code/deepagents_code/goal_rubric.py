"""Shared helpers for drafting rubric criteria from goal objectives."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage

GOAL_RUBRIC_SYSTEM_PROMPT = (
    "You draft acceptance criteria for a coding agent goal.\n\n"
    "Return only a concise markdown bullet list of criteria the user can review "
    "before work begins. Each criterion should be concrete, testable, and framed "
    "as a definition of done. Include criteria for tests, scope control, and "
    "user-visible behavior when relevant. Do not start implementing the goal."
)

GOAL_AMENDMENT_SYSTEM_PROMPT = (
    "You amend an existing coding-agent goal from user feedback. Preserve every "
    "unaffected acceptance criterion and explicit user constraint. Change only "
    "the objective and criteria needed to incorporate the feedback. Return a "
    "revised objective and a concise markdown bullet list of concrete, testable "
    "acceptance criteria. Do not start implementing the goal."
)


class GoalAmendment(TypedDict):
    """Structured proposed update to an existing goal."""

    objective: str
    criteria: str


def _goal_rubric_human_prompt(
    objective: str,
    *,
    feedback: str | None = None,
    previous_criteria: str | None = None,
) -> str:
    """Build the human prompt for goal criteria generation.

    Args:
        objective: Goal objective to turn into criteria.
        feedback: Optional user feedback for regenerating criteria.
        previous_criteria: Optional criteria the user rejected.

    Returns:
        Prompt text with user-controlled content in explicit boundaries.
    """
    parts = [
        "<goal>",
        objective,
        "</goal>",
    ]
    if feedback:
        parts.extend(
            [
                "",
                (
                    "The user rejected the previous criteria. Regenerate the "
                    "criteria entirely using this feedback; do not merely "
                    "patch the prior list."
                ),
            ]
        )
        if previous_criteria:
            parts.extend(
                [
                    "",
                    "<previous_criteria>",
                    previous_criteria,
                    "</previous_criteria>",
                ]
            )
        parts.extend(
            [
                "",
                "<user_feedback>",
                feedback,
                "</user_feedback>",
            ]
        )
    return "\n".join(parts)


def _goal_amendment_human_prompt(
    objective: str,
    criteria: str,
    feedback: str,
) -> str:
    """Build the bounded prompt for amending an accepted goal.

    Args:
        objective: Current accepted objective.
        criteria: Current accepted criteria.
        feedback: User-requested changes.

    Returns:
        Prompt text with each user-controlled value in an explicit boundary.
    """
    return (
        f"<current_goal>\n{objective}\n</current_goal>\n\n"
        f"<current_criteria>\n{criteria}\n</current_criteria>\n\n"
        f"<user_feedback>\n{feedback}\n</user_feedback>"
    )


def generate_goal_amendment(
    objective: str,
    criteria: str,
    feedback: str,
    *,
    model_spec: str | None,
    model_params: dict[str, Any] | None = None,
    profile_override: dict[str, Any] | None = None,
) -> GoalAmendment:
    """Generate a proposed objective and criteria amendment.

    Args:
        objective: Current accepted objective.
        criteria: Current accepted criteria.
        feedback: User-requested changes.
        model_spec: Model spec used to draft the amendment.
        model_params: Optional model constructor kwargs.
        profile_override: Optional profile metadata overrides.

    Returns:
        Proposed amended objective and criteria.
    """
    from deepagents_code.config import create_model

    result = create_model(
        model_spec,
        extra_kwargs=model_params,
        profile_overrides=profile_override,
    )
    model = result.model.with_structured_output(GoalAmendment)
    response = cast(
        "GoalAmendment",
        model.invoke(
            [
                SystemMessage(content=GOAL_AMENDMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=_goal_amendment_human_prompt(
                        objective,
                        criteria,
                        feedback,
                    )
                ),
            ]
        ),
    )
    return {
        "objective": str(response.get("objective", "")).strip(),
        "criteria": str(response.get("criteria", "")).strip(),
    }


def generate_goal_rubric(
    objective: str,
    *,
    model_spec: str | None,
    model_params: dict[str, Any] | None = None,
    profile_override: dict[str, Any] | None = None,
    feedback: str | None = None,
    previous_criteria: str | None = None,
) -> str:
    """Generate acceptance criteria for a goal objective.

    Args:
        objective: Goal objective to turn into criteria.
        model_spec: Model spec used to draft criteria.
        model_params: Optional model constructor kwargs.
        profile_override: Optional profile metadata overrides.
        feedback: Optional user feedback for regenerating criteria.
        previous_criteria: Optional criteria the user rejected.

    Returns:
        Proposed acceptance criteria text.
    """
    from deepagents_code.config import create_model

    result = create_model(
        model_spec,
        extra_kwargs=model_params,
        profile_overrides=profile_override,
    )
    response = result.model.invoke(
        [
            SystemMessage(content=GOAL_RUBRIC_SYSTEM_PROMPT),
            HumanMessage(
                content=_goal_rubric_human_prompt(
                    objective,
                    feedback=feedback,
                    previous_criteria=previous_criteria,
                )
            ),
        ],
    )
    # On real models `.text` is always a `str` (possibly empty), never `None`.
    # Strip and coerce so a whitespace-only response normalizes to `""` and the
    # caller's empty-rubric branch fires instead of activating a blank rubric.
    # The `or ""` also guards the `None` that test doubles may return.
    return (response.text or "").strip()
