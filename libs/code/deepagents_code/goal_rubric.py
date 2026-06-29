"""Shared helpers for drafting rubric criteria from goal objectives."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

GOAL_RUBRIC_SYSTEM_PROMPT = (
    "You draft acceptance criteria for a coding agent goal.\n\n"
    "Return only a concise markdown bullet list of criteria the user can review "
    "before work begins. Each criterion should be concrete, testable, and framed "
    "as a definition of done. Include criteria for tests, scope control, and "
    "user-visible behavior when relevant. Do not start implementing the goal."
)


def message_content_to_text(content: object) -> str:
    """Convert chat-model message content to plain text.

    Returns:
        Plain-text representation of message content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


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
    return message_content_to_text(response.content)
