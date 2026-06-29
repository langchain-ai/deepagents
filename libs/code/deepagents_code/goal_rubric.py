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


def generate_goal_rubric(
    objective: str,
    *,
    model_spec: str | None,
    model_params: dict[str, Any] | None = None,
    profile_override: dict[str, Any] | None = None,
) -> str:
    """Generate acceptance criteria for a goal objective.

    Args:
        objective: Goal objective to turn into criteria.
        model_spec: Model spec used to draft criteria.
        model_params: Optional model constructor kwargs.
        profile_override: Optional profile metadata overrides.

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
            HumanMessage(content=f"Goal:\n{objective}"),
        ],
    )
    return message_content_to_text(response.content)
