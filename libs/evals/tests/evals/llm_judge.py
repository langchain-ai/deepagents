"""LLM-as-judge assertion for agent trajectory evaluation.

Provides a `SuccessAssertion` subclass that uses a second LLM to grade the
agent's responses against a list of human-readable criteria. Each criterion
is scored pass/fail independently; the overall assertion fails when any single
criterion fails.

Ported from agent-builder-graphs eval suite and adapted for the deepagents
`TrajectoryScorer` framework.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as langchain_tool
from langsmith import testing as t
from pydantic import BaseModel, Field

from tests.evals.utils import AgentTrajectory, SuccessAssertion

_DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

_JUDGE_SYSTEM_PROMPT = """\
You are a strict grading assistant. You will receive a conversation between a \
user and an AI agent, and a numbered list of criteria. For each criterion, \
decide whether the conversation satisfies it (grade 1) or not (grade 0). \
Return your answer by calling the `grade_conversation` tool exactly once."""


class _CriteriaGrading(BaseModel):
    criteria_index: int = Field(ge=1)
    feedback: str
    grade: Literal[0, 1]


class _GradeConversation(BaseModel):
    grades: list[_CriteriaGrading] = Field(min_length=1)


@langchain_tool(
    "grade_conversation",
    args_schema=_GradeConversation,
    description="Grade the conversation against the criteria.",
)
def _grade_conversation_tool(**_kwargs: object) -> str:
    return "no-op"


@dataclass
class LLMJudge(SuccessAssertion):
    """Grade the agent's responses against criteria using an LLM judge."""

    criteria: tuple[str, ...]
    """Human-readable criteria the agent's responses must satisfy."""

    judge_model: str = _DEFAULT_JUDGE_MODEL
    """Model identifier for the judge LLM."""

    # Single-slot cache so check() and describe_failure() share one judge call.
    _last_grades: list[dict[str, object]] | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if not self.criteria:
            msg = "At least one criterion is required for LLM judge grading"
            raise ValueError(msg)

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Invoke the LLM judge and return True if all criteria pass.

        Args:
            trajectory: The agent trajectory to grade.

        Returns:
            Whether every criterion passed.
        """
        grades = self._grade(trajectory)
        self._last_grades = grades
        return all(g["grade"] == 1 for g in grades)

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Return a human-readable explanation of which criteria failed.

        Args:
            trajectory: The agent trajectory that failed.

        Returns:
            A failure description including per-criterion feedback.
        """
        grades = self._last_grades if self._last_grades is not None else self._grade(trajectory)
        failed = [g for g in grades if g["grade"] != 1]
        parts = [f"Criteria {g['criteria_index']}: {g['feedback']}" for g in failed]
        return f"{len(failed)}/{len(grades)} criteria failed — " + "; ".join(parts)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _grade(self, trajectory: AgentTrajectory) -> list[dict[str, object]]:
        """Call the judge model and return per-criterion grades.

        Args:
            trajectory: The agent trajectory to grade.

        Returns:
            A list of dicts with keys `criteria_index`, `feedback`, `grade`.
        """
        criteria_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(self.criteria))

        user_prompt = f"Criteria:\n{criteria_text}\n\nGrade the following conversation using the grade_conversation tool."

        conversation = [
            AIMessage(content=step.action.text) for step in trajectory.steps if step.action.text
        ]
        if not conversation:
            msg = "Cannot grade trajectory: no steps contain text content. The LLM judge requires at least one text response to evaluate."
            raise ValueError(msg)

        try:
            model = init_chat_model(self.judge_model, temperature=0)
            model_with_tools = model.bind_tools(
                [_grade_conversation_tool], tool_choice="grade_conversation"
            )
            result = model_with_tools.invoke(
                [
                    SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                    *conversation,
                ]
            )
        except Exception as exc:
            msg = f"LLM judge call failed (model={self.judge_model!r}, criteria_count={len(self.criteria)}): {exc}"
            raise RuntimeError(msg) from exc

        tool_calls = result.tool_calls
        if not tool_calls or len(tool_calls) != 1:
            msg = f"Judge model returned {len(tool_calls) if tool_calls else 0} tool calls, expected exactly 1"
            raise ValueError(msg)

        args = tool_calls[0].get("args")
        if not isinstance(args, dict) or "grades" not in args:
            msg = f"Judge tool call missing 'args' or 'grades' key: {repr(tool_calls[0])[:300]}"
            raise ValueError(msg)

        grades: list[dict[str, object]] = args["grades"]
        if len(grades) != len(self.criteria):
            msg = f"Judge returned {len(grades)} grades for {len(self.criteria)} criteria"
            raise ValueError(msg)

        for g in grades:
            for key in ("criteria_index", "feedback", "grade"):
                if key not in g:
                    msg = f"Judge grade dict missing required key '{key}': {g!r}"
                    raise ValueError(msg)

        # Log aggregate judge result to LangSmith.
        passed = sum(1 for g in grades if g.get("grade") == 1)
        failed = len(grades) - passed
        try:
            t.log_feedback(
                key="llm_judge_all_passed",
                score=1.0 if failed == 0 else 0.0,
                comment=f"{passed}/{len(grades)} criteria passed",
            )
        except Exception:  # noqa: BLE001
            warnings.warn(
                "Failed to log LLM judge feedback to LangSmith",
                stacklevel=2,
            )

        return grades


# ---------------------------------------------------------------------------
# Factory function (public API)
# ---------------------------------------------------------------------------


def llm_judge(
    *criteria: str,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
) -> LLMJudge:
    """Create an `LLMJudge` success assertion.

    Args:
        *criteria: One or more human-readable criteria strings.
        judge_model: Model identifier for the judge LLM.

    Returns:
        An `LLMJudge` assertion instance.

    Raises:
        ValueError: If no criteria are provided.
    """
    return LLMJudge(criteria=tuple(criteria), judge_model=judge_model)
