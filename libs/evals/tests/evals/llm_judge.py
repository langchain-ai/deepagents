"""LLM-as-judge assertion for agent trajectory evaluation.

Thin adapter around `openevals.llm.create_llm_as_judge` that exposes a
`SuccessAssertion` for the deepagents `TrajectoryScorer` framework. Each
criterion is evaluated independently; the overall assertion fails when any
single criterion fails.

Source: adapted from the agent-builder-graphs eval suite. Grading logic
delegated to openevals (https://github.com/langchain-ai/openevals).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from langsmith import testing as t
from openevals.llm import create_llm_as_judge

from tests.evals.utils import AgentTrajectory, SuccessAssertion

_DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

_RESPONSES_PROMPT = """You are a strict grading assistant. You will receive a
series of agent responses and a single criterion. Decide whether the agent's
responses satisfy the criterion.

<criterion>
{criterion}
</criterion>

<agent_responses>
{outputs}
</agent_responses>"""

_TRAJECTORY_PROMPT = """You are a strict grading assistant. You will receive an
agent trajectory (a sequence of steps) and a single criterion. Decide whether
the agent's trajectory satisfies the criterion.

Each step may contain:
- Text responses from the agent (shown as "text: ...")
- Tool calls the agent made (shown as "- tool_name {{args}}")

Tool calls are real actions the agent executed. Treat them as evidence
that the action was performed (e.g. an edit_file call means the agent
wrote to the file).

<criterion>
{criterion}
</criterion>

<agent_trajectory>
{outputs}
</agent_trajectory>"""


@dataclass
class LLMJudge(SuccessAssertion):
    """Grade agent output against criteria using openevals LLM judge.

    Each criterion is evaluated independently via `create_llm_as_judge`. All
    must pass for the assertion to succeed.

    When `include_tool_calls` is False (default), the judge sees only the
    agent's text responses — appropriate for criteria about what the agent
    *said*. When True, the full trajectory (tool calls + text) is shown —
    needed when criteria reference actions the agent *took* (e.g. writing
    to a file).
    """

    criteria: tuple[str, ...]
    """Human-readable criteria the agent's output must satisfy."""

    judge_model: str = _DEFAULT_JUDGE_MODEL
    """Model identifier for the judge LLM."""

    include_tool_calls: bool = False
    """Include tool calls in the context sent to the judge."""

    # Single-slot cache so check() and describe_failure() share one judge call.
    _last_results: list[dict[str, Any]] | None = field(
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
        results = self._grade(trajectory)
        self._last_results = results
        return all(r["score"] for r in results)

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Return a human-readable explanation of which criteria failed.

        Args:
            trajectory: The agent trajectory that failed.

        Returns:
            A failure description including per-criterion feedback.
        """
        results = self._last_results if self._last_results is not None else self._grade(trajectory)
        failed = [(i, r) for i, r in enumerate(results, 1) if not r["score"]]
        parts = [f"Criteria {i}: {r.get('comment') or 'no reason'}" for i, r in failed]
        return f"{len(failed)}/{len(results)} criteria failed — " + "; ".join(parts)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _serialize(self, trajectory: AgentTrajectory) -> str:
        """Serialize the trajectory for the judge prompt.

        Args:
            trajectory: The agent trajectory to serialize.

        Returns:
            A string representation of the trajectory.
        """
        if self.include_tool_calls:
            return trajectory.pretty()

        return "\n\n".join(
            f"[Agent]: {step.action.text}" for step in trajectory.steps if step.action.text
        )

    def _grade(self, trajectory: AgentTrajectory) -> list[dict[str, Any]]:
        """Call openevals judge per criterion and return results.

        Args:
            trajectory: The agent trajectory to grade.

        Returns:
            A list of `EvaluatorResult` dicts, one per criterion.
        """
        conversation = self._serialize(trajectory)
        if not conversation.strip():
            msg = (
                "Cannot grade trajectory: no steps contain content. "
                "The LLM judge requires at least one step to evaluate."
            )
            raise ValueError(msg)

        prompt = _TRAJECTORY_PROMPT if self.include_tool_calls else _RESPONSES_PROMPT
        evaluator = create_llm_as_judge(
            prompt=prompt,
            feedback_key="llm_judge_criterion",
            model=self.judge_model,
        )

        results: list[dict[str, Any]] = []
        for i, criterion in enumerate(self.criteria, 1):
            try:
                result = evaluator(outputs=conversation, criterion=criterion)
            except Exception as exc:
                msg = (
                    f"LLM judge failed on criterion {i}/{len(self.criteria)} "
                    f"(model={self.judge_model!r}): {criterion!r}"
                )
                raise RuntimeError(msg) from exc
            if not isinstance(result, dict) or "score" not in result:
                msg = (
                    f"openevals returned unexpected result for criterion "
                    f"{i}/{len(self.criteria)} {criterion!r}: {result!r}"
                )
                raise ValueError(msg)
            results.append(result)

        # Log aggregate judge result to LangSmith.
        passed = sum(1 for r in results if r["score"])
        try:
            t.log_feedback(
                key="llm_judge_all_passed",
                score=1.0 if passed == len(results) else 0.0,
                comment=f"{passed}/{len(results)} criteria passed",
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to log LLM judge feedback to LangSmith: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )

        return results


# ---------------------------------------------------------------------------
# Factory function (public API)
# ---------------------------------------------------------------------------


def llm_judge(
    *criteria: str,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    include_tool_calls: bool = False,
) -> LLMJudge:
    """Create an `LLMJudge` success assertion.

    Wraps `openevals.llm.create_llm_as_judge` to evaluate each criterion
    independently against the agent's output. All criteria must pass for the
    assertion to succeed.

    Args:
        *criteria: One or more human-readable criteria strings.
        judge_model: Model identifier for the judge LLM.
        include_tool_calls: When True, the judge sees the full trajectory
            (tool calls + text).

            When False (default), only text responses.

    Returns:
        An `LLMJudge` assertion instance.

    Raises:
        ValueError: If no criteria are provided.
    """
    return LLMJudge(
        criteria=tuple(criteria),
        judge_model=judge_model,
        include_tool_calls=include_tool_calls,
    )
