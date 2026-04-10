"""Oolong long-context aggregation eval.

Runs the Oolong-Synth benchmark (Bertsch et al., 2025) against a deepagents
agent with swarm-enabled subagents. The agent receives a large context as a
seeded file and must answer aggregation questions (counting, frequency,
temporal, user-based).

Reference: https://arxiv.org/abs/2511.02817
Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth

Usage:
    # All trec_coarse 131k tasks (default)
    uv run --group test pytest tests/evals/oolong/ -v

    # Specific tasks by pytest -k
    uv run --group test pytest tests/evals/oolong/ -k "task_42"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from deepagents.middleware.subagents import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DEFAULT_SUBAGENT_PROMPT,
)
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t

from tests.evals.oolong.data_utils import OolongTask, load_oolong_tasks
from tests.evals.oolong.scoring import Score, parse_gold, score_output
from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    arun_agent,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# ---------------------------------------------------------------------------
# Configuration — edit these to control which tasks are loaded
# ---------------------------------------------------------------------------

OOLONG_DATASET: str = "trec_coarse"
"""Source dataset to filter on (e.g. `"trec_coarse"`, `"spam"`, `"agnews"`)."""

OOLONG_CONTEXT_LEN: int = 131072
"""Context length bucket (1024, 4096, 32768, or 131072)."""

OOLONG_TASK_IDS: set[int] | None = {17000218}
"""Specific task IDs to run. Set to `None` to run all matching tasks."""

# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

SUBAGENT_MODEL: str = "claude-haiku-4-5-20251001"
"""Model for swarm subagents."""

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.eval_category("long_context"),
    pytest.mark.eval_tier("hillclimb"),
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise data analyst. You have access to a file /context.txt that \
contains labelled data.

When asked a question about this data, you MUST:
1. Read the file /context.txt to see the full dataset
2. Carefully analyse every single data point — do not skip, estimate, or approximate
3. Count and classify exactly as instructed by the question
4. Return ONLY the final answer in the exact format requested by the question — \
no explanation, no extra text

Be precise. The questions ask about aggregate statistics (counts, frequencies, \
comparisons). Every data point matters."""

# ---------------------------------------------------------------------------
# Custom success assertion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OolongCorrect(SuccessAssertion):
    """Assert that the agent's answer is correct per Oolong scoring."""

    gold_answer: str

    def check(self, trajectory: AgentTrajectory) -> bool:
        result = score_output(output=trajectory.answer, gold_answer=self.gold_answer)
        return result.correct

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        result = score_output(output=trajectory.answer, gold_answer=self.gold_answer)
        return (
            f"Oolong scoring failed: expected {result.gold!r}, "
            f"got prediction {result.pred!r} "
            f"(exact={result.exact_match}, normalized={result.normalized_match}, "
            f"contains={result.contains_match}, numeric={result.numeric_match})"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_tasks() -> list[OolongTask]:
    return load_oolong_tasks(
        dataset=OOLONG_DATASET,
        context_len=OOLONG_CONTEXT_LEN,
        task_ids=OOLONG_TASK_IDS,
    )


def _task_label(task: OolongTask) -> str:
    return f"task_{task.id}_{task.task}"


TASKS = _load_tasks()

if not TASKS:
    msg = (
        f"No Oolong tasks found for dataset={OOLONG_DATASET!r}, "
        f"context_len={OOLONG_CONTEXT_LEN}, task_ids={OOLONG_TASK_IDS}"
    )
    raise RuntimeError(msg)

# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


def _log_score(result: Score) -> None:
    """Log scoring details to LangSmith."""
    t.log_feedback(key="correct", score=1 if result.correct else 0)
    t.log_feedback(key="exact_match", score=1 if result.exact_match else 0)
    t.log_feedback(key="normalized_match", score=1 if result.normalized_match else 0)
    t.log_feedback(key="contains_match", score=1 if result.contains_match else 0)
    t.log_feedback(key="numeric_match", score=1 if result.numeric_match else 0)
    t.log_outputs({"prediction": result.pred, "gold_answer": result.gold})


@pytest.mark.langsmith
@pytest.mark.parametrize("task", TASKS, ids=[_task_label(t) for t in TASKS])
async def test_oolong(model: BaseChatModel, task: OolongTask) -> None:
    """Run a single Oolong aggregation task."""
    gold = parse_gold(task.answer)

    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        subagents=[
            {
                "name": "general-purpose",
                "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
                "system_prompt": DEFAULT_SUBAGENT_PROMPT,
                "model": SUBAGENT_MODEL,
            }
        ],
        enable_swarm=True,
        checkpointer=MemorySaver(),
    )

    trajectory = await arun_agent(
        agent,
        model=model,
        query=task.question,
        initial_files={"/context.txt": task.context_window_text},
        scorer=TrajectoryScorer().success(OolongCorrect(gold_answer=gold)),
        eval_metadata={
            "oolong_task_id": task.id,
            "oolong_dataset": task.dataset,
            "oolong_context_len": task.context_len,
            "oolong_task_type": task.task,
            "oolong_task_group": task.task_group,
            "oolong_answer_type": task.answer_type,
        },
    )

    result = score_output(output=trajectory.answer, gold_answer=gold)
    _log_score(result)
