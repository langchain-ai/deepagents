"""MemoryAgentBench evaluation tests for deepagents.

Runs the MemoryAgentBench benchmark (ICLR 2026) using the deepagents runner.
Data is loaded from the `ai-hyz/MemoryAgentBench` HuggingFace dataset.
Each test feeds context chunks to the agent, then poses questions and evaluates
responses against ground-truth answers using normalized exact-match, F1, and
substring-match metrics.

Reference: https://github.com/HUST-AI-HYZ/MemoryAgentBench

Usage:
    uv run --group test pytest tests/evals/memory_agent_bench/ -v
    uv run --group test pytest tests/evals/memory_agent_bench/ -k "cr_sh_6k"
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t

from deepagents import create_deep_agent
from tests.evals.memory_agent_bench.configs import (
    CI_CONFIGS,
    CONFLICT_RESOLUTION_CONFIGS,
    TEST_TIME_LEARNING_CONFIGS,
    DatasetConfig,
)
from tests.evals.memory_agent_bench.data_utils import (
    BenchmarkSample,
    chunk_text,
    load_benchmark_data,
)
from tests.evals.memory_agent_bench.eval_utils import calculate_metrics
from tests.evals.utils import AgentTrajectory, run_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_LANGSMITH_CONFIGURED = bool(os.environ.get("LANGSMITH_API_KEY"))

_langsmith_mark = pytest.mark.langsmith if _LANGSMITH_CONFIGURED else lambda f: f


def _log_feedback(*, key: str, value: object) -> None:
    """Log feedback to LangSmith when available, silently no-op otherwise."""
    with contextlib.suppress(ValueError, Exception):
        t.log_feedback(key=key, value=value)


def _require_memory_agent_bench_dependencies() -> None:
    """Skip the test when optional benchmark dependencies are unavailable."""
    pytest.importorskip("datasets", reason="MemoryAgentBench evals require the `datasets` package.")
    pytest.importorskip("nltk", reason="MemoryAgentBench evals require the `nltk` package.")
    pytest.importorskip("tiktoken", reason="MemoryAgentBench evals require the `tiktoken` package.")


MEMORIZE_PREFIX = "Please carefully memorize the following information. You will be asked questions about it later.\n\n"

QUERY_PREFIX = (
    "Based only on the information you have been given in this conversation, "
    "answer the following question as concisely as possible. "
    "Give a very short answer without extra explanation.\n\n"
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QAPrediction:
    """Raw prediction for a single QA pair, before any metric computation."""

    question: str
    prediction: str
    ground_truths: list[str]
    qa_pair_id: str | None = None


@dataclass(frozen=True)
class _SampleOutput:
    """Raw output from running a benchmark sample through the agent."""

    predictions: list[_QAPrediction] = field(default_factory=list)
    trajectories: list[AgentTrajectory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_benchmark_sample(
    sample: BenchmarkSample,
    config: DatasetConfig,
    model: BaseChatModel,
) -> _SampleOutput:
    """Execute a single MemoryAgentBench sample against deepagents.

    Feeds context chunks then poses each question, returning raw predictions
    without computing metrics or logging feedback.

    Args:
        sample: The benchmark sample to evaluate.
        config: Dataset configuration controlling chunk size.
        model: The chat model to use.

    Returns:
        Raw predictions and agent trajectories.
    """
    checkpointer = MemorySaver()
    agent = create_deep_agent(model=model, checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())

    chunks = chunk_text(sample.context, chunk_size=config.chunk_size)
    logger.info(
        "Sample source=%s: %d chunks, %d questions",
        sample.source,
        len(chunks),
        len(sample.questions),
    )

    for chunk in chunks:
        run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=MEMORIZE_PREFIX + chunk,
        )

    predictions: list[_QAPrediction] = []
    trajectories: list[AgentTrajectory] = []
    for idx, (question, answer) in enumerate(zip(sample.questions, sample.answers, strict=True)):
        trajectory = run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=QUERY_PREFIX + question,
        )
        trajectories.append(trajectory)

        ground_truths = answer if isinstance(answer, list) else [answer]
        qa_pair_id = sample.qa_pair_ids[idx] if idx < len(sample.qa_pair_ids) else None
        predictions.append(
            _QAPrediction(
                question=question,
                prediction=trajectory.answer,
                ground_truths=ground_truths,
                qa_pair_id=qa_pair_id,
            )
        )

    return _SampleOutput(predictions=predictions, trajectories=trajectories)


# ---------------------------------------------------------------------------
# Scorer (pure computation, no side effects)
# ---------------------------------------------------------------------------


def _score_predictions(output: _SampleOutput) -> list[dict[str, object]]:
    """Compute QA metrics for each prediction in a sample output.

    Args:
        output: Raw output from `_run_benchmark_sample`.

    Returns:
        List of dicts, one per question, each containing computed metrics
        and the raw prediction/answer pair.
    """
    results: list[dict[str, object]] = []
    for pred in output.predictions:
        metrics = calculate_metrics(pred.prediction, pred.ground_truths)
        results.append(
            {
                **metrics,
                "prediction": pred.prediction,
                "answer": pred.ground_truths,
                "question": pred.question,
                "qa_pair_id": pred.qa_pair_id,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Feedback logging (LangSmith side effects only)
# ---------------------------------------------------------------------------


def _log_sample_feedback(
    results: list[dict[str, object]],
    trajectories: list[AgentTrajectory],
    *,
    metric: str = "f1",
) -> None:
    """Log per-question and aggregate metrics to LangSmith.

    Does **not** fail the test — evals are tracking-only so regressions
    surface in dashboards rather than blocking CI.

    Args:
        results: Per-question result dicts from `_score_predictions`.
        trajectories: Agent trajectories from the query phase.
        metric: Which metric to aggregate.
    """
    for idx, result in enumerate(results):
        _log_feedback(key=f"q{idx}_exact_match", value=result["exact_match"])
        _log_feedback(key=f"q{idx}_f1", value=result["f1"])
        _log_feedback(key=f"q{idx}_substring_match", value=result["substring_exact_match"])

    total_steps = sum(len(traj.steps) for traj in trajectories)
    total_tool_calls = sum(len(s.action.tool_calls) for traj in trajectories for s in traj.steps)
    _log_feedback(key="agent_steps", value=total_steps)
    _log_feedback(key="tool_call_requests", value=total_tool_calls)

    if not results:
        _log_feedback(key="correctness", value=0)
        return

    scores = [float(r[metric]) for r in results]
    avg_score = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s > 0)
    ratio = passed / len(scores)

    _log_feedback(key=f"avg_{metric}", value=avg_score)
    _log_feedback(key="num_questions", value=len(results))
    _log_feedback(key="questions_passed", value=passed)
    _log_feedback(key="questions_total", value=len(scores))
    _log_feedback(key="correctness", value=ratio)


# ---------------------------------------------------------------------------
# Test parametrization helpers
# ---------------------------------------------------------------------------


def _config_id(cfg: DatasetConfig) -> str:
    """Generate a readable pytest ID from a DatasetConfig."""
    return cfg.source


# ---------------------------------------------------------------------------
# Conflict Resolution tests
# ---------------------------------------------------------------------------


@_langsmith_mark
@pytest.mark.parametrize("config", CONFLICT_RESOLUTION_CONFIGS, ids=_config_id)
def test_conflict_resolution(model: BaseChatModel, config: DatasetConfig) -> None:
    """Evaluate deepagents on MemoryAgentBench Conflict Resolution tasks.

    Tests the agent's ability to track and use the most recent information
    when facts change or contradict previous statements. Includes both
    single-hop (direct update) and multi-hop (derived update) scenarios.
    """
    _require_memory_agent_bench_dependencies()
    samples = load_benchmark_data(
        config.split,
        source_filter=config.source,
        max_samples=config.max_samples,
    )
    if not samples:
        pytest.skip(f"No samples found for source={config.source!r}")

    for sample in samples:
        output = _run_benchmark_sample(sample, config, model)
        results = _score_predictions(output)
        _log_sample_feedback(results, output.trajectories)


# ---------------------------------------------------------------------------
# Test-Time Learning tests
# ---------------------------------------------------------------------------


@_langsmith_mark
@pytest.mark.parametrize("config", TEST_TIME_LEARNING_CONFIGS, ids=_config_id)
def test_time_learning(model: BaseChatModel, config: DatasetConfig) -> None:
    """Evaluate deepagents on MemoryAgentBench Test-Time Learning tasks.

    Tests the agent's ability to learn new rules, patterns, or classification
    schemes from the context chunks and apply them to unseen examples.
    """
    _require_memory_agent_bench_dependencies()
    samples = load_benchmark_data(
        config.split,
        source_filter=config.source,
        max_samples=config.max_samples,
    )
    if not samples:
        pytest.skip(f"No samples found for source={config.source!r}")

    for sample in samples:
        output = _run_benchmark_sample(sample, config, model)
        results = _score_predictions(output)
        _log_sample_feedback(results, output.trajectories)


# ---------------------------------------------------------------------------
# CI-friendly subset
# ---------------------------------------------------------------------------


@_langsmith_mark
@pytest.mark.parametrize("config", CI_CONFIGS, ids=_config_id)
def test_memory_agent_bench_ci(model: BaseChatModel, config: DatasetConfig) -> None:
    """Small subset of MemoryAgentBench for regular CI runs.

    Includes the smallest Conflict Resolution configs (single-hop and
    multi-hop at 6k) and one Test-Time Learning config to keep cost low.
    """
    _require_memory_agent_bench_dependencies()
    samples = load_benchmark_data(
        config.split,
        source_filter=config.source,
        max_samples=config.max_samples,
    )
    if not samples:
        pytest.skip(f"No samples found for source={config.source!r}")

    for sample in samples:
        output = _run_benchmark_sample(sample, config, model)
        results = _score_predictions(output)
        _log_sample_feedback(results, output.trajectories)
