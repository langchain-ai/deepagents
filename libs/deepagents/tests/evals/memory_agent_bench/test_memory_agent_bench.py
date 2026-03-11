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
# Helpers
# ---------------------------------------------------------------------------


def _run_benchmark_sample(
    sample: BenchmarkSample,
    config: DatasetConfig,
    model: BaseChatModel,
) -> list[dict[str, object]]:
    """Execute a single MemoryAgentBench sample against deepagents.

    Feeds context chunks, poses each question, and returns per-question metrics.

    Args:
        sample: The benchmark sample to evaluate.
        config: Dataset configuration controlling chunk size.
        model: The chat model to use.

    Returns:
        List of dicts, one per question, each containing the computed metrics
        and the raw prediction/answer pair.
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

    # --- Memorization phase ---
    for chunk in chunks:
        run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=MEMORIZE_PREFIX + chunk,
        )

    # --- Query phase ---
    results: list[dict[str, object]] = []
    query_trajectories: list[AgentTrajectory] = []
    for idx, (question, answer) in enumerate(zip(sample.questions, sample.answers, strict=True)):
        trajectory = run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=QUERY_PREFIX + question,
        )
        query_trajectories.append(trajectory)

        prediction = trajectory.answer
        ground_truths = answer if isinstance(answer, list) else [answer]
        metrics = calculate_metrics(prediction, ground_truths)

        qa_pair_id = sample.qa_pair_ids[idx] if idx < len(sample.qa_pair_ids) else None
        result = {
            **metrics,
            "prediction": prediction,
            "answer": ground_truths,
            "question": question,
            "qa_pair_id": qa_pair_id,
        }
        results.append(result)

        _log_feedback(key=f"q{idx}_exact_match", value=metrics["exact_match"])
        _log_feedback(key=f"q{idx}_f1", value=metrics["f1"])
        _log_feedback(key=f"q{idx}_substring_match", value=metrics["substring_exact_match"])

    # Log aggregate efficiency data from the query-phase trajectories so they
    # appear in LangSmith alongside the standard eval keys.
    total_steps = sum(len(traj.steps) for traj in query_trajectories)
    total_tool_calls = sum(len(s.action.tool_calls) for traj in query_trajectories for s in traj.steps)
    _log_feedback(key="agent_steps", value=total_steps)
    _log_feedback(key="tool_call_requests", value=total_tool_calls)

    return results


def _assert_results(
    results: list[dict[str, object]],
    *,
    metric: str = "f1",
    pass_threshold: float = 0.5,
) -> None:
    """Assert that a majority of questions were answered correctly.

    Uses token-level F1 as the gate metric and requires more than half the
    questions to score above zero. Logs graduated correctness
    (``passed / total``) to LangSmith for trend tracking.

    Args:
        results: Per-question result dicts from `_run_benchmark_sample`.
        metric: Which metric to check for the per-question pass gate.
        pass_threshold: Fraction of questions that must score > 0 to pass
            the test overall.
    """
    if not results:
        _log_feedback(key="correctness", value=0)
        pytest.fail("MemoryAgentBench eval produced zero question results.")

    scores = [float(r[metric]) for r in results]
    avg_score = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s > 0)
    ratio = passed / len(scores)

    _log_feedback(key=f"avg_{metric}", value=avg_score)
    _log_feedback(key="num_questions", value=len(results))
    _log_feedback(key="questions_passed", value=passed)
    _log_feedback(key="questions_total", value=len(scores))
    _log_feedback(key="correctness", value=ratio)

    if ratio < pass_threshold:
        pytest.fail(
            f"MemoryAgentBench eval failed: {passed}/{len(scores)} questions passed "
            f"by {metric!r} ({ratio:.0%} < {pass_threshold:.0%} threshold, "
            f"avg_{metric}={avg_score:.3f})."
        )


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
        results = _run_benchmark_sample(sample, config, model)
        _assert_results(results)


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
        results = _run_benchmark_sample(sample, config, model)
        _assert_results(results)


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
        results = _run_benchmark_sample(sample, config, model)
        _assert_results(results)
