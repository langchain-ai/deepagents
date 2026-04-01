"""MemoryAgentBench evaluation tests for deepagents.

Runs the MemoryAgentBench benchmark (ICLR 2026) using the deepagents runner.
Data is loaded from the `ai-hyz/MemoryAgentBench` HuggingFace dataset.
Each test feeds context chunks to the agent, then poses questions and evaluates
responses against ground-truth answers using normalized substring matching.

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
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t

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
from tests.evals.memory_agent_bench.eval_utils import substring_match_any
from tests.evals.utils import run_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [pytest.mark.eval_category("memory")]

logger = logging.getLogger(__name__)

_LANGSMITH_CONFIGURED = bool(os.environ.get("LANGSMITH_API_KEY"))

_langsmith_mark = pytest.mark.langsmith if _LANGSMITH_CONFIGURED else lambda f: f


def _log_feedback(*, key: str, value: object) -> None:
    """Log feedback to LangSmith when available, silently no-op otherwise."""
    with contextlib.suppress(ValueError, Exception):
        t.log_feedback(key=key, value=value)


def _log_clean_inputs(model: BaseChatModel, config: DatasetConfig) -> None:
    """Override auto-captured inputs with clean metadata.

    Must be called before any `pytest.skip()` to prevent
    `@pytest.mark.langsmith` from serializing the raw `BaseChatModel`
    fixture as the dataset example inputs.

    Args:
        model: The chat model under evaluation.
        config: Dataset configuration for the current benchmark run.
    """
    with contextlib.suppress(Exception):
        t.log_inputs(
            {
                "config": {
                    "split": config.split,
                    "source": config.source,
                    "chunk_size": config.chunk_size,
                    "max_samples": config.max_samples,
                    "max_questions": config.max_questions,
                },
                "model": str(getattr(model, "model", None) or getattr(model, "model_name", "")),
            }
        )


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

# -- File-seeded (agentic retrieval) configuration ---------------------------

FILESEEDED_SYSTEM_PROMPT = (
    "You have access to a collection of text files in /data/ containing "
    "information relevant to answering questions. Use your file tools "
    "(grep, read_file, glob, ls) to search for and retrieve relevant "
    "information before answering. Do not assume you already know the answer — "
    "always search the files first."
)

FILESEEDED_QUERY_PREFIX = (
    "Answer the following question as concisely as possible. "
    "Use your file tools to search for the answer in the files under /data/. "
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
        Raw predictions for each question.
    """
    checkpointer = MemorySaver()
    agent = create_deep_agent(model=model, checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())

    chunks = chunk_text(sample.context, chunk_size=config.chunk_size)

    questions = sample.questions
    answers = sample.answers
    qa_pair_ids = sample.qa_pair_ids
    if config.max_questions is not None:
        questions = questions[: config.max_questions]
        answers = answers[: config.max_questions]
        qa_pair_ids = qa_pair_ids[: config.max_questions]

    logger.info(
        "Sample source=%s: %d chunks, %d questions",
        sample.source,
        len(chunks),
        len(questions),
    )

    for chunk in chunks:
        run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=MEMORIZE_PREFIX + chunk,
        )

    predictions: list[_QAPrediction] = []
    for idx, (question, answer) in enumerate(zip(questions, answers, strict=True)):
        trajectory = run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=QUERY_PREFIX + question,
        )

        ground_truths = answer if isinstance(answer, list) else [answer]
        qa_pair_id = qa_pair_ids[idx] if idx < len(qa_pair_ids) else None
        predictions.append(
            _QAPrediction(
                question=question,
                prediction=trajectory.answer,
                ground_truths=ground_truths,
                qa_pair_id=qa_pair_id,
            )
        )

    return _SampleOutput(predictions=predictions)


def _run_benchmark_sample_fileseeded(
    sample: BenchmarkSample,
    config: DatasetConfig,
    model: BaseChatModel,
) -> _SampleOutput:
    """Execute a MemoryAgentBench sample with chunks pre-loaded as files.

    Instead of feeding chunks as conversation messages (long-context strategy),
    this runner writes each chunk to a file in `/data/` and relies on the
    agent's file tools (grep, read_file, glob) for retrieval — the "agentic
    retrieval" strategy from the MemoryAgentBench paper (ICLR 2026).

    Args:
        sample: The benchmark sample to evaluate.
        config: Dataset configuration controlling chunk size.
        model: The chat model to use.

    Returns:
        Raw predictions for each question.
    """
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer,
        system_prompt=FILESEEDED_SYSTEM_PROMPT,
    )
    thread_id = str(uuid.uuid4())

    chunks = chunk_text(sample.context, chunk_size=config.chunk_size)
    initial_files = {f"/data/chunk_{i:04d}.txt": chunk for i, chunk in enumerate(chunks)}

    questions = sample.questions
    answers = sample.answers
    qa_pair_ids = sample.qa_pair_ids
    if config.max_questions is not None:
        questions = questions[: config.max_questions]
        answers = answers[: config.max_questions]
        qa_pair_ids = qa_pair_ids[: config.max_questions]

    logger.info(
        "Sample source=%s (fileseeded): %d chunks as files, %d questions",
        sample.source,
        len(chunks),
        len(questions),
    )

    predictions: list[_QAPrediction] = []
    for idx, (question, answer) in enumerate(zip(questions, answers, strict=True)):
        trajectory = run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            initial_files=initial_files if idx == 0 else None,
            query=FILESEEDED_QUERY_PREFIX + question,
        )

        ground_truths = answer if isinstance(answer, list) else [answer]
        qa_pair_id = qa_pair_ids[idx] if idx < len(qa_pair_ids) else None
        predictions.append(
            _QAPrediction(
                question=question,
                prediction=trajectory.answer,
                ground_truths=ground_truths,
                qa_pair_id=qa_pair_id,
            )
        )

    return _SampleOutput(predictions=predictions)


# ---------------------------------------------------------------------------
# Scorer (pure computation, no side effects)
# ---------------------------------------------------------------------------


def _score_predictions(output: _SampleOutput) -> list[bool]:
    """Compute substring match for each prediction in a sample output.

    Args:
        output: Raw output from `_run_benchmark_sample`.

    Returns:
        List of booleans, one per question, indicating substring match.
    """
    return [substring_match_any(pred.prediction, pred.ground_truths) for pred in output.predictions]


# ---------------------------------------------------------------------------
# Feedback logging (LangSmith side effects only)
# ---------------------------------------------------------------------------


def _log_sample_feedback(results: list[bool]) -> None:
    """Log aggregate question-level metrics to LangSmith.

    Logs correctness as a float (0.0-1.0) representing the fraction of
    questions answered correctly. Fails the test via `pytest.fail` when
    correctness is zero so that LangSmith's `pass` column reflects whether
    the agent answered any questions correctly. CI is not blocked because
    `pytest_reporter.py` overrides the exit status to 0.

    Args:
        results: Per-question substring match booleans from `_score_predictions`.
    """
    passed = sum(results)
    total = len(results)
    correctness = passed / total if total > 0 else 0.0
    _log_feedback(key="questions_passed", value=passed)
    _log_feedback(key="questions_total", value=total)
    _log_feedback(key="correctness", value=correctness)
    if correctness < 1.0:
        pytest.fail(
            f"{passed}/{total} questions answered correctly ({correctness:.0%})",
            pytrace=False,
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
    _log_clean_inputs(model, config)
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
        _log_sample_feedback(results)


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
    _log_clean_inputs(model, config)
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
        _log_sample_feedback(results)


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
    _log_clean_inputs(model, config)
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
        _log_sample_feedback(results)


# ---------------------------------------------------------------------------
# CI-friendly subset — file-seeded (agentic retrieval) strategy
# ---------------------------------------------------------------------------


@_langsmith_mark
@pytest.mark.parametrize("config", CI_CONFIGS, ids=_config_id)
def test_memory_agent_bench_ci_fileseeded(model: BaseChatModel, config: DatasetConfig) -> None:
    """File-seeded variant of the CI MemoryAgentBench suite.

    Same datasets as `test_memory_agent_bench_ci` but uses the agentic
    retrieval strategy: chunks are pre-loaded as files in `/data/` and the
    agent must use file tools (grep, read_file, etc.) to find answers
    instead of relying on conversation context.
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
        output = _run_benchmark_sample_fileseeded(sample, config, model)
        results = _score_predictions(output)
        _log_sample_feedback(results)
