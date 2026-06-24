"""OOLONG long-context aggregation benchmark — plain subagents vs. code interpreter.

Loads a small subset of `oolongbench/oolong-synth` from HuggingFace and
runs the same examples through two agent configurations:

- ``plain``: a deep agent that delegates chunk-level analysis to
  general-purpose subagents via the ``task`` tool. The long context is
  seeded as `/context.txt` in the agent workspace.
- ``code_interpreter``: a deep agent with the QuickJS
  ``CodeInterpreterMiddleware``, which exposes a JS ``eval`` tool the agent
  uses for aggregation. Subagents launched via ``task`` handle file reads
  from the shared workspace (no PTC required).

Dataset choice: the paper evaluates on the OOLONG ``trec_coarse`` split
(50 tasks at 131K-token context). The current ``oolongbench/oolong-synth``
HuggingFace release does not include ``trec_coarse``; the closest analog
is ``agnews`` — 4-way topic classification aggregation with the same
counting/user/timeline task groups. There are 50 ``agnews`` examples at
each context-length bucket (matching the paper's task count).

Scoring follows OOLONG paper conventions: extract ``Label: X``,
``Answer: N``, ``User: X``, or ``Answer: Month YYYY`` from the
trajectory's final answer and exact-match against the gold.

The module is self-contained: the pytest test module just
parameterizes over (arm, example) pairs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, get_args

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
from langsmith import testing as t

from tests.evals.utils import (
    run_agent,
)

if TYPE_CHECKING:
    from deepagents.middleware.subagents import SubAgent
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

#: HuggingFace dataset id for OOLONG-synth.
_HF_DATASET = "oolongbench/oolong-synth"

#: Input subset to filter on. The RLM paper uses ``trec_coarse``; the
#: current HF release of OOLONG-synth does not include it, so we use
#: ``agnews`` — the closest analog (4-way topic classification
#: aggregation, same task-group structure, 50 examples per
#: context-length bucket).
_INPUT_SUBSET = "agnews"

#: HF dataset split for each input subset.
#: ``trec_coarse`` is in ``validation``; ``agnews`` and others are in ``test``.
_SPLIT_FOR_SUBSET: dict[str, str] = {
    "trec_coarse": "validation",
}

#: Task groups across all supported subsets.
TASK_GROUPS = ("counting", "user", "timeline")

#: Arm names — keep in sync with `build_agent` and pytest marks.
#:
#: ``plain`` — `create_deep_agent` that delegates chunk-level analysis to
#: general-purpose subagents via the ``task`` tool; context is in the
#: workspace at `/context.txt`.
#: ``code_interpreter`` — `create_deep_agent` with the QuickJS
#: `CodeInterpreterMiddleware`; uses ``eval`` for JS aggregation and
#: ``task`` subagents for chunk classification.
Arm = Literal["plain", "code_interpreter"]


@dataclass(frozen=True)
class OolongExample:
    """One OOLONG-synth example, materialized into the shape the harness uses."""

    id: int
    input_subset: str
    task_group: str
    task: str
    context_len: int
    context_window_text: str
    question: str
    gold_answers: tuple[str, ...]
    answer_type: str


@lru_cache(maxsize=8)
def load_oolong_examples(
    *,
    input_subset: str = "agnews",
    context_len: int = 8192,
    n_examples: int | None = 5,
    split: str | None = None,
) -> tuple[OolongExample, ...]:
    """Load a deterministic OOLONG-synth subset at one context length.

    The paper uses ``trec_coarse`` (validation split) at 131K tokens, 50
    examples. ``agnews`` (test split) is the closest HF-available analog.

    Args:
        input_subset: Dataset name to filter on. Use ``"trec_coarse"`` for
            the paper's exact dataset, ``"agnews"`` for the default.
        context_len: Token-length bucket. OOLONG-synth has buckets at
            1024-4194304; 65536 and 131072 are the most useful.
        n_examples: How many examples to return. ``None`` means all 50.
            Otherwise distributed evenly across task groups by ``id``.
        split: HuggingFace split. Defaults to ``"validation"`` for
            ``trec_coarse``, ``"test"`` for everything else.

    Returns:
        A tuple of `OolongExample` instances, deterministic across calls.
    """
    from datasets import load_dataset  # noqa: PLC0415

    resolved_split = split or _SPLIT_FOR_SUBSET.get(input_subset, "test")
    dataset = load_dataset(_HF_DATASET, split=resolved_split)
    filtered = dataset.filter(
        lambda row: row["dataset"] == input_subset and row["context_len"] == context_len
    )
    filtered = filtered.sort("id")

    def _row_to_example(row: dict[str, Any]) -> OolongExample:
        return OolongExample(
            id=int(row["id"]),
            input_subset=str(row["dataset"]),
            task_group=str(row["task_group"]),
            task=str(row["task"]),
            context_len=int(row["context_len"]),
            context_window_text=str(row["context_window_text"]),
            question=str(row["question"]),
            gold_answers=_coerce_gold(row["answer"]),
            answer_type=str(row["answer_type"]),
        )

    if n_examples is None:
        return tuple(_row_to_example(row) for row in filtered)

    # Determine which task groups actually exist in this subset/bucket.
    actual_groups = sorted({str(row["task_group"]) for row in filtered})
    per_group = -(-n_examples // len(actual_groups))  # ceil div
    by_group: dict[str, list[OolongExample]] = {tg: [] for tg in actual_groups}
    for row in filtered:
        tg = row["task_group"]
        if tg in by_group and len(by_group[tg]) < per_group:
            by_group[tg].append(_row_to_example(row))
        if all(len(by_group[g]) >= per_group for g in actual_groups):
            break

    out: list[OolongExample] = []
    for tg in actual_groups:
        out.extend(by_group[tg])
    return tuple(out[:n_examples])


def resolve_arm() -> Arm:
    """Resolve the experiment arm from the ``OOLONG_ARM`` env var (default ``plain``)."""
    arm = os.environ.get("OOLONG_ARM", "plain")
    valid = get_args(Arm)
    if arm not in valid:
        msg = f"OOLONG_ARM must be one of {valid}, got {arm!r}"
        raise ValueError(msg)
    return arm  # type: ignore[return-value]


def _coerce_gold(raw: Any) -> tuple[str, ...]:  # noqa: ANN401
    """Coerce the dataset's `answer` field into a tuple of strings."""
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    if isinstance(raw, str):
        # Some HF preprocessors store python-list literals as strings.
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            import ast  # noqa: PLC0415

            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return tuple(str(x) for x in parsed)
            except (ValueError, SyntaxError):
                pass
        return (stripped,)
    return (str(raw),)


# ---------------------------------------------------------------------------
# Scoring — adapted from abertsch72/oolong src/eval/eval_helpers.py
# ---------------------------------------------------------------------------


def _parse_answer(text: str) -> str:
    """Extract the candidate answer from a model response.

    Mirrors ``synth_attempt_answer_parse`` from the official OOLONG harness:
    split on ``:``, take the last segment, strip markdown decorators.
    """
    if ":" not in text:
        return (text if len(text) < 20 else text.rsplit(maxsplit=1)[-1]).strip()
    candidate = text.rsplit(":", maxsplit=1)[-1].strip()
    candidate = candidate.replace("*", "").replace("[", "").replace("]", "")
    return candidate.strip()


def _parse_gold(gold: str, answer_type: str) -> object:
    """Parse the dataset's gold answer into a comparable Python value."""
    import ast  # noqa: PLC0415

    upper = answer_type.upper()
    if upper.endswith("DATE"):
        from datetime import datetime  # noqa: PLC0415

        try:
            return datetime.strptime(gold, "[datetime.date(%Y, %m, %d)]").replace(tzinfo=UTC)
        except ValueError:
            return gold
    try:
        parsed = ast.literal_eval(gold)
        return parsed[0] if isinstance(parsed, list) else parsed
    except (ValueError, SyntaxError):
        return gold


def _oolong_score(text: str, gold_answers: tuple[str, ...], answer_type: str) -> float:
    """Score a model response against gold, returning 0.0-1.0.

    Follows the official OOLONG scoring from ``synth_process_response``:
    - Exact string match for most types.
    - COMPARISON: substring check for "more common"/"less common"/"same frequency".
    - NUMERIC: partial credit ``0.75 ** abs(gold - pred)``.
    - DATE: flexible parse via ``dateutil`` then exact equality.
    """
    import dateutil.parser  # noqa: PLC0415

    candidate = _parse_answer(text)
    upper = answer_type.upper()

    for raw_gold in gold_answers:
        gold = _parse_gold(raw_gold, upper)

        if upper.endswith("COMPARISON"):
            for phrase in ("more common", "less common", "same frequency"):
                if phrase in candidate.lower():
                    normalized = phrase
                    break
            else:
                normalized = candidate.lower()
            if normalized in str(gold).lower() or str(gold).lower() in normalized:
                return 1.0
            continue

        if upper.endswith("DATE"):
            try:
                parsed_dt = dateutil.parser.parse(candidate)
                if parsed_dt == gold:
                    return 1.0
            except (ValueError, OverflowError):
                pass
            continue

        if upper.endswith("NUMERIC"):
            if str(candidate) == str(gold):
                return 1.0
            try:
                return 0.75 ** abs(int(str(gold)) - int(candidate))
            except (ValueError, TypeError):
                pass
            continue

        if str(candidate).lower() == str(gold).lower():
            return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

_PLAIN_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "The document you need to analyze is at `/context.txt` in your workspace. "
    "Use the task tool to delegate parts of the analysis to subagents."
)

_CODE_INTERPRETER_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "The document you need to analyze is at `/context.txt` in your workspace. "
    "Use the eval and task tools to analyze the document."
)


def build_agent(
    arm: Arm,
    *,
    root_model: BaseChatModel,
    sub_model_id: str,
    context: str,
) -> tuple[CompiledStateGraph, dict[str, str]]:
    """Build a deep agent configured for the given arm."""
    # Both arms spawn task subagents — configure them to use the sub-model
    # (GPT-5-mini in paper setup) so sub-calls are cheaper and match paper specs.
    subagent_cfg: list[SubAgent] = [
        {
            "name": "general-purpose",
            "description": "Reads a range of /context.txt and extracts facts as directed.",
            "system_prompt": (
                "You are a helpful assistant. "
                "Read the assigned range of /context.txt using read_file, "
                "extract the requested facts, and return concise results. "
                "Do not delegate further."
            ),
            "model": sub_model_id,
        }
    ]

    if arm == "plain":
        agent = create_deep_agent(
            model=root_model,
            system_prompt=_PLAIN_SYSTEM_PROMPT,
            subagents=subagent_cfg,
        )
        return agent, {"/context.txt": context}

    if arm == "code_interpreter":
        agent = create_deep_agent(
            model=root_model,
            system_prompt=_CODE_INTERPRETER_SYSTEM_PROMPT,
            middleware=[CodeInterpreterMiddleware()],
            subagents=subagent_cfg,
        )
        return agent, {"/context.txt": context}

    msg = f"unknown arm: {arm!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Public entry point used by the pytest module
# ---------------------------------------------------------------------------


def run_oolong_case(
    example: OolongExample,
    arm: Arm,
    *,
    root_model: BaseChatModel,
    sub_model_id: str,
) -> None:
    """Run a single OOLONG example through the given arm and score it.

    Uses the standard `run_agent` LangSmith wiring (so examples, latency and
    inputs are recorded the normal way), but scores *softly*: the answer is
    graded into ``correctness`` / ``partial_score`` feedback rather than
    hard-failing the test. This matters for a pairwise benchmark — a
    ``pytest.fail`` leaves the LangSmith root run ``pending`` and never syncs the
    dataset example, which would drop the wrong-answer arm out of the
    side-by-side comparison entirely. Logged shape:

    - **inputs** (`eval_inputs`): the question + task metadata, identical across
      arms so both experiments share one dataset example;
    - **reference output**: the gold answer;
    - **feedback**: ``correctness`` / ``partial_score`` / ``agent_steps`` /
      ``tool_call_requests`` (numeric scores).

    The arm, root model and sub-model travel as metadata, not inputs.
    """
    t.log_reference_outputs({"gold_answer": list(example.gold_answers)})

    agent, initial_files = build_agent(
        arm,
        root_model=root_model,
        sub_model_id=sub_model_id,
        context=example.context_window_text,
    )

    # The context is uploaded to the agent's filesystem at `/context.txt` (via
    # `initial_files`); the question alone is the human message (the system
    # prompt already points the agent at `/context.txt`).
    trajectory = run_agent(
        agent,
        model=root_model,
        query=example.question,
        initial_files=initial_files,
        eval_inputs={
            "question": example.question,
            "oolong_id": example.id,
            "task_group": example.task_group,
            "task_type": example.task,
            "answer_type": example.answer_type,
            "context_len": example.context_len,
            "input_subset": example.input_subset,
        },
    )

    # Soft scoring: log the grade as feedback; never `pytest.fail` (see above).
    # `score=` (not `value=`) so the metrics surface as columns in the compare view.
    score = _oolong_score(trajectory.answer, example.gold_answers, example.answer_type)
    t.log_feedback(key="correctness", score=1.0 if score >= 1.0 else 0.0)
    t.log_feedback(key="partial_score", score=score)
    t.log_feedback(key="agent_steps", score=len(trajectory.steps))
    t.log_feedback(
        key="tool_call_requests",
        score=sum(len(step.action.tool_calls) for step in trajectory.steps),
    )
