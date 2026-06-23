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

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent,
)

if TYPE_CHECKING:
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
# Scoring
# ---------------------------------------------------------------------------

# Labels in OOLONG can contain `/` (e.g. agnews "Sci/Tech") and digits
# (rare). Capture everything up to a comma, period, newline, or end of
# string to avoid truncating multi-token labels.
_LABEL_RE = re.compile(r"Label\s*:\s*([^\n.,;]+?)\s*(?:[.,;\n]|$)", re.IGNORECASE)
_USER_RE = re.compile(r"User\s*:\s*\[?(\d+)\]?", re.IGNORECASE)
_NUMBER_RE = re.compile(r"Answer\s*:\s*\[?(-?\d+(?:\.\d+)?)\]?", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(
    r"Answer\s*:\s*([A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"Answer\s*:\s*[A-Za-z_]+\s+is\s+([a-z ]+?)\s+[A-Za-z_0-9-]+",
    re.IGNORECASE,
)
# DATE: match MM/DD/YYYY or YYYY-MM-DD in the model's answer
_DATE_MDY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# Gold DATE is stored as e.g. "datetime.date(2023, 2, 6)"
_GOLD_DATE_RE = re.compile(r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)")


def _extract_answer(text: str, answer_type: str) -> str | None:
    """Pull the formatted answer out of the model's free-form response."""
    upper = answer_type.upper()
    if upper.endswith("LABEL"):
        m = _LABEL_RE.search(text)
        return m.group(1).lower() if m else None
    if upper.endswith("USER"):
        m = _USER_RE.search(text)
        return m.group(1) if m else None
    if upper.endswith("NUMERIC"):
        m = _NUMBER_RE.search(text)
        return m.group(1) if m else None
    if upper.endswith("MONTH_YEAR"):
        m = _MONTH_YEAR_RE.search(text)
        return m.group(1).strip().lower() if m else None
    if upper.endswith("COMPARISON"):
        m = _COMPARISON_RE.search(text)
        return m.group(1).strip().lower() if m else None
    if upper.endswith("DATE"):
        m = _DATE_MDY_RE.search(text)
        if m:
            return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
        m = _DATE_ISO_RE.search(text)
        if m:
            return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"
        return None
    return None


def _normalize_gold(gold: str, answer_type: str) -> str:
    """Mirror the normalization applied to the extracted answer."""
    upper = answer_type.upper()
    g = gold.strip()
    if upper.endswith("DATE"):
        # Gold is stored as "datetime.date(YYYY, M, D)" or "[datetime.date(...)]"
        m = _GOLD_DATE_RE.search(g)
        if m:
            return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"
        return g.lower()
    if upper.endswith(("LABEL", "MONTH_YEAR", "COMPARISON")):
        return g.lower()
    return g


@dataclass(frozen=True)
class _OolongAnswerMatches(SuccessAssertion):
    """Fail unless the trajectory's final answer matches any gold answer."""

    gold_answers: tuple[str, ...]
    answer_type: str

    def check(self, trajectory: AgentTrajectory) -> bool:
        extracted = _extract_answer(trajectory.answer, self.answer_type)
        if extracted is None:
            return False
        normalized = extracted.strip().lower()
        return any(
            normalized == _normalize_gold(gold, self.answer_type) for gold in self.gold_answers
        )

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        extracted = _extract_answer(trajectory.answer, self.answer_type)
        return (
            f"OOLONG answer mismatch (answer_type={self.answer_type!r}). "
            f"Extracted={extracted!r}, "
            f"gold={list(self.gold_answers)!r}, "
            f"final_text={trajectory.answer[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

_PLAIN_SYSTEM_PROMPT = (
    "You are answering an aggregation question over a long document.\n"
    "The full document is in your workspace at `/context.txt`.\n"
    "\n"
    "Use the `task` tool to delegate chunk-level analysis to subagents. "
    "Tell each subagent which lines of `/context.txt` to read and what to extract. "
    "Aggregate the results and reply with ONLY the final answer in the exact format "
    "requested (e.g. `Label: Business`, `Answer: 4`, `User: 76063`)."
)

_CODE_INTERPRETER_SYSTEM_PROMPT = (
    "You are answering an aggregation question over a long document.\n"
    "The full document is in the workspace at `/context.txt`.\n"
    "\n"
    "Solve this as a workflow:\n"
    "1. Use `task` to launch subagents in parallel — each subagent reads a "
    "specific range of `/context.txt` (e.g. lines 0-100) and returns the "
    "atomic facts needed (e.g. per-article label, user ID, or date).\n"
    "2. Use `eval` to aggregate the structured results from all subagents "
    "in JavaScript (e.g. count, sort, find the majority).\n"
    "3. Reply with ONLY the final answer in the exact format the "
    "question requests (e.g. `Label: Business`, `Answer: 4`, "
    "`User: 76063`)."
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
    subagent_cfg = [
        {
            "name": "general-purpose",
            "description": (
                "Reads a specified range of /context.txt and extracts atomic "
                "facts (labels, dates, user IDs, counts). Used for chunk-level "
                "analysis delegated by the orchestrator."
            ),
            "system_prompt": (
                "You are a document analysis subagent. Read the assigned range of "
                "/context.txt using read_file, extract the requested facts, and "
                "return concise structured results. Do not delegate further."
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
    """Run a single OOLONG example through the given arm and score it."""
    agent, initial_files = build_agent(
        arm,
        root_model=root_model,
        sub_model_id=sub_model_id,
        context=example.context_window_text,
    )

    query = f"The document is at `/context.txt` in your workspace.\n\n{example.question}"

    run_agent(
        agent,
        model=root_model,
        query=query,
        initial_files=initial_files,
        scorer=TrajectoryScorer().success(
            _OolongAnswerMatches(
                gold_answers=example.gold_answers,
                answer_type=example.answer_type,
            )
        ),
        eval_metadata={
            "arm": arm,
            "oolong_id": example.id,
            "task_group": example.task_group,
            "task": example.task,
            "context_len": example.context_len,
            "answer_type": example.answer_type,
            "origin_benchmark": "oolong-synth",
            "input_subset": example.input_subset,
            "sub_model": sub_model_id,
        },
    )
