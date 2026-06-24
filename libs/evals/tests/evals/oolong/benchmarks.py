"""OOLONG long-context aggregation benchmark — plain subagents vs. code interpreter.

Loads a subset of `oolongbench/oolong-synth` from HuggingFace and runs the same
examples through two agent configurations:

- ``plain``: a deep agent that delegates chunk-level analysis to
  general-purpose subagents via the ``task`` tool. The long context is seeded
  as `/context.txt` in the agent workspace.
- ``code_interpreter``: a deep agent with the QuickJS
  ``CodeInterpreterMiddleware``, which exposes a JS ``eval`` tool the agent uses
  for aggregation. Subagents launched via ``task`` handle file reads from the
  shared workspace.

Sustainable by design — `load_oolong_examples` pulls *any* (``dataset``,
``context_len``) bucket via the HuggingFace datasets-server ``/filter`` endpoint
(only the matching rows, cached to JSONL) instead of downloading the whole split,
so adding a new eval set is just new env vars (``OOLONG_DATASET`` /
``OOLONG_CONTEXT_LEN``), never new code. See ``README.md`` for the run matrix.

Scoring uses the **official** OOLONG scorer (see `official_scorer`), vendored
verbatim, so ``score`` matches the paper and the north-star LangSmith dataset.

LangSmith shape (mirrors the ``oolong-trec_coarse`` north-star dataset, plus the
full HF row content so each example is self-contained and inspectable in the UI):

- **inputs**: every field of the HF record — the identifiers (``dataset`` /
  ``task_id`` / ``task_type`` / ``task_group`` / ``answer_type`` /
  ``context_len`` / ``input_subset`` / ``context_window_id`` / ``num_labels``)
  *and* the content (``question`` / ``context_window_text`` /
  ``context_window_text_with_labels``). These are display-only — the agent reads
  the document from its workspace, not from the logged inputs — and identical
  across arms, so both experiments share one example;
- **reference output**: ``{"answer": "['less common than']"}`` (raw gold);
- **run outputs**: the official per-example record — ``attempted_parse`` /
  ``parse_confidence`` / ``score`` / ``gold_answer`` (+ ``final_text``);
- **feedback**: ``score`` — the only OOLONG metric (0-1, NUMERIC partial credit;
  the paper reports its mean) — plus ``agent_steps`` / ``tool_call_requests``
  (harness efficiency telemetry, the point of plain-vs-RLM; not OOLONG metrics).

The arm is read from ``OOLONG_ARM`` (not a pytest parameter) so both arms
produce the same example identity and line up for side-by-side comparison.
"""

from __future__ import annotations

import ast
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
from langsmith import testing as t

from tests.evals.oolong.official_scorer import synth_process_response
from tests.evals.utils import run_agent

if TYPE_CHECKING:
    from deepagents.middleware.subagents import SubAgent
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

#: HuggingFace dataset id for OOLONG-synth.
_HF_DATASET = "oolongbench/oolong-synth"

#: Default dataset (subset) to filter on. The RLM paper and the north-star
#: LangSmith dataset both use ``trec_coarse`` — 6-way question classification
#: aggregation, present in the ``validation`` split with 50 examples per
#: context-length bucket.
_DEFAULT_DATASET = "trec_coarse"

#: HF dataset split per dataset (subset). ``trec_coarse`` lives in
#: ``validation``; ``agnews`` / ``spam`` / others live in ``test``.
_SPLIT_FOR_SUBSET: dict[str, str] = {
    "trec_coarse": "validation",
}

#: HuggingFace datasets-server ``/filter`` endpoint. We fetch *only* the rows
#: matching ``dataset`` + ``context_len`` (server-side filter) instead of
#: ``datasets.load_dataset``, which would download the entire split parquet
#: (~2 GB for ``validation``, ~10 GB for ``test``) just to keep 50 rows.
_FILTER_URL = "https://datasets-server.huggingface.co/filter"

#: Max rows per ``/filter`` request (the endpoint caps ``length`` at 100).
_PAGE_SIZE = 100

#: On-disk cache for fetched rows, one JSONL per (split, subset, context_len).
_CACHE_DIR = Path(__file__).parent / ".cache" / "oolong"


def _get_json_with_retry(url: str, *, attempts: int = 4) -> dict[str, Any]:
    """GET a JSON payload, retrying the datasets-server's intermittent 5xx."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(1.5 * (i + 1))
    msg = f"datasets-server request failed after {attempts} attempts: {url}\n{last_err}"
    raise RuntimeError(msg)


def _fetch_oolong_rows(subset: str, context_len: int | None, split: str) -> list[dict[str, Any]]:
    """Fetch all rows for one (subset, context_len) bucket via the filter API.

    Paginates server-side until the full matching set is retrieved. A bucket is
    typically 50 rows, so this transfers a few hundred KB (or ~35 MB at the
    131K-token bucket) rather than the multi-GB whole-split download.
    """
    where = f"\"dataset\"='{subset}'"
    if context_len is not None:
        where += f' AND "context_len"={context_len}'

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "dataset": _HF_DATASET,
                "config": "default",
                "split": split,
                "where": where,
                "offset": offset,
                "length": _PAGE_SIZE,
            }
        )
        data = _get_json_with_retry(f"{_FILTER_URL}?{params}")
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= (data.get("num_rows_total") or 0):
            break
    return rows


def _load_rows_cached(subset: str, context_len: int | None, split: str) -> list[dict[str, Any]]:
    """Return the bucket's rows, fetching + caching to JSONL on first access."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ctx_key = context_len if context_len is not None else "all"
    cache_path = _CACHE_DIR / f"{split}__{subset}__ctx{ctx_key}.jsonl"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rows = _fetch_oolong_rows(subset, context_len, split)
    tmp_path = cache_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(cache_path)  # atomic publish so partial fetches never cache
    return rows


#: Task groups across all supported subsets.
TASK_GROUPS = ("counting", "user", "timeline")

#: Arm names — keep in sync with `build_agent` and pytest marks.
Arm = Literal["plain", "code_interpreter"]


@dataclass(frozen=True)
class OolongExample:
    """One OOLONG-synth example, materialized into the harness shape.

    Field names mirror the north-star LangSmith dataset inputs so the logged
    example lines up 1:1 with it (``task_id`` ← HF ``id``, ``dataset`` ← HF
    ``dataset``, ``task_type`` ← HF ``task``).
    """

    task_id: int
    dataset: str
    task_group: str
    task_type: str
    context_len: int
    context_window_id: int
    context_window_text: str
    context_window_text_with_labels: str
    num_labels: int
    question: str
    gold_answers: tuple[str, ...]
    gold_answer_raw: str
    answer_type: str
    input_subset: bool


@lru_cache(maxsize=8)
def load_oolong_examples(
    *,
    dataset: str = _DEFAULT_DATASET,
    context_len: int = 65536,
    n_examples: int | None = 5,
    split: str | None = None,
) -> tuple[OolongExample, ...]:
    """Load a deterministic OOLONG-synth subset at one context length.

    Defaults match the north-star LangSmith dataset: ``trec_coarse``
    (validation split) at the 65536-token bucket, 50 examples total.

    Rows are fetched targeted via the HuggingFace datasets-server ``/filter``
    endpoint (only the matching ``(dataset, context_len)`` bucket, cached to
    JSONL) — *not* the whole split — so changing ``context_len`` to any bucket
    stays cheap.

    Args:
        dataset: HF subset to filter on (the ``dataset`` column). Defaults to
            ``"trec_coarse"`` — the paper's exact dataset.
        context_len: Token-length bucket. OOLONG-synth has buckets at
            1024-4194304; 65536 and 131072 are the most useful.
        n_examples: How many examples to return. ``None`` means all 50.
            Otherwise distributed evenly across task groups by ``id``.
        split: HuggingFace split. Defaults to ``"validation"`` for
            ``trec_coarse``, ``"test"`` for everything else.

    Returns:
        A tuple of `OolongExample` instances, deterministic across calls.
    """
    resolved_split = split or _SPLIT_FOR_SUBSET.get(dataset, "test")
    rows = _load_rows_cached(dataset, context_len, resolved_split)
    rows = sorted(rows, key=lambda r: int(r["id"]))

    def _row_to_example(row: dict[str, Any]) -> OolongExample:
        return OolongExample(
            task_id=int(row["id"]),
            dataset=str(row["dataset"]),
            task_group=str(row["task_group"]),
            task_type=str(row["task"]),
            context_len=int(row["context_len"]),
            context_window_id=int(row["context_window_id"]),
            context_window_text=str(row["context_window_text"]),
            context_window_text_with_labels=str(row["context_window_text_with_labels"]),
            num_labels=int(row["num_labels"]),
            question=str(row["question"]),
            gold_answers=_coerce_gold(row["answer"]),
            gold_answer_raw=str(row["answer"]),
            answer_type=str(row["answer_type"]),
            input_subset=_coerce_bool(row["input_subset"]),
        )

    if n_examples is None:
        return tuple(_row_to_example(row) for row in rows)

    # Determine which task groups actually exist in this subset/bucket.
    actual_groups = sorted({str(row["task_group"]) for row in rows})
    per_group = -(-n_examples // len(actual_groups))  # ceil div
    by_group: dict[str, list[OolongExample]] = {tg: [] for tg in actual_groups}
    for row in rows:
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


def _coerce_bool(raw: Any) -> bool:  # noqa: ANN401
    """Coerce the dataset's `input_subset` field into a bool.

    The HF release stores it as a string (``"True"``/``"False"``); the
    north-star dataset logs it as a real boolean.
    """
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return bool(raw)


def _coerce_gold(raw: Any) -> tuple[str, ...]:  # noqa: ANN401
    """Coerce the dataset's `answer` field into a tuple of strings."""
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return tuple(str(x) for x in parsed)
            except (ValueError, SyntaxError):
                pass
        return (stripped,)
    return (str(raw),)


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

_PLAIN_SYSTEM_PROMPT = (
    "You are a precise data analyst. The document to analyze is at `/context.txt`.\n"
    "\n"
    "Work in two phases:\n"
    "1. UNDERSTAND: read enough of `/context.txt` (the header plus a few lines) to "
    "learn its exact format and how many data lines it has. Do not try to read or "
    "classify the whole document yourself — it is too large to do accurately in one pass.\n"
    "2. ORCHESTRATE: split the data lines into contiguous chunks (~40-60 lines each) and "
    "dispatch them to `general-purpose` subagents *in parallel* — emit multiple `task` calls "
    "in a single turn, one per chunk. Each `task` tells its subagent the exact line range to "
    "read from `/context.txt` and to return structured per-line results (the label for each "
    "line in its range). Subagents read their own range with `read_file`.\n"
    "\n"
    "Then AGGREGATE the per-line results from all chunks yourself: concatenate them and "
    "compute the answer (count labels, take the min/max, etc.). Never ask one subagent to "
    "process the whole document, and never let a subagent return the final aggregate — do the "
    "final computation yourself. Finish by stating the final answer in the exact format the "
    "question requests.\n"
    "\n"
    "Keep the orchestration to a SINGLE fan-out round: one batch of parallel `task` calls over "
    "fixed contiguous chunks, then aggregate. Do NOT run validation-and-repair passes or "
    "re-dispatch subagents for missing/duplicate lines — if a few lines are missing, just "
    "aggregate what you have. One round keeps latency bounded."
)

_CODE_INTERPRETER_SYSTEM_PROMPT = (
    "You are a precise data analyst. The document to analyze is at `/context.txt`.\n"
    "\n"
    "Work in two phases:\n"
    "1. UNDERSTAND: read enough of `/context.txt` (the header plus a few lines) to "
    "learn its exact format and how many data lines it has. Do not try to read or "
    "classify the whole document yourself — it is too large to do accurately in one pass.\n"
    "2. ORCHESTRATE IN THE REPL: do ALL of the heavy work inside a single `eval` program. "
    "Split the data lines into contiguous chunks (~40-60 lines each) and fan the chunks out "
    "to `general-purpose` subagents *in parallel* with `Promise.all([...])` of `task(...)` "
    "calls. Each `task` tells its subagent the exact line range to read from `/context.txt` "
    "and uses a `responseSchema` so it returns structured per-line results (e.g. the label "
    "for each line in its range). Subagents read their own range with `read_file`.\n"
    "\n"
    "Then AGGREGATE deterministically in JavaScript: concatenate the per-line results from "
    "all chunks and compute the answer with code (count labels, take the min/max, etc.). "
    "Never ask one subagent to process the whole document, and never let a subagent return "
    "the final aggregate — the counting and the final computation must happen in your "
    "JavaScript so they are exact. Finish by stating the final answer in the exact format "
    "the question requests.\n"
    "\n"
    "Keep the orchestration to a SINGLE fan-out round: one `Promise.all` of `task` calls over "
    "fixed contiguous chunks, then aggregate. Do NOT run validation-and-repair passes or "
    "re-dispatch subagents for missing/duplicate lines — if a few lines are missing, just "
    "aggregate what you have. One round keeps latency bounded."
)

#: Shared subagent prompt (both arms use the same general-purpose subagent, so
#: only the orchestrator's aggregation substrate differs).
_SUBAGENT_SYSTEM_PROMPT = (
    "You analyze one assigned range of /context.txt. "
    "Read the ENTIRE assigned line range with read_file (paginate with "
    "offset/limit if needed), then process every line in the range — do not "
    "skip, sample, or approximate. Return exactly the structured per-line "
    "result the task asks for (one entry per data line in your range). "
    "Do not compute aggregates or totals; return the raw per-line data so the "
    "caller can aggregate. Do not delegate further."
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
            "description": "Reads a range of /context.txt and extracts per-line facts as directed.",
            "system_prompt": _SUBAGENT_SYSTEM_PROMPT,
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
            # Long timeout: a single `eval` awaits a Promise.all of `task()`
            # subagent dispatches, which take much longer than the 5s default.
            middleware=[CodeInterpreterMiddleware(timeout=600.0)],
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

    Scores *softly*: the answer is graded into feedback rather than
    hard-failing. A ``pytest.fail`` would leave the LangSmith root run
    ``pending`` and never sync the dataset example, dropping the wrong-answer
    arm out of the side-by-side comparison entirely.

    The ``score`` feedback — the only OOLONG metric — comes from the
    **official** OOLONG scorer (0-1 with NUMERIC partial credit). The arm, root
    model and sub-model travel as run metadata, never as inputs (which would
    split the example by arm and break the comparison).
    """
    # Reference output: raw stringified gold list under key ``answer`` — the
    # exact north-star shape (e.g. ``"['less common than']"``).
    t.log_reference_outputs({"answer": example.gold_answer_raw})

    agent, initial_files = build_agent(
        arm,
        root_model=root_model,
        sub_model_id=sub_model_id,
        context=example.context_window_text,
    )

    # `log_result_as_output=False` so this function is the sole outputs logger —
    # `RunTree.add_outputs` merges, so logging the raw agent state too would
    # pollute the clean north-star output dict below.
    #
    # `eval_inputs` is display-only (logged as the example inputs, visible in the
    # LangSmith UI) — it does NOT feed the agent, which gets the document via
    # `initial_files`. So we log the *full* OOLONG row here (document text,
    # labeled oracle, counts, ids), making each example a faithful, self-contained
    # copy of the HF record rather than just the 8 scalar identifiers. Both arms
    # log identical inputs, so they still hash to one shared example.
    trajectory = run_agent(
        agent,
        model=root_model,
        query=example.question,
        initial_files=initial_files,
        log_result_as_output=False,
        eval_inputs={
            "dataset": example.dataset,
            "task_id": example.task_id,
            "question": example.question,
            "task_type": example.task_type,
            "task_group": example.task_group,
            "answer_type": example.answer_type,
            "context_len": example.context_len,
            "input_subset": example.input_subset,
            "context_window_id": example.context_window_id,
            "num_labels": example.num_labels,
            "context_window_text": example.context_window_text,
            "context_window_text_with_labels": example.context_window_text_with_labels,
        },
        eval_metadata={"arm": arm, "sub_model": sub_model_id, "origin_benchmark": "oolong-synth"},
    )

    # Official scorer — `score` is the *only* OOLONG metric (0-1, with numeric
    # partial credit folded in; the paper reports its mean). We deliberately do
    # NOT log a separate binary "correct" — not in the OOLONG spec, and it would
    # discard partial credit on NUMERIC tasks.
    model_id = str(getattr(root_model, "model", None) or getattr(root_model, "model_name", ""))
    datapoint = {
        "id": example.task_id,
        "context_window_id": example.context_window_id,
        "dataset": example.dataset,
        "answer": example.gold_answer_raw,
        "answer_type": example.answer_type,
    }
    graded = synth_process_response(datapoint, trajectory.answer, model_id)
    score = float(graded["score"])

    # Feedback — `score=` (not `value=`) so metrics surface as compare-view columns.
    # `agent_steps` / `tool_call_requests` are harness efficiency telemetry
    # (the point of plain-vs-RLM), orthogonal to the OOLONG score.
    t.log_feedback(key="score", score=score)
    t.log_feedback(key="agent_steps", score=len(trajectory.steps))
    t.log_feedback(
        key="tool_call_requests",
        score=sum(len(step.action.tool_calls) for step in trajectory.steps),
    )

    # Run outputs — mirror the official harness's per-example record.
    t.log_outputs(
        {
            "attempted_parse": graded["attempted_parse"],
            "parse_confidence": graded["parse_confidence"],
            "score": score,
            "gold_answer": list(example.gold_answers),
            "final_text": trajectory.answer,
        }
    )
