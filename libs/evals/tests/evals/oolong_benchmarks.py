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
(50 tasks at 131K-token context). ``trec_coarse`` *is* available in the
``oolongbench/oolong-synth`` HuggingFace release — it lives in the
``validation`` split (50 tasks per context-length bucket). Select it with
``OOLONG_DATASET=trec_coarse`` to match the paper exactly. The default is
``agnews`` (``test`` split) — a 4-way topic classification analog with the
same counting/user/timeline task groups — kept as the cheap default.

Scoring is a faithful port of the official OOLONG harness
(`abertsch72/oolong` ``src/eval/eval_helpers.py``: ``synth_attempt_answer_parse``
+ ``synth_process_response``) so results are directly comparable to the
paper: parse the model's answer (split on ``:``, strip markdown/bracket
artifacts), then exact-match — with substring matching for COMPARISON,
``0.75 ** |gold - pred|`` partial credit for NUMERIC, and flexible
``dateutil`` parsing for DATE.

The module is self-contained: the pytest test module just
parameterizes over (arm, example) pairs.
"""

from __future__ import annotations

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

#: Default input subset to filter on. The RLM paper uses ``trec_coarse``
#: (available in the ``validation`` split — set ``OOLONG_DATASET=trec_coarse``
#: to match the paper). ``agnews`` is the cheap default: a 4-way topic
#: classification analog with the same task-group structure and 50 examples
#: per context-length bucket.
_INPUT_SUBSET = "agnews"

#: HF dataset split for each input subset.
#: ``trec_coarse`` is in ``validation``; ``agnews`` and others are in ``test``.
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
    #: Raw HF ``answer`` field, verbatim (e.g. ``"['spam']"`` or
    #: ``"[datetime.date(2024, 1, 15)]"``). The official scorer parses this
    #: directly via ``ast.literal_eval`` / ``strptime``.
    answer_raw: str
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
    examples. ``agnews`` (test split) is a same-shape analog used as the
    cheap default.

    Rows are fetched targeted via the HuggingFace datasets-server ``/filter``
    endpoint (only the matching bucket, cached to JSONL) — *not* the whole
    split, which would be a 2-10 GB download for 50 rows.

    Args:
        input_subset: Dataset name to filter on. Use ``"trec_coarse"`` for
            the paper's exact dataset, ``"agnews"`` for the default.
        context_len: Token-length bucket. OOLONG-synth has buckets at
            1024-4194304; 65536 and 131072 are the most useful.
        n_examples: How many examples to return. ``None`` means the whole
            bucket (50). Otherwise distributed evenly across task groups by ``id``.
        split: HuggingFace split. Defaults to ``"validation"`` for
            ``trec_coarse``, ``"test"`` for everything else.

    Returns:
        A tuple of `OolongExample` instances, deterministic across calls.
    """
    resolved_split = split or _SPLIT_FOR_SUBSET.get(input_subset, "test")
    rows = _load_rows_cached(input_subset, context_len, resolved_split)
    filtered = sorted(rows, key=lambda row: int(row["id"]))

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
            answer_raw=str(row["answer"]),
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
# Scoring — faithful port of abertsch72/oolong src/eval/eval_helpers.py
# (``synth_attempt_answer_parse`` + ``synth_process_response``). Kept verbatim
# in logic — including the ``parse_confidence`` tiers — so the record we log
# matches the official harness and results are directly comparable to the paper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OolongScore:
    """The OOLONG per-example scoring record (mirrors ``synth_process_response``)."""

    #: 0.0-1.0; the paper's headline metric, with numeric partial credit folded in.
    score: float
    #: The parsed candidate answer (official ``attempted_parse``).
    prediction: str
    #: Parse heuristic confidence: ``low`` / ``med`` / ``high`` / ``vhigh``.
    parse_confidence: str
    #: The gold answer, stringified (official ``answer``).
    gold: str


def _synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Parse a model response into ``(candidate, parse_confidence)``.

    Mirrors the official ``synth_attempt_answer_parse``: if there is no ``:``,
    return the whole answer when short else its last word; otherwise take the
    segment after the last ``:`` and strip ``*``/``[``/``]`` formatting
    artifacts (models like to bold answers or wrap them in brackets). A long
    candidate containing a comparison phrase is normalized to that phrase. The
    confidence tier is reported but never affects the score.
    """
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        return answer.rsplit(maxsplit=1)[-1], parse_confidence

    candidate = answer.rsplit(":", maxsplit=1)[-1].strip()
    candidate = candidate.replace("*", "").replace("[", "").replace("]", "")

    parse_confidence = "med"
    if "User:" in answer or "Answer:" in answer or "Date:" in answer or "Label" in answer:
        parse_confidence = "high"
    if len(candidate) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate:
        candidate = "more common"
    elif "less common" in candidate:
        candidate = "less common"
    elif "same frequency" in candidate:
        candidate = "same frequency"
    return candidate, parse_confidence


def _synth_process_response(output: str, answer_raw: str, answer_type: str) -> OolongScore:
    """Score a model response against the raw gold answer.

    Faithful port of the official ``synth_process_response``:

    - gold is ``ast.literal_eval(answer)[0]`` unless the raw answer is a
      ``datetime.date(...)`` literal, in which case it is parsed via ``strptime``;
    - exact string match scores 1.0;
    - a parsed answer of "more common"/"less common"/"same frequency" scores 1.0
      when contained in the gold (COMPARISON wording differences);
    - ``ANSWER_TYPE.NUMERIC`` gets ``0.75 ** |gold - pred|`` partial credit;
    - ``ANSWER_TYPE.DATE`` is parsed with ``dateutil`` and compared for equality.

    Args:
        output: The model's final answer text.
        answer_raw: The dataset's raw ``answer`` field, verbatim.
        answer_type: The dataset's ``answer_type`` (e.g. ``ANSWER_TYPE.NUMERIC``).

    Returns:
        An `OolongScore` with the score plus the parsed prediction, confidence,
        and gold — the same fields the official harness records per example.
    """
    import ast  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415

    import dateutil.parser  # noqa: PLC0415

    # ``answer_raw`` is a trusted dataset field, not model output; literal_eval
    # only ever sees the HF gold string (a list literal or a date literal).
    if "datetime" not in answer_raw:
        gold: object = ast.literal_eval(answer_raw)[0]
    else:
        gold = datetime.strptime(answer_raw, "[datetime.date(%Y, %m, %d)]")  # noqa: DTZ007

    trimmed, parse_confidence = _synth_attempt_answer_parse(output)

    score = 0.0
    if str(trimmed) == str(gold):
        score = 1.0
    elif str(trimmed) in ("more common", "less common", "same frequency"):
        score = 1.0 if str(trimmed) in str(gold) else 0.0
    elif answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            score = 0.75 ** abs(int(gold) - int(trimmed))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            score = 0.0
    elif answer_type == "ANSWER_TYPE.DATE":
        try:
            score = 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except (ValueError, OverflowError, TypeError):
            score = 0.0

    return OolongScore(
        score=score,
        prediction=str(trimmed),
        parse_confidence=parse_confidence,
        gold=str(gold),
    )


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
    graded into feedback rather than hard-failing the test. This matters for a
    pairwise benchmark — a ``pytest.fail`` leaves the LangSmith root run
    ``pending`` and never syncs the dataset example, which would drop the
    wrong-answer arm out of the side-by-side comparison entirely. Logged shape:

    - **inputs** (`eval_inputs`): the question + task metadata, identical across
      arms so both experiments share one dataset example;
    - **reference output**: the gold answer;
    - **feedback**: ``score`` (the *only* OOLONG metric — 0-1 with numeric partial
      credit; the paper reports its mean) plus ``agent_steps`` /
      ``tool_call_requests`` (harness efficiency telemetry, not part of OOLONG);
    - **outputs**: the OOLONG per-example record — ``attempted_parse`` (parsed
      prediction), ``parse_confidence``, ``score`` and ``gold_answer``.

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
    result = _synth_process_response(trajectory.answer, example.answer_raw, example.answer_type)
    # `score` is the *only* OOLONG metric: 0-1 with numeric partial credit folded
    # in. The paper reports its mean. We deliberately do not log a separate binary
    # "correctness" — it is not in the OOLONG spec and would discard partial credit
    # on NUMERIC tasks. `agent_steps` / `tool_call_requests` are harness efficiency
    # telemetry, orthogonal to the OOLONG score.
    t.log_feedback(key="score", score=result.score)
    t.log_feedback(key="agent_steps", score=len(trajectory.steps))
    t.log_feedback(
        key="tool_call_requests",
        score=sum(len(step.action.tool_calls) for step in trajectory.steps),
    )
    # Mirror the official harness's per-example record for debuggability.
    t.log_outputs(
        {
            "attempted_parse": result.prediction,
            "parse_confidence": result.parse_confidence,
            "score": result.score,
            "gold_answer": result.gold,
        }
    )
