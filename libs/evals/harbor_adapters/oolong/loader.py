"""OOLONG-synth dataset loader — targeted HuggingFace fetch, no agent code.

Vendored from PR #4213 (``tests/evals/oolong/benchmarks.py``), keeping only the
dataset-loading half: pull a ``(dataset, context_len)`` bucket from the
HuggingFace datasets-server ``/filter`` endpoint (only the matching rows, cached
to JSONL) and materialize each row into an ``OolongExample``. The agent
factories live with the runtime harness, not here — this module is import-light
(stdlib only) so the task generator can run without the agent stack.

Adding a new eval set is just new args (``dataset`` / ``context_len``), never new
code. When PR #4213 merges, dedupe against the canonical loader there.
"""

from __future__ import annotations

import ast
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

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

#: Task groups across all supported subsets.
TASK_GROUPS = ("counting", "user", "timeline")


def _get_json_with_retry(url: str, *, attempts: int = 4) -> dict[str, Any]:
    """GET a JSON payload, retrying the datasets-server's intermittent 5xx."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
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

    if n_examples is None:
        return tuple(_row_to_example(row) for row in rows)

    # Determine which task groups actually exist in this subset/bucket.
    actual_groups = sorted({str(row["task_group"]) for row in rows})
    if not actual_groups:
        return ()
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
