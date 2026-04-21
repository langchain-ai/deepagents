"""Load Oolong benchmark tasks from HuggingFace.

Port of ``evals/oolong/load-oolong.ts`` in ``langchain-ai/deepagentsjs``.
Env-var knobs (``OOLONG_MAX_PER_DATASET``, ``OOLONG_CONTEXT_LEN``) are
identical so JS and Python runs share configuration.

Streams the validation split via ``datasets.load_dataset(streaming=True)``
and filters in-memory. ``streaming=True`` avoids materializing the
whole DatasetDict (incl. the ~5k test split we don't use) to disk.

One caveat: ``datasets`` streams rows in parquet-shard order, which
groups by source-dataset and then by increasing ``task_id``. Early
``task_id`` ranges correspond to low ``context_len`` buckets, so
setting ``OOLONG_MAX_PER_DATASET=10`` without an ``OOLONG_CONTEXT_LEN``
filter will hand you the ten easiest rows — every one at
``context_len=1024``. The JS harness's rows-server endpoint happens
to interleave context lengths, so the defaults produce a harder
mix there. To run a comparable slice across Python and JS, pin
``OOLONG_CONTEXT_LEN`` explicitly (e.g. ``65536``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

HF_DATASET = "oolongbench/oolong-synth"
HF_SPLIT = "validation"

DEFAULT_MAX_PER_DATASET = 10
"""Same default the JS side uses. Set ``OOLONG_MAX_PER_DATASET=0`` to
disable the cap — only useful for non-streaming scans, since there's
no natural stopping point otherwise."""


@dataclass(frozen=True)
class OolongTask:
    """One Oolong aggregation task.

    Field names mirror the HuggingFace columns after snake_case
    normalization. The JS side camel-cases them; we keep snake_case to
    match Python convention.

    The dataset ships ``input_subset`` as the string ``"True"`` /
    ``"False"`` rather than a boolean — coerced here so the test
    harness doesn't have to.
    """

    id: int
    dataset: str
    context_len: int
    context_window_text: str
    question: str
    task_group: str
    task: str
    answer: str
    answer_type: str
    input_subset: bool
    num_labels: int
    context_window_id: int


def _coerce_bool(raw: object) -> bool:
    """Coerce a dataset ``input_subset`` cell to a bool.

    HF stores these as ``"True"`` / ``"False"`` strings. Guard against
    the shape drifting to actual bools, numeric flags, or mixed case.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() == "true"


def _parse_row(row: dict[str, object]) -> OolongTask:
    """Convert one HF row dict into a typed :class:`OolongTask`."""
    return OolongTask(
        id=int(row["id"]),  # type: ignore[arg-type]
        dataset=str(row["dataset"]),
        context_len=int(row["context_len"]),  # type: ignore[arg-type]
        context_window_text=str(row["context_window_text"]),
        question=str(row["question"]),
        task_group=str(row["task_group"]),
        task=str(row["task"]),
        answer=str(row["answer"]),
        answer_type=str(row["answer_type"]),
        input_subset=_coerce_bool(row["input_subset"]),
        num_labels=int(row["num_labels"]),  # type: ignore[arg-type]
        context_window_id=int(row["context_window_id"]),  # type: ignore[arg-type]
    )


def _resolve_max_per_dataset(override: int | None) -> int:
    """Resolve the per-dataset cap, preferring env > arg > default.

    ``0`` means "no cap" — matches JS. Negative values are treated as
    "no cap" too since they can't be meaningful and silently trimming
    would surprise.
    """
    env = os.environ.get("OOLONG_MAX_PER_DATASET")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            logger.warning("Ignoring invalid OOLONG_MAX_PER_DATASET=%r", env)
    return override if override is not None else DEFAULT_MAX_PER_DATASET


def _resolve_context_len(override: int | None) -> int | None:
    """Resolve the context-len filter, preferring env > arg > None."""
    env = os.environ.get("OOLONG_CONTEXT_LEN")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            logger.warning("Ignoring invalid OOLONG_CONTEXT_LEN=%r", env)
    return override


def _iter_rows() -> Iterable[dict[str, object]]:
    """Stream the OOLONG validation split from HF.

    ``streaming=True`` returns an ``IterableDataset`` that yields rows
    lazily without downloading the full DatasetDict. The caller is
    responsible for stopping early once it has enough rows.
    """
    from datasets import load_dataset

    return load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)


def load_oolong_tasks(
    dataset_name: str,
    *,
    max_per_dataset: int | None = None,
    context_len: int | None = None,
) -> list[OolongTask]:
    """Load up to ``max_per_dataset`` tasks for one source dataset.

    Iterates the cached rows-server fetch in row-index order
    and stops as soon as the cap is hit.

    Args:
        dataset_name: Source dataset name (e.g. ``"trec_coarse"``,
            ``"multinli"``, ``"metaphors"``).
        max_per_dataset: Cap the examples pulled. ``None`` → env var
            or default. ``0`` → no cap, returns every row for the
            dataset.
        context_len: Filter to one context-length bucket (1024, 4096,
            32768, 65536, or 131072). ``None`` → all buckets.

    Returns:
        A list of up to ``max_per_dataset`` tasks, in rows-server order.

    Raises:
        ValueError: No tasks matched the filters.
    """
    cap = _resolve_max_per_dataset(max_per_dataset)
    resolved_len = _resolve_context_len(context_len)
    tasks: list[OolongTask] = []

    for row in _iter_rows():
        if str(row["dataset"]) != dataset_name:
            continue
        if resolved_len is not None and int(row["context_len"]) != resolved_len:  # type: ignore[arg-type]
            continue
        tasks.append(_parse_row(row))
        if cap and cap > 0 and len(tasks) >= cap:
            break

    if not tasks:
        msg = (
            f"No Oolong tasks matched filters "
            f"(dataset_name={dataset_name!r}, "
            f"context_len={resolved_len}, max_per_dataset={cap})"
        )
        raise ValueError(msg)
    logger.info(
        "Loaded %d Oolong tasks for dataset=%s (cap=%s, context_len=%s)",
        len(tasks),
        dataset_name,
        cap,
        resolved_len,
    )
    return tasks


__all__ = [
    "DEFAULT_MAX_PER_DATASET",
    "HF_DATASET",
    "HF_SPLIT",
    "OolongTask",
    "load_oolong_tasks",
]
