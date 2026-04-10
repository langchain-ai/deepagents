"""Oolong-Synth dataset loader.

Downloads tasks from the HuggingFace `oolongbench/oolong-synth` dataset
(validation split) and caches them locally as JSONL. Supports filtering
by source dataset, context length, and task ID.

Reference: https://arxiv.org/abs/2511.02817
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_PATH = _CACHE_DIR / "tasks.jsonl"

_HF_DATASET = "oolongbench/oolong-synth"
_HF_SPLIT = "validation"


class OolongTask:
    """A single Oolong-Synth aggregation task."""

    __slots__ = (
        "id",
        "dataset",
        "context_len",
        "context_window_text",
        "question",
        "task_group",
        "task",
        "answer",
        "answer_type",
        "context_window_id",
    )

    def __init__(self, row: dict[str, Any]) -> None:
        self.id: int = row["id"]
        self.dataset: str = row["dataset"]
        self.context_len: int = row["context_len"]
        self.context_window_text: str = row["context_window_text"]
        self.question: str = row["question"]
        self.task_group: str = row["task_group"]
        self.task: str = row["task"]
        self.answer: str = row["answer"]
        self.answer_type: str = row["answer_type"]
        self.context_window_id: int = row["context_window_id"]


def _fetch_and_cache() -> None:
    """Download the full validation split and cache as JSONL."""

    logger.info("Downloading %s (%s split) from HuggingFace...", _HF_DATASET, _HF_SPLIT)
    ds = load_dataset(_HF_DATASET, split=_HF_SPLIT)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    logger.info("Cached %d tasks -> %s", len(ds), _CACHE_PATH)


def _load_cache() -> list[dict[str, Any]]:
    """Read cached JSONL rows."""
    rows: list[dict[str, Any]] = []
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_oolong_tasks(
    *,
    dataset: str | None = None,
    context_len: int | None = None,
    task_ids: set[int] | None = None,
) -> list[OolongTask]:
    """Load Oolong-Synth tasks with optional filtering.

    Downloads from HuggingFace on first call, then reads from a local
    JSONL cache.

    Args:
        dataset: Filter to a source dataset (e.g. `"trec_coarse"`).
        context_len: Filter to a specific context length bucket
            (e.g. `131072`).
        task_ids: If provided, only return tasks whose `id` is in this
            set. Applied after dataset/context_len filtering.

    Returns:
        List of matching `OolongTask` objects.
    """
    if not _CACHE_PATH.exists():
        _fetch_and_cache()

    rows = _load_cache()

    if dataset is not None:
        rows = [r for r in rows if r.get("dataset") == dataset]

    if context_len is not None:
        rows = [r for r in rows if r.get("context_len") == context_len]

    if task_ids is not None:
        rows = [r for r in rows if r["id"] in task_ids]

    return [OolongTask(r) for r in rows]
