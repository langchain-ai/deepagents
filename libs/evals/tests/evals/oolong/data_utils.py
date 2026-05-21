"""Oolong-Synth dataset loader.

Reads pre-built JSONL cache files from ``.cache/``. Each file contains
rows for a single (dataset, context_len) pair, named
``{dataset}_{context_len}.jsonl``.

Cache files are built offline by running ``build_cache.py``::

    cd libs/evals
    uv run --group test python tests/evals/oolong/build_cache.py

See ``build_cache.py`` for details on how the cache is generated.

Reference: https://arxiv.org/abs/2511.02817
Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / ".cache"


class OolongTask:
    """A single Oolong-Synth aggregation task."""

    __slots__ = (
        "answer",
        "answer_type",
        "context_len",
        "context_window_id",
        "context_window_text",
        "dataset",
        "id",
        "question",
        "task",
        "task_group",
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


def _cache_path(dataset: str, context_len: int) -> Path:
    """Return the cache file path for a (dataset, context_len) pair.

    Args:
        dataset: Source dataset name.
        context_len: Context length bucket.

    Returns:
        Path to the JSONL cache file.
    """
    return _CACHE_DIR / f"{dataset}_{context_len}.jsonl"


def load_oolong_tasks(
    *,
    dataset: str,
    context_len: int,
    task_ids: set[int] | None = None,
) -> list[OolongTask]:
    """Load Oolong-Synth tasks from a pre-built cache file.

    Args:
        dataset: Source dataset name (e.g. `"spam"`, `"agnews"`).
        context_len: Context length bucket (e.g. `65536`).
        task_ids: If provided, only return tasks whose `id` is in this
            set.

    Returns:
        List of matching `OolongTask` objects.

    Raises:
        FileNotFoundError: If the cache file does not exist. Run
            ``build_cache.py`` first.
        RuntimeError: If ``task_ids`` is provided but some IDs are
            missing from the cache file.
    """
    path = _cache_path(dataset, context_len)
    if not path.exists():
        msg = (
            f"Cache file not found: {path}\n"
            f"Run build_cache.py first:\n"
            f"  cd libs/evals\n"
            f"  uv run --group test python tests/evals/oolong/build_cache.py"
        )
        raise FileNotFoundError(msg)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # noqa: PLW2901  # intentional re-strip of loop variable
            if line:
                rows.append(json.loads(line))

    if task_ids is not None:
        found_ids = {r["id"] for r in rows}
        missing = task_ids - found_ids
        if missing:
            msg = f"Task IDs {missing} not found in {path.name}"
            raise RuntimeError(msg)
        rows = [r for r in rows if r["id"] in task_ids]

    return [OolongTask(r) for r in rows]
