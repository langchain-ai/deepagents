"""Offline script to build chunked Oolong-Synth cache files.

Streams the oolongbench/oolong-synth dataset (validation + test splits)
using the HuggingFace ``datasets`` library and writes one JSONL file per
(dataset, context_len) pair into ``.cache/``.  Rows are written
incrementally to avoid loading the full dataset into memory.

File naming convention::

    .cache/{dataset}_{context_len}.jsonl

For example::

    .cache/spam_65536.jsonl
    .cache/trec_coarse_16384.jsonl

Usage::

    cd libs/evals
    uv run --group test python tests/evals/oolong/build_cache.py

This will pull all rows from the dataset, although ongoing evals only uses a small
subset.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HF_DATASET = "oolongbench/oolong-synth"
_CACHE_DIR = Path(__file__).parent / ".cache"


def main() -> None:
    """Download dataset and write chunked JSONL files."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    handles: dict[tuple[str, int], Any] = {}
    counts: dict[tuple[str, int], int] = defaultdict(int)
    total = 0

    try:
        for split in ("validation", "test"):
            logger.info("Streaming %s (%s split)...", _HF_DATASET, split)
            ds = load_dataset(_HF_DATASET, split=split, streaming=True)
            for row in ds:
                row_dict = dict(row)
                key = (row_dict["dataset"], row_dict["context_len"])
                if key not in handles:
                    path = _CACHE_DIR / f"{key[0]}_{key[1]}.jsonl"
                    handles[key] = path.open("w", encoding="utf-8")
                handles[key].write(json.dumps(row_dict, ensure_ascii=False) + "\n")
                counts[key] += 1
                total += 1
                if total % 1000 == 0:
                    logger.info("  %s split: %d rows streamed...", split, total)
    finally:
        for fh in handles.values():
            fh.close()

    logger.info("Streamed %d rows across %d buckets", total, len(counts))
    for (dataset, context_len), count in sorted(counts.items()):
        logger.info("  %s_%s.jsonl: %d rows", dataset, context_len, count)
    logger.info("Done. Cache directory: %s", _CACHE_DIR)


if __name__ == "__main__":
    main()
