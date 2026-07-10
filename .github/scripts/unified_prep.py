"""Prep step for the unified multi-model Harbor evals orchestrator.

Parses a free-form comma-separated model CSV, validates it via models.py,
buckets specs by provider, clamps shard_parallel to satisfy the two
concurrency invariants, maps each category to its Harbor dataset, and emits
per-provider matrices to GITHUB_OUTPUT.

Invariants (see docs/superpowers/specs/2026-07-10-unified-evals-ci-design.md §6):
  per model:  concurrency * shard_parallel <= 40
  global:     num_providers * shard_parallel <= 64
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402  (models.py in same dir)

MAX_TASKS_PER_MODEL = 40
MAX_RUNNERS = 64

KNOWN_PROVIDERS = {
    "anthropic", "baseten", "fireworks", "google_genai", "groq",
    "nvidia", "ollama", "openai", "openrouter", "xai",
}

CATEGORY_MAP: dict[str, dict] = {
    "autonomous": {
        "dataset": "harbor-index/harbor-index-1.0",
        "dataset_path": "",
        "agent_impl": "dcode",
        "ls_dataset": "harbor-index",
    },
    "conversation": {
        "dataset": "tau3-subset",
        "dataset_path": "",
        "agent_impl": "tau3",
        "ls_dataset": "tau3-subset",
    },
    "context": {
        "dataset": "",
        "dataset_path": "datasets/context-retrieval-evals",
        "agent_impl": "dcode",
        "ls_dataset": "context-retrieval-evals",
    },
}

DEFAULT_N_SHARDS = {"autonomous": 10, "conversation": 3, "context": 3}


def slugify(spec: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", spec).strip("-").lower()


def provider_of(spec: str, known: set[str] = KNOWN_PROVIDERS) -> str:
    prefix = spec.split(":", 1)[0]
    return prefix if prefix in known else "other"


def clamp_shard_parallel(requested: int, num_providers: int, concurrency: int) -> int:
    sp = min(
        requested,
        MAX_RUNNERS // max(num_providers, 1),
        MAX_TASKS_PER_MODEL // max(concurrency, 1),
    )
    return max(sp, 1)


def build_provider_matrices(
    models_list: list[str],
    categories: list[str],
    shard_parallel: int,
    n_shards_by_cat: dict[str, int],
) -> dict[str, list[dict]]:
    matrices: dict[str, list[dict]] = {}
    for spec in models_list:
        prov = provider_of(spec)
        for cat in categories:
            cm = CATEGORY_MAP[cat]
            entry = {
                "model": spec,
                "provider": prov,
                "category": cat,
                "dataset": cm["dataset"],
                "dataset_path": cm["dataset_path"],
                "agent_impl": cm["agent_impl"],
                "langsmith_dataset": f"{cm['ls_dataset']}__{slugify(spec)}",
                "n_shards": n_shards_by_cat.get(cat, DEFAULT_N_SHARDS[cat]),
                "shard_parallel": shard_parallel,
            }
            matrices.setdefault(prov, []).append(entry)
    return matrices
