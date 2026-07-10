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


def _emit(github_output: str | None, outputs: dict[str, object]) -> None:
    if not github_output:
        for k, v in outputs.items():
            payload = v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
            print(f"{k}={payload}")
        return
    with open(github_output, "a") as f:
        for k, v in outputs.items():
            payload = v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
            f.write(f"{k}={payload}\n")


def main(argv: list[str] | None = None) -> int:
    selection = os.environ.get("UNIFIED_MODELS", "").strip()
    categories = [c.strip() for c in os.environ.get("UNIFIED_CATEGORIES", "autonomous,conversation,context").split(",") if c.strip()]
    concurrency = int(os.environ.get("UNIFIED_CONCURRENCY", "4"))
    requested_sp = int(os.environ.get("UNIFIED_SHARD_PARALLEL", "10"))
    n_shards_by_cat = {
        "autonomous": int(os.environ.get("UNIFIED_N_SHARDS_AUTONOMOUS", DEFAULT_N_SHARDS["autonomous"])),
        "conversation": int(os.environ.get("UNIFIED_N_SHARDS_CONVERSATION", DEFAULT_N_SHARDS["conversation"])),
        "context": int(os.environ.get("UNIFIED_N_SHARDS_CONTEXT", DEFAULT_N_SHARDS["context"])),
    }

    unknown = [c for c in categories if c not in CATEGORY_MAP]
    if unknown:
        raise SystemExit(f"Unknown categor(y/ies): {unknown}. Valid: {sorted(CATEGORY_MAP)}")
    if concurrency < 1 or concurrency > MAX_TASKS_PER_MODEL:
        raise SystemExit(f"concurrency must be 1..{MAX_TASKS_PER_MODEL}, got {concurrency}")

    # Validate + dedupe the free-form CSV via the shared resolver.
    try:
        model_specs = models._resolve_models("harbor", selection)
    except ValueError as exc:
        raise SystemExit(str(exc))

    providers_present: list[str] = []
    seen = set()
    for spec in model_specs:
        p = provider_of(spec)
        if p not in seen:
            seen.add(p)
            providers_present.append(p)

    shard_parallel = clamp_shard_parallel(requested_sp, len(providers_present), concurrency)
    # clamp guarantees both invariants; assert as a safety net.
    assert concurrency * shard_parallel <= MAX_TASKS_PER_MODEL
    assert len(providers_present) * shard_parallel <= MAX_RUNNERS

    matrices = build_provider_matrices(model_specs, categories, shard_parallel, n_shards_by_cat)

    outputs: dict[str, object] = {
        "effective_shard_parallel": str(shard_parallel),
        "providers": providers_present,
    }
    for prov in sorted(KNOWN_PROVIDERS | {"other"}):
        include = matrices.get(prov, [])
        outputs[f"{prov}_has_models"] = "true" if include else "false"
        if include:
            outputs[f"{prov}_matrix"] = {"include": include}

    _emit(os.environ.get("GITHUB_OUTPUT"), outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
