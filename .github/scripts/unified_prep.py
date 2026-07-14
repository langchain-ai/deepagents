"""Prep step for the unified multi-model Harbor evals orchestrator.

Parses a free-form comma-separated model CSV, validates it via models.py,
buckets specs by provider, clamps shard_parallel to satisfy the two
concurrency invariants, maps each category to its Harbor dataset, and emits
per-provider matrices to GITHUB_OUTPUT.

Concurrency invariants:
  per model:  concurrency * shard_parallel <= 40
  global:     num_providers * shard_parallel <= 64
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lite_tasks  # noqa: E402  (lite_tasks.py in same dir)
import models  # noqa: E402  (models.py in same dir)
import shard_matrix  # noqa: E402  (shard_matrix.py in same dir)

MAX_TASKS_PER_MODEL = 40
MAX_RUNNERS = 64

KNOWN_PROVIDERS = {
    "anthropic",
    "baseten",
    "fireworks",
    "google_genai",
    "groq",
    "nvidia",
    "ollama",
    "openai",
    "openrouter",
    "xai",
}

CATEGORY_MAP: dict[str, dict] = {
    "autonomous": {
        "dataset": "harbor-index/harbor-index-1.0",
        "dataset_path": "",
        "agent_impl": "dcode",
    },
    "conversation": {
        "dataset": "tau3-subset",
        "dataset_path": "",
        "agent_impl": "tau3",
    },
    "context": {
        "dataset": "",
        "dataset_path": "datasets/context-retrieval-evals",
        "agent_impl": "dcode",
    },
}

DEFAULT_N_SHARDS = {"autonomous": 10, "conversation": 3, "context": 3}

# Harnesses selectable for the code categories (autonomous, context) via the
# `agent_impl` input. Conversation is always tau3 and is never overridden.
CODE_AGENT_IMPLS = {"dcode", "bare"}

# Run profiles: "full" = every task in each category; "lite" = the frozen
# high-signal subset from lite_tasks.py (fewer tasks, full rollouts).
PROFILES = {"full", "lite"}


def parse_int_input(
    name: str,
    raw: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Parse an integer input constrained to an inclusive range.

    Args:
        name: Input name to include in validation errors.
        raw: Raw input value.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or `None` for no upper bound.

    Returns:
        The parsed integer.

    Raises:
        SystemExit: If `raw` is not an integer in the accepted range.
    """
    accepted_range = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
    try:
        value = int(raw.strip())
    except ValueError:
        msg = f"{name} must be an integer in {accepted_range}, got {raw!r}"
        raise SystemExit(msg) from None
    if value < minimum or (maximum is not None and value > maximum):
        msg = f"{name} must be an integer in {accepted_range}, got {raw!r}"
        raise SystemExit(msg)
    return value


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
    dcode_impl: str = "dcode",
    profile: str = "full",
) -> dict[str, list[dict]]:
    matrices: dict[str, list[dict]] = {}
    for spec in models_list:
        prov = provider_of(spec)
        for cat in categories:
            cm = CATEGORY_MAP[cat]
            # `dcode_impl` swaps the harness for the code categories only; the
            # conversation category's tau3 harness is left untouched.
            agent_impl = dcode_impl if cm["agent_impl"] == "dcode" else cm["agent_impl"]
            # `lite` runs a frozen high-signal subset (full rollouts, fewer tasks).
            # Spread ~2 tasks per shard, capped at shard_parallel so every shard
            # runs at once: this maxes concurrency (categories are serialized per
            # provider, so a category never exceeds shard_parallel*concurrency
            # trials in flight) and scatters heavy task images one-per-runner
            # instead of concentrating them in a fat shard.
            include = lite_tasks.include_tasks(cat) if profile == "lite" else ""
            n_shards = n_shards_by_cat.get(cat, DEFAULT_N_SHARDS[cat])
            if include:
                n_tasks = len(include.split())
                n_shards = min(shard_parallel, max(1, (n_tasks + 1) // 2))
            entry = {
                "model": spec,
                "provider": prov,
                "category": cat,
                "dataset": cm["dataset"],
                "dataset_path": cm["dataset_path"],
                "agent_impl": agent_impl,
                "include_tasks": include,
                # Empty: the leaf derives the canonical shared dataset name per
                # category (harbor-index/harbor-index-1.0, tau3-subset,
                # local/...), so every model attaches as an experiment on the one
                # shared dataset. unified_summary remains the cross-model surface.
                "langsmith_dataset": "",
                "n_shards": n_shards,
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
    # Order-preserving dedupe so a repeated category can't produce duplicate
    # (model, category) entries with colliding artifact/dataset names.
    categories = list(
        dict.fromkeys(
            c.strip()
            for c in os.environ.get(
                "UNIFIED_CATEGORIES", "autonomous,conversation,context"
            ).split(",")
            if c.strip()
        )
    )
    concurrency = parse_int_input(
        "UNIFIED_CONCURRENCY",
        os.environ.get("UNIFIED_CONCURRENCY", "4"),
        minimum=1,
        maximum=MAX_TASKS_PER_MODEL,
    )
    requested_sp = parse_int_input(
        "UNIFIED_SHARD_PARALLEL",
        os.environ.get("UNIFIED_SHARD_PARALLEL", "10"),
        minimum=1,
    )
    n_shards_by_cat = {
        category: parse_int_input(
            f"UNIFIED_N_SHARDS_{category.upper()}",
            os.environ.get(f"UNIFIED_N_SHARDS_{category.upper()}", str(default)),
            minimum=1,
            maximum=shard_matrix.MAX_SHARDS,
        )
        for category, default in DEFAULT_N_SHARDS.items()
    }

    # Empty defaults to "dcode" (the historical per-category default).
    dcode_impl = os.environ.get("UNIFIED_AGENT_IMPL", "").strip() or "dcode"
    if dcode_impl not in CODE_AGENT_IMPLS:
        raise SystemExit(
            f"UNIFIED_AGENT_IMPL must be one of {sorted(CODE_AGENT_IMPLS)}, "
            f"got {dcode_impl!r}"
        )

    profile = os.environ.get("UNIFIED_PROFILE", "").strip() or "full"
    if profile not in PROFILES:
        raise SystemExit(
            f"UNIFIED_PROFILE must be one of {sorted(PROFILES)}, got {profile!r}"
        )

    if not categories:
        raise SystemExit(f"No categories selected. Choose from {sorted(CATEGORY_MAP)}.")
    unknown = [c for c in categories if c not in CATEGORY_MAP]
    if unknown:
        raise SystemExit(
            f"Unknown categor(y/ies): {unknown}. Valid: {sorted(CATEGORY_MAP)}"
        )
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

    shard_parallel = clamp_shard_parallel(
        requested_sp, len(providers_present), concurrency
    )
    # clamp guarantees both invariants; assert as a safety net.
    assert concurrency * shard_parallel <= MAX_TASKS_PER_MODEL
    assert len(providers_present) * shard_parallel <= MAX_RUNNERS

    matrices = build_provider_matrices(
        model_specs, categories, shard_parallel, n_shards_by_cat, dcode_impl, profile
    )

    outputs: dict[str, object] = {
        "effective_shard_parallel": str(shard_parallel),
        # The expected grid, so the combiner can flag missing/incomplete leaves
        # instead of silently ranking a model on fewer categories.
        "models": model_specs,
        "categories": categories,
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
