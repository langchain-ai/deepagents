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

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        "agent_impl": "bare",
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
        "agent_impl": "bare",
        "ls_dataset": "context-retrieval-evals",
    },
}

DEFAULT_N_SHARDS = {"autonomous": 10, "conversation": 3, "context": 3}

DEEPAGENT_IMPLS = {"bare", "dcode"}
"""Deep-agent harnesses eligible for the `agent_impl` override."""

DEFAULT_AGENT_IMPL = "bare"
"""Deep-agent harness used when `UNIFIED_AGENT_IMPL` is unset or blank."""

KNOWN_AGENT_IMPLS = DEEPAGENT_IMPLS | {"tau3"}
"""Every harness a category may pin (deep-agent harnesses plus `tau3`)."""

# A typo in a CATEGORY_MAP `agent_impl` (e.g. "bear") would silently make that
# category ineligible for the override *and* run it on a nonexistent harness.
# Fail loudly at import instead.
assert all(cm["agent_impl"] in KNOWN_AGENT_IMPLS for cm in CATEGORY_MAP.values()), (
    f"every CATEGORY_MAP agent_impl must be one of {sorted(KNOWN_AGENT_IMPLS)}"
)
# The default must itself be a selectable deep-agent harness, otherwise every
# default run (UNIFIED_AGENT_IMPL unset) would fail validation in main().
assert DEFAULT_AGENT_IMPL in DEEPAGENT_IMPLS, (
    f"DEFAULT_AGENT_IMPL must be one of {sorted(DEEPAGENT_IMPLS)}"
)


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


def slugify(spec: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", spec).strip("-").lower()


def _short_hash(value: str) -> str:
    # slugify is lossy (e.g. "foo/bar" and "foo-bar" collapse to the same slug),
    # so distinct specs can produce identical names. Append a short hash of the
    # raw value to keep dataset/artifact names unique.
    return hashlib.sha256(value.encode()).hexdigest()[:8]


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
    *,
    agent_impl: str | None = None,
) -> dict[str, list[dict]]:
    """Cross-product models and categories into per-provider matrix entries.

    Args:
        models_list: Resolved `provider:model` specs.
        categories: Capability categories to run (keys of `CATEGORY_MAP`).
        shard_parallel: Parallel shards per `(model, category)` leaf.
        n_shards_by_cat: Shard count per category, falling back to
            `DEFAULT_N_SHARDS`.
        agent_impl: Deep-agent harness override. `None` keeps each category's
            `CATEGORY_MAP` default; a value in `DEEPAGENT_IMPLS` replaces the
            default only for categories already pinned to a deep-agent harness
            (a category pinned to a non-deep-agent harness such as `tau3` is
            never overridden).

    Returns:
        A mapping of provider to its list of matrix entries.

    Raises:
        ValueError: If `agent_impl` is neither `None` nor in `DEEPAGENT_IMPLS`.
    """
    # Defense in depth for direct callers: main() validates UNIFIED_AGENT_IMPL,
    # but this public helper must not silently route a run to an unknown harness.
    if agent_impl and agent_impl not in DEEPAGENT_IMPLS:
        msg = (
            f"agent_impl must be one of {sorted(DEEPAGENT_IMPLS)} or None, "
            f"got {agent_impl!r}"
        )
        raise ValueError(msg)
    matrices: dict[str, list[dict]] = {}
    for spec in models_list:
        prov = provider_of(spec)
        for cat in categories:
            cm = CATEGORY_MAP[cat]
            # The override selects the deep-agents harness; it never applies to a
            # category pinned to a non-deep-agents harness (e.g. tau3).
            resolved_impl = (
                agent_impl
                if agent_impl and cm["agent_impl"] in DEEPAGENT_IMPLS
                else cm["agent_impl"]
            )
            entry = {
                "model": spec,
                "provider": prov,
                "category": cat,
                "dataset": cm["dataset"],
                "dataset_path": cm["dataset_path"],
                "agent_impl": resolved_impl,
                # Per-model datasets isolate runs; unified_summary is the
                # cross-model comparison surface.
                "langsmith_dataset": f"{cm['ls_dataset']}__{slugify(spec)}-{_short_hash(spec)}",
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

    agent_impl = (
        os.environ.get("UNIFIED_AGENT_IMPL", DEFAULT_AGENT_IMPL).strip()
        or DEFAULT_AGENT_IMPL
    )

    if not categories:
        raise SystemExit(f"No categories selected. Choose from {sorted(CATEGORY_MAP)}.")
    unknown = [c for c in categories if c not in CATEGORY_MAP]
    if unknown:
        raise SystemExit(
            f"Unknown categor(y/ies): {unknown}. Valid: {sorted(CATEGORY_MAP)}"
        )
    if agent_impl not in DEEPAGENT_IMPLS:
        raise SystemExit(
            f"UNIFIED_AGENT_IMPL must be one of {sorted(DEEPAGENT_IMPLS)}, "
            f"got {agent_impl!r}"
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
        model_specs,
        categories,
        shard_parallel,
        n_shards_by_cat,
        agent_impl=agent_impl,
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
