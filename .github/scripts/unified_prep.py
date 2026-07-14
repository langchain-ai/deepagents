"""Prep step for the unified multi-model Harbor evals orchestrator.

Parses a free-form comma-separated model CSV, validates it via models.py,
maps each category to its Harbor dataset, and emits a per-model flat matrix
(one entry per single-task shard, spanning every category) to GITHUB_OUTPUT.

Pool sizing is derived by `derive_pool`, not clamped after the fact: given
`concurrency` (trials in flight per task) and `rollouts` (trials per task),
`per_shard = min(concurrency, rollouts)` is the peak concurrent trials a
single 1-task shard ever runs. `max_parallel = MAX_TASKS_PER_MODEL //
per_shard` saturates the per-model 40-trial budget, and `model_parallel =
MAX_RUNNERS // max_parallel` bounds how many models run at once so total
runners stay within MAX_RUNNERS. Both invariants hold by construction:
  per model:  concurrency * max_parallel <= MAX_TASKS_PER_MODEL (40)
  global:     model_parallel * max_parallel <= MAX_RUNNERS (80)
`total_job_guard` separately caps the total job count (n_models * est_tasks)
against a fixed budget so an oversized selection fails fast instead of
launching a firehose.
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
MAX_RUNNERS = 80

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

# Harnesses selectable for the code categories (autonomous, context) via the
# `agent_impl` input. Conversation is always tau3 and is never overridden.
CODE_AGENT_IMPLS = {"dcode", "bare"}

# Run profiles: "full" = every task in each category; "lite" = the frozen
# high-signal subset from lite_tasks.py (fewer tasks, full rollouts).
PROFILES = {"full", "lite"}

TOTAL_JOB_BUDGET = 400


def total_job_guard(n_models: int, est_tasks_per_model: int) -> None:
    """Fail when the flat matrix would generate too many total jobs.

    At 1-task/shard the run generates n_models * est_tasks jobs. GitHub-hosted
    Actions become unreliable well before that count needs to be large, so cap
    it and point at the worker-pool escalation instead of silently launching a
    firehose.
    """
    total = n_models * est_tasks_per_model
    if total > TOTAL_JOB_BUDGET:
        raise SystemExit(
            f"Flat matrix would generate ~{total} jobs "
            f"({n_models} models x ~{est_tasks_per_model} tasks), over "
            f"TOTAL_JOB_BUDGET={TOTAL_JOB_BUDGET}. Reduce the model set or task "
            "count, or move to a worker pool orchestrator (see the flat-pool spec)."
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


def provider_of(spec: str, known: set[str] = KNOWN_PROVIDERS) -> str:
    prefix = spec.split(":", 1)[0]
    return prefix if prefix in known else "other"


def derive_pool(
    concurrency: int, rollouts: int, n_shards: int, n_models: int
) -> tuple[int, int]:
    """Derive (max_parallel, model_parallel) from concurrency and rollouts.

    per_shard is the peak concurrent trials in one 1-task shard, which a shard
    can never exceed: min(concurrency, rollouts). max_parallel saturates the
    per-model 40-trial budget; model_parallel bounds how many models run at once
    so total runners stay within MAX_RUNNERS. Both hold the invariants by
    construction, so no separate clamp/assert is needed.
    """
    per_shard = max(1, min(concurrency, rollouts))
    max_parallel = max(1, min(MAX_TASKS_PER_MODEL // per_shard, n_shards))
    model_parallel = max(1, min(MAX_RUNNERS // max_parallel, n_models))
    return max_parallel, model_parallel


def build_flat_matrix(
    model: str,
    categories: list[str],
    tasks_by_cat: dict[str, list[str]],
    dcode_impl: str = "dcode",
) -> list[dict]:
    """One flat matrix of single-`harbor run` shards spanning all categories.

    Each category's task list is packed into <= MAX_SHARDS groups (1 task each
    below the cap); each group is one matrix entry carrying its own dataset and
    agent so the leaf's max-parallel pool drains the mixed queue across category
    boundaries. `provider` is retained only as an aggregation tag.
    """
    prov = provider_of(model)
    entries: list[dict] = []
    for cat in categories:
        cm = CATEGORY_MAP[cat]
        agent_impl = dcode_impl if cm["agent_impl"] == "dcode" else cm["agent_impl"]
        for group in shard_matrix.pack_tasks(tasks_by_cat.get(cat, [])):
            entries.append(
                {
                    "model": model,
                    "provider": prov,
                    "category": cat,
                    "dataset": cm["dataset"],
                    "dataset_path": cm["dataset_path"],
                    "agent_impl": agent_impl,
                    "include_tasks": " ".join(group),
                    "langsmith_dataset": "",
                    "n_shards": 1,
                    "shard": 0,
                }
            )
    return entries


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
    rollouts = parse_int_input(
        "UNIFIED_ROLLOUTS", os.environ.get("UNIFIED_ROLLOUTS", "3"), minimum=1
    )

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

    # Resolve the per-category task lists.
    if profile == "lite":
        tasks_by_cat = {c: list(lite_tasks.LITE_TASKS.get(c, [])) for c in categories}
    else:
        tasks_json = os.environ.get("UNIFIED_TASKS_JSON", "").strip()
        if not tasks_json:
            raise SystemExit("full profile requires UNIFIED_TASKS_JSON (enumerated tasks).")
        with open(tasks_json) as f:
            tasks_by_cat = json.load(f)

    n_models = len(model_specs)
    est_tasks = max((sum(len(tasks_by_cat.get(c, [])) for c in categories), 1))
    total_job_guard(n_models, est_tasks)

    # n_shards for the pool = number of single-task shards (pre-pack); derive pool.
    n_shards = est_tasks
    max_parallel, model_parallel = derive_pool(concurrency, rollouts, n_shards, n_models)

    outputs: dict[str, object] = {
        # The expected grid, so the combiner can flag missing/incomplete leaves
        # instead of silently ranking a model on fewer categories.
        "models": model_specs,
        "categories": categories,
        "max_parallel": str(max_parallel),
        "model_parallel": str(model_parallel),
        "model_slugs": [_slug(m) for m in model_specs],
    }
    for idx, model in enumerate(model_specs):
        entries = build_flat_matrix(model, categories, tasks_by_cat, dcode_impl)
        outputs[f"model_{idx}_matrix"] = {"include": entries}
    _emit(os.environ.get("GITHUB_OUTPUT"), outputs)
    return 0


def _slug(model: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "-", model.replace("/", "-").replace(":", "-"))


if __name__ == "__main__":
    raise SystemExit(main())
