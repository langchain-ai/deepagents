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
        "agent_impl": "bare",
    },
    "conversation": {
        "dataset": "tau3-subset",
        "dataset_path": "",
        "agent_impl": "tau3",
    },
    "context": {
        "dataset": "",
        "dataset_path": "datasets/context-retrieval-evals",
        "agent_impl": "bare",
    },
}

# Harnesses selectable for the code categories (autonomous, context) via the
# `agent_impl` input. Conversation is always tau3 and is never overridden.
CODE_AGENT_IMPLS = {"dcode", "bare"}

# Harness used when the `agent_impl` input (UNIFIED_AGENT_IMPL) is unset or blank.
DEFAULT_AGENT_IMPL = "bare"

# Every harness a category may pin in CATEGORY_MAP (code harnesses plus tau3).
KNOWN_AGENT_IMPLS = CODE_AGENT_IMPLS | {"tau3"}

# A typo in a CATEGORY_MAP `agent_impl` (e.g. "bear") would silently make that
# category ineligible for the override and route it to a nonexistent harness, so
# validate at import. raise (not assert): asserts are stripped under `python -O`.
if not all(cm["agent_impl"] in KNOWN_AGENT_IMPLS for cm in CATEGORY_MAP.values()):
    raise RuntimeError(
        f"every CATEGORY_MAP agent_impl must be one of {sorted(KNOWN_AGENT_IMPLS)}"
    )
if DEFAULT_AGENT_IMPL not in CODE_AGENT_IMPLS:
    raise RuntimeError(f"DEFAULT_AGENT_IMPL must be one of {sorted(CODE_AGENT_IMPLS)}")

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


def _allocate_shard_budgets(counts: dict[str, int], cap: int) -> dict[str, int]:
    """Split `cap` shards across categories proportional to `counts`.

    Each category gets `max(1, cap * count // total)` shards. That floor
    allocation can round up to more than `cap` in total when several small
    categories each get bumped to the 1-shard floor, so any resulting excess is
    trimmed one shard at a time from the currently-largest budget (ties broken
    by category order) until the sum is exactly `<= cap`.
    """
    total = sum(counts.values())
    budgets = {cat: max(1, (cap * n) // total) for cat, n in counts.items()}
    excess = sum(budgets.values()) - cap
    while excess > 0:
        shrinkable = [cat for cat, b in budgets.items() if b > 1]
        if not shrinkable:
            break
        largest = max(shrinkable, key=lambda cat: budgets[cat])
        budgets[largest] -= 1
        excess -= 1
    return budgets


def build_flat_matrix(
    model: str,
    categories: list[str],
    tasks_by_cat: dict[str, list[str]],
    code_impls: list[str] | None = None,
) -> list[dict]:
    """One flat matrix of single-`harbor run` shards spanning categories x configs.

    Code categories (their CATEGORY_MAP agent_impl is in CODE_AGENT_IMPLS) emit one
    shard group per (category, config) across `code_impls`. A non-code category
    (conversation / tau3) emits one group with its pinned agent_impl and is never
    multiplied by configs. The per-model entry count is bounded by
    `shard_matrix.MAX_SHARDS`: the 1-task/shard packing applies when the combined
    (category, config) task count fits under the cap, otherwise MAX_SHARDS is
    allocated across the groups proportional to their task counts and each group is
    packed into its own budget, so the total never exceeds MAX_SHARDS.
    """
    if code_impls is None:
        code_impls = [DEFAULT_AGENT_IMPL]
    prov = provider_of(model)

    # (category, agent_impl, tasks) groups, code categories fanned out over configs.
    groups: list[tuple[str, str, list[str]]] = []
    for cat in categories:
        cm = CATEGORY_MAP[cat]
        tasks = tasks_by_cat.get(cat, [])
        if not tasks:
            continue
        if cm["agent_impl"] in CODE_AGENT_IMPLS:
            for impl in code_impls:
                groups.append((cat, impl, tasks))
        else:
            groups.append((cat, cm["agent_impl"], tasks))

    counts = {(cat, impl): len(tasks) for cat, impl, tasks in groups}
    total = sum(counts.values())
    if total > shard_matrix.MAX_SHARDS:
        budgets = _allocate_shard_budgets(counts, shard_matrix.MAX_SHARDS)
    else:
        budgets = dict.fromkeys(counts, shard_matrix.MAX_SHARDS)

    entries: list[dict] = []
    for cat, impl, tasks in groups:
        cm = CATEGORY_MAP[cat]
        budget = budgets.get((cat, impl), shard_matrix.MAX_SHARDS)
        for group in shard_matrix.pack_tasks(tasks, budget):
            entries.append(
                {
                    "model": model,
                    "provider": prov,
                    "category": cat,
                    "dataset": cm["dataset"],
                    "dataset_path": cm["dataset_path"],
                    "agent_impl": impl,
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

    # Empty defaults to the bare create_deep_agent harness for the code categories.
    code_impl = os.environ.get("UNIFIED_AGENT_IMPL", "").strip() or DEFAULT_AGENT_IMPL
    if code_impl not in CODE_AGENT_IMPLS:
        raise SystemExit(
            f"UNIFIED_AGENT_IMPL must be one of {sorted(CODE_AGENT_IMPLS)}, "
            f"got {code_impl!r}"
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
    est_tasks = max(sum(len(tasks_by_cat.get(c, [])) for c in categories), 1)
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
    }
    # One eval_matrix entry per model, each carrying its own pre-serialized
    # flat matrix. GitHub job outputs are statically declared, so a single
    # matrixable output (rather than one `model_<idx>_matrix` output per
    # model) is what lets the `eval` job's `strategy.matrix` scale to an
    # arbitrary model count.
    eval_include = [
        {
            "model": m,
            "slug": _slug(m),
            "flat_matrix": json.dumps(
                {"include": build_flat_matrix(m, categories, tasks_by_cat, [code_impl])},
                separators=(",", ":"),
            ),
        }
        for m in model_specs
    ]
    outputs["eval_matrix"] = {"include": eval_include}
    _emit(os.environ.get("GITHUB_OUTPUT"), outputs)
    return 0


def _slug(model: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "-", model.replace("/", "-").replace(":", "-"))


if __name__ == "__main__":
    raise SystemExit(main())
