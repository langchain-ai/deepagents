"""Prep step for the unified multi-model Harbor evals orchestrator.

Parses a free-form comma-separated model CSV, validates it via models.py,
maps each category to its Harbor dataset, and emits a per-model flat matrix
(one entry per shard, spanning every category) to GITHUB_OUTPUT.

Pool sizing is derived by `derive_pool`, not clamped after the fact: given
`concurrency` (trials in flight per shard job), `max_parallel =
MAX_TASKS_PER_MODEL // concurrency` is the per-model concurrent-shard budget.
The inner parallelism divides that budget across compared branches, while
the outer parallelism bounds concurrent `(model, branch)` jobs. Both invariants
hold by construction:
  per model:  branches * concurrency * inner <= MAX_TASKS_PER_MODEL (40)
  global:     outer * inner <= MAX_RUNNERS (80)
`total_job_guard` separately caps the total post-pack job count (summed across
models and branches) against a fixed budget so an oversized selection fails
fast instead of launching a firehose.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lite_tasks  # noqa: E402  (lite_tasks.py in same dir)
import models  # noqa: E402  (models.py in same dir)
import shard_matrix  # noqa: E402  (shard_matrix.py in same dir)
from unified_types import LeafKey  # noqa: E402

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
# `agent_impls` input. Conversation is always tau3 and is never overridden.
CODE_AGENT_IMPLS = {"dcode", "bare"}

# Harness used when the `agent_impls` input (UNIFIED_AGENT_IMPLS) is unset or blank.
DEFAULT_AGENT_IMPL = "bare"

# Every harness a category may pin in CATEGORY_MAP (code harnesses plus tau3).
KNOWN_AGENT_IMPLS = CODE_AGENT_IMPLS | {"tau3"}

# A typo in a CATEGORY_MAP `agent_impl` (e.g. "bear") would silently make that
# category ineligible for the override and route it to a nonexistent harness, so
# validate at import. raise (not assert): asserts are stripped under `python -O`.
if not all(cm["agent_impl"] in KNOWN_AGENT_IMPLS for cm in CATEGORY_MAP.values()):
    raise RuntimeError(f"every CATEGORY_MAP agent_impl must be one of {sorted(KNOWN_AGENT_IMPLS)}")
if DEFAULT_AGENT_IMPL not in CODE_AGENT_IMPLS:
    raise RuntimeError(f"DEFAULT_AGENT_IMPL must be one of {sorted(CODE_AGENT_IMPLS)}")

# Run profiles: "full" = every task in each category; "lite" = the frozen
# high-signal subset from lite_tasks.py (fewer tasks, full rollouts).
PROFILES = {"full", "lite"}

TOTAL_JOB_BUDGET = 400


def total_job_guard(total_jobs: int) -> None:
    """Fail when the built flat matrices would generate too many total jobs.

    `total_jobs` is the actual post-pack entry count summed across models and
    branches (what GitHub launches), not the pre-pack task count. Packing bounds
    each model at MAX_SHARDS, so this reflects the real matrix size. GitHub-hosted
    Actions become unreliable well before an unbounded count, so cap it and point
    at the worker-pool escalation instead of silently launching a firehose.
    """
    if total_jobs <= 0:
        raise SystemExit("Flat matrix would generate no jobs; select at least one task.")
    if total_jobs > TOTAL_JOB_BUDGET:
        raise SystemExit(
            f"Flat matrix would generate {total_jobs} jobs, over "
            f"TOTAL_JOB_BUDGET={TOTAL_JOB_BUDGET}. Reduce the model set, config "
            "count, or task count, or move to a worker pool orchestrator (see the "
            "flat-pool spec)."
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
    concurrency: int,
    rollouts: int,
    n_shards: int,
    n_groups: int,
    n_branches: int = 1,
) -> tuple[int, int]:
    """Derive (inner_max_parallel, outer_parallel).

    A packed shard can run multiple tasks and therefore uses its full
    `concurrency` even when `rollouts` is lower, so the per-model concurrent-shard
    budget divides MAX_TASKS_PER_MODEL by `concurrency`. Dividing that budget by
    `n_branches` keeps a model's concurrently-running branch jobs summed within it,
    so per-model provider load is unchanged by the branch axis. outer_parallel
    bounds how many (model, branch) jobs run at once so total runners stay within
    MAX_RUNNERS. With n_branches == 1 and n_groups == n_models this is the
    pre-branch behavior. `rollouts` stays in the signature for callers that supply
    all run limits.
    """
    del rollouts
    per_model = max(1, MAX_TASKS_PER_MODEL // max(1, concurrency))
    inner = max(1, min(per_model // n_branches, n_shards))
    outer = max(1, min(MAX_RUNNERS // inner, n_groups))
    return inner, outer


def _load_tasks_json(path: str) -> dict[str, list[str]]:
    """Load the full-profile task mapping from an enumerated JSON file."""
    msg = "UNIFIED_TASKS_JSON must be a JSON object mapping category names to lists of task strings"
    try:
        with open(path) as f:
            raw: object = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(msg) from exc
    if not isinstance(raw, dict) or not all(
        isinstance(category, str)
        and isinstance(tasks, list)
        and all(isinstance(task, str) for task in tasks)
        for category, tasks in raw.items()
    ):
        raise SystemExit(msg)
    return cast(dict[str, list[str]], raw)


def _resolve_branch_sha(branch: str) -> str:
    """Resolve a validated remote ref to an immutable commit SHA."""
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or branch.startswith("-") or ".." in branch:
        raise SystemExit(f"Invalid branch ref: {branch!r}")
    try:
        result = subprocess.run(
            ["git", "ls-remote", "origin", branch],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        msg = f"Could not resolve branch ref {branch!r} from origin."
        raise SystemExit(msg) from exc
    line = result.stdout.splitlines()
    if not line:
        raise SystemExit(f"Branch ref {branch!r} was not found on origin.")
    sha = line[0].split(maxsplit=1)[0]
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise SystemExit(f"Origin returned an invalid SHA for branch ref {branch!r}.")
    return sha


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
    code_impls = list(dict.fromkeys(code_impls))
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
            for c in os.environ.get("UNIFIED_CATEGORIES", "autonomous,conversation,context").split(
                ","
            )
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

    # Comma list of code harnesses; empty defaults to the bare create_deep_agent
    # harness. Conversation is always tau3 and is never taken from this input.
    raw_impls = os.environ.get("UNIFIED_AGENT_IMPLS", "").strip()
    code_impls = list(dict.fromkeys(s.strip() for s in raw_impls.split(",") if s.strip())) or [
        DEFAULT_AGENT_IMPL
    ]
    unknown_impls = [i for i in code_impls if i not in CODE_AGENT_IMPLS]
    if unknown_impls:
        raise SystemExit(
            f"UNIFIED_AGENT_IMPLS entries must be in {sorted(CODE_AGENT_IMPLS)}, "
            f"got unknown {unknown_impls}"
        )

    # Comma list of git refs to pull agent source from; empty means the current
    # checkout only (the sentinel "current" runs no overlay in the leaf).
    raw_branches = os.environ.get("UNIFIED_BRANCHES", "").strip()
    branches = list(dict.fromkeys(b.strip() for b in raw_branches.split(",") if b.strip())) or [
        "current"
    ]

    profile = os.environ.get("UNIFIED_PROFILE", "").strip() or "full"
    if profile not in PROFILES:
        raise SystemExit(f"UNIFIED_PROFILE must be one of {sorted(PROFILES)}, got {profile!r}")

    if not categories:
        raise SystemExit(f"No categories selected. Choose from {sorted(CATEGORY_MAP)}.")
    unknown = [c for c in categories if c not in CATEGORY_MAP]
    if unknown:
        raise SystemExit(f"Unknown categor(y/ies): {unknown}. Valid: {sorted(CATEGORY_MAP)}")
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
        tasks_by_cat = _load_tasks_json(tasks_json)

    empty_categories = [category for category in categories if not tasks_by_cat.get(category)]
    if empty_categories:
        raise SystemExit(f"No tasks resolved for requested categor(y/ies): {empty_categories}")

    n_models = len(model_specs)

    # Build every model's flat matrix up front so the job guard and pool sizing use
    # the actual post-pack entry counts (packing can shrink these below the pre-pack
    # task totals when a large config x task grid packs multiple tasks per shard).
    per_model_matrices = {
        m: build_flat_matrix(m, categories, tasks_by_cat, code_impls) for m in model_specs
    }
    outer_entries = len(model_specs) * len(branches)
    if outer_entries > shard_matrix.GITHUB_MATRIX_MAX:
        raise SystemExit(
            f"eval matrix would have {outer_entries} (model, branch) entries, over "
            f"GitHub's {shard_matrix.GITHUB_MATRIX_MAX}-entry matrix cap "
            f"({len(model_specs)} models x {len(branches)} branches). Reduce models or branches."
        )
    # Every branch runs the same post-pack per-model matrix, so the actual job
    # count is the per-model total multiplied by the branch axis.
    total_jobs = sum(len(entries) for entries in per_model_matrices.values()) * len(branches)
    total_job_guard(total_jobs)

    # Pool sizing stays per-model. n_shards is the largest per-model entry count
    # (what one model's shared pool drains); derive_pool caps max_parallel so
    # per-model concurrency is unchanged by the config axis.
    n_shards = max((len(v) for v in per_model_matrices.values()), default=1)
    n_branches = len(branches)
    # A packed shard uses full `concurrency`, so the per-model concurrent-shard
    # budget divides by concurrency (not min(concurrency, rollouts)).
    budget_shards = max(1, MAX_TASKS_PER_MODEL // concurrency)
    if n_branches > budget_shards:
        raise SystemExit(
            f"branches_to_compare has {n_branches} branches but the per-model "
            f"concurrent-shard budget is {budget_shards} (at concurrency="
            f"{concurrency}). Reduce branches or lower concurrency so branches "
            "can share the per-model budget."
        )
    max_parallel, model_parallel = derive_pool(
        concurrency, rollouts, n_shards, n_models * n_branches, n_branches
    )
    branch_shas = {
        branch: "" if branch == "current" else _resolve_branch_sha(branch) for branch in branches
    }

    expected_keys: list[LeafKey] = []
    seen_leaves: set[LeafKey] = set()
    for m, entries in per_model_matrices.items():
        for b in branches:
            for e in entries:
                key = LeafKey(m, b, e["agent_impl"], e["category"])
                if key not in seen_leaves:
                    seen_leaves.add(key)
                    expected_keys.append(key)

    outputs: dict[str, object] = {
        "models": model_specs,
        "categories": categories,
        "configs": code_impls,
        "branches": branches,
        "expected_leaves": [key._asdict() for key in expected_keys],
        "max_parallel": str(max_parallel),
        "model_parallel": str(model_parallel),
    }
    # GitHub job outputs are statically declared, so one matrixable output keeps
    # the outer (model, branch) axis scalable without per-model output names.
    eval_include = [
        {
            "model": m,
            "branch": b,
            "branch_sha": branch_shas[b],
            "flat_matrix": json.dumps({"include": per_model_matrices[m]}, separators=(",", ":")),
        }
        for m in model_specs
        for b in branches
    ]
    outputs["eval_matrix"] = {"include": eval_include}
    _emit(os.environ.get("GITHUB_OUTPUT"), outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
