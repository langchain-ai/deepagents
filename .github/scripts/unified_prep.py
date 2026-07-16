"""Prepare matrices for ordinary and multi-branch Unified Harbor evaluations.

Every inner-matrix entry contains exactly one task. GitHub accepts at most 256
entries in a matrix, so oversized selections fail explicitly instead of packing
multiple tasks into one shard. A resolved source list expands the existing model
axis to immutable source-version x model evaluation groups.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import TypedDict, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_agent_configs as agent_configs  # noqa: E402
import lite_tasks  # noqa: E402
import models  # noqa: E402
import shard_matrix  # noqa: E402

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

# Compatibility aliases used by existing tests and scripts.
CODE_AGENT_IMPLS = set(agent_configs.CODE_CONFIGS)
DEFAULT_AGENT_IMPL = agent_configs.DEFAULT_CODE_CONFIG
KNOWN_AGENT_IMPLS = set(agent_configs.RUNTIME_CONFIGS)
PROFILES = {"full", "lite"}

if not all(cm["agent_impl"] in KNOWN_AGENT_IMPLS for cm in CATEGORY_MAP.values()):
    raise RuntimeError(
        f"every CATEGORY_MAP agent_impl must be one of {sorted(KNOWN_AGENT_IMPLS)}"
    )


class Source(TypedDict):
    """Immutable product source used by one evaluation version."""

    version_id: str
    branch: str
    sha: str
    product_artifact: str


def parse_int_input(
    name: str,
    raw: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Parse an integer constrained to an inclusive range."""
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
    """Return a known provider prefix or the fallback provider bucket."""
    prefix = spec.split(":", 1)[0]
    return prefix if prefix in known else "other"


def derive_pool(
    concurrency: int, rollouts: int, n_shards: int, n_models: int
) -> tuple[int, int]:
    """Derive inner shard and outer source/model parallelism.

    One shard carries one task, so its peak concurrent trials are bounded by
    `min(concurrency, rollouts)`. The existing 40-trial group cap determines the
    inner pool. The 80-runner workflow cap then determines how many source/model
    groups may run their inner pools simultaneously.
    """
    per_shard = max(1, min(concurrency, rollouts))
    max_parallel = max(1, min(MAX_TASKS_PER_MODEL // per_shard, n_shards))
    group_parallel = max(1, min(MAX_RUNNERS // max_parallel, n_models))
    return max_parallel, group_parallel


def build_flat_matrix(
    model: str,
    categories: list[str],
    tasks_by_cat: dict[str, list[str]],
    code_impls: list[str] | None = None,
) -> list[dict]:
    """Build one single-task matrix spanning categories and selected configs."""
    configs = list(dict.fromkeys(code_impls or [DEFAULT_AGENT_IMPL]))
    provider = provider_of(model)
    groups: list[tuple[str, str, list[str]]] = []
    for category in categories:
        category_config = CATEGORY_MAP[category]
        tasks = tasks_by_cat.get(category, [])
        if not tasks:
            continue
        if category_config["agent_impl"] in CODE_AGENT_IMPLS:
            for config in configs:
                groups.append(
                    (
                        category,
                        agent_configs.runtime_for_code_config(config),
                        tasks,
                    )
                )
        else:
            conversation_runtimes = list(
                dict.fromkeys(
                    agent_configs.conversation_runtime_for(config) for config in configs
                )
            )
            groups.extend(
                (category, runtime, tasks) for runtime in conversation_runtimes
            )

    total = sum(len(tasks) for _category, _runtime, tasks in groups)
    if total > shard_matrix.MAX_SHARDS:
        raise SystemExit(
            f"Model {model!r} requires {total} matrix entries, over GitHub's "
            f"{shard_matrix.MAX_SHARDS}-entry matrix limit. Reduce tasks, "
            "categories, or agent configs. Unified Evals does not pack tasks."
        )

    entries: list[dict] = []
    for category, runtime, tasks in groups:
        category_config = CATEGORY_MAP[category]
        for task in tasks:
            entries.append(
                {
                    "model": model,
                    "provider": provider,
                    "category": category,
                    "dataset": category_config["dataset"],
                    "dataset_path": category_config["dataset_path"],
                    "agent_impl": runtime,
                    "include_tasks": task,
                    "langsmith_dataset": "",
                    "n_shards": 1,
                    "shard": 0,
                }
            )
    return entries


def parse_sources(raw: str) -> list[Source]:
    """Parse ordered branch/SHA records resolved by the workflow."""
    if not raw.strip():
        sha = os.environ.get("GITHUB_SHA", "0" * 40)
        branch = os.environ.get("GITHUB_REF_NAME", "controller")
        return [
            {
                "version_id": "",
                "branch": branch,
                "sha": sha,
                "product_artifact": f"unified-products-current-{sha[:12]}",
            }
        ]

    msg = "UNIFIED_SOURCES_JSON must contain ordered branch/SHA source objects"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(msg) from exc
    if not isinstance(value, list) or not value:
        raise SystemExit(msg)
    comparison = len(value) > 1
    sources: list[Source] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise SystemExit(msg)
        expected_id = f"v{index}" if comparison else ""
        version_id = item.get("version_id")
        branch = item.get("branch")
        sha = item.get("sha")
        if (
            version_id != expected_id
            or not isinstance(branch, str)
            or not branch
            or not isinstance(sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", sha) is None
        ):
            raise SystemExit(msg)
        artifact_version = version_id or "current"
        sources.append(
            cast(
                Source,
                {
                    "version_id": version_id,
                    "branch": branch,
                    "sha": sha,
                    "product_artifact": (
                        f"unified-products-{artifact_version}-{sha[:12]}"
                    ),
                },
            )
        )
    if len({source["sha"] for source in sources}) != len(sources):
        raise SystemExit("comparison branches must resolve to distinct commit SHAs")
    return sources


def filter_tasks(
    tasks: dict[str, list[str]], categories: list[str], raw: str
) -> dict[str, list[str]]:
    """Apply one exact task selection to every evaluated source."""
    selected = list(dict.fromkeys(raw.split()))
    if not selected:
        return tasks
    available = {task for category in categories for task in tasks.get(category, [])}
    unknown = [task for task in selected if task not in available]
    if unknown:
        raise SystemExit(f"UNIFIED_INCLUDE_TASKS contains unknown tasks: {unknown}")
    filtered = {
        category: [task for task in selected if task in tasks.get(category, [])]
        for category in categories
    }
    empty = [category for category, names in filtered.items() if not names]
    if empty:
        raise SystemExit(
            f"UNIFIED_INCLUDE_TASKS must select a task in every category: {empty}"
        )
    return filtered


def _emit(github_output: str | None, outputs: dict[str, object]) -> None:
    """Write compact outputs to GitHub or stdout."""
    if not github_output:
        for key, value in outputs.items():
            payload = (
                value
                if isinstance(value, str)
                else json.dumps(value, separators=(",", ":"))
            )
            print(f"{key}={payload}")
        return
    with open(github_output, "a") as handle:
        for key, value in outputs.items():
            payload = (
                value
                if isinstance(value, str)
                else json.dumps(value, separators=(",", ":"))
            )
            handle.write(f"{key}={payload}\n")


def main(argv: list[str] | None = None) -> int:
    """Validate inputs and emit build/evaluation matrices."""
    del argv
    selection = os.environ.get("UNIFIED_MODELS", "").strip()
    categories = list(
        dict.fromkeys(
            category.strip()
            for category in os.environ.get(
                "UNIFIED_CATEGORIES", "autonomous,conversation,context"
            ).split(",")
            if category.strip()
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
    try:
        code_configs = agent_configs.parse_code_configs(
            os.environ.get("UNIFIED_AGENT_IMPLS", "")
        )
    except ValueError as exc:
        raise SystemExit(
            f"UNIFIED_AGENT_IMPLS entries must be in {sorted(CODE_AGENT_IMPLS)}: {exc}"
        ) from exc

    profile = os.environ.get("UNIFIED_PROFILE", "").strip() or "full"
    if profile not in PROFILES:
        raise SystemExit(
            f"UNIFIED_PROFILE must be one of {sorted(PROFILES)}, got {profile!r}"
        )
    if not categories:
        raise SystemExit(f"No categories selected. Choose from {sorted(CATEGORY_MAP)}.")
    unknown_categories = [
        category for category in categories if category not in CATEGORY_MAP
    ]
    if unknown_categories:
        raise SystemExit(
            f"Unknown categor(y/ies): {unknown_categories}. "
            f"Valid: {sorted(CATEGORY_MAP)}"
        )
    try:
        model_specs = models._resolve_models("harbor", selection)  # noqa: SLF001
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if profile == "lite":
        tasks_by_category = {
            category: list(lite_tasks.LITE_TASKS.get(category, []))
            for category in categories
        }
    else:
        tasks_path = os.environ.get("UNIFIED_TASKS_JSON", "").strip()
        if not tasks_path:
            raise SystemExit(
                "full profile requires UNIFIED_TASKS_JSON (enumerated tasks)."
            )
        with open(tasks_path) as handle:
            tasks_by_category = json.load(handle)
    tasks_by_category = filter_tasks(
        tasks_by_category,
        categories,
        os.environ.get("UNIFIED_INCLUDE_TASKS", ""),
    )
    sources = parse_sources(os.environ.get("UNIFIED_SOURCES_JSON", ""))

    per_model_matrices = {
        model: build_flat_matrix(
            model, categories, tasks_by_category, code_impls=code_configs
        )
        for model in model_specs
    }
    eval_group_count = len(sources) * len(model_specs)
    if eval_group_count > shard_matrix.MAX_SHARDS:
        raise SystemExit(
            f"branch x model matrix has {eval_group_count} entries, over GitHub's "
            f"{shard_matrix.MAX_SHARDS}-entry matrix limit"
        )
    largest_matrix = max(
        (len(entries) for entries in per_model_matrices.values()), default=1
    )
    max_parallel, version_model_parallel = derive_pool(
        concurrency, rollouts, largest_matrix, eval_group_count
    )

    expected_leaves: list[dict] = []
    seen_leaves: set[tuple[str, str, str]] = set()
    runtimes: list[str] = []
    for model, entries in per_model_matrices.items():
        for entry in entries:
            runtime = entry["agent_impl"]
            if runtime not in runtimes:
                runtimes.append(runtime)
            identity = (model, runtime, entry["category"])
            if identity not in seen_leaves:
                seen_leaves.add(identity)
                expected_leaves.append(
                    {"model": model, "config": runtime, "category": entry["category"]}
                )

    packages = agent_configs.required_packages(runtimes)
    eval_matrix = {
        "include": [
            {
                **source,
                "model": model,
                "slug": _slug(model),
                "flat_matrix": json.dumps(
                    {"include": per_model_matrices[model]}, separators=(",", ":")
                ),
            }
            for source in sources
            for model in model_specs
        ]
    }
    outputs: dict[str, object] = {
        "models": model_specs,
        "categories": categories,
        "configs": code_configs,
        "sources": sources,
        "comparison_mode": "true" if len(sources) > 1 else "false",
        "conversation_sources": {
            config: agent_configs.conversation_runtime_for(config)
            for config in code_configs
        },
        "expected_leaves": expected_leaves,
        "max_parallel": str(max_parallel),
        # Keep the old output name while callers transition to the clearer name.
        "model_parallel": str(version_model_parallel),
        "version_model_parallel": str(version_model_parallel),
        "total_jobs": str(
            len(sources) * sum(len(entries) for entries in per_model_matrices.values())
        ),
        "build_matrix": {
            "include": [{**source, "packages": packages} for source in sources]
        },
        "eval_matrix": eval_matrix,
    }
    _emit(os.environ.get("GITHUB_OUTPUT"), outputs)
    return 0


def _slug(model: str) -> str:
    """Return a GitHub-safe model identifier."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model.replace("/", "-").replace(":", "-"))


if __name__ == "__main__":
    raise SystemExit(main())
