#!/usr/bin/env python3
"""Build the two-version matrix for unified branch comparisons."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TypedDict, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_prep as unified  # noqa: E402


class Version(TypedDict):
    """Immutable product source selected for one side of a comparison."""

    version_id: str
    branch: str
    sha: str


def _versions(raw: str) -> list[Version]:
    """Parse and validate the exactly-two comparison sources."""
    msg = "UNIFIED_VERSIONS_JSON must contain distinct v1/v2 branch+SHA entries"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(msg) from exc
    if not isinstance(value, list) or len(value) != 2:
        raise SystemExit(msg)
    versions: list[Version] = []
    for item in value:
        if not isinstance(item, dict):
            raise SystemExit(msg)
        version_id = item.get("version_id")
        branch = item.get("branch")
        sha = item.get("sha")
        if (
            version_id not in {"v1", "v2"}
            or not isinstance(branch, str)
            or not branch
            or not isinstance(sha, str)
            or len(sha) != 40
            or any(char not in "0123456789abcdef" for char in sha)
        ):
            raise SystemExit(msg)
        versions.append(
            cast(Version, {"version_id": version_id, "branch": branch, "sha": sha})
        )
    if {version["version_id"] for version in versions} != {"v1", "v2"}:
        raise SystemExit(msg)
    if versions[0]["sha"] == versions[1]["sha"]:
        raise SystemExit("Version 1 and Version 2 resolve to the same commit SHA")
    return sorted(versions, key=lambda version: version["version_id"])


def _tasks(categories: list[str], profile: str) -> dict[str, list[str]]:
    """Load the one shared task manifest assigned to both versions."""
    if profile == "lite":
        return {
            category: list(unified.lite_tasks.LITE_TASKS.get(category, []))
            for category in categories
        }
    tasks_path = os.environ.get("UNIFIED_TASKS_JSON", "").strip()
    if not tasks_path:
        raise SystemExit("full profile requires UNIFIED_TASKS_JSON")
    raw = json.loads(Path(tasks_path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit("UNIFIED_TASKS_JSON must contain an object")
    tasks: dict[str, list[str]] = {}
    for category in categories:
        values = raw.get(category)
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise SystemExit(
                f"Task manifest for {category!r} must be a list of strings"
            )
        tasks[category] = values
    return tasks


def _filter_tasks(
    tasks: dict[str, list[str]], categories: list[str], raw: str
) -> dict[str, list[str]]:
    """Restrict every selected category to an exact shared task subset."""
    selected = list(dict.fromkeys(raw.split()))
    if not selected:
        return tasks
    available = {
        task for category in categories for task in tasks.get(category, [])
    }
    unknown = [task for task in selected if task not in available]
    if unknown:
        raise SystemExit(
            f"UNIFIED_INCLUDE_TASKS contains tasks outside the selected profile: {unknown}"
        )
    filtered = {
        category: [task for task in selected if task in tasks.get(category, [])]
        for category in categories
    }
    empty = [category for category, values in filtered.items() if not values]
    if empty:
        raise SystemExit(
            "UNIFIED_INCLUDE_TASKS must select at least one task in every category: "
            f"{empty}"
        )
    return filtered


def build_outputs() -> dict[str, object]:
    """Validate dispatch inputs and return GitHub output values."""
    versions = _versions(os.environ.get("UNIFIED_VERSIONS_JSON", ""))
    categories = list(
        dict.fromkeys(
            category.strip()
            for category in os.environ.get(
                "UNIFIED_CATEGORIES", "autonomous,conversation,context"
            ).split(",")
            if category.strip()
        )
    )
    if not categories:
        raise SystemExit("No categories selected")
    unknown = [
        category for category in categories if category not in unified.CATEGORY_MAP
    ]
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")

    profile = os.environ.get("UNIFIED_PROFILE", "").strip() or "full"
    if profile not in unified.PROFILES:
        raise SystemExit(f"UNIFIED_PROFILE must be one of {sorted(unified.PROFILES)}")
    agent_impl = os.environ.get("UNIFIED_AGENT_IMPL", "").strip() or "dcode"
    if agent_impl not in unified.CODE_AGENT_IMPLS:
        raise SystemExit(
            f"UNIFIED_AGENT_IMPL must be one of {sorted(unified.CODE_AGENT_IMPLS)}"
        )
    concurrency = unified.parse_int_input(
        "UNIFIED_CONCURRENCY",
        os.environ.get("UNIFIED_CONCURRENCY", "4"),
        minimum=1,
        maximum=unified.MAX_TASKS_PER_MODEL,
    )
    rollouts = unified.parse_int_input(
        "UNIFIED_ROLLOUTS", os.environ.get("UNIFIED_ROLLOUTS", "3"), minimum=1
    )
    try:
        models = unified.models._resolve_models(  # noqa: SLF001
            "harbor", os.environ.get("UNIFIED_MODELS", "").strip()
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tasks = _filter_tasks(
        _tasks(categories, profile),
        categories,
        os.environ.get("UNIFIED_INCLUDE_TASKS", ""),
    )
    task_count = max(sum(len(tasks.get(category, [])) for category in categories), 1)
    subject_count = len(versions) * len(models)
    unified.total_job_guard(subject_count, task_count)
    max_parallel, subject_parallel = unified.derive_pool(
        concurrency, rollouts, task_count, subject_count
    )

    flat_by_model = {
        model: json.dumps(
            {
                "include": unified.build_flat_matrix(
                    model, categories, tasks, dcode_impl=agent_impl
                )
            },
            separators=(",", ":"),
        )
        for model in models
    }
    include = [
        {
            "version_id": version["version_id"],
            "source_branch": version["branch"],
            "source_sha": version["sha"],
            "model": model,
            "flat_matrix": flat_by_model[model],
        }
        for version in versions
        for model in models
    ]
    return {
        "eval_matrix": {"include": include},
        "max_parallel": str(max_parallel),
        "subject_parallel": str(subject_parallel),
        "models": models,
        "categories": categories,
        "versions": versions,
    }


def main() -> int:
    """Write comparison matrix values to `GITHUB_OUTPUT`."""
    unified._emit(os.environ.get("GITHUB_OUTPUT"), build_outputs())  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
