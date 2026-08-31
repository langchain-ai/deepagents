"""Enumerate a category's task names for the flat shard matrix.

Reuses the exact task-name resolution the run step uses:
  - local (--path) datasets: subdir basenames with a task.toml (sorted, stable),
  - synthetic datasets (e.g. "tau3-subset"): a committed task list, since the
    name is not a registry package and would fail a registry lookup,
  - registry datasets: PackageDatasetClient manifest task_ids -> get_name().

Registry lookups need harbor installed and network; the local path is pure
stdlib. Run as a script, main() emits newline-joined names.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shard_matrix  # noqa: E402


def local_task_names(local_dir: str) -> list[str]:
    names = sorted(
        entry.name
        for entry in os.scandir(local_dir)
        if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "task.toml"))
    )
    if not names:
        raise SystemExit(f"No local Harbor tasks (dirs with task.toml) under {local_dir}")
    return names


def registry_task_names(dataset_ref: str) -> list[str]:
    import asyncio

    from harbor.registry.client.package import PackageDatasetClient

    md = asyncio.run(
        PackageDatasetClient().get_dataset_metadata(f"{dataset_ref}@latest")
    )
    names = [x for x in (shard_matrix.task_display_name(t) for t in md.task_ids) if x]
    if not names:
        raise SystemExit(
            f"Resolved 0 usable task names from {dataset_ref}@latest "
            f"({len(md.task_ids)} task ids); unexpected task-id shape."
        )
    return names


def tau3_subset_task_names() -> list[str]:
    # "tau3-subset" is a synthetic curated view, not a registry package, so it
    # cannot be resolved via the registry client. Pull its committed 30-task
    # list from the same module the harbor leaf uses (single source of truth).
    from deepagents_evals.tau3_subset import INCLUDE_TASKS

    names = INCLUDE_TASKS.split()
    if not names:
        raise SystemExit(
            "tau3-subset resolved to 0 tasks; "
            "deepagents_evals.tau3_subset.INCLUDE_TASKS is empty."
        )
    return names


# Synthetic datasets resolve to a committed task list instead of a registry
# lookup, because their names are not `org/name` registry packages.
SYNTHETIC_DATASETS = {"tau3-subset": tau3_subset_task_names}


def main() -> int:
    dataset_path = os.environ.get("ENUM_DATASET_PATH", "").strip()
    dataset = os.environ.get("ENUM_DATASET", "").strip()
    if dataset_path:
        names = local_task_names(dataset_path)
    elif dataset in SYNTHETIC_DATASETS:
        names = SYNTHETIC_DATASETS[dataset]()
    else:
        names = registry_task_names(dataset)
    out = os.environ.get("ENUM_OUTPUT")
    text = "\n".join(names) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
