"""Shard math for the Harbor evals workflow (`.github/workflows/harbor.yml`).

Two concerns, two consumers, one tested source of truth:

* `expand_matrix` — used by the `prep` job to cross-product the model matrix
    with the shard axis. Guards the result against GitHub Actions' hard 256-job
    matrix cap (a sharded `all` run is `len(models) * n_shards`).
* `select_shard_tasks` — used by the per-shard `harbor` run step to pick this
    shard's disjoint slice of the dataset. It reproduces the *include-glob +
    `n_tasks`* subset of Harbor's task-selection pipeline (`_filter_task_ids` in
    `harbor/models/job/config.py`, as of the pinned Harbor) so that a sharded run
    executes exactly the tasks an unsharded `include_tasks`/`n_tasks` run would
  — just split across runners. It intentionally does **not** model Harbor's
    `exclude_task_names` stage or its empty-include `ValueError` (the workflow
    exposes no `exclude_tasks` input and fail-fasts on an empty include match
    itself); see `select_shard_tasks` for the full divergence note.
* `task_display_name` — used by the run step to turn a Harbor manifest task id
    into the `org/name` string the shard filter and `--include-task-name` expect,
    delegating to the task id's own `get_name()` so every id variant resolves.

Run as a script, `main()` drives `expand_matrix` from env vars and writes the
matrix to `$GITHUB_OUTPUT` (mirroring `.github/scripts/models.py`).
"""

from __future__ import annotations

import json
import os
import sys
from fnmatch import fnmatch

# GitHub Actions refuses to start a job matrix with more than this many entries.
# https://docs.github.com/actions/using-jobs/using-a-matrix-for-your-jobs
GITHUB_MATRIX_MAX = 256

# Upper bound on the shard axis itself, independent of the matrix cap. Keeps a
# fat-fingered dispatch (e.g. n_shards=1000) from producing a nonsensical run.
MAX_SHARDS = 64


class ShardConfigError(Exception):
    """Raised for an invalid shard configuration.

    An explicit exception (never `assert`) so the check survives `python -O`
    and `main()` can render it as a GitHub `::error::` annotation.
    """


def expand_matrix(model_matrix: dict, n_shards: int) -> dict:
    """Cross-product each model entry in `model_matrix` with the shard axis.

    `model_matrix` is the `{"include": [...]}` payload emitted by
    `models.py harbor`. Each entry gains a `shard` key in `0..n_shards-1`.
    `n_shards == 1` is a no-op cross-product (`shard: 0` on every entry),
    identical to the pre-sharding matrix.

    Raises:
        ShardConfigError: if `n_shards` is out of range, or the expanded
            matrix would exceed GitHub's job cap.
    """
    if not isinstance(n_shards, int) or not (1 <= n_shards <= MAX_SHARDS):
        msg = f"Invalid n_shards (must be an integer 1..{MAX_SHARDS}): {n_shards!r}"
        raise ShardConfigError(msg)

    include = model_matrix.get("include", [])
    total = len(include) * n_shards
    if total > GITHUB_MATRIX_MAX:
        msg = (
            f"Sharded matrix is {len(include)} models x {n_shards} shards = "
            f"{total} jobs, over GitHub's {GITHUB_MATRIX_MAX}-job matrix limit. "
            "Reduce n_shards or select a smaller model set."
        )
        raise ShardConfigError(msg)

    expanded = [
        {**entry, "shard": shard} for entry in include for shard in range(n_shards)
    ]
    return {"include": expanded}


def effective_shards(n_shards: int, n_tasks: int) -> int:
    """Cap the shard axis to the amount of selectable work.

    Sharding more ways than there are tasks just spawns empty no-op jobs. When
    `n_tasks > 0` the selection is at most `n_tasks` tasks, so the useful
    shard count is `min(n_shards, n_tasks)` — this keeps prep from emitting
    shard jobs that can only be empty (e.g. `n_tasks=1 n_shards=4` -> 1 shard).

    `n_tasks == 0` means "all tasks", so no cap is applied. Smaller selections
    produced by `include_tasks` globs can't be sized without the dataset
    manifest (which prep doesn't resolve); those rarer residual empty shards are
    handled as a successful no-op in the run job instead.
    """
    if n_tasks > 0:
        return min(n_shards, n_tasks)
    return n_shards


def task_display_name(task: object) -> str | None:
    """Return a Harbor manifest task's `org/name` display string, or `None`.

    The dataset manifest yields task ids of several shapes (`PackageTaskId`,
    `GitTaskId`, `LocalTaskId`), each of which implements `get_name()` — the
    canonical name Harbor itself filters and reports on. Delegating to it is what
    makes every variant resolve: a manual `f"{org}/{name}"` reconstruction only
    works for `PackageTaskId` and silently returns `None` for the git/local
    ids, which would drop every task and run an empty shard.

    Falls back to dict-shaped (`{"org", "name"}`) and bare `org`/`name`
    attribute access for manifests that don't expose `get_name()`. Returns
    `None` only when no name can be derived, so callers can filter unusable
    entries (and fail loudly if *every* entry is unusable).
    """
    getter = getattr(task, "get_name", None)
    if callable(getter):
        name = getter()
        return name or None
    if isinstance(task, dict):
        org = task.get("org")
        name = task.get("name")
    else:
        org = getattr(task, "org", None)
        name = getattr(task, "name", None)
    if name and org:
        return f"{org}/{name}"
    return name or None


def select_shard_tasks(
    names: list[str],
    include_globs: list[str],
    n_tasks: int,
    n_shards: int,
    shard_index: int,
) -> list[str]:
    """Return this shard's slice of the dataset's task names.

    `names` MUST be in the dataset's native manifest order (the order of
    `get_dataset_metadata().task_ids`) — the same order Harbor filters at run
    time. This reproduces the *include-glob + `n_tasks`* subset of Harbor's
    `_filter_task_ids` (`harbor/models/job/config.py`, as of the pinned
    Harbor):

    1. keep names matching any `include_globs` (`fnmatch`, order preserved);
        empty `include_globs` keeps everything,
    2. if `n_tasks > 0`, take the first `n_tasks` (a **total** cap, applied
        before sharding — NOT per shard),
    3. partition with `j % n_shards == shard_index`.

    Because the cap is applied to the native-order list before partitioning, the
    union of every shard's result equals exactly the task set an unsharded
    `include_tasks`/`n_tasks` run would execute. Do not sort `names`:
    Harbor's `--n-tasks` slices in native order, so sorting would select a
    different N.

    Two stages of `_filter_task_ids` are intentionally **not** reproduced, so
    keep this in sync if a maintainer wires the corresponding inputs into the
    workflow:

    * `exclude_task_names` — Harbor applies it between the include filter and
        the `n_tasks` cap. The workflow exposes no `exclude_tasks` input, so it
        is omitted. (Parity here is by construction, not test-enforced; a Harbor
        refactor that renames `_filter_task_ids` won't fail any test here.)
    * the empty-include `ValueError` — Harbor raises when an include glob
        matches nothing; this returns an empty selection instead. The run step
        fail-fasts on an empty include match before calling this, so the divergence
        is unreachable in the workflow.

    Raises:
        ShardConfigError: if `n_shards`/`shard_index` are out of range.
    """
    if not isinstance(n_shards, int) or n_shards < 1:
        msg = f"Invalid n_shards (must be >= 1): {n_shards!r}"
        raise ShardConfigError(msg)
    if not isinstance(shard_index, int) or not (0 <= shard_index < n_shards):
        msg = f"Invalid shard_index {shard_index!r} for {n_shards} shards"
        raise ShardConfigError(msg)

    selected = [n for n in names if n]
    if include_globs:
        selected = [n for n in selected if any(fnmatch(n, g) for g in include_globs)]
    if n_tasks > 0:
        selected = selected[:n_tasks]
    return [name for j, name in enumerate(selected) if j % n_shards == shard_index]


def main() -> None:
    """Entry point for the prep job: expand the model matrix by shard.

    Reads `MODEL_MATRIX` (JSON from `models.py harbor`), `N_SHARDS` and
    `N_TASKS`, caps the shard axis to the selectable work, and writes both
    `matrix=<json>` and the effective `n_shards=<int>` to `$GITHUB_OUTPUT`
    (or stdout when unset). The harbor job reads back the effective `n_shards`
    so its per-shard partition matches the matrix. Config errors become a GitHub
    `::error::` annotation + exit 1.
    """
    raw_shards = os.environ.get("N_SHARDS", "1").strip() or "1"
    if not raw_shards.isdigit():
        print(
            f"::error::Invalid n_shards (must be an integer): {raw_shards!r}",
            file=sys.stderr,
        )  # noqa: T201
        sys.exit(1)
    raw_tasks = os.environ.get("N_TASKS", "0").strip() or "0"
    if not raw_tasks.isdigit():
        print(
            f"::error::Invalid n_tasks (must be an integer): {raw_tasks!r}",
            file=sys.stderr,
        )  # noqa: T201
        sys.exit(1)

    try:
        model_matrix = json.loads(os.environ["MODEL_MATRIX"])
        n_shards = effective_shards(int(raw_shards), int(raw_tasks))
        matrix = expand_matrix(model_matrix, n_shards)
    except ShardConfigError as exc:
        print(f"::error::{exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    lines = [
        "matrix=" + json.dumps(matrix, separators=(",", ":")),
        f"n_shards={n_shards}",
    ]
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:  # noqa: PTH123
            f.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))  # noqa: T201


if __name__ == "__main__":
    main()
