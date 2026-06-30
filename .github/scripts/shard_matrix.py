"""Shard math for the Harbor evals workflow (`.github/workflows/harbor.yml`).

Two concerns, two consumers, one tested source of truth:

* ``expand_matrix`` — used by the ``prep`` job to cross-product the model matrix
  with the shard axis. Guards the result against GitHub Actions' hard 256-job
  matrix cap (a sharded ``all`` run is ``len(models) * n_shards``).
* ``select_shard_tasks`` — used by the per-shard ``harbor`` run step to pick this
  shard's disjoint slice of the dataset. It mirrors Harbor's own task-selection
  pipeline (``_filter_task_ids`` in ``harbor/models/job/config.py``) so that a
  sharded run executes exactly the tasks an unsharded ``include_tasks``/``n_tasks``
  run would — just split across runners.

Run as a script, ``main()`` drives ``expand_matrix`` from env vars and writes the
matrix to ``$GITHUB_OUTPUT`` (mirroring ``.github/scripts/models.py``).
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

    An explicit exception (never ``assert``) so the check survives ``python -O``
    and ``main()`` can render it as a GitHub ``::error::`` annotation.
    """


def expand_matrix(model_matrix: dict, n_shards: int) -> dict:
    """Cross-product each model entry in ``model_matrix`` with the shard axis.

    ``model_matrix`` is the ``{"include": [...]}`` payload emitted by
    ``models.py harbor``. Each entry gains a ``shard`` key in ``0..n_shards-1``.
    ``n_shards == 1`` is a no-op cross-product (``shard: 0`` on every entry),
    identical to the pre-sharding matrix.

    Raises:
        ShardConfigError: if ``n_shards`` is out of range, or the expanded
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
        {**entry, "shard": shard}
        for entry in include
        for shard in range(n_shards)
    ]
    return {"include": expanded}


def select_shard_tasks(
    names: list[str],
    include_globs: list[str],
    n_tasks: int,
    n_shards: int,
    shard_index: int,
) -> list[str]:
    """Return this shard's slice of the dataset's task names.

    ``names`` MUST be in the dataset's native manifest order (the order of
    ``get_dataset_metadata().task_ids``) — the same order Harbor filters at run
    time. This mirrors Harbor's ``_filter_task_ids``:

    1. keep names matching any ``include_globs`` (``fnmatch``, order preserved);
       empty ``include_globs`` keeps everything,
    2. if ``n_tasks > 0``, take the first ``n_tasks`` (a **total** cap, applied
       before sharding — NOT per shard),
    3. partition with ``j % n_shards == shard_index``.

    Because the cap is applied to the native-order list before partitioning, the
    union of every shard's result equals exactly the task set an unsharded
    ``include_tasks``/``n_tasks`` run would execute. Do not sort ``names``:
    Harbor's ``--n-tasks`` slices in native order, so sorting would select a
    different N.

    Raises:
        ShardConfigError: if ``n_shards``/``shard_index`` are out of range.
    """
    if not isinstance(n_shards, int) or n_shards < 1:
        msg = f"Invalid n_shards (must be >= 1): {n_shards!r}"
        raise ShardConfigError(msg)
    if not isinstance(shard_index, int) or not (0 <= shard_index < n_shards):
        msg = f"Invalid shard_index {shard_index!r} for {n_shards} shards"
        raise ShardConfigError(msg)

    selected = [n for n in names if n]
    if include_globs:
        selected = [
            n for n in selected if any(fnmatch(n, g) for g in include_globs)
        ]
    if n_tasks > 0:
        selected = selected[:n_tasks]
    return [name for j, name in enumerate(selected) if j % n_shards == shard_index]


def main() -> None:
    """Entry point for the prep job: expand the model matrix by shard.

    Reads ``MODEL_MATRIX`` (JSON from ``models.py harbor``) and ``N_SHARDS``,
    writes ``matrix=<json>`` to ``$GITHUB_OUTPUT`` (or stdout when unset).
    Config errors become a GitHub ``::error::`` annotation + exit 1.
    """
    raw_shards = os.environ.get("N_SHARDS", "1").strip() or "1"
    if not raw_shards.isdigit():
        print(f"::error::Invalid n_shards (must be an integer): {raw_shards!r}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    try:
        model_matrix = json.loads(os.environ["MODEL_MATRIX"])
        matrix = expand_matrix(model_matrix, int(raw_shards))
    except ShardConfigError as exc:
        print(f"::error::{exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    payload = "matrix=" + json.dumps(matrix, separators=(",", ":"))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:  # noqa: PTH123
            f.write(payload + "\n")
    else:
        print(payload)  # noqa: T201


if __name__ == "__main__":
    main()
