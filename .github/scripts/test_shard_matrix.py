"""Tests for the Harbor shard-matrix helper (`.github/scripts/shard_matrix.py`).

Mirrors `test_models.py`: import-by-path, stdlib + pytest only, so it runs under
CI's `pytest .github/scripts/test_*.py` (see `.github/workflows/ci.yml`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARD_SCRIPT = REPO_ROOT / ".github" / "scripts" / "shard_matrix.py"


def _load_shard_script() -> ModuleType:
    """Load `.github/scripts/shard_matrix.py` as a module.

    The script lives outside any importable package, so import-by-path is the
    only way to exercise its internals from a test.
    """
    spec = importlib.util.spec_from_file_location("gha_shard_matrix", SHARD_SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {SHARD_SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shard() -> ModuleType:
    """Module-scoped handle to the loaded `shard_matrix.py` script."""
    return _load_shard_script()


def _models(n: int) -> dict:
    """A model matrix with `n` entries, shaped like `models.py harbor` output."""
    return {"include": [{"model": f"p:m{i}", "artifact_key": f"p-m{i}"} for i in range(n)]}


# --------------------------------------------------------------------------- #
# expand_matrix — model x shard cross-product + GitHub 256-job cap (comment #1)
# --------------------------------------------------------------------------- #


def test_expand_matrix_single_shard_is_backward_compatible(shard: ModuleType) -> None:
    """n_shards=1 -> one job per model, each tagged shard 0 (no behavior change)."""
    out = shard.expand_matrix(_models(3), 1)

    assert len(out["include"]) == 3
    assert all(entry["shard"] == 0 for entry in out["include"])
    # Original keys are preserved verbatim alongside the new `shard` key.
    assert out["include"][0] == {"model": "p:m0", "artifact_key": "p-m0", "shard": 0}


def test_expand_matrix_cross_products_models_and_shards(shard: ModuleType) -> None:
    """n_shards=3 -> len(models) * 3 entries with shard indices 0..2 per model."""
    out = shard.expand_matrix(_models(2), 3)

    assert len(out["include"]) == 2 * 3
    # Each model appears once per shard index.
    for model in ("p:m0", "p:m1"):
        shards = sorted(e["shard"] for e in out["include"] if e["model"] == model)
        assert shards == [0, 1, 2]


def test_expand_matrix_accepts_matrix_at_the_256_cap(shard: ModuleType) -> None:
    """len(models) * n_shards == 256 is allowed (boundary)."""
    out = shard.expand_matrix(_models(128), 2)  # 128 * 2 == 256
    assert len(out["include"]) == 256


def test_expand_matrix_rejects_matrix_over_the_256_cap(shard: ModuleType) -> None:
    """257 jobs is one over the cap and must be rejected before any run."""
    with pytest.raises(shard.ShardConfigError, match="256-job"):
        shard.expand_matrix(_models(129), 2)  # 129 * 2 == 258


def test_expand_matrix_rejects_real_all_set_overflow(shard: ModuleType) -> None:
    """The reviewer's concrete case: 56-model `all` set x 5 shards = 280 > 256."""
    with pytest.raises(shard.ShardConfigError, match="280 jobs"):
        shard.expand_matrix(_models(56), 5)


@pytest.mark.parametrize("bad", [0, 65, -1])
def test_expand_matrix_rejects_out_of_range_shards(shard: ModuleType, bad: int) -> None:
    """n_shards must be an integer in 1..64."""
    with pytest.raises(shard.ShardConfigError, match="Invalid n_shards"):
        shard.expand_matrix(_models(2), bad)


# --------------------------------------------------------------------------- #
# effective_shards — cap the shard axis so empty jobs aren't spawned
# --------------------------------------------------------------------------- #


def test_effective_shards_no_cap_when_running_all_tasks(shard: ModuleType) -> None:
    """n_tasks=0 means all tasks: the shard count is used as-is."""
    assert shard.effective_shards(4, 0) == 4


def test_effective_shards_no_cap_when_tasks_exceed_shards(shard: ModuleType) -> None:
    """n_tasks >= n_shards: every shard gets work, so no reduction."""
    assert shard.effective_shards(4, 10) == 4
    assert shard.effective_shards(4, 4) == 4


def test_effective_shards_caps_to_task_count(shard: ModuleType) -> None:
    """n_tasks < n_shards: cap to n_tasks so trailing shards aren't spawned.

    The cited case (n_tasks=1 n_shards=4) collapses to a single shard job.
    """
    assert shard.effective_shards(4, 1) == 1
    assert shard.effective_shards(4, 3) == 3


def test_effective_shards_then_expand_emits_only_useful_jobs(shard: ModuleType) -> None:
    """The capped count feeds expand_matrix: 2 models x 1 effective shard = 2 jobs."""
    eff = shard.effective_shards(4, 1)
    out = shard.expand_matrix(_models(2), eff)
    assert len(out["include"]) == 2
    assert all(entry["shard"] == 0 for entry in out["include"])


# --------------------------------------------------------------------------- #
# select_shard_tasks — filter + cap + partition (comment #2)
# --------------------------------------------------------------------------- #


def test_partition_is_disjoint_and_covers_the_whole_selection(shard: ModuleType) -> None:
    """Across all shards, every task runs exactly once (no gaps, no overlap)."""
    names = [f"org/t{i}" for i in range(10)]
    n_shards = 3

    slices = [
        shard.select_shard_tasks(names, [], 0, n_shards, i) for i in range(n_shards)
    ]

    flat = [name for s in slices for name in s]
    assert sorted(flat) == sorted(names)  # covers everything
    assert len(flat) == len(set(flat))  # no task in two shards


def test_include_globs_filter_before_partitioning(shard: ModuleType) -> None:
    """Only tasks matching an include glob are sharded; others are dropped."""
    names = ["org/foo-1", "org/bar-1", "org/foo-2", "org/baz"]
    n_shards = 2

    kept = [
        name
        for i in range(n_shards)
        for name in shard.select_shard_tasks(names, ["org/foo-*"], 0, n_shards, i)
    ]

    assert sorted(kept) == ["org/foo-1", "org/foo-2"]


def test_n_tasks_caps_total_across_shards_not_per_shard(shard: ModuleType) -> None:
    """n_tasks is a global cap: the shards together run exactly n_tasks tasks.

    This is the comment-#2 regression: an unsharded `--n-tasks 10` must not
    become 10-per-shard once sharded.
    """
    names = [f"org/t{i}" for i in range(20)]
    n_tasks, n_shards = 10, 4

    total = sum(
        len(shard.select_shard_tasks(names, [], n_tasks, n_shards, i))
        for i in range(n_shards)
    )

    assert total == n_tasks  # not n_tasks * n_shards (== 40)


def test_n_tasks_takes_native_order_not_sorted(shard: ModuleType) -> None:
    """The cap slices the list as given (native manifest order), never sorted.

    Harbor's `--n-tasks` is `filtered_ids[:n_tasks]` in native order, so a
    sharded run must pick the same first-N as an unsharded run would.
    """
    # Deliberately not alphabetical; sorted() would pick a different first 2.
    names = ["org/zebra", "org/apple", "org/mango", "org/cherry"]
    n_tasks, n_shards = 2, 2

    selected = [
        name
        for i in range(n_shards)
        for name in shard.select_shard_tasks(names, [], n_tasks, n_shards, i)
    ]

    # First two in NATIVE order, not sorted(['apple', 'cherry', ...]).
    assert sorted(selected) == ["org/apple", "org/zebra"]


def test_include_and_n_tasks_compose(shard: ModuleType) -> None:
    """include_globs applies first, then the n_tasks cap, then partitioning."""
    names = [f"org/keep-{i}" for i in range(6)] + [f"org/drop-{i}" for i in range(6)]
    n_tasks, n_shards = 3, 2

    selected = [
        name
        for i in range(n_shards)
        for name in shard.select_shard_tasks(names, ["org/keep-*"], n_tasks, n_shards, i)
    ]

    assert len(selected) == 3
    assert all(name.startswith("org/keep-") for name in selected)


def test_n_shards_one_returns_full_selection(shard: ModuleType) -> None:
    """n_shards=1 is the whole (filtered/capped) list — the unsharded path."""
    names = [f"org/t{i}" for i in range(5)]
    assert shard.select_shard_tasks(names, [], 0, 1, 0) == names


def test_fewer_tasks_than_shards_yields_empty_trailing_shards(shard: ModuleType) -> None:
    """n_tasks < n_shards: early shards get the work, later shards are empty.

    The workflow relies on this to no-op (not fail) empty shard jobs. e.g.
    n_tasks=1 n_shards=4 -> shard 0 runs the task, shards 1-3 are empty.
    """
    names = [f"org/t{i}" for i in range(10)]
    n_tasks, n_shards = 1, 4

    slices = [
        shard.select_shard_tasks(names, [], n_tasks, n_shards, i) for i in range(n_shards)
    ]

    assert slices[0] == ["org/t0"]
    assert slices[1] == slices[2] == slices[3] == []
    # Union is still exactly the selected task set (no task lost, none duplicated).
    assert [name for s in slices for name in s] == ["org/t0"]


def test_select_shard_tasks_rejects_bad_shard_index(shard: ModuleType) -> None:
    """shard_index must be in range(n_shards)."""
    with pytest.raises(shard.ShardConfigError, match="shard_index"):
        shard.select_shard_tasks(["org/t0"], [], 0, 2, 2)
