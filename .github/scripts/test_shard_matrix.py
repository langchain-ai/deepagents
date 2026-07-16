"""Tests for the Harbor shard-matrix helper (`.github/scripts/shard_matrix.py`).

Mirrors `test_models.py`: import-by-path, stdlib + pytest only, so it runs under
CI's `pytest .github/scripts/test_*.py` (see `.github/workflows/ci.yml`).
"""

from __future__ import annotations

import importlib.util
import json
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
    return {
        "include": [{"model": f"p:m{i}", "artifact_key": f"p-m{i}"} for i in range(n)]
    }


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


def test_expand_matrix_accepts_max_shards_boundary(shard: ModuleType) -> None:
    """n_shards == MAX_SHARDS is accepted (boundary)."""
    m = {"include": [{"model": "x"}]}
    out = shard.expand_matrix(m, shard.MAX_SHARDS)
    assert len(out["include"]) == shard.MAX_SHARDS


def test_expand_matrix_rejects_over_max_shards(shard: ModuleType) -> None:
    """n_shards == MAX_SHARDS + 1 is rejected (out of range)."""
    m = {"include": [{"model": "x"}]}
    with pytest.raises(shard.ShardConfigError, match=r"1\.\.256"):
        shard.expand_matrix(m, shard.MAX_SHARDS + 1)


@pytest.mark.parametrize("bad", [0, 257, -1])
def test_expand_matrix_rejects_out_of_range_shards(shard: ModuleType, bad: int) -> None:
    """n_shards must be an integer in 1..256."""
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


def test_partition_is_disjoint_and_covers_the_whole_selection(
    shard: ModuleType,
) -> None:
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
        for name in shard.select_shard_tasks(
            names, ["org/keep-*"], n_tasks, n_shards, i
        )
    ]

    assert len(selected) == 3
    assert all(name.startswith("org/keep-") for name in selected)


def test_n_shards_one_returns_full_selection(shard: ModuleType) -> None:
    """n_shards=1 is the whole (filtered/capped) list — the unsharded path."""
    names = [f"org/t{i}" for i in range(5)]
    assert shard.select_shard_tasks(names, [], 0, 1, 0) == names


def test_fewer_tasks_than_shards_yields_empty_trailing_shards(
    shard: ModuleType,
) -> None:
    """n_tasks < n_shards: early shards get the work, later shards are empty.

    The workflow relies on this to no-op (not fail) empty shard jobs. e.g.
    n_tasks=1 n_shards=4 -> shard 0 runs the task, shards 1-3 are empty.
    """
    names = [f"org/t{i}" for i in range(10)]
    n_tasks, n_shards = 1, 4

    slices = [
        shard.select_shard_tasks(names, [], n_tasks, n_shards, i)
        for i in range(n_shards)
    ]

    assert slices[0] == ["org/t0"]
    assert slices[1] == slices[2] == slices[3] == []
    # Union is still exactly the selected task set (no task lost, none duplicated).
    assert [name for s in slices for name in s] == ["org/t0"]


def test_select_shard_tasks_rejects_bad_shard_index(shard: ModuleType) -> None:
    """shard_index must be in range(n_shards)."""
    with pytest.raises(shard.ShardConfigError, match="shard_index"):
        shard.select_shard_tasks(["org/t0"], [], 0, 2, 2)


# --------------------------------------------------------------------------- #
# task_display_name — resolve every Harbor task-id shape to org/name
# --------------------------------------------------------------------------- #


class _GetName:
    """A task id that exposes `get_name()` (every real Harbor TaskId does)."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get_name(self) -> str:
        return self._value


class _OrgName:
    """A `get_name()`-less id carrying bare `org`/`name` attributes (fallback)."""

    def __init__(self, org: str | None, name: str | None) -> None:
        self.org = org
        self.name = name


def test_task_display_name_prefers_get_name(shard: ModuleType) -> None:
    """`get_name()` is authoritative — the same name Harbor filters/reports on.

    This is the regression guard: a `PackageTaskId` returns `org/name` from
    `get_name()`, but a `GitTaskId`/`LocalTaskId` returns `path.name` with no
    `org`/`name` attributes. Delegating to `get_name()` resolves all of them,
    where the old `f"{org}/{name}"` reconstruction returned `None` for git/local
    ids and silently dropped every task.
    """
    assert shard.task_display_name(_GetName("acme/widget")) == "acme/widget"
    assert shard.task_display_name(_GetName("my-task")) == "my-task"


def test_task_display_name_dict_fallback(shard: ModuleType) -> None:
    """Dict-shaped manifest entries fall back to `org`/`name` keys."""
    assert shard.task_display_name({"org": "acme", "name": "widget"}) == "acme/widget"
    assert shard.task_display_name({"name": "solo"}) == "solo"


def test_task_display_name_attr_fallback(shard: ModuleType) -> None:
    """Objects without `get_name()` fall back to `org`/`name` attributes."""
    assert shard.task_display_name(_OrgName("acme", "widget")) == "acme/widget"
    assert shard.task_display_name(_OrgName(None, "solo")) == "solo"


def test_task_display_name_returns_none_when_underivable(shard: ModuleType) -> None:
    """No derivable name -> None, so the caller can filter and fail loudly.

    An empty `get_name()`, an empty dict, and an attr-less object must all map to
    `None` rather than `""` or `"None/None"`, so the run step's
    `[x for ... if x]` filter drops them and the empty-manifest fail-fast fires.
    """
    assert shard.task_display_name(_GetName("")) is None
    assert shard.task_display_name({}) is None
    assert shard.task_display_name(_OrgName(None, None)) is None
    assert shard.task_display_name(object()) is None


# --------------------------------------------------------------------------- #
# main — env-driven entry point: matrix + effective n_shards to $GITHUB_OUTPUT
# --------------------------------------------------------------------------- #


def _run_main(
    shard: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    **env: str,
) -> dict[str, str]:
    """Run `main()` with `env` set and return the parsed `$GITHUB_OUTPUT` lines."""
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    shard.main()
    written = output_file.read_text().splitlines()
    return dict(line.split("=", 1) for line in written)


def test_main_writes_matrix_and_effective_shards(
    shard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`main()` emits both the `matrix=` and `n_shards=` keys the harbor job reads.

    The two-key contract is load-bearing: the run job reads back `n_shards` to
    partition, so a dropped/renamed key would silently mismatch the matrix and
    drop tasks. Mirrors `test_models.py`'s `main()` output test.
    """
    matrix_in = json.dumps({"include": [{"model": "p:m0"}, {"model": "p:m1"}]})
    keyed = _run_main(
        shard, monkeypatch, tmp_path, MODEL_MATRIX=matrix_in, N_SHARDS="3", N_TASKS="0"
    )

    assert keyed["n_shards"] == "3"
    matrix = json.loads(keyed["matrix"])
    assert len(matrix["include"]) == 2 * 3  # cross-product, capped to nothing
    assert sorted(e["shard"] for e in matrix["include"] if e["model"] == "p:m0") == [
        0,
        1,
        2,
    ]


def test_main_output_is_single_line_per_key(
    shard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub Actions parses `key=value\\n`; guards against an `indent=2` refactor."""
    matrix_in = json.dumps({"include": [{"model": "p:m0"}]})
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("MODEL_MATRIX", matrix_in)

    shard.main()

    lines = output_file.read_text().splitlines()
    assert all(line.count("=") >= 1 and "\n" not in line for line in lines)
    assert {line.split("=", 1)[0] for line in lines} == {"matrix", "n_shards"}


def test_main_applies_effective_shard_cap(
    shard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N_TASKS < N_SHARDS collapses to n_tasks shards end-to-end (cap reaches output).

    `effective_shards` is unit-tested in isolation; this proves the cap actually
    flows through `main()` into the emitted matrix (1 shard, not 4).
    """
    matrix_in = json.dumps({"include": [{"model": "p:m0"}, {"model": "p:m1"}]})
    keyed = _run_main(
        shard, monkeypatch, tmp_path, MODEL_MATRIX=matrix_in, N_SHARDS="4", N_TASKS="1"
    )

    assert keyed["n_shards"] == "1"
    matrix = json.loads(keyed["matrix"])
    assert len(matrix["include"]) == 2  # 2 models x 1 effective shard
    assert all(entry["shard"] == 0 for entry in matrix["include"])


def test_main_defaults_to_single_unsharded_matrix(
    shard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset N_SHARDS/N_TASKS -> n_shards=1, one shard-0 job per model (no change)."""
    monkeypatch.delenv("N_SHARDS", raising=False)
    monkeypatch.delenv("N_TASKS", raising=False)
    matrix_in = json.dumps({"include": [{"model": "p:m0"}, {"model": "p:m1"}]})
    keyed = _run_main(shard, monkeypatch, tmp_path, MODEL_MATRIX=matrix_in)

    assert keyed["n_shards"] == "1"
    matrix = json.loads(keyed["matrix"])
    assert len(matrix["include"]) == 2
    assert all(entry["shard"] == 0 for entry in matrix["include"])


@pytest.mark.parametrize("var", ["N_SHARDS", "N_TASKS"])
def test_main_rejects_non_integer_env(
    shard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    """A non-integer N_SHARDS/N_TASKS exits 1 (not a stack trace)."""
    monkeypatch.setenv("MODEL_MATRIX", json.dumps({"include": [{"model": "p:m0"}]}))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    monkeypatch.setenv(var, "not-a-number")

    with pytest.raises(SystemExit) as excinfo:
        shard.main()
    assert excinfo.value.code == 1


def test_main_rejects_matrix_overflow_as_exit(
    shard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `ShardConfigError` (matrix over the 256 cap) becomes exit 1, not a traceback."""
    big = json.dumps({"include": [{"model": f"p:m{i}"} for i in range(129)]})
    monkeypatch.setenv("MODEL_MATRIX", big)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    monkeypatch.setenv("N_SHARDS", "2")  # 129 * 2 = 258 > 256

    with pytest.raises(SystemExit) as excinfo:
        shard.main()
    assert excinfo.value.code == 1


# --------------------------------------------------------------------------- #
# pack_tasks — split ordered task list into contiguous groups (comment #3)
# --------------------------------------------------------------------------- #


def test_pack_tasks_one_per_shard_under_cap(shard: ModuleType) -> None:
    """When len(names) <= max_shards, one task per group (1-task shards)."""
    names = [f"t{i}" for i in range(5)]
    assert shard.pack_tasks(names, max_shards=200) == [
        ["t0"],
        ["t1"],
        ["t2"],
        ["t3"],
        ["t4"],
    ]


def test_pack_tasks_packs_above_cap_contiguously(shard: ModuleType) -> None:
    """Above the cap, pack ceil(n/max_shards) tasks per group contiguously.

    cap 2 -> ceil(5/2)=3 per group -> [t0,t1,t2],[t3,t4]
    """
    names = [f"t{i}" for i in range(5)]
    assert shard.pack_tasks(names, max_shards=2) == [["t0", "t1", "t2"], ["t3", "t4"]]


def test_pack_tasks_covers_every_task_disjointly(shard: ModuleType) -> None:
    """Order-preserving, no dupes, no drops (covers exactly the input set)."""
    names = [f"t{i}" for i in range(203)]
    groups = shard.pack_tasks(names, max_shards=200)
    assert len(groups) <= 200
    flat = [n for g in groups for n in g]
    assert flat == names  # order-preserving, no dupes, no drops


def test_pack_tasks_empty(shard: ModuleType) -> None:
    """Empty input yields empty output."""
    assert shard.pack_tasks([], max_shards=200) == []
