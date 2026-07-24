"""Tests for the Harbor single-model + resource-limit gate.

Mirrors `test_shard_matrix.py`: import-by-path, stdlib + pytest only, so it runs
under CI's `pytest .github/scripts/test_*.py` (see `.github/workflows/ci.yml`).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "validate_harbor_limits.py"
UNIFIED_PREP_SCRIPT = REPO_ROOT / ".github" / "scripts" / "unified_prep.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {path}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load(SCRIPT, "gha_validate_harbor_limits")


@pytest.fixture(scope="module")
def unified_prep() -> ModuleType:
    return _load(UNIFIED_PREP_SCRIPT, "gha_unified_prep")


def _one_model(name: str = "openai:gpt-5.4") -> list[dict]:
    return [{"model": name, "provider": "openai", "artifact_key": "openai-gpt-5.4"}]


def test_single_model_at_caps_is_valid(mod: ModuleType) -> None:
    """One model, n_shards at the shard cap, concurrency at its own cap, passes."""
    assert (
        mod.validate_limits(
            _one_model(), mod.shard_matrix.MAX_SHARDS, mod.MAX_CONCURRENCY, 1
        )
        == []
    )


def test_zero_models_rejected(mod: ModuleType) -> None:
    errors = mod.validate_limits([], 1, 1, 1)
    assert any("single model" in e for e in errors)


def test_two_models_rejected_and_names_reported(mod: ModuleType) -> None:
    models = [{"model": "openai:gpt-5.4"}, {"model": "anthropic:claude-sonnet-4-6"}]
    errors = mod.validate_limits(models, 1, 1, 1)
    assert any("single model" in e for e in errors)
    # The offending model names are surfaced so the failure is self-diagnosing.
    assert any("anthropic:claude-sonnet-4-6" in e for e in errors)


def test_n_shards_over_cap_rejected(mod: ModuleType) -> None:
    errors = mod.validate_limits(_one_model(), mod.shard_matrix.MAX_SHARDS + 1, 1, 1)
    assert any("n_shards" in e for e in errors)
    # Boundary: exactly the cap (200, from shard_matrix.MAX_SHARDS) is allowed.
    assert mod.validate_limits(_one_model(), mod.shard_matrix.MAX_SHARDS, 1, 1) == []


def test_concurrency_over_cap_rejected(mod: ModuleType) -> None:
    errors = mod.validate_limits(_one_model(), 1, mod.MAX_CONCURRENCY + 1, 1)
    assert any("concurrency" in e for e in errors)
    assert mod.validate_limits(_one_model(), 1, mod.MAX_CONCURRENCY, 1) == []


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "", "  ", "1.5"])
def test_parse_positive_rejects_non_positive_ints(mod: ModuleType, bad: str) -> None:
    with pytest.raises(mod.LimitError):
        mod.parse_positive("n_shards", bad)


def test_parse_positive_accepts_valid(mod: ModuleType) -> None:
    assert mod.parse_positive("n_shards", " 7 ") == 7


def test_derive_shard_parallel_saturates_pool_and_clamps_to_n_shards(
    mod: ModuleType,
) -> None:
    # Packed shards can use all 4 slots; 40 // 4 = 10, under n_shards=100.
    assert mod.derive_shard_parallel(concurrency=4, rollouts=3, n_shards=100) == 10
    # Same concurrency/rollouts, but n_shards=5 clamps the pool down to 5.
    assert mod.derive_shard_parallel(concurrency=4, rollouts=3, n_shards=5) == 5
    # concurrency 1 allows 40 shard jobs, using the full budget.
    assert mod.derive_shard_parallel(concurrency=1, rollouts=3, n_shards=100) == 40


def test_derive_shard_parallel_uses_concurrency_when_rollouts_are_lower(
    mod: ModuleType,
) -> None:
    """Multiple tasks let a shard fill concurrency beyond one task's rollouts."""
    pool = mod.derive_shard_parallel(concurrency=4, rollouts=2, n_shards=100)
    assert pool == 10
    assert pool * 4 == mod.MAX_TASKS_PER_MODEL


def test_derive_shard_parallel_unchanged_at_todays_n_shards_10(mod: ModuleType) -> None:
    """No-change property: at n_shards=10 with today's typical inputs, the
    derived shard_parallel equals n_shards itself — identical to the current
    (undifferentiated) behavior of passing n_shards straight through.
    """
    assert mod.derive_shard_parallel(concurrency=4, rollouts=3, n_shards=10) == 10
    assert mod.derive_shard_parallel(concurrency=4, rollouts=4, n_shards=10) == 10


@pytest.mark.parametrize(
    ("concurrency", "rollouts", "n_shards"),
    [
        (4, 3, 100),
        (4, 3, 10),
        (4, 4, 10),
        (1, 3, 100),
        (4, 2, 100),
        (1, 3, 8),
        (4, 3, 5),
        (2, 5, 34),
    ],
)
def test_derive_shard_parallel_matches_unified_prep_derive_pool(
    mod: ModuleType,
    unified_prep: ModuleType,
    concurrency: int,
    rollouts: int,
    n_shards: int,
) -> None:
    """Drift guard: pins this module's derivation to unified_prep.derive_pool's
    single-model max_parallel so the two formulas can't silently diverge.
    """
    expected = unified_prep.derive_pool(concurrency, rollouts, n_shards, 1)[0]
    assert mod.derive_shard_parallel(concurrency, rollouts, n_shards) == expected


def test_main_exits_nonzero_on_multiple_models(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = {"include": [{"model": "a:1"}, {"model": "b:2"}]}
    monkeypatch.setenv("MODEL_MATRIX", json.dumps(matrix))
    monkeypatch.setenv("N_SHARDS", "10")
    monkeypatch.setenv("CONCURRENCY", "4")
    monkeypatch.setenv("ROLLOUTS", "3")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)


def test_main_exits_nonzero_on_bad_shards(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_MATRIX", json.dumps({"include": [{"model": "a:1"}]}))
    monkeypatch.setenv("N_SHARDS", "0")
    monkeypatch.setenv("CONCURRENCY", "4")
    monkeypatch.setenv("ROLLOUTS", "3")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)


def test_main_exits_nonzero_on_bad_rollouts(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_MATRIX", json.dumps({"include": [{"model": "a:1"}]}))
    monkeypatch.setenv("N_SHARDS", "10")
    monkeypatch.setenv("CONCURRENCY", "4")
    monkeypatch.setenv("ROLLOUTS", "0")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)


def test_main_succeeds_for_valid_single_model(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODEL_MATRIX", json.dumps({"include": [{"model": "a:1"}]}))
    monkeypatch.setenv("N_SHARDS", "10")
    monkeypatch.setenv("CONCURRENCY", "4")
    monkeypatch.setenv("ROLLOUTS", "3")
    github_output = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    # No SystemExit == success.
    mod.main()
    # The derived shard_parallel is emitted for harbor.yml to consume.
    assert "shard_parallel=10" in github_output.read_text()
