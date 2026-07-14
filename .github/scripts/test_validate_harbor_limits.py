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


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gha_validate_harbor_limits", SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load()


def _one_model(name: str = "openai:gpt-5.4") -> list[dict]:
    return [{"model": name, "provider": "openai", "artifact_key": "openai-gpt-5.4"}]


def test_single_model_at_caps_is_valid(mod: ModuleType) -> None:
    """One model, n_shards == 10, concurrency == 4 (the documented maximum) passes."""
    assert mod.validate_limits(_one_model(), mod.MAX_SHARDS, mod.MAX_CONCURRENCY) == []


def test_zero_models_rejected(mod: ModuleType) -> None:
    errors = mod.validate_limits([], 1, 1)
    assert any("single model" in e for e in errors)


def test_two_models_rejected_and_names_reported(mod: ModuleType) -> None:
    models = [{"model": "openai:gpt-5.4"}, {"model": "anthropic:claude-sonnet-4-6"}]
    errors = mod.validate_limits(models, 1, 1)
    assert any("single model" in e for e in errors)
    # The offending model names are surfaced so the failure is self-diagnosing.
    assert any("anthropic:claude-sonnet-4-6" in e for e in errors)


def test_n_shards_over_cap_rejected(mod: ModuleType) -> None:
    errors = mod.validate_limits(_one_model(), mod.MAX_SHARDS + 1, 1)
    assert any("n_shards" in e for e in errors)
    # Boundary: exactly the cap is allowed.
    assert mod.validate_limits(_one_model(), mod.MAX_SHARDS, 1) == []


def test_concurrency_over_cap_rejected(mod: ModuleType) -> None:
    errors = mod.validate_limits(_one_model(), 1, mod.MAX_CONCURRENCY + 1)
    assert any("concurrency" in e for e in errors)
    assert mod.validate_limits(_one_model(), 1, mod.MAX_CONCURRENCY) == []


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "", "  ", "1.5"])
def test_parse_positive_rejects_non_positive_ints(mod: ModuleType, bad: str) -> None:
    with pytest.raises(mod.LimitError):
        mod.parse_positive("n_shards", bad)


def test_parse_positive_accepts_valid(mod: ModuleType) -> None:
    assert mod.parse_positive("n_shards", " 7 ") == 7


def test_main_exits_nonzero_on_multiple_models(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = {"include": [{"model": "a:1"}, {"model": "b:2"}]}
    monkeypatch.setenv("MODEL_MATRIX", json.dumps(matrix))
    monkeypatch.setenv("N_SHARDS", "10")
    monkeypatch.setenv("CONCURRENCY", "4")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)


def test_main_exits_nonzero_on_bad_shards(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_MATRIX", json.dumps({"include": [{"model": "a:1"}]}))
    monkeypatch.setenv("N_SHARDS", "0")
    monkeypatch.setenv("CONCURRENCY", "4")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)


def test_main_succeeds_for_valid_single_model(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_MATRIX", json.dumps({"include": [{"model": "a:1"}]}))
    monkeypatch.setenv("N_SHARDS", "10")
    monkeypatch.setenv("CONCURRENCY", "4")
    # No SystemExit == success.
    mod.main()
