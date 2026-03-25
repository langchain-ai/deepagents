"""Tests for LangSmith feedback helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from deepagents_harbor.langsmith import _extract_reward

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def trial_dir(tmp_path: Path) -> Path:
    """Return a temporary trial directory."""
    return tmp_path


def _write_result(trial_dir: Path, data: dict[str, Any]) -> None:
    (trial_dir / "result.json").write_text(json.dumps(data))


class TestExtractReward:
    """Tests for _extract_reward."""

    def test_normal_reward(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": 0.75}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.75
        assert comment is None

    def test_zero_reward(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": 0.0}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is None

    def test_negative_reward(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": -0.5}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == -0.5
        assert comment is None

    def test_integer_reward_returned_as_float(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": 1}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 1.0
        assert isinstance(reward, float)
        assert comment is None

    def test_missing_verifier_result_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"some_other_key": True})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None
        assert "verifier_result" in comment

    def test_empty_verifier_result_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {}})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_none_verifier_result_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": None})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_missing_rewards_key_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"something_else": 1}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None
        assert "reward" in comment

    def test_empty_rewards_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {"rewards": {}}})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_string_reward_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": "high"}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None
        assert "str" in comment

    def test_null_reward_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": None}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_list_reward_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": [1, 2]}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_missing_file_raises(self, trial_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _extract_reward(trial_dir)

    def test_malformed_json_raises(self, trial_dir: Path) -> None:
        (trial_dir / "result.json").write_text("{bad json")
        with pytest.raises(ValueError, match="malformed JSON"):
            _extract_reward(trial_dir)

    def test_malformed_json_preserves_cause(self, trial_dir: Path) -> None:
        (trial_dir / "result.json").write_text("{bad json")
        with pytest.raises(ValueError, match="malformed JSON") as exc_info:
            _extract_reward(trial_dir)
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
