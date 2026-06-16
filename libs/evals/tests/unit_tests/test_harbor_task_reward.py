"""Tests for the `write-file-simple` Harbor task scorer.

Exercises `compute_rewards` from
`tests/harbor_tasks/write-file-simple/tests/score.py` against synthetic ATIF
trajectories, proving the pytest `TrajectoryScorer` metrics (correctness,
step / tool-call counts, efficiency ratios) can be derived inside a Harbor
trial without running a container.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SCORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "harbor_tasks"
    / "write-file-simple"
    / "tests"
    / "score.py"
)


def _load_score() -> ModuleType:
    spec = importlib.util.spec_from_file_location("write_file_simple_score", _SCORE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = _load_score()

_NAME = "Foo Bar"


def _user_step() -> dict[str, Any]:
    return {"step_id": 1, "source": "user", "message": "Write your name..."}


def _write_step(*, target: str = "/app/name.txt") -> dict[str, Any]:
    return {
        "step_id": 2,
        "source": "agent",
        "message": "",
        "tool_calls": [
            {
                "tool_call_id": "t1",
                "function_name": "write_file",
                "arguments": {"file_path": target, "content": _NAME},
            }
        ],
    }


def _final_step(text: str, *, step_id: int = 3) -> dict[str, Any]:
    return {"step_id": step_id, "source": "agent", "message": text}


def _trajectory(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "ATIF-v1.2", "steps": list(steps)}


def test_full_success() -> None:
    trajectory = _trajectory(_user_step(), _write_step(), _final_step("My name is Foo Bar."))
    rewards = score.compute_rewards(trajectory, {"/app/name.txt": "Foo Bar"})

    assert rewards["reward"] == 1.0
    assert rewards["correctness"] == 1.0
    assert rewards["file_name_contains"] == 1.0
    assert rewards["final_text_contains"] == 1.0
    assert rewards["wrote_name_file"] == 1.0
    assert rewards["agent_steps"] == 2.0
    assert rewards["tool_call_requests"] == 1.0
    assert rewards["step_ratio"] == 1.0
    assert rewards["tool_call_ratio"] == 1.0


def test_missing_file_fails_correctness() -> None:
    trajectory = _trajectory(_user_step(), _write_step(), _final_step("My name is Foo Bar."))
    rewards = score.compute_rewards(trajectory, {})

    assert rewards["file_name_contains"] == 0.0
    assert rewards["final_text_contains"] == 1.0
    assert rewards["correctness"] == 0.0
    assert rewards["reward"] == 0.0


def test_wrong_final_text_fails_correctness() -> None:
    trajectory = _trajectory(_user_step(), _write_step(), _final_step("Done."))
    rewards = score.compute_rewards(trajectory, {"/app/name.txt": "Foo Bar"})

    assert rewards["file_name_contains"] == 1.0
    assert rewards["final_text_contains"] == 0.0
    assert rewards["correctness"] == 0.0


def test_extra_step_inflates_step_ratio() -> None:
    trajectory = _trajectory(
        _user_step(),
        _write_step(),
        _final_step("thinking out loud", step_id=3),
        _final_step("My name is Foo Bar.", step_id=4),
    )
    rewards = score.compute_rewards(trajectory, {"/app/name.txt": "Foo Bar"})

    assert rewards["agent_steps"] == 3.0
    assert rewards["step_ratio"] == 1.5
    assert rewards["correctness"] == 1.0


def test_message_content_parts_are_flattened() -> None:
    final = {
        "step_id": 3,
        "source": "agent",
        "message": [
            {"type": "text", "text": "My name is "},
            {"type": "text", "text": "Foo Bar."},
        ],
    }
    trajectory = _trajectory(_user_step(), _write_step(), final)
    rewards = score.compute_rewards(trajectory, {"/app/name.txt": "Foo Bar"})

    assert rewards["final_text_contains"] == 1.0
    assert rewards["correctness"] == 1.0


def test_no_tool_calls() -> None:
    trajectory = _trajectory(_user_step(), _final_step("My name is Foo Bar."))
    rewards = score.compute_rewards(trajectory, {"/app/name.txt": "Foo Bar"})

    assert rewards["wrote_name_file"] == 0.0
    assert rewards["tool_call_requests"] == 0.0
    assert rewards["tool_call_ratio"] == 0.0
    assert rewards["agent_steps"] == 1.0


def test_malformed_trajectory_scores_zero_without_crashing() -> None:
    rewards = score.compute_rewards("not a dict", None)

    assert rewards["correctness"] == 0.0
    assert rewards["agent_steps"] == 0.0
    assert rewards["tool_call_requests"] == 0.0
    # Ratios are still emitted (baselines are positive), and are zero.
    assert rewards["step_ratio"] == 0.0
    assert rewards["tool_call_ratio"] == 0.0


def test_ratios_omitted_when_baseline_is_zero() -> None:
    trajectory = _trajectory(_user_step(), _final_step("My name is Foo Bar."))
    rewards = score.compute_rewards(
        trajectory,
        {"/app/name.txt": "Foo Bar"},
        expected_steps=0,
        expected_tool_calls=0,
    )

    assert "step_ratio" not in rewards
    assert "tool_call_ratio" not in rewards


@pytest.mark.parametrize("target", ["/app/name.txt", "name.txt", "/app/./name.txt"])
def test_wrote_name_file_matches_name_txt_targets(target: str) -> None:
    trajectory = _trajectory(
        _user_step(), _write_step(target=target), _final_step("My name is Foo Bar.")
    )
    rewards = score.compute_rewards(trajectory, {"/app/name.txt": "Foo Bar"})

    assert rewards["wrote_name_file"] == 1.0
