"""Tests for the trial analysis script (`scripts/analyze.py`).

The script lives outside any importable package, so it is loaded by path. These
tests pin the I/O-reuse refactor: that helpers tolerate missing/corrupt inputs
without crashing, that the task-dir index is first-match and cached, and that a
malformed trajectory still yields exit-code-based failure classification (the
raw text must survive a JSON parse failure).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze.py"
_MODULE_NAME = "_analyze_under_test"


def _load_analyze() -> ModuleType:
    """Import `scripts/analyze.py` as a module without polluting `sys.path`."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"could not load spec for {_SCRIPT}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze()


@pytest.fixture(autouse=True)
def _clear_index_cache() -> Iterator[None]:
    """Clear the process-lifetime `_task_dir_index` cache between tests."""
    analyze._task_dir_index.cache_clear()
    yield
    analyze._task_dir_index.cache_clear()


class TestReadJson:
    """Tests for `_read_json`."""

    def test_valid_object(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"a": 1}))
        assert analyze._read_json(path) == {"a": 1}

    def test_missing_file_is_silent_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert analyze._read_json(tmp_path / "absent.json") is None
        # Missing files are expected; no warning should be emitted.
        assert capsys.readouterr().out == ""

    def test_corrupt_json_warns_and_returns_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        assert analyze._read_json(path) is None
        assert "malformed JSON" in capsys.readouterr().out

    def test_non_object_json_returns_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A valid-but-non-object JSON (e.g. a list) must not crash callers that
        # immediately call `.get(...)` on the result.
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        assert analyze._read_json(path) is None
        assert "expected a JSON object" in capsys.readouterr().out


class TestTaskDirIndex:
    """Tests for `_task_dir_index` and `find_task_directory`."""

    def test_indexes_all_tasks(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        (source / "hashA" / "task-1").mkdir(parents=True)
        (source / "hashB" / "task-2").mkdir(parents=True)

        index = analyze._task_dir_index(source)

        assert set(index) == {"task-1", "task-2"}
        assert index["task-1"] == source / "hashA" / "task-1"

    def test_missing_source_returns_empty(self, tmp_path: Path) -> None:
        assert analyze._task_dir_index(tmp_path / "nope") == {}

    def test_duplicate_task_name_keeps_single_first_match(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        (source / "hashA" / "dup").mkdir(parents=True)
        (source / "hashB" / "dup").mkdir(parents=True)

        index = analyze._task_dir_index(source)

        # First match wins (one entry, pointing at a real candidate directory).
        assert list(index) == ["dup"]
        assert index["dup"] in {source / "hashA" / "dup", source / "hashB" / "dup"}
        # Cached: a second call returns the identical mapping.
        assert analyze._task_dir_index(source)["dup"] == index["dup"]

    def test_find_task_directory(self, tmp_path: Path) -> None:
        # trial_dir.parent.parent is the jobs root; task source sits beside jobs.
        (tmp_path / "src" / "hashA" / "my-task").mkdir(parents=True)
        trial_dir = tmp_path / "jobs" / "trial-1"
        trial_dir.mkdir(parents=True)

        found = analyze.find_task_directory(trial_dir, "my-task", "src")

        assert found == tmp_path / "src" / "hashA" / "my-task"

    def test_find_task_directory_absent(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "jobs" / "trial-1"
        trial_dir.mkdir(parents=True)
        assert analyze.find_task_directory(trial_dir, "missing", "src") is None


class TestCountToolUsage:
    """Tests for `count_tool_usage`."""

    def test_counts_by_function_name(self) -> None:
        data = {
            "steps": [
                {"tool_calls": [{"function_name": "read"}, {"function_name": "read"}]},
                {"tool_calls": [{"function_name": "write"}]},
                {"tool_calls": [{}]},  # missing name defaults to "unknown"
                {"source": "assistant"},  # no tool_calls
            ]
        }
        assert analyze.count_tool_usage(data) == {"read": 2, "write": 1, "unknown": 1}

    def test_empty_trajectory(self) -> None:
        assert analyze.count_tool_usage({}) == {}


class TestExtractTaskInstructions:
    """Tests for `extract_task_instructions`."""

    def test_returns_first_user_message(self) -> None:
        data = {
            "steps": [
                {"source": "system", "message": "sys"},
                {"source": "user", "message": "do the thing"},
                {"source": "user", "message": "ignored"},
            ]
        }
        assert analyze.extract_task_instructions(data) == "do the thing"

    def test_user_step_without_message_returns_empty_string(self) -> None:
        assert analyze.extract_task_instructions({"steps": [{"source": "user"}]}) == ""

    def test_no_user_step_returns_none(self) -> None:
        assert analyze.extract_task_instructions({"steps": [{"source": "assistant"}]}) is None


def _make_trial(
    trial_dir: Path,
    *,
    trajectory: str | None = None,
    reward: str | None = None,
    config: dict | None = None,
) -> None:
    """Write a minimal trial directory layout for `analyze_trial`."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (trial_dir / "config.json").write_text(json.dumps(config))
    if trajectory is not None:
        (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
        (trial_dir / "agent" / "trajectory.json").write_text(trajectory)
    if reward is not None:
        (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
        (trial_dir / "verifier" / "reward.txt").write_text(reward)


class TestAnalyzeTrial:
    """Tests for `analyze_trial`."""

    def test_completed_trial_counts_tools(self, tmp_path: Path) -> None:
        trajectory = json.dumps(
            {"steps": [{"tool_calls": [{"function_name": "bash"}, {"function_name": "bash"}]}]}
        )
        _make_trial(tmp_path, trajectory=trajectory, reward="1", config={"task": {"path": "t"}})

        trial = analyze.analyze_trial(tmp_path)

        assert trial.status is analyze.TrialStatus.COMPLETED
        assert trial.reward is True
        assert trial.tool_usage == {"bash": 2}

    def test_corrupt_trajectory_preserves_exit_code_classification(self, tmp_path: Path) -> None:
        # Regression guard: a malformed trajectory.json must not poison the raw
        # text used for exit-code extraction. The exit code 137 (OOM) is only
        # recoverable from the raw text via regex fallback, so failure_category
        # must be INFRA_OOM — not CAPABILITY (which is what an empty exit-code
        # list would produce here, since there is no exception.txt).
        corrupt = 'this is not valid json {"exit_code": 137}'
        _make_trial(tmp_path, trajectory=corrupt, reward="0", config={"task": {"path": "t"}})

        trial = analyze.analyze_trial(tmp_path)

        assert trial.status is analyze.TrialStatus.FAILED
        assert trial.tool_usage == {}  # parse failed, counted nothing
        assert trial.failure_category is analyze.FailureCategory.INFRA_OOM

    def test_pending_trial_when_no_reward_or_exception(self, tmp_path: Path) -> None:
        _make_trial(tmp_path, trajectory=json.dumps({"steps": []}), config={"task": {"path": "t"}})

        trial = analyze.analyze_trial(tmp_path)

        assert trial.status is analyze.TrialStatus.PENDING
        assert trial.reward is None

    def test_uses_solution_mapping(self, tmp_path: Path) -> None:
        solution = tmp_path / "solve.sh"
        solution.write_text("#!/bin/sh\n")
        _make_trial(tmp_path, trajectory=json.dumps({"steps": []}), config={"task": {"path": "t"}})

        trial = analyze.analyze_trial(tmp_path, solution_mapping={"t": solution})

        assert trial.solution_path == solution


class TestScanJobsDirectory:
    """Tests for `scan_jobs_directory` (concurrent analysis entry point)."""

    async def test_scans_all_trials_concurrently(self, tmp_path: Path) -> None:
        for name in ("trial-a", "trial-b", "trial-c"):
            _make_trial(
                tmp_path / name,
                trajectory=json.dumps({"steps": []}),
                reward="1",
                config={"task": {"path": name}},
            )

        trials = await analyze.scan_jobs_directory(tmp_path)

        assert len(trials) == 3
        assert {t.trial_id for t in trials} == {"trial-a", "trial-b", "trial-c"}

    async def test_missing_jobs_directory_returns_empty(self, tmp_path: Path) -> None:
        assert await analyze.scan_jobs_directory(tmp_path / "absent") == []
