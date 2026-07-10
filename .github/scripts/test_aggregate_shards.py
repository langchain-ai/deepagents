"""Tests for aggregate_shards.py.

Runs under pytest, and also standalone via `python3 test_aggregate_shards.py`
(a minimal runner at the bottom provides a temp dir to tests that need one).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aggregate_shards as agg


def _write_trial(dirpath: Path, task, reward=None, errored=False, model="m1", job_id="job1"):
    """Create a trial folder with a result.json shaped like Harbor's."""
    dirpath.mkdir(parents=True, exist_ok=True)
    result = {
        "task_name": task,
        "config": {"agent": {"model_name": model}, "job_id": job_id},
        "verifier_result": None if errored else {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": "SomeError"} if errored else None,
    }
    (dirpath / "result.json").write_text(json.dumps(result))


def test_aggregate_and_summary(tmp_path: Path):
    specs = {
        "taskA": [1.0, 0.0, 0.0],  # 1 of 3
        "taskB": [0.0, 0.0, 0.0],  # 0 of 3
        "taskC": [1.0, 1.0, 1.0],  # 3 of 3
    }
    i = 0
    # A job-level result.json (no task_name) that must be ignored.
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "result.json").write_text(json.dumps({"stats": {"n": 9}}))
    for task, rewards in specs.items():
        for reward in rewards:
            _write_trial(tmp_path / f"{task}__{i}", task, reward=reward)
            i += 1

    models, job_ids, by_task = agg.aggregate(tmp_path)
    assert by_task["taskA"] == {"trials": 3, "passed": 1, "errored": 0}
    assert by_task["taskB"] == {"trials": 3, "passed": 0, "errored": 0}
    assert by_task["taskC"] == {"trials": 3, "passed": 3, "errored": 0}
    assert models == {"m1"}
    assert job_ids == {"job1"}

    dataset_passk, avg_at_k, totals, per_task = agg.build_summary(by_task, 3)
    # pass@K (K=3), scalar: mean over tasks of "passed at least once" = (1+0+1)/3.
    assert abs(dataset_passk - (1 + 0 + 1) / 3) < 1e-6
    assert totals == {
        "tasks": 3, "trials": 9, "expected_trials": 9, "passed": 4, "errored": 0,
    }
    # avg@K: passing trials / expected trials = 4 / (3 tasks * 3 rollouts) = 4/9.
    assert abs(avg_at_k - 4 / 9) < 1e-6
    assert len(per_task) == 3
    # per-task pass@K is a scalar under a dynamic "pass@{K}" key.
    assert {r["task"]: r["pass@3"] for r in per_task} == {
        "taskA": 1.0, "taskB": 0.0, "taskC": 1.0,
    }


def test_errored_and_missing_count_as_fail(tmp_path: Path):
    _write_trial(tmp_path / "t__0", "taskX", reward=1.0)
    _write_trial(tmp_path / "t__1", "taskX", errored=True)      # exception -> fail + errored
    _write_trial(tmp_path / "t__2", "taskX", reward=None)       # no verifier reward -> fail + errored
    _, _, by_task = agg.aggregate(tmp_path)
    assert by_task["taskX"] == {"trials": 3, "passed": 1, "errored": 2}


def test_partial_reward_is_not_a_pass(tmp_path: Path):
    _write_trial(tmp_path / "t__0", "taskY", reward=0.5)
    _, _, by_task = agg.aggregate(tmp_path)
    assert by_task["taskY"] == {"trials": 1, "passed": 0, "errored": 0}


def test_end_to_end_writes_files(tmp_path: Path):
    _write_trial(tmp_path / "a__0", "taskA", reward=1.0)
    _write_trial(tmp_path / "a__1", "taskA", reward=0.0)
    out = tmp_path / "out"
    rc = agg.main([str(tmp_path), "--rollouts", "2", "--out-dir", str(out), "--dataset", "ds/x"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dataset"] == "ds/x"
    assert summary["model"] == "m1"
    assert summary["totals"] == {
        "tasks": 1, "trials": 2, "expected_trials": 2, "passed": 1, "errored": 0,
    }
    assert summary["pass@2"] == 1.0  # pass@2: taskA passed at least once
    assert summary["avg@2"] == 0.5  # 1 passing trial of 2 expected
    assert summary["incomplete"] is False  # 2 trials == 2 expected, no shard gate
    rows = [json.loads(line) for line in (out / "per_task.jsonl").read_text().splitlines()]
    assert rows == [
        {"task": "taskA", "trials": 2, "passed": 1, "errored": 0, "pass@2": 1.0}
    ]


def test_missing_rollouts_count_as_failures(tmp_path: Path):
    # taskA ran only 1 of 3 rollouts (a shard died mid-task) and it passed.
    _write_trial(tmp_path / "a__0", "taskA", reward=1.0)
    out = tmp_path / "out"
    rc = agg.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    # pass@3 is 1.0 (it did pass once), but avg@3 must be 1/3, not 1/1.
    assert summary["pass@3"] == 1.0
    assert abs(summary["avg@3"] - 1 / 3) < 1e-6
    assert summary["totals"]["trials"] == 1
    assert summary["totals"]["expected_trials"] == 3
    assert summary["incomplete"] is True  # 1 trial < 3 expected


def test_missing_shards_flag_incomplete(tmp_path: Path):
    # A complete task, but the run expected 2 shards and only 1 uploaded.
    _write_trial(tmp_path / "a__0", "taskA", reward=1.0, job_id="job1")
    _write_trial(tmp_path / "a__1", "taskA", reward=1.0, job_id="job1")
    out = tmp_path / "out"
    rc = agg.main(
        [str(tmp_path), "--rollouts", "2", "--expected-shards", "2", "--out-dir", str(out)]
    )
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["shards_found"] == 1
    assert summary["expected_shards"] == 2
    assert summary["incomplete"] is True  # 1 shard < 2 expected


def test_multiple_models_is_rejected(tmp_path: Path):
    _write_trial(tmp_path / "a__0", "taskA", reward=1.0, model="m1")
    _write_trial(tmp_path / "a__1", "taskA", reward=0.0, model="m2")
    try:
        agg.main([str(tmp_path), "--rollouts", "1", "--out-dir", str(tmp_path / "o")])
    except SystemExit as exc:
        assert exc.code not in (0, None)
        return
    raise AssertionError("expected SystemExit for multiple models")


def test_empty_tree_is_no_op(tmp_path: Path):
    rc = agg.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(tmp_path)])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["totals"]["tasks"] == 0
    assert summary["pass@3"] is None
    assert summary["avg@3"] == 0.0
    assert summary["incomplete"] is False  # nothing expected, nothing missing


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as tmp:
                    test(Path(tmp))
            else:
                test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
