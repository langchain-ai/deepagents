"""Tests for Version 1 versus Version 2 result comparison."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_unified_compare as compare  # noqa: E402


VERSIONS = [
    {"version_id": "v1", "branch": "branch/one", "sha": "a" * 40},
    {"version_id": "v2", "branch": "branch/two", "sha": "b" * 40},
]


def _artifact(
    root: Path,
    version: str,
    category: str,
    pass_k: float,
    avg_k: float,
    task_passes: dict[str, int],
) -> None:
    artifact = root / f"harbor-combined-compare-{version}-{category}-model-slug"
    artifact.mkdir()
    tasks = len(task_passes)
    summary = {
        "model": "anthropic:sonnet",
        "category": category,
        "dataset": "dataset",
        "rollouts_per_task": 3,
        "incomplete": False,
        "totals": {
            "tasks": tasks,
            "trials": tasks * 3,
            "expected_trials": tasks * 3,
            "passed": sum(task_passes.values()),
            "errored": 0,
        },
        "pass@3": pass_k,
        "avg@3": avg_k,
    }
    (artifact / "summary.json").write_text(json.dumps(summary))
    rows = [
        {
            "task": task,
            "trials": 3,
            "passed": passed,
            "errored": 0,
            "pass@3": float(passed > 0),
        }
        for task, passed in task_passes.items()
    ]
    (artifact / "per_task.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_compare_calculates_signed_deltas_winners_and_radar(tmp_path):
    categories = ["autonomous", "conversation", "context"]
    for category in categories:
        _artifact(tmp_path, "v1", category, 0.5, 0.4, {"a": 1, "b": 0})
        _artifact(tmp_path, "v2", category, 1.0, 0.6, {"a": 1, "b": 1})

    leaves, tasks = compare.discover(tmp_path, 3)
    result = compare.compare(leaves, tasks, VERSIONS, ["anthropic:sonnet"], categories)

    model = result["comparisons"]["anthropic:sonnet"]
    assert model["categories"]["autonomous"] == {
        "pass_at_k": 0.5,
        "avg_at_k": 0.19999999999999996,
    }
    assert model["tasks"]["autonomous"] == {
        "v1_wins": 0,
        "v2_wins": 1,
        "ties": 1,
        "missing": 0,
    }
    markdown = compare.render_markdown(result, 3)
    assert "**1.000**" in markdown
    assert "+0.500" in markdown
    assert "Version 2" in markdown

    out = tmp_path / "out"
    compare.write_outputs(result, 3, out)
    radar = list((out / "radar-inputs").glob("*.json"))
    assert len(radar) == 1
    labels = [entry["model"] for entry in json.loads(radar[0].read_text())]
    assert labels == ["Version 1 — branch/one", "Version 2 — branch/two"]


def test_incomplete_version_suppresses_deltas_and_radar(tmp_path):
    _artifact(tmp_path, "v1", "autonomous", 0.5, 0.4, {"a": 1})
    leaves, tasks = compare.discover(tmp_path, 3)
    result = compare.compare(
        leaves,
        tasks,
        VERSIONS,
        ["anthropic:sonnet"],
        ["autonomous", "conversation", "context"],
    )

    assert result["comparisons"]["anthropic:sonnet"]["complete"] is False
    assert result["comparisons"]["anthropic:sonnet"]["macro"]["pass_at_k"] is None
    out = tmp_path / "out"
    compare.write_outputs(result, 3, out)
    assert not (out / "radar-inputs").exists()
