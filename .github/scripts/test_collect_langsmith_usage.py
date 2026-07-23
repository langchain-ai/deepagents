from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_langsmith_usage as usage  # noqa: E402


@dataclass
class FakeRun:
    tags: list[str] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_cost: Decimal | None = None
    error: str | None = None


class FakeClient:
    def __init__(self, responses: list[list[FakeRun] | Exception]) -> None:
        self.responses = responses
        self.calls = 0

    def list_runs(
        self, *, project_name: str, is_root: bool, select: list[str]
    ) -> list[FakeRun]:
        assert project_name == "experiment"
        assert is_root is True
        assert select == usage.SELECT_FIELDS
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def _run(
    *,
    prompt: int = 10,
    completion: int = 5,
    cost: str | None = "0.25",
    error: str | None = None,
) -> FakeRun:
    return FakeRun(
        tags=["harbor", "harbor-trial"],
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        total_cost=Decimal(cost) if cost is not None else None,
        error=error,
    )


def test_summarize_runs_uses_only_harbor_roots_and_includes_errors() -> None:
    child = _run(prompt=100, completion=100, cost="10")
    child.tags = ["harbor-phase"]
    result = usage.summarize_runs(
        [_run(), _run(prompt=20, completion=10, cost="0.75", error="failed"), child],
        expected_rollouts=2,
    )

    assert result["status"] == "complete"
    assert result["coverage"] == {
        "expected_rollouts": 2,
        "observed_rollouts": 2,
        "token_rollouts": 2,
        "priced_rollouts": 2,
        "completed_rollouts": 1,
        "errored_rollouts": 1,
    }
    # True spend counts every rollout, including the errored one.
    assert result["totals"] == {
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
        "cost_usd": 1.0,
    }


def test_completed_totals_exclude_errored_rollouts() -> None:
    result = usage.summarize_runs(
        [
            _run(prompt=10, completion=5, cost="0.25"),
            _run(prompt=20, completion=10, cost="0.75", error="boom"),
        ],
        expected_rollouts=2,
    )

    # Completed-only totals drop the errored rollout; true spend keeps both.
    assert result["completed_totals"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost_usd": 0.25,
    }
    assert result["totals"]["cost_usd"] == 1.0
    assert result["coverage"]["completed_rollouts"] == 1
    assert result["coverage"]["errored_rollouts"] == 1


def test_summarize_runs_keeps_tokens_when_price_is_missing() -> None:
    result = usage.summarize_runs([_run(cost=None)], expected_rollouts=1)

    assert result["status"] == "partial"
    assert result["coverage"]["token_rollouts"] == 1
    assert result["coverage"]["priced_rollouts"] == 0
    assert result["totals"]["total_tokens"] == 15
    assert result["totals"]["cost_usd"] is None
    assert result["completed_totals"]["cost_usd"] is None


def test_summarize_runs_rejects_non_numeric_tokens_and_costs() -> None:
    bad = FakeRun(
        tags=["harbor-trial"],
        prompt_tokens=-1,
        completion_tokens=5,
        total_tokens=4,
        total_cost=Decimal("nan"),
    )
    result = usage.summarize_runs([bad], expected_rollouts=1)

    assert result["coverage"]["observed_rollouts"] == 1
    assert result["coverage"]["token_rollouts"] == 0
    assert result["coverage"]["priced_rollouts"] == 0
    assert result["totals"]["total_tokens"] is None
    assert result["totals"]["cost_usd"] is None


def test_collect_experiment_retries_until_usage_and_price_are_complete() -> None:
    client = FakeClient(
        [
            RuntimeError("temporary"),
            [_run(cost=None)],
            [_run(), _run(prompt=20, completion=10, cost="0.75")],
        ]
    )
    sleeps: list[float] = []

    result = usage.collect_experiment(
        client,
        "experiment",
        2,
        attempts=3,
        sleep=sleeps.append,
        delays=(0.0,),
    )

    assert client.calls == 3
    assert sleeps == [0.0, 0.0]
    assert result["status"] == "complete"
    assert result["coverage"]["priced_rollouts"] == 2


def test_collect_experiment_retains_best_partial_result() -> None:
    client = FakeClient([[_run()], RuntimeError("temporary")])

    result = usage.collect_experiment(
        client,
        "experiment",
        2,
        attempts=2,
        sleep=lambda _delay: None,
    )

    assert result["status"] == "partial"
    assert result["coverage"]["observed_rollouts"] == 1


def test_collect_all_marks_experiments_unavailable_without_client() -> None:
    result = usage.collect_all({"experiment": 2}, None)

    assert result["schema_version"] == 1
    assert result["experiments"]["experiment"]["status"] == "unavailable"
    assert result["experiments"]["experiment"]["totals"]["cost_usd"] is None
    assert result["experiments"]["experiment"]["completed_totals"]["cost_usd"] is None


def _write_leaf(root: Path, name: str, summary: dict[str, object]) -> None:
    leaf = root / name
    leaf.mkdir(parents=True)
    (leaf / "summary.json").write_text(json.dumps(summary))


def test_discover_experiments_reads_leaf_summaries(tmp_path: Path) -> None:
    _write_leaf(
        tmp_path,
        "harbor-combined-a",
        {
            "langsmith_experiment": "experiment",
            "expected_shards": 3,
            "rollouts_per_task": 2,
        },
    )
    # A second leaf sharing the same experiment (same expected count) dedupes.
    _write_leaf(
        tmp_path,
        "harbor-combined-b",
        {
            "langsmith_experiment": "experiment",
            "expected_shards": 3,
            "rollouts_per_task": 2,
        },
    )

    assert usage.discover_experiments(tmp_path) == {"experiment": 6}


def test_discover_experiments_falls_back_to_expected_trials(tmp_path: Path) -> None:
    _write_leaf(
        tmp_path,
        "harbor-combined-c",
        {
            "langsmith_experiment": "other",
            "rollouts_per_task": 2,
            "totals": {"expected_trials": 8},
        },
    )

    assert usage.discover_experiments(tmp_path) == {"other": 8}


def test_discover_experiments_skips_leaves_without_experiment(tmp_path: Path) -> None:
    _write_leaf(tmp_path, "harbor-combined-d", {"rollouts_per_task": 2})

    assert usage.discover_experiments(tmp_path) == {}


def test_discover_experiments_rejects_conflicting_counts(tmp_path: Path) -> None:
    _write_leaf(
        tmp_path,
        "harbor-combined-e",
        {"langsmith_experiment": "dup", "expected_shards": 3, "rollouts_per_task": 2},
    )
    _write_leaf(
        tmp_path,
        "harbor-combined-f",
        {"langsmith_experiment": "dup", "expected_shards": 4, "rollouts_per_task": 2},
    )

    try:
        usage.discover_experiments(tmp_path)
    except ValueError as exc:
        assert "conflicting expected rollout counts" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a ValueError for conflicting counts")


def test_main_without_key_writes_unavailable(tmp_path: Path, monkeypatch) -> None:
    _write_leaf(
        tmp_path,
        "harbor-combined-g",
        {"langsmith_experiment": "experiment", "expected_shards": 1, "rollouts_per_task": 1},
    )
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    out = tmp_path / "usage" / "langsmith_usage.json"

    assert usage.main([str(tmp_path), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["schema_version"] == 1
    assert written["experiments"]["experiment"]["status"] == "unavailable"
