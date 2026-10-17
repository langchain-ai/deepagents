from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import collect_langsmith_usage as usage
import pytest


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


def test_collect_all_retries_until_usage_and_price_are_complete() -> None:
    client = FakeClient(
        [
            RuntimeError("temporary"),
            [_run(cost=None)],
            [_run(), _run(prompt=20, completion=10, cost="0.75")],
        ]
    )
    sleeps: list[float] = []

    result = usage.collect_all(
        {"experiment": 2},
        client,
        attempts=3,
        sleep=sleeps.append,
        delays=(0.0,),
    )
    experiment = result["experiments"]["experiment"]

    assert client.calls == 3
    assert sleeps == [0.0, 0.0]
    assert experiment["status"] == "complete"
    assert experiment["coverage"]["priced_rollouts"] == 2


def test_collect_all_retains_best_partial_result() -> None:
    client = FakeClient([[_run()], RuntimeError("temporary")])

    result = usage.collect_all(
        {"experiment": 2},
        client,
        attempts=2,
        sleep=lambda _delay: None,
    )
    experiment = result["experiments"]["experiment"]

    assert experiment["status"] == "partial"
    assert experiment["coverage"]["observed_rollouts"] == 1


def test_collect_all_marks_experiments_unavailable_without_client() -> None:
    result = usage.collect_all({"experiment": 2}, None)

    assert result["schema_version"] == 1
    assert result["experiments"]["experiment"]["status"] == "unavailable"
    assert result["experiments"]["experiment"]["totals"]["cost_usd"] is None
    assert result["experiments"]["experiment"]["completed_totals"]["cost_usd"] is None


def _write_experiments(path: Path, mapping: dict[str, object]) -> Path:
    path.write_text(json.dumps(mapping))
    return path


def test_load_experiments_reads_name_to_expected_map(tmp_path: Path) -> None:
    f = _write_experiments(tmp_path / "e.json", {"exp-a": 6, "exp-b": 8})
    assert usage.load_experiments(f) == {"exp-a": 6, "exp-b": 8}


def test_load_experiments_nulls_bad_expected_and_skips_empty_names(tmp_path: Path) -> None:
    f = _write_experiments(
        tmp_path / "e.json",
        {"exp-a": None, "exp-b": "nope", "exp-c": -1, "": 5},
    )
    # Non-int / negative expected -> None (best-effort coverage); empty name dropped.
    assert usage.load_experiments(f) == {"exp-a": None, "exp-b": None, "exp-c": None}


def test_load_experiments_rejects_non_object(tmp_path: Path) -> None:
    f = _write_experiments(tmp_path / "e.json", [1, 2, 3])
    with pytest.raises(ValueError, match="must be a JSON object"):
        usage.load_experiments(f)


def test_main_without_key_writes_unavailable(tmp_path: Path, monkeypatch) -> None:
    exp = _write_experiments(tmp_path / "e.json", {"experiment": 4})
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    out = tmp_path / "usage" / "langsmith_usage.json"

    assert usage.main(["--experiments-json", str(exp), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["schema_version"] == 1
    assert written["experiments"]["experiment"]["status"] == "unavailable"
    assert written["experiments"]["experiment"]["coverage"]["expected_rollouts"] == 4
