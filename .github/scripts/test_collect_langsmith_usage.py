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
        "errored_rollouts": 1,
    }
    assert result["totals"] == {
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
        "cost_usd": 1.0,
    }
    assert result["averages"]["total_tokens_per_rollout"] == 22.5
    assert result["averages"]["cost_usd_per_rollout"] == 0.5


def test_summarize_runs_keeps_tokens_when_price_is_missing() -> None:
    result = usage.summarize_runs([_run(cost=None)], expected_rollouts=1)

    assert result["status"] == "partial"
    assert result["coverage"]["token_rollouts"] == 1
    assert result["coverage"]["priced_rollouts"] == 0
    assert result["averages"]["total_tokens_per_rollout"] == 15
    assert result["averages"]["cost_usd_per_rollout"] is None


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


def test_discover_experiments_deduplicates_shared_category_leaf(tmp_path: Path) -> None:
    for config in ("bare", "dcode"):
        bundle = tmp_path / config
        leaf = bundle / "categories" / "conversation"
        leaf.mkdir(parents=True)
        (leaf / "summary.json").write_text(
            json.dumps(
                {
                    "langsmith_experiment": "experiment",
                    "expected_shards": 3,
                    "rollouts_per_task": 2,
                }
            )
        )
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "categories": {
                        "conversation": {
                            "runtime": "tau3",
                            "path": "categories/conversation",
                        }
                    },
                }
            )
        )

    assert usage.discover_experiments(tmp_path) == {"experiment": 6}
