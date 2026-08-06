"""Tests for Switchyard stats aggregation and pricing inputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from switchyard import collect_stats

if TYPE_CHECKING:
    from pathlib import Path


def test_usage_metadata_supports_uncached_pricing() -> None:
    stats = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cached_tokens": 60,
        "cache_creation_tokens": 10,
        "reasoning_tokens": 5,
    }

    cache_aware = collect_stats._usage_metadata(stats, pricing="cache-aware")
    uncached = collect_stats._usage_metadata(stats, pricing="uncached")

    assert cache_aware["input_token_details"] == {
        "cache_read": 60,
        "cache_creation": 10,
    }
    assert uncached["input_token_details"] == {
        "cache_read": 0,
        "cache_creation": 0,
    }
    assert uncached["input_tokens"] == 100
    assert uncached["output_tokens"] == 20


def test_custom_cost_uses_per_million_rates() -> None:
    stats = {"prompt_tokens": 2_000_000, "completion_tokens": 500_000}

    cost = collect_stats._custom_cost(
        stats,
        "private/nvidia/nemotron-3.5-nano-30b-a3b",
        {"nemotron-3.5-nano": (0.05, 0.20)},
    )

    assert cost == 0.2


def test_aggregate_snapshots_merges_model_and_classifier_usage(tmp_path: Path) -> None:
    first = tmp_path / "first" / "switchyard-stats.json"
    second = tmp_path / "second" / "switchyard-stats.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(
        json.dumps(
            {
                "total_requests": 2,
                "total_errors": 1,
                "models": {"nano": {"calls": 2, "prompt_tokens": 100}},
                "classifier": {
                    "total_requests": 1,
                    "models": {"judge": {"calls": 1, "completion_tokens": 10}},
                },
                "switchyard_config": "routes-nano.toml",
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "total_requests": 3,
                "total_errors": 2,
                "models": {"nano": {"calls": 3, "prompt_tokens": 200}},
                "classifier": {
                    "total_requests": 2,
                    "models": {"judge": {"calls": 2, "completion_tokens": 20}},
                },
                "switchyard_config": "routes-nano.toml",
            }
        )
    )

    paths = collect_stats._snapshot_paths([tmp_path])
    merged = collect_stats._aggregate_snapshots(paths)

    assert merged["snapshot_count"] == 2
    assert merged["total_requests"] == 5
    assert merged["total_errors"] == 3
    assert merged["models"]["nano"]["calls"] == 5
    assert merged["models"]["nano"]["prompt_tokens"] == 300
    assert merged["models"]["nano"]["avg_prompt_tokens"] == 60
    assert merged["classifier"]["models"]["judge"]["completion_tokens"] == 30
    assert merged["switchyard_configs"] == ["routes-nano.toml"]


def _write_trial(
    job_dir: Path,
    *,
    exception_info: object | None = None,
    verifier_result: object | None = True,
    total_requests: int = 1,
    total_errors: int = 0,
) -> None:
    trial = job_dir / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "exception_info": exception_info,
                "verifier_result": verifier_result,
            }
        )
    )
    (artifacts / "switchyard-stats.json").write_text(
        json.dumps(
            {
                "total_requests": total_requests,
                "total_errors": total_errors,
            }
        )
    )


def test_validate_job_accepts_successful_routed_trial(tmp_path: Path) -> None:
    _write_trial(tmp_path)

    count, failures = collect_stats._validate_job(tmp_path)

    assert count == 1
    assert failures == []


def test_validate_job_rejects_routed_upstream_errors(tmp_path: Path) -> None:
    _write_trial(tmp_path, total_errors=1)

    _, failures = collect_stats._validate_job(tmp_path)

    assert failures == ["trial: Switchyard recorded upstream errors"]


def test_validate_job_rejects_trial_exception_even_with_stats(tmp_path: Path) -> None:
    _write_trial(tmp_path, exception_info={"type": "AgentError"})

    _, failures = collect_stats._validate_job(tmp_path)

    assert failures == ["trial: trial recorded an exception"]


def test_validate_job_rejects_missing_verifier_result(tmp_path: Path) -> None:
    _write_trial(tmp_path, verifier_result=None)

    _, failures = collect_stats._validate_job(tmp_path)

    assert failures == ["trial: verifier result is missing"]
