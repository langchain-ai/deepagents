"""Tests for the two-version unified comparison matrix."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_compare_prep as prep  # noqa: E402


VERSIONS = [
    {"version_id": "v1", "branch": "feature/with-todos", "sha": "a" * 40},
    {"version_id": "v2", "branch": "feature/without-todos", "sha": "b" * 40},
]


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIFIED_VERSIONS_JSON", json.dumps(VERSIONS))
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:sonnet")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous,conversation,context")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    monkeypatch.setenv("UNIFIED_CONCURRENCY", "4")
    monkeypatch.setenv("UNIFIED_ROLLOUTS", "3")
    monkeypatch.delenv("UNIFIED_INCLUDE_TASKS", raising=False)


def test_build_outputs_duplicates_one_shared_matrix_per_version(monkeypatch):
    _env(monkeypatch)
    outputs = prep.build_outputs()

    assert outputs["max_parallel"] == "13"
    assert outputs["subject_parallel"] == "2"
    include = outputs["eval_matrix"]["include"]
    assert [(entry["version_id"], entry["source_sha"]) for entry in include] == [
        ("v1", "a" * 40),
        ("v2", "b" * 40),
    ]
    assert include[0]["flat_matrix"] == include[1]["flat_matrix"]
    flat = json.loads(include[0]["flat_matrix"])["include"]
    assert {entry["category"] for entry in flat} == {
        "autonomous",
        "conversation",
        "context",
    }


def test_versions_must_resolve_to_distinct_commits(monkeypatch):
    _env(monkeypatch)
    duplicate = [VERSIONS[0], {**VERSIONS[1], "sha": "a" * 40}]
    monkeypatch.setenv("UNIFIED_VERSIONS_JSON", json.dumps(duplicate))

    with pytest.raises(SystemExit, match="same commit"):
        prep.build_outputs()


def test_include_tasks_builds_identical_exact_subset_for_both_versions(monkeypatch):
    _env(monkeypatch)
    selected = [
        "harbor-index/gso-speedup-pydantic-enum",
        "harbor-index/swebenchpro-fix-file-suffix-chooser",
    ]
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_CONCURRENCY", "2")
    monkeypatch.setenv("UNIFIED_ROLLOUTS", "2")
    monkeypatch.setenv("UNIFIED_INCLUDE_TASKS", " ".join(selected))

    outputs = prep.build_outputs()

    assert outputs["max_parallel"] == "2"
    assert outputs["subject_parallel"] == "2"
    include = outputs["eval_matrix"]["include"]
    assert include[0]["flat_matrix"] == include[1]["flat_matrix"]
    flat = json.loads(include[0]["flat_matrix"])["include"]
    assert [entry["include_tasks"] for entry in flat] == selected


def test_include_tasks_rejects_unknown_profile_task(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_INCLUDE_TASKS", "harbor-index/not-a-task")

    with pytest.raises(SystemExit, match="outside the selected profile"):
        prep.build_outputs()


def test_total_job_budget_counts_both_versions(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv(
        "UNIFIED_MODELS", ",".join(f"openai:model-{i}" for i in range(6))
    )

    with pytest.raises(SystemExit, match="worker pool"):
        prep.build_outputs()
