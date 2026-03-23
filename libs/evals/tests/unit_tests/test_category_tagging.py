from __future__ import annotations

import json

import pytest

from deepagents_evals.radar import CATEGORY_LABELS, EVAL_CATEGORIES

# ---------------------------------------------------------------------------
# Category definitions consistency
# ---------------------------------------------------------------------------


# All eval test modules that define pytestmark with eval_category.
# Maps category name -> list of test module basenames.
EXPECTED_CATEGORY_MODULES: dict[str, list[str]] = {
    "file_operations": ["test_file_operations"],
    "skills": ["test_skills"],
    "hitl": ["test_hitl"],
    "memory": ["test_memory", "test_memory_multiturn"],
    "summarization": ["test_summarization"],
    "subagents": ["test_subagents"],
    "system_prompt": ["test_system_prompt"],
    "tool_usage": ["test_tool_usage_relational", "test_tool_selection"],
    "followup_quality": ["test_followup_quality"],
    "external_benchmarks": ["test_external_benchmarks"],
    "tau2_airline": ["test_tau2_airline"],
    "memory_agent_bench": ["test_memory_agent_bench"],
}


def test_all_categories_have_labels():
    for cat in EVAL_CATEGORIES:
        assert cat in CATEGORY_LABELS, f"Missing label for category {cat!r}"


def test_all_labeled_categories_are_registered():
    for cat in CATEGORY_LABELS:
        assert cat in EVAL_CATEGORIES, f"Label defined for unregistered category {cat!r}"


def test_expected_categories_match_eval_categories():
    assert set(EXPECTED_CATEGORY_MODULES.keys()) == set(EVAL_CATEGORIES)


# ---------------------------------------------------------------------------
# Reporter per-category scoring logic
# ---------------------------------------------------------------------------


def test_category_scores_computation():
    from tests.evals.pytest_reporter import _CATEGORY_RESULTS

    # Save original state and restore after test.
    original = dict(_CATEGORY_RESULTS)
    try:
        _CATEGORY_RESULTS.clear()
        _CATEGORY_RESULTS["memory"] = {"passed": 3, "failed": 1, "total": 4}
        _CATEGORY_RESULTS["hitl"] = {"passed": 5, "failed": 0, "total": 5}
        _CATEGORY_RESULTS["tool_usage"] = {"passed": 0, "failed": 2, "total": 2}

        scores: dict[str, float] = {}
        for cat, counts in sorted(_CATEGORY_RESULTS.items()):
            if counts["total"] > 0:
                scores[cat] = round(counts["passed"] / counts["total"], 2)

        assert scores == {"hitl": 1.0, "memory": 0.75, "tool_usage": 0.0}
    finally:
        _CATEGORY_RESULTS.clear()
        _CATEGORY_RESULTS.update(original)


# ---------------------------------------------------------------------------
# Radar loader reads category_scores
# ---------------------------------------------------------------------------


def test_load_results_with_category_scores(tmp_path):
    from deepagents_evals.radar import load_results_from_summary

    data = [
        {
            "model": "test:model-a",
            "category_scores": {"memory": 0.90, "hitl": 0.80},
        },
    ]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    results = load_results_from_summary(path)
    assert len(results) == 1
    assert results[0].scores == {"memory": 0.90, "hitl": 0.80}


def test_load_results_missing_category_scores_raises(tmp_path):
    from deepagents_evals.radar import load_results_from_summary

    data = [{"model": "test:model-b", "correctness": 0.72}]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KeyError):
        load_results_from_summary(path)


def test_load_results_empty_category_scores(tmp_path):
    from deepagents_evals.radar import load_results_from_summary

    data = [{"model": "test:model-c", "category_scores": {}}]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    results = load_results_from_summary(path)
    assert results[0].scores == {}


# ---------------------------------------------------------------------------
# conftest --eval-category filtering
# ---------------------------------------------------------------------------


def test_eval_category_is_valid_mark_name():
    assert "eval_category".isidentifier()
