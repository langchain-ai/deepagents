from __future__ import annotations

import json

import pytest

from deepagents_evals.radar import ALL_CATEGORIES, CATEGORY_LABELS, EVAL_CATEGORIES

# ---------------------------------------------------------------------------
# Category definitions consistency
# ---------------------------------------------------------------------------


# All eval test modules that define pytestmark with eval_category.
# Maps category name -> list of test module basenames.
EXPECTED_CATEGORY_MODULES: dict[str, list[str]] = {
    "file_operations": ["test_file_operations"],
    "retrieval": ["test_file_operations", "test_external_benchmarks"],
    "tool_use": [
        "test_tool_selection",
        "test_tool_usage_relational",
        "test_todos",
        "test_external_benchmarks",
    ],
    "memory": ["test_memory", "test_memory_multiturn", "test_memory_agent_bench"],
    "conversation": ["test_followup_quality", "test_tau2_airline"],
    "summarization": ["test_summarization"],
    "unit_test": [
        "test_system_prompt",
        "test_hitl",
        "test_subagents",
        "test_skills",
    ],
}


def test_all_categories_have_labels():
    for cat in ALL_CATEGORIES:
        assert cat in CATEGORY_LABELS, f"Missing label for category {cat!r}"


def test_all_labeled_categories_are_registered():
    for cat in CATEGORY_LABELS:
        assert cat in ALL_CATEGORIES, f"Label defined for unregistered category {cat!r}"


def test_expected_categories_match_all_categories():
    assert set(EXPECTED_CATEGORY_MODULES.keys()) == set(ALL_CATEGORIES)


def test_radar_categories_are_subset_of_all():
    assert set(EVAL_CATEGORIES) <= set(ALL_CATEGORIES)


def test_unit_test_excluded_from_radar():
    assert "unit_test" not in EVAL_CATEGORIES


def _is_marker_call(node: object, marker_name: str) -> str | None:
    """Return the marker value if *node* is a `pytest.mark.<marker_name>("value")` call, else `None`."""
    import ast

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == marker_name
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ):
        return None
    return str(node.args[0].value)


def _is_eval_category_call(node: object) -> str | None:
    """Return the category name if *node* is a `pytest.mark.eval_category("name")` call, else `None`."""
    return _is_marker_call(node, "eval_category")


def test_expected_modules_match_filesystem():
    """Discover eval_category markers on disk and assert they match `EXPECTED_CATEGORY_MODULES`.

    Scans both module-level ``pytestmark`` assignments and per-function
    ``@pytest.mark.eval_category(...)`` decorators so that files with
    mixed per-function categories (e.g. test_external_benchmarks,
    test_file_operations) are detected correctly.
    """
    import ast
    from pathlib import Path

    evals_dir = Path(__file__).resolve().parent.parent / "evals"
    discovered: dict[str, set[str]] = {}

    def _record(cat: str, stem: str) -> None:
        discovered.setdefault(cat, set()).add(stem)

    for path in sorted(evals_dir.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.iter_child_nodes(tree):
            # Module-level pytestmark = [pytest.mark.eval_category("...")]
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "pytestmark" in targets:
                    for elt in ast.walk(node.value):
                        cat = _is_eval_category_call(elt)
                        if cat:
                            _record(cat, path.stem)

            # Function-level decorator: @pytest.mark.eval_category("...")
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for decorator in node.decorator_list:
                    cat = _is_eval_category_call(decorator)
                    if cat:
                        _record(cat, path.stem)

    # Compare as sets so insertion order in EXPECTED_CATEGORY_MODULES doesn't matter.
    expected = {cat: set(modules) for cat, modules in EXPECTED_CATEGORY_MODULES.items()}
    assert discovered == expected, (
        f"Mismatch between eval test files on disk and EXPECTED_CATEGORY_MODULES.\n"
        f"  On disk:  {dict(discovered)}\n"
        f"  Expected: {expected}"
    )


def _has_eval_tier_marker(tree: object) -> bool:
    """Return True if the AST tree has any `eval_tier` marker anywhere in the module.

    Walks the entire AST to catch eval_tier in pytestmark lists, function
    decorators, and helper functions like `_tiered_params`.
    """
    import ast

    if not isinstance(tree, ast.Module):
        return False

    has_test_functions = any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_")
        for node in ast.iter_child_nodes(tree)
    )

    # Files with no test functions (e.g. empty placeholder files) are fine.
    if not has_test_functions:
        return True

    # Walk entire module — catches pytestmark, decorators, and helper functions.
    # Check both constant-arg calls (via _is_marker_call) and dynamic-arg calls
    # (e.g. conditional expressions in _tiered_params) by looking for any
    # pytest.mark.eval_tier attribute access.
    for node in ast.walk(tree):
        if _is_marker_call(node, "eval_tier"):
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "eval_tier"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            return True
    return False


def test_all_eval_modules_have_eval_tier():
    """Every eval test module must have at least one `eval_tier` marker.

    Ensures new eval files cannot silently lack tier annotations, which would
    cause them to be excluded when running `--eval-tier baseline`.
    """
    import ast
    from pathlib import Path

    evals_dir = Path(__file__).resolve().parent.parent / "evals"
    missing: list[str] = []

    for path in sorted(evals_dir.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _has_eval_tier_marker(tree):
            missing.append(str(path.relative_to(evals_dir)))

    assert not missing, (
        f"Eval test modules missing eval_tier marker: {missing}. "
        f"Add @pytest.mark.eval_tier('baseline') or @pytest.mark.eval_tier('hillclimb') "
        f"to each test function or module-level pytestmark."
    )


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
        _CATEGORY_RESULTS["unit_test"] = {"passed": 5, "failed": 0, "total": 5}
        _CATEGORY_RESULTS["tool_use"] = {"passed": 0, "failed": 2, "total": 2}

        scores: dict[str, float] = {}
        for cat, counts in sorted(_CATEGORY_RESULTS.items()):
            if counts["total"] > 0:
                scores[cat] = round(counts["passed"] / counts["total"], 2)

        assert scores == {"memory": 0.75, "tool_use": 0.0, "unit_test": 1.0}
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
            "category_scores": {"memory": 0.90, "tool_use": 0.80},
        },
    ]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    results = load_results_from_summary(path)
    assert len(results) == 1
    assert results[0].scores == {"memory": 0.90, "tool_use": 0.80}


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
