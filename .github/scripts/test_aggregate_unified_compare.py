import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_unified_compare as compare  # noqa: E402


ROLLOUTS = 2


def _leaf(
    root: Path,
    *,
    branch: str,
    sha: str,
    model: str,
    config: str,
    category: str,
    pass_k: float,
    avg_k: float,
    tasks: dict[str, float],
    incomplete: bool = False,
    write_tasks: bool = True,
) -> None:
    leaf = (
        root
        / f"{branch.replace('/', '-')}-{model.replace(':', '-')}-{config}-{category}"
    )
    leaf.mkdir(parents=True)
    passed = round(avg_k * len(tasks) * ROLLOUTS)
    (leaf / "summary.json").write_text(
        json.dumps(
            {
                "model": model,
                "branch": branch,
                "source_sha": sha,
                "config": config,
                "category": category,
                "dataset": "dataset",
                "rollouts_per_task": ROLLOUTS,
                "incomplete": incomplete,
                "totals": {
                    "tasks": len(tasks),
                    "trials": len(tasks) * ROLLOUTS,
                    "expected_trials": len(tasks) * ROLLOUTS,
                    "passed": passed,
                    "errored": 0,
                },
                f"pass@{ROLLOUTS}": pass_k,
                f"avg@{ROLLOUTS}": avg_k,
            }
        )
    )
    if write_tasks:
        (leaf / "per_task.jsonl").write_text(
            "".join(
                json.dumps({"task": task, f"pass@{ROLLOUTS}": score}) + "\n"
                for task, score in tasks.items()
            )
        )


def _expected(
    sources: list[dict[str, str]],
    *,
    model: str = "model",
    allocations: dict[str, list[str]],
) -> list[dict[str, str]]:
    return [
        {
            "model": model,
            "branch": source["branch"],
            "source_sha": source["sha"],
            "config": config,
            "category": category,
        }
        for source in sources
        for config, categories in allocations.items()
        for category in categories
    ]


def _populate(
    root: Path,
    sources: list[dict[str, str]],
    allocations: dict[str, list[str]],
    *,
    model: str = "model",
) -> None:
    for source_index, source in enumerate(sources):
        for config_index, (config, categories) in enumerate(allocations.items()):
            for category in categories:
                candidate_wins = source_index > 0 or config_index > 0
                _leaf(
                    root,
                    branch=source["branch"],
                    sha=source["sha"],
                    model=model,
                    config=config,
                    category=category,
                    pass_k=1.0 if candidate_wins else 0.5,
                    avg_k=0.75 if candidate_wins else 0.25,
                    tasks={"win": 1.0 if candidate_wins else 0.0, "tie": 1.0},
                )


def test_compare_covers_cross_branch_and_shared_within_branch_configs(
    tmp_path: Path,
) -> None:
    sources = [
        {"branch": "main", "sha": "a" * 40},
        {"branch": "feature", "sha": "b" * 40},
    ]
    allocations = {
        "bare": ["autonomous", "context"],
        "dcode": ["autonomous", "context"],
        "tau3": ["conversation"],
    }
    _populate(tmp_path, sources, allocations)

    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=_expected(sources, allocations=allocations),
        categories=["autonomous", "conversation", "context"],
        rollouts=ROLLOUTS,
    )

    comparisons = result["comparisons"]
    cross_branch = [item for item in comparisons if item["kind"] == "cross_branch"]
    within_branch = [item for item in comparisons if item["kind"] == "within_branch"]
    assert {item["baseline"]["config"] for item in cross_branch} == {
        "bare",
        "dcode",
        "tau3",
    }
    assert len(within_branch) == 2
    assert all(
        {item["baseline"]["config"], item["candidate"]["config"]} == {"bare", "dcode"}
        for item in within_branch
    )
    assert all(
        item["shared_categories"] == ["autonomous", "context"] for item in within_branch
    )
    assert len(result["not_comparable"]) == 4
    assert {item["reason"] for item in result["not_comparable"]} == {
        "no_shared_categories"
    }

    bare = next(item for item in cross_branch if item["baseline"]["config"] == "bare")
    autonomous = bare["metrics"]["categories"]["autonomous"]
    assert autonomous["pass_at_k"]["delta"] == 0.5
    assert autonomous["avg_at_k"]["delta"] == 0.5
    assert bare["task_outcomes"]["autonomous"] == {
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "missing": 0,
    }


def test_compare_never_pairs_different_models(tmp_path: Path) -> None:
    sources = [
        {"branch": "main", "sha": "a" * 40},
        {"branch": "feature", "sha": "b" * 40},
    ]
    allocations = {"bare": ["context"]}
    expected: list[dict[str, str]] = []
    for model in ("model-a", "model-b"):
        expected.extend(_expected(sources, model=model, allocations=allocations))
        _populate(tmp_path, sources, allocations, model=model)

    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=expected,
        categories=["context"],
        rollouts=ROLLOUTS,
    )

    assert len(result["comparisons"]) == 2
    assert all(
        item["baseline"]["model"] == item["candidate"]["model"]
        for item in result["comparisons"]
    )


def test_future_conversation_config_compares_with_tau3(tmp_path: Path) -> None:
    sources = [{"branch": "main", "sha": "a" * 40}]
    allocations = {"tau3": ["conversation"], "new-conversation": ["conversation"]}
    _populate(tmp_path, sources, allocations)

    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=_expected(sources, allocations=allocations),
        categories=["conversation"],
        rollouts=ROLLOUTS,
    )

    assert len(result["comparisons"]) == 1
    comparison = result["comparisons"][0]
    assert comparison["kind"] == "within_branch"
    assert comparison["shared_categories"] == ["conversation"]
    assert result["not_comparable"] == []


def test_incomplete_subject_still_emits_diagnostic_delta(tmp_path: Path) -> None:
    sources = [
        {"branch": "main", "sha": "a" * 40},
        {"branch": "feature", "sha": "b" * 40},
    ]
    allocations = {"bare": ["context"]}
    _leaf(
        tmp_path,
        branch="main",
        sha="a" * 40,
        model="model",
        config="bare",
        category="context",
        pass_k=0.0,
        avg_k=0.0,
        tasks={"task": 0.0},
        incomplete=True,
    )
    _leaf(
        tmp_path,
        branch="feature",
        sha="b" * 40,
        model="model",
        config="bare",
        category="context",
        pass_k=1.0,
        avg_k=1.0,
        tasks={"task": 1.0},
    )

    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=_expected(sources, allocations=allocations),
        categories=["context"],
        rollouts=ROLLOUTS,
    )

    comparison = result["comparisons"][0]
    assert comparison["status"] == "incomplete"
    assert comparison["metrics"]["macro"]["pass_at_k"]["delta"] == 1.0


def test_missing_task_file_marks_comparison_incomplete(tmp_path: Path) -> None:
    sources = [
        {"branch": "main", "sha": "a" * 40},
        {"branch": "feature", "sha": "b" * 40},
    ]
    allocations = {"bare": ["context"]}
    for source in sources:
        _leaf(
            tmp_path,
            branch=source["branch"],
            sha=source["sha"],
            model="model",
            config="bare",
            category="context",
            pass_k=0.5,
            avg_k=0.5,
            tasks={"task": 1.0},
            write_tasks=source["branch"] != "main",
        )

    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=_expected(sources, allocations=allocations),
        categories=["context"],
        rollouts=ROLLOUTS,
    )

    comparison = result["comparisons"][0]
    assert comparison["status"] == "incomplete"
    assert comparison["task_outcomes"]["context"]["missing"] == 1


def test_compare_rejects_actual_source_sha_mismatch(tmp_path: Path) -> None:
    sources = [{"branch": "main", "sha": "a" * 40}]
    allocations = {"bare": ["context"], "dcode": ["context"]}
    _populate(tmp_path, [{"branch": "main", "sha": "b" * 40}], allocations)

    with pytest.raises(ValueError, match="actual source SHA mismatch"):
        compare.compare(
            tmp_path,
            sources=sources,
            expected_leaves=_expected(sources, allocations=allocations),
            categories=["context"],
            rollouts=ROLLOUTS,
        )


def test_compare_rejects_malformed_per_task_data(tmp_path: Path) -> None:
    sources = [{"branch": "main", "sha": "a" * 40}]
    allocations = {"bare": ["context"], "dcode": ["context"]}
    _populate(tmp_path, sources, allocations)
    path = next(tmp_path.rglob("per_task.jsonl"))
    path.write_text("not-json\n")

    with pytest.raises(ValueError, match="invalid per-task JSON"):
        compare.compare(
            tmp_path,
            sources=sources,
            expected_leaves=_expected(sources, allocations=allocations),
            categories=["context"],
            rollouts=ROLLOUTS,
        )


def test_write_outputs_skips_single_subject(tmp_path: Path) -> None:
    sources = [{"branch": "main", "sha": "a" * 40}]
    allocations = {"bare": ["context"]}
    _populate(tmp_path, sources, allocations)
    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=_expected(sources, allocations=allocations),
        categories=["context"],
        rollouts=ROLLOUTS,
    )

    out = tmp_path / "out"
    assert compare.write_outputs(result, out) is False
    assert not out.exists()


def test_write_outputs_emits_json_markdown_and_step_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [
        {"branch": "main", "sha": "a" * 40},
        {"branch": "feature", "sha": "b" * 40},
    ]
    allocations = {"bare": ["context"]}
    _populate(tmp_path, sources, allocations)
    result = compare.compare(
        tmp_path,
        sources=sources,
        expected_leaves=_expected(sources, allocations=allocations),
        categories=["context"],
        rollouts=ROLLOUTS,
    )
    summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    out = tmp_path / "out"
    assert compare.write_outputs(result, out) is True
    assert (out / "comparison_summary.json").is_file()
    assert "Different models are never compared" in (out / "comparison.md").read_text()
    assert "deterministic comparisons" in summary.read_text()
