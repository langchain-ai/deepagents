import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_unified_compare as compare  # noqa: E402


def _bundle(
    root: Path,
    *,
    version: str,
    branch: str,
    sha: str,
    model: str,
    config: str,
    scores: dict[str, tuple[float, float, dict[str, float]]],
) -> None:
    bundle = root / f"{version}-{config}"
    records = {}
    for category, (pass_k, avg_k, tasks) in scores.items():
        leaf = bundle / "categories" / category
        leaf.mkdir(parents=True)
        (leaf / "summary.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "config": "tau3" if category == "conversation" else config,
                    "category": category,
                    "rollouts_per_task": 2,
                    "incomplete": False,
                    "totals": {
                        "tasks": len(tasks),
                        "passed": sum(round(value * 2) for value in tasks.values()),
                    },
                    "pass@2": pass_k,
                    "avg@2": avg_k,
                }
            )
        )
        (leaf / "per_task.jsonl").write_text(
            "".join(
                json.dumps({"task": task, "pass@2": value}) + "\n"
                for task, value in tasks.items()
            )
        )
        records[category] = {
            "runtime": "tau3" if category == "conversation" else config,
            "path": f"categories/{category}",
        }
    bundle.mkdir(exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version_id": version,
                "source_branch": branch,
                "source_sha": sha,
                "model": model,
                "config": config,
                "categories": records,
            }
        )
    )


def test_compare_emits_highlights_pairwise_deltas_and_task_outcomes(
    tmp_path: Path,
) -> None:
    categories = ["autonomous", "conversation", "context"]
    _bundle(
        tmp_path,
        version="v1",
        branch="without-todos",
        sha="a" * 40,
        model="openai:gpt",
        config="bare",
        scores={
            category: (0.5, 0.25, {"same": 1.0, "changed": 0.0})
            for category in categories
        },
    )
    _bundle(
        tmp_path,
        version="v2",
        branch="with-todos",
        sha="b" * 40,
        model="openai:gpt",
        config="bare",
        scores={
            category: (1.0, 0.75, {"same": 1.0, "changed": 1.0})
            for category in categories
        },
    )
    sources = [
        {"version_id": "v1", "branch": "without-todos", "sha": "a" * 40},
        {"version_id": "v2", "branch": "with-todos", "sha": "b" * 40},
    ]

    result = compare.compare(
        tmp_path,
        sources=sources,
        models=["openai:gpt"],
        configs=["bare"],
        categories=categories,
        rollouts=2,
    )

    assert len(result["rows"]) == 2
    pair = result["pairwise"][0]
    assert pair["deltas"]["macro.pass_at_k"] == 0.5
    assert pair["task_outcomes"]["autonomous"] == {
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "missing": 0,
    }
    markdown = compare.render_markdown(result)
    assert "**1.000**/**0.750**" in markdown
    assert "+0.500/+0.500" in markdown

    out = tmp_path / "out"
    compare.write_outputs(result, out)
    assert (out / "comparison_summary.json").is_file()
    assert len(list((out / "radar_by_config").glob("*.json"))) == 1


def test_compare_materializes_missing_expected_bundle(tmp_path: Path) -> None:
    sources = [
        {"version_id": "v1", "branch": "a", "sha": "a" * 40},
        {"version_id": "v2", "branch": "b", "sha": "b" * 40},
    ]
    result = compare.compare(
        tmp_path,
        sources=sources,
        models=["openai:gpt"],
        configs=["bare", "dcode"],
        categories=["autonomous"],
        rollouts=2,
    )
    assert len(result["rows"]) == 4
    assert all(row["incomplete"] for row in result["rows"])
    assert all(row["missing_categories"] == ["autonomous"] for row in result["rows"])


def test_compare_rejects_bundle_with_wrong_source_sha(tmp_path: Path) -> None:
    import pytest

    _bundle(
        tmp_path,
        version="v1",
        branch="branch",
        sha="f" * 40,
        model="model",
        config="bare",
        scores={"context": (1.0, 1.0, {"task": 1.0})},
    )
    with pytest.raises(ValueError, match="source mismatch"):
        compare.compare(
            tmp_path,
            sources=[{"version_id": "v1", "branch": "branch", "sha": "a" * 40}],
            models=["model"],
            configs=["bare"],
            categories=["context"],
            rollouts=2,
        )
