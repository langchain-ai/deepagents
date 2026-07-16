import json
import os
import sys
from pathlib import Path

import pytest

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
    invalid: bool = False,
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
                    "langsmith_experiment": (
                        f"{version}-tau3-conversation"
                        if category == "conversation"
                        else f"{version}-{config}-{category}"
                    ),
                    "rollouts_per_task": 2,
                    "expected_shards": len(tasks),
                    "invalid": invalid,
                    "incomplete": invalid,
                    "totals": {
                        "tasks": len(tasks),
                        "expected_trials": len(tasks) * 2,
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


def _usage_block(
    *, expected: int, prompt: int, completion: int, cost: float
) -> dict[str, object]:
    total = prompt + completion
    return {
        "status": "complete",
        "coverage": {
            "expected_rollouts": expected,
            "observed_rollouts": expected,
            "token_rollouts": expected,
            "priced_rollouts": expected,
            "errored_rollouts": 0,
        },
        "totals": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cost_usd": cost,
        },
        "averages": {
            "prompt_tokens_per_rollout": prompt / expected,
            "completion_tokens_per_rollout": completion / expected,
            "total_tokens_per_rollout": total / expected,
            "cost_usd_per_rollout": cost / expected,
        },
    }


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


def test_compare_excludes_invalid_trials_from_deltas_highlights_and_radar(
    tmp_path: Path,
) -> None:
    categories = ["autonomous", "conversation", "context"]
    sources = [
        {"version_id": "v1", "branch": "a", "sha": "a" * 40},
        {"version_id": "v2", "branch": "b", "sha": "b" * 40},
    ]
    _bundle(
        tmp_path,
        version="v1",
        branch="a",
        sha="a" * 40,
        model="model",
        config="dcode",
        scores={category: (0.5, 0.5, {"task": 1.0}) for category in categories},
    )
    _bundle(
        tmp_path,
        version="v2",
        branch="b",
        sha="b" * 40,
        model="model",
        config="dcode",
        scores={category: (0.0, 0.0, {"task": 0.0}) for category in categories},
        invalid=True,
    )

    result = compare.compare(
        tmp_path,
        sources=sources,
        models=["model"],
        configs=["dcode"],
        categories=categories,
        rollouts=2,
    )

    assert result["rows"][1]["invalid"] is True
    assert result["pairwise"] == []
    markdown = compare.render_markdown(result)
    assert "dcode ❌ invalid" in markdown
    assert "Invalid results were excluded" in markdown
    out = tmp_path / "out"
    compare.write_outputs(result, out)
    radar = json.loads((out / "radar_results.json").read_text())
    assert [item["model"] for item in radar] == ["v1 a / model / dcode"]


def test_compare_rejects_bundle_with_wrong_source_sha(tmp_path: Path) -> None:
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


def test_usage_is_weighted_by_rollouts_and_compared_across_branches(
    tmp_path: Path,
) -> None:
    categories = ["autonomous", "context"]
    sources = [
        {"version_id": "v1", "branch": "a", "sha": "a" * 40},
        {"version_id": "v2", "branch": "b", "sha": "b" * 40},
    ]
    for source in sources:
        _bundle(
            tmp_path,
            version=source["version_id"],
            branch=source["branch"],
            sha=source["sha"],
            model="model",
            config="bare",
            scores={
                "autonomous": (1.0, 1.0, {"one": 1.0}),
                "context": (1.0, 1.0, {"one": 1.0, "two": 1.0}),
            },
        )
    experiment_usage = {
        "v1-bare-autonomous": _usage_block(
            expected=2, prompt=120, completion=80, cost=1.0
        ),
        "v1-bare-context": _usage_block(
            expected=4, prompt=600, completion=400, cost=6.0
        ),
        "v2-bare-autonomous": _usage_block(
            expected=2, prompt=180, completion=120, cost=1.5
        ),
        "v2-bare-context": _usage_block(
            expected=4, prompt=900, completion=600, cost=9.0
        ),
    }

    result = compare.compare(
        tmp_path,
        sources=sources,
        models=["model"],
        configs=["bare"],
        categories=categories,
        rollouts=2,
        usage_by_experiment=experiment_usage,
    )

    assert result["schema_version"] == 2
    first_overall = result["rows"][0]["usage"]["overall"]
    assert first_overall["coverage"]["observed_rollouts"] == 6
    assert first_overall["averages"]["total_tokens_per_rollout"] == 200
    assert first_overall["averages"]["cost_usd_per_rollout"] == 7 / 6
    cross_branch = result["usage_comparisons"][0]
    overall = cross_branch["scopes"]["overall"]["metrics"]
    assert overall["total_tokens_per_rollout"]["absolute"] == 100
    assert overall["total_tokens_per_rollout"]["percent"] == 50
    assert overall["cost_usd_per_rollout"]["absolute"] == pytest.approx(3.5 / 6)


def test_usage_marks_shared_tau3_experiment_within_branch(tmp_path: Path) -> None:
    source = {"version_id": "v1", "branch": "a", "sha": "a" * 40}
    for config in ("bare", "dcode"):
        _bundle(
            tmp_path,
            version="v1",
            branch="a",
            sha="a" * 40,
            model="model",
            config=config,
            scores={"conversation": (1.0, 1.0, {"task": 1.0})},
        )
    experiment_usage = {
        "v1-tau3-conversation": _usage_block(
            expected=2, prompt=100, completion=50, cost=0.5
        )
    }

    result = compare.compare(
        tmp_path,
        sources=[source],
        models=["model"],
        configs=["bare", "dcode"],
        categories=["conversation"],
        rollouts=2,
        usage_by_experiment=experiment_usage,
    )

    within_branch = result["usage_comparisons"][0]
    assert within_branch["kind"] == "within_branch"
    assert within_branch["scopes"]["conversation"]["shared_experiment"] is True
    markdown = compare.render_markdown(result)
    assert "bare, dcode" in markdown
    assert "| yes | shared | shared | shared | shared | shared | shared |" in markdown
