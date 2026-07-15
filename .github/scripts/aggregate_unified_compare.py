#!/usr/bin/env python3
"""Compare two unified-eval versions and render tables, deltas, and radar inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import TypedDict, cast

import aggregate_unified as unified


class Version(TypedDict):
    """Display metadata for one comparison side."""

    version_id: str
    branch: str
    sha: str


def _load_json_list(raw: str, label: str) -> list[object]:
    """Load a JSON list or fail with a concise input error."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be a JSON list") from exc
    if not isinstance(value, list):
        raise SystemExit(f"{label} must be a JSON list")
    return value


def _string_list(raw: str, label: str) -> list[str]:
    """Load a non-empty JSON list containing only strings."""
    value = _load_json_list(raw, label)
    if not value or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"{label} must be a non-empty JSON list of strings")
    return cast(list[str], value)


def _versions(raw: str) -> list[Version]:
    """Parse the v1/v2 mapping emitted by comparison preparation."""
    versions: list[Version] = []
    for item in _load_json_list(raw, "--versions-json"):
        if not isinstance(item, dict):
            raise SystemExit("--versions-json entries must be objects")
        version_id = item.get("version_id")
        branch = item.get("branch")
        sha = item.get("sha")
        if (
            version_id not in {"v1", "v2"}
            or not isinstance(branch, str)
            or not isinstance(sha, str)
        ):
            raise SystemExit("--versions-json must contain v1/v2 branch+SHA entries")
        versions.append(cast(Version, item))
    if len(versions) != 2 or {version["version_id"] for version in versions} != {
        "v1",
        "v2",
    }:
        raise SystemExit("--versions-json must contain exactly v1 and v2")
    return sorted(versions, key=lambda version: version["version_id"])


def _artifact_version(path: Path) -> str | None:
    """Read the version identity from a comparison artifact directory name."""
    match = re.search(r"harbor-combined-compare-(v1|v2)-", path.name)
    return match.group(1) if match else None


def discover(
    root: Path, rollouts: int
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Discover versioned leaf summaries and task rows from downloaded artifacts."""
    leaves: dict[str, list[dict]] = {"v1": [], "v2": []}
    tasks: dict[str, dict] = {}
    for artifact in sorted(path for path in root.iterdir() if path.is_dir()):
        version_id = _artifact_version(artifact)
        if version_id is None or not (artifact / "summary.json").is_file():
            continue
        leaf = unified.read_leaf(artifact, expected_rollouts=rollouts)
        leaves[version_id].append(leaf)
        per_task = artifact / "per_task.jsonl"
        if not per_task.is_file():
            continue
        for line in per_task.read_text().splitlines():
            row = json.loads(line)
            key = (version_id, leaf["model"], leaf["category"], row["task"])
            tasks["\x1f".join(key)] = row
    return leaves, tasks


def _fmt(value: float | None, *, signed: bool = False) -> str:
    """Format a metric or an unavailable marker."""
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _pair(
    left: float | None, right: float | None, *, highlight: bool = True
) -> tuple[str, str]:
    """Format a v1/v2 metric pair and bold the strictly higher value."""
    left_text, right_text = _fmt(left), _fmt(right)
    if highlight and left is not None and right is not None:
        if left > right:
            left_text = f"**{left_text}**"
        elif right > left:
            right_text = f"**{right_text}**"
    return left_text, right_text


def _task_pass_score(row: dict) -> float:
    """Return the dynamic per-task `pass@K` value."""
    keys = [key for key in row if isinstance(key, str) and key.startswith("pass@")]
    if len(keys) != 1:
        raise ValueError("per_task row must contain exactly one pass@K metric")
    return float(row[keys[0]])


def compare(
    leaves: dict[str, list[dict]],
    tasks: dict[str, dict],
    versions: list[Version],
    models: list[str],
    categories: list[str],
) -> dict:
    """Build per-version results, signed deltas, and paired task diagnostics."""
    combined = {
        version_id: unified.combine(leaves[version_id], models, categories)
        for version_id in ("v1", "v2")
    }
    comparisons: dict[str, dict] = {}
    for model in models:
        left = combined["v1"]["models"][model]
        right = combined["v2"]["models"][model]
        complete = not left["incomplete"] and not right["incomplete"]
        category_deltas: dict[str, dict[str, float | None]] = {}
        task_stats: dict[str, dict[str, int]] = {}
        for category in categories:
            left_cat = left["categories"].get(category)
            right_cat = right["categories"].get(category)
            cat_complete = bool(
                left_cat
                and right_cat
                and not left_cat["incomplete"]
                and not right_cat["incomplete"]
            )
            category_deltas[category] = {
                metric: (
                    right_cat[metric] - left_cat[metric]
                    if cat_complete
                    and right_cat[metric] is not None
                    and left_cat[metric] is not None
                    else None
                )
                for metric in ("pass_at_k", "avg_at_k")
            }
            names = {
                key.split("\x1f", 3)[3]
                for key in tasks
                if key.startswith(f"v1\x1f{model}\x1f{category}\x1f")
                or key.startswith(f"v2\x1f{model}\x1f{category}\x1f")
            }
            stats = {"v1_wins": 0, "v2_wins": 0, "ties": 0, "missing": 0}
            for name in names:
                left_row = tasks.get("\x1f".join(("v1", model, category, name)))
                right_row = tasks.get("\x1f".join(("v2", model, category, name)))
                if left_row is None or right_row is None:
                    stats["missing"] += 1
                    continue
                left_score = _task_pass_score(left_row)
                right_score = _task_pass_score(right_row)
                if left_score > right_score:
                    stats["v1_wins"] += 1
                elif right_score > left_score:
                    stats["v2_wins"] += 1
                else:
                    stats["ties"] += 1
            task_stats[category] = stats
        comparisons[model] = {
            "complete": complete,
            "categories": category_deltas,
            "macro": {
                metric: (
                    right["macro"][metric] - left["macro"][metric]
                    if complete
                    and right["macro"][metric] is not None
                    and left["macro"][metric] is not None
                    else None
                )
                for metric in ("pass_at_k", "avg_at_k")
            },
            "micro": {
                metric: (
                    right["micro"][metric] - left["micro"][metric]
                    if complete
                    and right["micro"][metric] is not None
                    and left["micro"][metric] is not None
                    else None
                )
                for metric in ("pass_at_k", "avg_at_k")
            },
            "tasks": task_stats,
        }
    return {
        "versions": versions,
        "models": models,
        "categories": categories,
        "results": combined,
        "comparisons": comparisons,
    }


def render_markdown(result: dict, k: int) -> str:
    """Render branch mapping, highlighted scores, deltas, and task outcomes."""
    versions = {version["version_id"]: version for version in result["versions"]}
    lines = [
        "## Unified evals — Version 1 vs Version 2",
        "",
        "| Version | Branch | Commit |",
        "|---|---|---|",
        f"| Version 1 | `{versions['v1']['branch']}` | `{versions['v1']['sha'][:12]}` |",
        f"| Version 2 | `{versions['v2']['branch']}` | `{versions['v2']['sha'][:12]}` |",
        "",
    ]
    for model in result["models"]:
        left = result["results"]["v1"]["models"][model]
        right = result["results"]["v2"]["models"][model]
        header = ["Version"] + [
            f"{cat} pass@{k}/avg@{k}" for cat in result["categories"]
        ]
        header += [f"macro pass@{k}/avg@{k}", f"micro pass@{k}/avg@{k}"]
        rows = {"v1": ["Version 1"], "v2": ["Version 2"]}
        for category in result["categories"]:
            left_cat = left["categories"].get(category)
            right_cat = right["categories"].get(category)
            highlight = bool(
                left_cat
                and right_cat
                and not left_cat["incomplete"]
                and not right_cat["incomplete"]
            )
            pass_pair = _pair(
                left_cat["pass_at_k"] if left_cat else None,
                right_cat["pass_at_k"] if right_cat else None,
                highlight=highlight,
            )
            avg_pair = _pair(
                left_cat["avg_at_k"] if left_cat else None,
                right_cat["avg_at_k"] if right_cat else None,
                highlight=highlight,
            )
            rows["v1"].append(f"{pass_pair[0]}/{avg_pair[0]}")
            rows["v2"].append(f"{pass_pair[1]}/{avg_pair[1]}")
        for scope in ("macro", "micro"):
            highlight = not left["incomplete"] and not right["incomplete"]
            pass_pair = _pair(
                left[scope]["pass_at_k"],
                right[scope]["pass_at_k"],
                highlight=highlight,
            )
            avg_pair = _pair(
                left[scope]["avg_at_k"],
                right[scope]["avg_at_k"],
                highlight=highlight,
            )
            rows["v1"].append(f"{pass_pair[0]}/{avg_pair[0]}")
            rows["v2"].append(f"{pass_pair[1]}/{avg_pair[1]}")
        lines += [
            f"### `{model}`",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
            "| " + " | ".join(rows["v1"]) + " |",
            "| " + " | ".join(rows["v2"]) + " |",
            "",
        ]
        comparison = result["comparisons"][model]
        delta_header = ["V2 − V1"] + list(result["categories"]) + ["macro", "micro"]
        for metric, label in (("pass_at_k", f"pass@{k}"), ("avg_at_k", f"avg@{k}")):
            values = [
                _fmt(comparison["categories"][cat][metric], signed=True)
                for cat in result["categories"]
            ]
            values += [
                _fmt(comparison["macro"][metric], signed=True),
                _fmt(comparison["micro"][metric], signed=True),
            ]
            if metric == "pass_at_k":
                lines += [
                    "| " + " | ".join(delta_header) + " |",
                    "|" + "|".join(["---"] * len(delta_header)) + "|",
                ]
            lines.append("| " + " | ".join([label, *values]) + " |")
        lines += [
            "",
            "| Category | V1 wins | V2 wins | Ties | Missing |",
            "|---|---:|---:|---:|---:|",
        ]
        for category in result["categories"]:
            stats = comparison["tasks"][category]
            lines.append(
                f"| {category} | {stats['v1_wins']} | {stats['v2_wins']} | "
                f"{stats['ties']} | {stats['missing']} |"
            )
        if not comparison["complete"]:
            lines += [
                "",
                "> ⚠️ Delta values are unavailable where either version is incomplete.",
            ]
        lines.append("")
    return "\n".join(lines) + "\n"


def _slug(value: str) -> str:
    """Return a collision-resistant filename slug."""
    safe = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "model"
    return f"{safe[:56]}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def write_outputs(result: dict, k: int, out_dir: Path) -> None:
    """Write JSON, Markdown, and one two-series radar input per complete model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    markdown = render_markdown(result, k)
    (out_dir / "comparison.md").write_text(markdown)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as summary:
            summary.write(markdown)
    radar_dir = out_dir / "radar-inputs"
    chart_manifest: dict[str, str] = {}
    versions = {version["version_id"]: version for version in result["versions"]}
    for model in result["models"]:
        if (
            not result["comparisons"][model]["complete"]
            or len(result["categories"]) < 3
        ):
            continue
        radar_dir.mkdir(exist_ok=True)
        payload = []
        for version_id, label in (("v1", "Version 1"), ("v2", "Version 2")):
            model_result = result["results"][version_id]["models"][model]
            payload.append(
                {
                    "model": f"{label} — {versions[version_id]['branch']}",
                    "scores": {
                        category: model_result["categories"][category]["pass_at_k"]
                        for category in result["categories"]
                    },
                }
            )
        slug = _slug(model)
        (radar_dir / f"{slug}.json").write_text(json.dumps(payload, indent=2) + "\n")
        chart_manifest[slug] = model
    if chart_manifest:
        (out_dir / "chart_manifest.json").write_text(
            json.dumps(chart_manifest, indent=2) + "\n"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--versions-json", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--categories-json", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    versions = _versions(args.versions_json)
    models = _string_list(args.models_json, "--models-json")
    categories = _string_list(args.categories_json, "--categories-json")
    leaves, tasks = discover(args.root, args.rollouts)
    result = compare(leaves, tasks, versions, models, categories)
    write_outputs(result, args.rollouts, args.out_dir)
    incomplete = [
        model for model in models if not result["comparisons"][model]["complete"]
    ]
    if incomplete:
        print(
            f"::error::Incomplete Version 1/Version 2 comparison for: {', '.join(incomplete)}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
