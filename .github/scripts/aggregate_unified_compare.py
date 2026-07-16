"""Compare self-contained Unified Eval bundles across immutable branch versions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from itertools import combinations
from pathlib import Path
from typing import cast

import aggregate_unified as unified


def _json_list(raw: str, label: str) -> list[object]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return cast(list[object], value)


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"invalid run bundle manifest: {path}")
    required = {
        "version_id",
        "source_branch",
        "source_sha",
        "model",
        "config",
        "categories",
    }
    if not required <= set(value):
        raise ValueError(f"run bundle manifest is missing fields: {path}")
    return cast(dict[str, object], value)


def _discover(root: Path) -> dict[tuple[str, str, str], tuple[Path, dict[str, object]]]:
    bundles: dict[tuple[str, str, str], tuple[Path, dict[str, object]]] = {}
    for path in sorted(root.rglob("manifest.json")):
        manifest = _load_manifest(path)
        identity_values = (
            manifest["version_id"],
            manifest["model"],
            manifest["config"],
        )
        if not all(isinstance(value, str) and value for value in identity_values):
            raise ValueError(f"invalid bundle identity: {path}")
        identity = cast(tuple[str, str, str], identity_values)
        if identity in bundles:
            raise ValueError(f"duplicate run bundle for {identity!r}")
        bundles[identity] = (path.parent, manifest)
    return bundles


def _task_results(path: Path, k: int) -> dict[str, float]:
    results: dict[str, float] = {}
    if not path.is_file():
        return results
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("task"), str):
            raise ValueError(f"invalid per-task row at {path}:{number}")
        score = value.get(f"pass@{k}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"invalid pass@{k} at {path}:{number}")
        numeric = float(score)
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError(f"invalid pass@{k} at {path}:{number}")
        results[cast(str, value["task"])] = numeric
    return results


def _row_from_bundle(
    bundle: tuple[Path, dict[str, object]] | None,
    *,
    source: dict[str, str],
    model: str,
    config: str,
    categories: list[str],
    rollouts: int,
) -> dict[str, object]:
    leaves: list[dict] = []
    tasks: dict[str, dict[str, float]] = {}
    if bundle is not None:
        root, manifest = bundle
        if (
            manifest["source_branch"] != source["branch"]
            or manifest["source_sha"] != source["sha"]
        ):
            raise ValueError(
                f"bundle source mismatch for {source['version_id']}/{model}/{config}"
            )
        records = manifest["categories"]
        if not isinstance(records, dict):
            raise ValueError("bundle categories must be an object")
        for category in categories:
            record = records.get(category)
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            leaf_dir = root / cast(str, record["path"])
            leaf = unified.read_leaf(leaf_dir, expected_rollouts=rollouts)
            leaf["config"] = config
            leaves.append(leaf)
            tasks[category] = _task_results(leaf_dir / "per_task.jsonl", rollouts)

    expected = [
        {"model": model, "config": config, "category": category}
        for category in categories
    ]
    combined = unified.combine(leaves, expected, categories)
    result = combined["rows"][0]
    return {
        "version_id": source["version_id"],
        "branch": source["branch"],
        "sha": source["sha"],
        "model": model,
        "config": config,
        **result,
        "tasks": tasks,
    }


def _metric(row: dict[str, object], key: str) -> float | None:
    scope, metric = key.split(".", 1)
    if scope in {"macro", "micro"}:
        block = cast(dict[str, object], row[scope])
    else:
        categories = cast(dict[str, dict[str, object]], row["categories"])
        block = categories.get(scope, {})
    value = block.get(metric)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _metric_keys(categories: list[str]) -> list[str]:
    return [
        *(
            f"{category}.{metric}"
            for category in categories
            for metric in ("pass_at_k", "avg_at_k")
        ),
        "macro.pass_at_k",
        "macro.avg_at_k",
        "micro.pass_at_k",
        "micro.avg_at_k",
    ]


def _pairwise(
    rows: list[dict[str, object]], sources: list[dict[str, str]], categories: list[str]
) -> list[dict[str, object]]:
    by_subject = {
        (
            cast(str, row["version_id"]),
            cast(str, row["model"]),
            cast(str, row["config"]),
        ): row
        for row in rows
    }
    subjects = sorted(
        {(cast(str, row["model"]), cast(str, row["config"])) for row in rows}
    )
    output: list[dict[str, object]] = []
    for first, second in combinations(sources, 2):
        for model, config in subjects:
            a = by_subject[(first["version_id"], model, config)]
            b = by_subject[(second["version_id"], model, config)]
            deltas: dict[str, float | None] = {}
            for key in _metric_keys(categories):
                av = _metric(a, key)
                bv = _metric(b, key)
                deltas[key] = None if av is None or bv is None else bv - av
            task_outcomes: dict[str, dict[str, int]] = {}
            for category in categories:
                a_tasks = cast(dict[str, dict[str, float]], a["tasks"]).get(
                    category, {}
                )
                b_tasks = cast(dict[str, dict[str, float]], b["tasks"]).get(
                    category, {}
                )
                counts = {"wins": 0, "losses": 0, "ties": 0, "missing": 0}
                for task in sorted(set(a_tasks) | set(b_tasks)):
                    if task not in a_tasks or task not in b_tasks:
                        counts["missing"] += 1
                    elif b_tasks[task] > a_tasks[task]:
                        counts["wins"] += 1
                    elif b_tasks[task] < a_tasks[task]:
                        counts["losses"] += 1
                    else:
                        counts["ties"] += 1
                task_outcomes[category] = counts
            output.append(
                {
                    "from": first,
                    "to": second,
                    "model": model,
                    "config": config,
                    "deltas": deltas,
                    "task_outcomes": task_outcomes,
                }
            )
    return output


def compare(
    root: Path,
    *,
    sources: list[dict[str, str]],
    models: list[str],
    configs: list[str],
    categories: list[str],
    rollouts: int,
) -> dict[str, object]:
    """Build every expected subject row and all source-pair deltas."""
    bundles = _discover(root)
    rows = [
        _row_from_bundle(
            bundles.get((source["version_id"], model, config)),
            source=source,
            model=model,
            config=config,
            categories=categories,
            rollouts=rollouts,
        )
        for source in sources
        for model in models
        for config in configs
    ]
    return {
        "schema_version": 1,
        "rollouts_per_task": rollouts,
        "sources": sources,
        "models": models,
        "configs": configs,
        "categories": categories,
        "rows": rows,
        "pairwise": _pairwise(rows, sources, categories),
    }


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(result: dict[str, object]) -> str:
    """Render highlighted absolute metrics and every pairwise delta."""
    rows = cast(list[dict[str, object]], result["rows"])
    categories = cast(list[str], result["categories"])
    k = cast(int, result["rollouts_per_task"])
    keys = _metric_keys(categories)
    maxima: dict[tuple[str, str, str], float] = {}
    for row in rows:
        group = (cast(str, row["model"]), cast(str, row["config"]))
        for key in keys:
            value = _metric(row, key)
            if value is not None:
                maxima[(*group, key)] = max(value, maxima.get((*group, key), value))

    header = (
        ["Version", "Branch", "Model", "Config"]
        + [f"{category} pass@{k}/avg@{k}" for category in categories]
        + [f"macro pass@{k}/avg@{k}", f"micro pass@{k}/avg@{k}"]
    )
    lines = [
        "## Compared sources",
        "",
        "| Version | Branch | Commit |",
        "|---|---|---|",
    ]
    for source in cast(list[dict[str, str]], result["sources"]):
        lines.append(
            f"| {_md(source['version_id'])} | {_md(source['branch'])} | `{_md(source['sha'])}` |"
        )
    lines.extend(
        [
            "",
            "## Unified evals — branch comparison",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
    )

    def highlighted(row: dict[str, object], key: str) -> str:
        value = _metric(row, key)
        text = _fmt(value)
        group_key = (cast(str, row["model"]), cast(str, row["config"]), key)
        return (
            f"**{text}**"
            if value is not None and value == maxima.get(group_key)
            else text
        )

    for row in rows:
        cells = [
            _md(row["version_id"]),
            _md(row["branch"]),
            _md(row["model"]),
            _md(row["config"]) + (" ⚠️" if row["incomplete"] else ""),
        ]
        cells.extend(
            f"{highlighted(row, f'{category}.pass_at_k')}/{highlighted(row, f'{category}.avg_at_k')}"
            for category in categories
        )
        cells.extend(
            [
                f"{highlighted(row, 'macro.pass_at_k')}/{highlighted(row, 'macro.avg_at_k')}",
                f"{highlighted(row, 'micro.pass_at_k')}/{highlighted(row, 'micro.avg_at_k')}",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")

    pairwise = cast(list[dict[str, object]], result["pairwise"])
    if pairwise:
        lines.extend(
            [
                "",
                "## Pairwise deltas",
                "",
                "Positive means the second version scored higher.",
                "",
            ]
        )
        delta_header = (
            ["Comparison", "Model", "Config"]
            + [f"Δ {category} pass/avg" for category in categories]
            + ["Δ macro pass/avg", "Δ micro pass/avg"]
        )
        lines.extend(
            [
                "| " + " | ".join(delta_header) + " |",
                "|" + "|".join(["---"] * len(delta_header)) + "|",
            ]
        )
        for pair in pairwise:
            first = cast(dict[str, str], pair["from"])
            second = cast(dict[str, str], pair["to"])
            deltas = cast(dict[str, float | None], pair["deltas"])
            cells = [
                f"{_md(first['version_id'])} → {_md(second['version_id'])}",
                _md(pair["model"]),
                _md(pair["config"]),
            ]
            cells.extend(
                f"{_fmt(deltas[f'{category}.pass_at_k'], signed=True)}/{_fmt(deltas[f'{category}.avg_at_k'], signed=True)}"
                for category in categories
            )
            cells.extend(
                [
                    f"{_fmt(deltas['macro.pass_at_k'], signed=True)}/{_fmt(deltas['macro.avg_at_k'], signed=True)}",
                    f"{_fmt(deltas['micro.pass_at_k'], signed=True)}/{_fmt(deltas['micro.avg_at_k'], signed=True)}",
                ]
            )
            lines.append("| " + " | ".join(cells) + " |")

        lines.extend(
            [
                "",
                "## Per-task outcomes",
                "",
                "Wins/losses are from the second version’s perspective.",
                "",
                "| Comparison | Model | Config | Category | Wins | Losses | Ties | Missing |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for pair in pairwise:
            first = cast(dict[str, str], pair["from"])
            second = cast(dict[str, str], pair["to"])
            outcomes = cast(dict[str, dict[str, int]], pair["task_outcomes"])
            for category in categories:
                counts = outcomes[category]
                lines.append(
                    f"| {_md(first['version_id'])} → {_md(second['version_id'])} | {_md(pair['model'])} | {_md(pair['config'])} | {_md(category)} | {counts['wins']} | {counts['losses']} | {counts['ties']} | {counts['missing']} |"
                )
    return "\n".join(lines) + "\n"


def _slug(value: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "value"
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{safe[:48]}-{digest}"


def write_outputs(result: dict[str, object], out_dir: Path) -> None:
    """Write machine-readable comparison, radar inputs, and job summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(result)
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)

    categories = cast(list[str], result["categories"])
    if len(categories) < 3:
        return
    rows = cast(list[dict[str, object]], result["rows"])
    radar = [
        {
            "model": f"{row['version_id']} {row['branch']} / {row['model']} / {row['config']}",
            "scores": {
                category: cast(dict[str, object], row["categories"])
                .get(category, {})
                .get("pass_at_k")
                for category in categories
                if cast(dict[str, object], row["categories"])
                .get(category, {})
                .get("pass_at_k")
                is not None
            },
        }
        for row in rows
    ]
    (out_dir / "radar_results.json").write_text(
        json.dumps(radar, indent=2) + "\n", encoding="utf-8"
    )
    by_config = out_dir / "radar_by_config"
    by_config.mkdir()
    for config in cast(list[str], result["configs"]):
        selected = [
            item
            for item, row in zip(radar, rows, strict=True)
            if row["config"] == config
        ]
        (by_config / f"{_slug(config)}.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )


def _sources(raw: str) -> list[dict[str, str]]:
    values = _json_list(raw, "--sources-json")
    sources: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("--sources-json entries must be objects")
        source = {key: value.get(key) for key in ("version_id", "branch", "sha")}
        if not all(isinstance(item, str) and item for item in source.values()):
            raise ValueError(
                "--sources-json entries require version_id, branch, and sha"
            )
        sources.append(cast(dict[str, str], source))
    return sources


def _strings(raw: str, label: str) -> list[str]:
    values = _json_list(raw, label)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{label} must contain non-empty strings")
    return cast(list[str], values)


def main(argv: list[str] | None = None) -> int:
    """CLI for the comparison workflow's final aggregation job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources-json", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--configs-json", required=True)
    parser.add_argument("--categories-json", required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")
    result = compare(
        args.root,
        sources=_sources(args.sources_json),
        models=_strings(args.models_json, "--models-json"),
        configs=_strings(args.configs_json, "--configs-json"),
        categories=_strings(args.categories_json, "--categories-json"),
        rollouts=args.rollouts,
    )
    write_outputs(result, args.out_dir)
    rows = cast(list[dict[str, object]], result["rows"])
    return 1 if rows and all(row["incomplete"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
