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
import collect_langsmith_usage as langsmith_usage


USAGE_METRICS = (
    "prompt_tokens_per_rollout",
    "completion_tokens_per_rollout",
    "total_tokens_per_rollout",
    "cost_usd_per_rollout",
)


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


def _load_usage(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    value = _load_json_object(path, "usage summary")
    if value.get("schema_version") != 1 or not isinstance(
        value.get("experiments"), dict
    ):
        raise ValueError(f"invalid usage summary: {path}")
    experiments = cast(dict[str, object], value["experiments"])
    if not all(
        isinstance(name, str) and isinstance(item, dict)
        for name, item in experiments.items()
    ):
        raise ValueError(f"invalid experiment usage entries: {path}")
    return cast(dict[str, dict[str, object]], experiments)


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, object], value)


def _category_usage(
    leaf_dir: Path,
    runtime: str | None,
    experiments: dict[str, dict[str, object]],
) -> dict[str, object]:
    summary = _load_json_object(leaf_dir / "summary.json", "category summary")
    experiment = summary.get("langsmith_experiment")
    expected = langsmith_usage.expected_rollouts(summary)
    if not isinstance(experiment, str) or not experiment:
        block = langsmith_usage.unavailable_usage(expected)
        return {"runtime": runtime, "experiment": None, **block}
    block = experiments.get(experiment) or langsmith_usage.unavailable_usage(expected)
    return {"runtime": runtime, "experiment": experiment, **block}


def _usage_int(block: dict[str, object], field: str) -> int:
    coverage = cast(dict[str, object], block["coverage"])
    value = coverage.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage_total(block: dict[str, object], field: str) -> float | int | None:
    totals = cast(dict[str, object], block["totals"])
    value = totals.get(field)
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _overall_usage(categories: dict[str, dict[str, object]]) -> dict[str, object]:
    unique: dict[str, dict[str, object]] = {}
    for block in categories.values():
        experiment = block.get("experiment")
        if isinstance(experiment, str) and experiment:
            unique.setdefault(experiment, block)
    if not unique:
        return {"experiments": [], **langsmith_usage.unavailable_usage(None)}

    blocks = list(unique.values())
    expected_values = [
        cast(dict[str, object], block["coverage"]).get("expected_rollouts")
        for block in blocks
    ]
    expected = (
        sum(cast(list[int], expected_values))
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in expected_values
        )
        else None
    )
    observed = sum(_usage_int(block, "observed_rollouts") for block in blocks)
    token_rollouts = sum(_usage_int(block, "token_rollouts") for block in blocks)
    priced_rollouts = sum(_usage_int(block, "priced_rollouts") for block in blocks)
    errored_rollouts = sum(_usage_int(block, "errored_rollouts") for block in blocks)

    prompt = sum(
        cast(int, value)
        for block in blocks
        if (value := _usage_total(block, "prompt_tokens")) is not None
    )
    completion = sum(
        cast(int, value)
        for block in blocks
        if (value := _usage_total(block, "completion_tokens")) is not None
    )
    total = sum(
        cast(int, value)
        for block in blocks
        if (value := _usage_total(block, "total_tokens")) is not None
    )
    cost = sum(
        float(value)
        for block in blocks
        if (value := _usage_total(block, "cost_usd")) is not None
    )
    statuses = {block.get("status") for block in blocks}
    status = (
        "complete"
        if statuses == {"complete"}
        else "partial"
        if observed
        else "unavailable"
    )
    return {
        "experiments": sorted(unique),
        "status": status,
        "coverage": {
            "expected_rollouts": expected,
            "observed_rollouts": observed,
            "token_rollouts": token_rollouts,
            "priced_rollouts": priced_rollouts,
            "errored_rollouts": errored_rollouts,
        },
        "totals": {
            "prompt_tokens": prompt if token_rollouts else None,
            "completion_tokens": completion if token_rollouts else None,
            "total_tokens": total if token_rollouts else None,
            "cost_usd": cost if priced_rollouts else None,
        },
        "averages": {
            "prompt_tokens_per_rollout": (
                prompt / token_rollouts if token_rollouts else None
            ),
            "completion_tokens_per_rollout": (
                completion / token_rollouts if token_rollouts else None
            ),
            "total_tokens_per_rollout": (
                total / token_rollouts if token_rollouts else None
            ),
            "cost_usd_per_rollout": cost / priced_rollouts if priced_rollouts else None,
        },
    }


def _row_from_bundle(
    bundle: tuple[Path, dict[str, object]] | None,
    *,
    source: dict[str, str],
    model: str,
    config: str,
    categories: list[str],
    rollouts: int,
    usage_by_experiment: dict[str, dict[str, object]],
) -> dict[str, object]:
    leaves: list[dict] = []
    tasks: dict[str, dict[str, float]] = {}
    category_usage: dict[str, dict[str, object]] = {}
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
        category_records = cast(dict[str, object], records)
        for category in categories:
            record_value = category_records.get(category)
            record = (
                cast(dict[str, object], record_value)
                if isinstance(record_value, dict)
                else None
            )
            runtime = (
                cast(str, record.get("runtime"))
                if record is not None and isinstance(record.get("runtime"), str)
                else None
            )
            if record is None or not isinstance(record.get("path"), str):
                category_usage[category] = {
                    "runtime": runtime,
                    "experiment": None,
                    **langsmith_usage.unavailable_usage(None),
                }
                continue
            leaf_dir = root / cast(str, record["path"])
            leaf = unified.read_leaf(leaf_dir, expected_rollouts=rollouts)
            leaf["config"] = config
            leaves.append(leaf)
            tasks[category] = _task_results(leaf_dir / "per_task.jsonl", rollouts)
            category_usage[category] = _category_usage(
                leaf_dir, runtime, usage_by_experiment
            )

    for category in categories:
        category_usage.setdefault(
            category,
            {
                "runtime": None,
                "experiment": None,
                **langsmith_usage.unavailable_usage(None),
            },
        )

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
        "usage": {
            "categories": category_usage,
            "overall": _overall_usage(category_usage),
        },
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
            # Trial errors are infrastructure/runtime failures, not scores. Do not
            # manufacture a delta or win/loss result from an invalid subject.
            if a.get("invalid") or b.get("invalid"):
                continue
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


def _usage_scope(row: dict[str, object], scope: str) -> dict[str, object]:
    usage = cast(dict[str, object], row["usage"])
    if scope == "overall":
        return cast(dict[str, object], usage["overall"])
    categories = cast(dict[str, dict[str, object]], usage["categories"])
    return categories[scope]


def _usage_average(block: dict[str, object], metric: str) -> float | None:
    averages = cast(dict[str, object], block["averages"])
    value = averages.get(metric)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _usage_metric_delta(
    first: dict[str, object], second: dict[str, object], metric: str
) -> dict[str, float | None]:
    baseline = _usage_average(first, metric)
    candidate = _usage_average(second, metric)
    absolute = (
        candidate - baseline if baseline is not None and candidate is not None else None
    )
    percent = (
        absolute / baseline * 100
        if absolute is not None and baseline not in (None, 0.0)
        else None
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute": absolute,
        "percent": percent,
    }


def _usage_scopes(
    first: dict[str, object], second: dict[str, object], categories: list[str]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for scope in [*categories, "overall"]:
        first_block = _usage_scope(first, scope)
        second_block = _usage_scope(second, scope)
        first_experiment = first_block.get("experiment")
        second_experiment = second_block.get("experiment")
        shared = (
            scope != "overall"
            and isinstance(first_experiment, str)
            and first_experiment == second_experiment
        )
        output[scope] = {
            "shared_experiment": shared,
            "from_coverage": first_block["coverage"],
            "to_coverage": second_block["coverage"],
            "metrics": {
                metric: _usage_metric_delta(first_block, second_block, metric)
                for metric in USAGE_METRICS
            },
        }
    return output


def _usage_identity(row: dict[str, object]) -> dict[str, str]:
    return {
        "version_id": cast(str, row["version_id"]),
        "branch": cast(str, row["branch"]),
        "config": cast(str, row["config"]),
    }


def _usage_comparisons(
    rows: list[dict[str, object]],
    sources: list[dict[str, str]],
    models: list[str],
    configs: list[str],
    categories: list[str],
) -> list[dict[str, object]]:
    by_subject = {
        (
            cast(str, row["version_id"]),
            cast(str, row["model"]),
            cast(str, row["config"]),
        ): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for first_source, second_source in combinations(sources, 2):
        for model in models:
            for config in configs:
                first = by_subject[(first_source["version_id"], model, config)]
                second = by_subject[(second_source["version_id"], model, config)]
                output.append(
                    {
                        "kind": "cross_branch",
                        "model": model,
                        "from": _usage_identity(first),
                        "to": _usage_identity(second),
                        "scopes": _usage_scopes(first, second, categories),
                    }
                )
    for source in sources:
        for model in models:
            for first_config, second_config in combinations(configs, 2):
                first = by_subject[(source["version_id"], model, first_config)]
                second = by_subject[(source["version_id"], model, second_config)]
                output.append(
                    {
                        "kind": "within_branch",
                        "model": model,
                        "from": _usage_identity(first),
                        "to": _usage_identity(second),
                        "scopes": _usage_scopes(first, second, categories),
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
    usage_by_experiment: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build every expected subject row and all source-pair deltas."""
    bundles = _discover(root)
    experiment_usage = usage_by_experiment or {}
    rows = [
        _row_from_bundle(
            bundles.get((source["version_id"], model, config)),
            source=source,
            model=model,
            config=config,
            categories=categories,
            rollouts=rollouts,
            usage_by_experiment=experiment_usage,
        )
        for source in sources
        for model in models
        for config in configs
    ]
    return {
        "schema_version": 2,
        "rollouts_per_task": rollouts,
        "sources": sources,
        "models": models,
        "configs": configs,
        "categories": categories,
        "rows": rows,
        "pairwise": _pairwise(rows, sources, categories),
        "usage_comparisons": _usage_comparisons(
            rows, sources, models, configs, categories
        ),
    }


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt_usage(value: float | None, *, cost: bool = False, signed: bool = False) -> str:
    if value is None:
        return "—"
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.6f}" if cost else f"{prefix}{value:.1f}"


def _fmt_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def _coverage_text(block: dict[str, object]) -> tuple[str, str, str, str]:
    coverage = cast(dict[str, object], block["coverage"])
    observed = coverage.get("observed_rollouts")
    expected = coverage.get("expected_rollouts")
    token = coverage.get("token_rollouts")
    priced = coverage.get("priced_rollouts")
    errored = coverage.get("errored_rollouts")
    expected_text = str(expected) if isinstance(expected, int) else "—"
    return (
        f"{observed}/{expected_text}",
        f"{token}/{observed}",
        f"{priced}/{observed}",
        str(errored),
    )


def _usage_markdown(result: dict[str, object]) -> list[str]:
    rows = cast(list[dict[str, object]], result["rows"])
    categories = cast(list[str], result["categories"])
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        usage = cast(dict[str, object], row["usage"])
        category_blocks = cast(dict[str, dict[str, object]], usage["categories"])
        for category in categories:
            block = category_blocks[category]
            experiment = block.get("experiment")
            key = (
                cast(str, row["version_id"]),
                cast(str, row["branch"]),
                cast(str, row["model"]),
                category,
                experiment if isinstance(experiment, str) else "",
                "" if isinstance(experiment, str) else cast(str, row["config"]),
            )
            entry = grouped.setdefault(
                key,
                {
                    "version_id": row["version_id"],
                    "branch": row["branch"],
                    "model": row["model"],
                    "scope": category,
                    "runtime": block.get("runtime"),
                    "configs": [],
                    "block": block,
                },
            )
            cast(list[str], entry["configs"]).append(cast(str, row["config"]))
        overall = cast(dict[str, object], usage["overall"])
        grouped[
            (
                cast(str, row["version_id"]),
                cast(str, row["branch"]),
                cast(str, row["model"]),
                "overall",
                cast(str, row["config"]),
                "",
            )
        ] = {
            "version_id": row["version_id"],
            "branch": row["branch"],
            "model": row["model"],
            "scope": "overall",
            "runtime": "—",
            "configs": [row["config"]],
            "block": overall,
        }

    lines = [
        "",
        "## Usage and cost",
        "",
        "Averages use only rollouts carrying that metric; coverage columns show the denominators.",
        "",
        "| Version | Branch | Model | Config(s) | Scope | Runtime | Status | Traces | Token coverage | Price coverage | Errors | Avg input tokens | Avg output tokens | Avg total tokens | Avg cost (USD) |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in grouped.values():
        block = cast(dict[str, object], entry["block"])
        averages = cast(dict[str, object], block["averages"])
        traces, tokens, prices, errors = _coverage_text(block)
        configs = ", ".join(dict.fromkeys(cast(list[str], entry["configs"])))
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(entry["version_id"]),
                    _md(entry["branch"]),
                    _md(entry["model"]),
                    _md(configs),
                    _md(entry["scope"]),
                    _md(entry["runtime"] or "—"),
                    _md(block["status"]),
                    traces,
                    tokens,
                    prices,
                    errors,
                    _fmt_usage(
                        cast(float | None, averages["prompt_tokens_per_rollout"])
                    ),
                    _fmt_usage(
                        cast(float | None, averages["completion_tokens_per_rollout"])
                    ),
                    _fmt_usage(
                        cast(float | None, averages["total_tokens_per_rollout"])
                    ),
                    _fmt_usage(
                        cast(float | None, averages["cost_usd_per_rollout"]),
                        cost=True,
                    ),
                ]
            )
            + " |"
        )

    comparisons = cast(list[dict[str, object]], result["usage_comparisons"])
    if not comparisons:
        return lines
    lines.extend(
        [
            "",
            "## Usage and cost deltas",
            "",
            "Delta is candidate minus baseline; percentage uses the baseline denominator.",
            "",
            "| Kind | Comparison | Model | Scope | Shared experiment | Δ avg input | Δ avg output | Δ avg total | Δ total % | Δ avg cost (USD) | Δ cost % |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in comparisons:
        first = cast(dict[str, str], comparison["from"])
        second = cast(dict[str, str], comparison["to"])
        scopes = cast(dict[str, dict[str, object]], comparison["scopes"])
        label = (
            f"{first['version_id']}/{first['config']} → "
            f"{second['version_id']}/{second['config']}"
        )
        for scope, block in scopes.items():
            shared = bool(block["shared_experiment"])
            metrics = cast(dict[str, dict[str, float | None]], block["metrics"])
            if shared:
                input_delta = output_delta = total_delta = total_percent = "shared"
                cost_delta = cost_percent = "shared"
            else:
                input_delta = _fmt_usage(
                    metrics["prompt_tokens_per_rollout"]["absolute"], signed=True
                )
                output_delta = _fmt_usage(
                    metrics["completion_tokens_per_rollout"]["absolute"], signed=True
                )
                total_delta = _fmt_usage(
                    metrics["total_tokens_per_rollout"]["absolute"], signed=True
                )
                total_percent = _fmt_percent(
                    metrics["total_tokens_per_rollout"]["percent"]
                )
                cost_delta = _fmt_usage(
                    metrics["cost_usd_per_rollout"]["absolute"],
                    cost=True,
                    signed=True,
                )
                cost_percent = _fmt_percent(metrics["cost_usd_per_rollout"]["percent"])
            lines.append(
                f"| {_md(comparison['kind'])} | {_md(label)} | "
                f"{_md(comparison['model'])} | {_md(scope)} | "
                f"{'yes' if shared else 'no'} | {input_delta} | {output_delta} | "
                f"{total_delta} | {total_percent} | {cost_delta} | {cost_percent} |"
            )
    return lines


def render_markdown(result: dict[str, object]) -> str:
    """Render highlighted absolute metrics and every pairwise delta."""
    rows = cast(list[dict[str, object]], result["rows"])
    categories = cast(list[str], result["categories"])
    k = cast(int, result["rollouts_per_task"])
    keys = _metric_keys(categories)
    maxima: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("invalid"):
            continue
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
        status = (
            " ❌ invalid" if row.get("invalid") else " ⚠️" if row["incomplete"] else ""
        )
        cells = [
            _md(row["version_id"]),
            _md(row["branch"]),
            _md(row["model"]),
            _md(row["config"]) + status,
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

    invalids = [row for row in rows if row.get("invalid")]
    if invalids:
        lines.extend(
            [
                "",
                "> ❌ **Invalid results were excluded from highlights, deltas, and charts because one or more trials errored:**",
            ]
        )
        for row in invalids:
            lines.append(
                f"> - {_md(row['version_id'])} / `{_md(row['branch'])}` / "
                f"`{_md(row['model'])}` / `{_md(row['config'])}`"
            )

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
    lines.extend(_usage_markdown(result))
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
    radar_pairs = [
        (
            row,
            {
                "model": f"{row['version_id']} {row['branch']} / {row['model']} / {row['config']}",
                "scores": {
                    category: category_row.get("pass_at_k")
                    for category in categories
                    if (
                        category_row := cast(
                            dict[str, dict[str, object]], row["categories"]
                        ).get(category, {})
                    ).get("pass_at_k")
                    is not None
                },
            },
        )
        for row in rows
        if not row.get("invalid")
    ]
    radar = [item for _row, item in radar_pairs]
    (out_dir / "radar_results.json").write_text(
        json.dumps(radar, indent=2) + "\n", encoding="utf-8"
    )
    by_config = out_dir / "radar_by_config"
    by_config.mkdir()
    for config in cast(list[str], result["configs"]):
        selected = [item for row, item in radar_pairs if row["config"] == config]
        (by_config / f"{_slug(config)}.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )


def _sources(raw: str) -> list[dict[str, str]]:
    values = _json_list(raw, "--sources-json")
    sources: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("--sources-json entries must be objects")
        source_value = cast(dict[str, object], value)
        source = {key: source_value.get(key) for key in ("version_id", "branch", "sha")}
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
    parser.add_argument("--usage-json", type=Path)
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
        usage_by_experiment=_load_usage(args.usage_json),
    )
    write_outputs(result, args.out_dir)
    rows = cast(list[dict[str, object]], result["rows"])
    return 1 if rows and all(row["incomplete"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
