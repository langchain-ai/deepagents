"""Compare Unified Eval results across branches and active agent configs."""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, cast

import aggregate_unified as unified
from unified_types import LeafKey


METRICS = ("pass_at_k", "avg_at_k")


class SubjectKey(NamedTuple):
    """Identity of one branch, model, and active agent config."""

    branch: str
    model: str
    config: str


class LeafData(NamedTuple):
    """Validated aggregate leaf plus its per-task pass results."""

    leaf: dict
    tasks: dict[str, float]
    has_tasks: bool


def _json_list(raw: str, label: str) -> list[object]:
    """Decode a JSON list argument with a consistent error."""
    value = json.loads(raw)
    if not isinstance(value, list):
        msg = f"{label} must be a JSON list"
        raise ValueError(msg)
    return cast(list[object], value)


def parse_sources(raw: str) -> list[dict[str, str]]:
    """Parse ordered branch and immutable-SHA identities from prep output."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in _json_list(raw, "--sources-json"):
        if not isinstance(value, dict):
            msg = "--sources-json entries must be objects"
            raise ValueError(msg)
        source = cast(dict[str, object], value)
        branch = source.get("branch")
        sha = source.get("sha")
        if not isinstance(branch, str) or not branch or not isinstance(sha, str):
            msg = "--sources-json entries require a branch and string sha"
            raise ValueError(msg)
        if branch in seen:
            msg = f"duplicate source branch: {branch!r}"
            raise ValueError(msg)
        seen.add(branch)
        sources.append({"branch": branch, "sha": sha})
    if not sources:
        msg = "--sources-json must contain at least one source"
        raise ValueError(msg)
    return sources


def parse_expected_leaves(raw: str) -> list[dict[str, str]]:
    """Parse the authoritative branch/model/config/category allocation table."""
    fields = {"model", "branch", "source_sha", "config", "category"}
    leaves: list[dict[str, str]] = []
    seen: set[LeafKey] = set()
    for value in _json_list(raw, "--expected-leaves-json"):
        if not isinstance(value, dict) or not fields <= set(value):
            msg = (
                "--expected-leaves-json entries require model, branch, "
                "source_sha, config, and category"
            )
            raise ValueError(msg)
        raw_leaf = cast(dict[str, object], value)
        leaf = {field: raw_leaf[field] for field in fields}
        if not all(isinstance(item, str) for item in leaf.values()):
            msg = "--expected-leaves-json fields must be strings"
            raise ValueError(msg)
        typed = cast(dict[str, str], leaf)
        key = LeafKey(
            typed["model"], typed["branch"], typed["config"], typed["category"]
        )
        if key in seen:
            msg = f"duplicate expected leaf: {key!r}"
            raise ValueError(msg)
        seen.add(key)
        leaves.append(typed)
    if not leaves:
        msg = "--expected-leaves-json must contain at least one leaf"
        raise ValueError(msg)
    return leaves


def parse_categories(raw: str) -> list[str]:
    """Parse the ordered category list from prep output."""
    values = _json_list(raw, "--categories-json")
    if not all(isinstance(value, str) and value for value in values):
        msg = "--categories-json must contain non-empty strings"
        raise ValueError(msg)
    return list(dict.fromkeys(cast(list[str], values)))


def _read_tasks(path: Path, rollouts: int) -> tuple[dict[str, float], bool]:
    """Read per-task pass@K values; a missing file is incomplete, not malformed."""
    if not path.is_file():
        return {}, False
    tasks: dict[str, float] = {}
    field = f"pass@{rollouts}"
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except ValueError as exc:
            msg = f"invalid per-task JSON at {path}:{number}"
            raise ValueError(msg) from exc
        if not isinstance(value, dict):
            msg = f"per-task row must be an object at {path}:{number}"
            raise ValueError(msg)
        row = cast(dict[str, object], value)
        task = row.get("task")
        score = row.get(field)
        if not isinstance(task, str) or not task:
            msg = f"per-task row requires a task at {path}:{number}"
            raise ValueError(msg)
        if task in tasks:
            msg = f"duplicate task {task!r} in {path}"
            raise ValueError(msg)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            msg = f"{field} must be numeric at {path}:{number}"
            raise ValueError(msg)
        numeric = float(score)
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            msg = f"{field} must be in [0, 1] at {path}:{number}"
            raise ValueError(msg)
        tasks[task] = numeric
    return tasks, True


def _actual_leaves(root: Path, rollouts: int) -> dict[LeafKey, LeafData]:
    """Index validated result leaves by their complete evaluation identity."""
    output: dict[LeafKey, LeafData] = {}
    for record in unified.discover_leaf_records(root, expected_rollouts=rollouts):
        leaf = record.leaf
        key = LeafKey(
            cast(str, leaf["model"]),
            cast(str, leaf["branch"]),
            cast(str, leaf["config"]),
            cast(str, leaf["category"]),
        )
        if key in output:
            msg = f"duplicate actual leaf: {key!r}"
            raise ValueError(msg)
        tasks, has_tasks = _read_tasks(record.path / "per_task.jsonl", rollouts)
        output[key] = LeafData(leaf, tasks, has_tasks)
    return output


def _source_index(sources: list[dict[str, str]]) -> dict[str, str]:
    """Return branch-to-SHA lookup preserving validation in one place."""
    return {source["branch"]: source["sha"] for source in sources}


def _allocation(
    expected_leaves: list[dict[str, str]], sources: list[dict[str, str]]
) -> tuple[list[SubjectKey], dict[SubjectKey, list[str]]]:
    """Build ordered subjects and their assigned categories from expected leaves."""
    source_shas = _source_index(sources)
    order: list[SubjectKey] = []
    categories: dict[SubjectKey, list[str]] = {}
    for leaf in expected_leaves:
        branch = leaf["branch"]
        if branch not in source_shas:
            msg = f"expected leaf references unknown source branch {branch!r}"
            raise ValueError(msg)
        if leaf["source_sha"] != source_shas[branch]:
            msg = f"expected source SHA mismatch for branch {branch!r}"
            raise ValueError(msg)
        key = SubjectKey(branch, leaf["model"], leaf["config"])
        if key not in categories:
            order.append(key)
            categories[key] = []
        categories[key].append(leaf["category"])
    return order, categories


def _subject_identity(subject: dict[str, object]) -> dict[str, str]:
    """Return the stable identity fields used in comparison records."""
    return {
        "branch": cast(str, subject["branch"]),
        "source_sha": cast(str, subject["source_sha"]),
        "model": cast(str, subject["model"]),
        "config": cast(str, subject["config"]),
    }


def _build_subjects(
    actual: dict[LeafKey, LeafData],
    expected_leaves: list[dict[str, str]],
    sources: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    dict[SubjectKey, dict[str, dict[str, float]]],
    dict[SubjectKey, dict[str, bool]],
]:
    """Materialize scorecard subjects plus per-category task data."""
    order, allocation = _allocation(expected_leaves, sources)
    expected_by_subject: dict[SubjectKey, list[dict[str, str]]] = {
        key: [] for key in order
    }
    for leaf in expected_leaves:
        key = SubjectKey(leaf["branch"], leaf["model"], leaf["config"])
        expected_by_subject[key].append(leaf)

    source_shas = _source_index(sources)
    subjects: list[dict[str, object]] = []
    tasks_by_subject: dict[SubjectKey, dict[str, dict[str, float]]] = {}
    task_files_by_subject: dict[SubjectKey, dict[str, bool]] = {}
    for key in order:
        leaf_data: list[dict] = []
        task_data: dict[str, dict[str, float]] = {}
        task_files: dict[str, bool] = {}
        for category in allocation[key]:
            leaf_key = LeafKey(key.model, key.branch, key.config, category)
            record = actual.get(leaf_key)
            if record is None:
                task_data[category] = {}
                task_files[category] = False
                continue
            actual_sha = cast(str, record.leaf.get("source_sha", ""))
            if actual_sha != source_shas[key.branch]:
                msg = (
                    f"actual source SHA mismatch for {key.branch!r}, "
                    f"{key.model!r}, {key.config!r}, {category!r}"
                )
                raise ValueError(msg)
            leaf_data.append(record.leaf)
            task_data[category] = record.tasks
            task_files[category] = record.has_tasks

        combined = unified.combine(
            leaf_data,
            cast(list[LeafKey | dict[str, str]], expected_by_subject[key]),
            allocation[key],
        )
        row = cast(dict[str, object], combined["rows"][0])
        subjects.append(
            {
                **row,
                "assigned_categories": allocation[key],
                "task_data_complete": all(
                    task_files.get(category, False) for category in allocation[key]
                ),
            }
        )
        tasks_by_subject[key] = task_data
        task_files_by_subject[key] = task_files
    return subjects, tasks_by_subject, task_files_by_subject


def _metric_value(
    subject: dict[str, object], category: str, metric: str
) -> float | None:
    categories = cast(dict[str, dict[str, object]], subject["categories"])
    value = categories.get(category, {}).get(metric)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _aggregate_metric(
    subject: dict[str, object], categories: list[str], metric: str, *, micro: bool
) -> float | None:
    """Compute a macro or task-weighted metric over exactly the shared categories."""
    blocks = cast(dict[str, dict[str, object]], subject["categories"])
    values: list[tuple[float, int]] = []
    for category in categories:
        block = blocks.get(category)
        if block is None:
            return None
        value = block.get(metric)
        tasks = block.get("tasks")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or isinstance(tasks, bool)
            or not isinstance(tasks, int)
            or tasks < 1
        ):
            return None
        values.append((float(value), tasks))
    if not values:
        return None
    if micro:
        total = sum(tasks for _value, tasks in values)
        return sum(value * tasks for value, tasks in values) / total
    return sum(value for value, _tasks in values) / len(values)


def _metric_delta(
    baseline: float | None, candidate: float | None
) -> dict[str, float | None]:
    """Return both absolute values and candidate-minus-baseline delta."""
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": (
            candidate - baseline
            if baseline is not None and candidate is not None
            else None
        ),
    }


def _comparison_metrics(
    baseline: dict[str, object],
    candidate: dict[str, object],
    categories: list[str],
) -> dict[str, object]:
    """Compute category, macro, and micro metrics over the shared allocation."""
    category_metrics = {
        category: {
            metric: _metric_delta(
                _metric_value(baseline, category, metric),
                _metric_value(candidate, category, metric),
            )
            for metric in METRICS
        }
        for category in categories
    }
    return {
        "categories": category_metrics,
        "macro": {
            metric: _metric_delta(
                _aggregate_metric(baseline, categories, metric, micro=False),
                _aggregate_metric(candidate, categories, metric, micro=False),
            )
            for metric in METRICS
        },
        "micro": {
            metric: _metric_delta(
                _aggregate_metric(baseline, categories, metric, micro=True),
                _aggregate_metric(candidate, categories, metric, micro=True),
            )
            for metric in METRICS
        },
    }


def _task_outcomes(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    categories: list[str],
) -> dict[str, dict[str, int]]:
    """Count candidate wins, losses, ties, and unmatched tasks by category."""
    output: dict[str, dict[str, int]] = {}
    for category in categories:
        first = baseline.get(category, {})
        second = candidate.get(category, {})
        counts = {"wins": 0, "losses": 0, "ties": 0, "missing": 0}
        for task in sorted(set(first) | set(second)):
            if task not in first or task not in second:
                counts["missing"] += 1
            elif second[task] > first[task]:
                counts["wins"] += 1
            elif second[task] < first[task]:
                counts["losses"] += 1
            else:
                counts["ties"] += 1
        output[category] = counts
    return output


def _not_comparable(
    kind: str,
    baseline: dict[str, str],
    candidate: dict[str, str],
    reason: str,
) -> dict[str, object]:
    """Describe an intentionally omitted pair without treating it as an error."""
    return {
        "kind": kind,
        "baseline": baseline,
        "candidate": candidate,
        "reason": reason,
    }


def _make_comparison(
    kind: str,
    baseline: dict[str, object],
    candidate: dict[str, object],
    baseline_tasks: dict[str, dict[str, float]],
    candidate_tasks: dict[str, dict[str, float]],
    baseline_task_files: dict[str, bool],
    candidate_task_files: dict[str, bool],
    category_order: list[str],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Build one fair comparison or a no-shared-category audit record."""
    baseline_categories = set(cast(list[str], baseline["assigned_categories"]))
    candidate_categories = set(cast(list[str], candidate["assigned_categories"]))
    shared = [
        category
        for category in category_order
        if category in baseline_categories and category in candidate_categories
    ]
    first_identity = _subject_identity(baseline)
    second_identity = _subject_identity(candidate)
    if not shared:
        return None, _not_comparable(
            kind, first_identity, second_identity, "no_shared_categories"
        )
    incomplete = (
        bool(baseline["incomplete"])
        or bool(candidate["incomplete"])
        or any(not baseline_task_files.get(category, False) for category in shared)
        or any(not candidate_task_files.get(category, False) for category in shared)
    )
    return (
        {
            "kind": kind,
            "baseline": first_identity,
            "candidate": second_identity,
            "shared_categories": shared,
            "status": "incomplete" if incomplete else "complete",
            "metrics": _comparison_metrics(baseline, candidate, shared),
            "task_outcomes": _task_outcomes(baseline_tasks, candidate_tasks, shared),
        },
        None,
    )


def compare(
    root: Path,
    *,
    sources: list[dict[str, str]],
    expected_leaves: list[dict[str, str]],
    categories: list[str],
    rollouts: int,
) -> dict[str, object]:
    """Build every controlled branch and config comparison from the run allocation."""
    actual = _actual_leaves(root, rollouts)
    subjects, tasks, task_files = _build_subjects(actual, expected_leaves, sources)
    by_key = {
        SubjectKey(
            cast(str, subject["branch"]),
            cast(str, subject["model"]),
            cast(str, subject["config"]),
        ): subject
        for subject in subjects
    }
    model_order = list(
        dict.fromkeys(cast(str, subject["model"]) for subject in subjects)
    )
    comparisons_out: list[dict[str, object]] = []
    not_comparable: list[dict[str, object]] = []

    # Change only the branch: model and config remain fixed.
    for first_source, second_source in combinations(sources, 2):
        first_branch = first_source["branch"]
        second_branch = second_source["branch"]
        for model in model_order:
            configs = list(
                dict.fromkeys(
                    key.config
                    for key in by_key
                    if key.model == model
                    and key.branch in {first_branch, second_branch}
                )
            )
            for config in configs:
                first_key = SubjectKey(first_branch, model, config)
                second_key = SubjectKey(second_branch, model, config)
                if first_key not in by_key or second_key not in by_key:
                    baseline = {
                        "branch": first_branch,
                        "source_sha": first_source["sha"],
                        "model": model,
                        "config": config,
                    }
                    candidate = {
                        "branch": second_branch,
                        "source_sha": second_source["sha"],
                        "model": model,
                        "config": config,
                    }
                    not_comparable.append(
                        _not_comparable(
                            "cross_branch",
                            baseline,
                            candidate,
                            "config_not_active_on_both_branches",
                        )
                    )
                    continue
                comparison, omitted = _make_comparison(
                    "cross_branch",
                    by_key[first_key],
                    by_key[second_key],
                    tasks[first_key],
                    tasks[second_key],
                    task_files[first_key],
                    task_files[second_key],
                    categories,
                )
                if comparison is not None:
                    comparisons_out.append(comparison)
                if omitted is not None:
                    not_comparable.append(omitted)

    # Change only the config: branch and model remain fixed.
    for source in sources:
        branch = source["branch"]
        for model in model_order:
            keys = [
                key for key in by_key if key.branch == branch and key.model == model
            ]
            for first_key, second_key in combinations(keys, 2):
                comparison, omitted = _make_comparison(
                    "within_branch",
                    by_key[first_key],
                    by_key[second_key],
                    tasks[first_key],
                    tasks[second_key],
                    task_files[first_key],
                    task_files[second_key],
                    categories,
                )
                if comparison is not None:
                    comparisons_out.append(comparison)
                if omitted is not None:
                    not_comparable.append(omitted)

    return {
        "schema_version": 1,
        "rollouts_per_task": rollouts,
        "sources": sources,
        "subjects": subjects,
        "comparisons": comparisons_out,
        "not_comparable": not_comparable,
    }


def _fmt(value: float | None, *, signed: bool = False) -> str:
    """Format one comparison metric for Markdown."""
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _md(value: object) -> str:
    """Escape a scalar for a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _identity_label(identity: dict[str, str]) -> str:
    """Render a compact branch/config identity."""
    return f"{identity['branch']}/{identity['config']}"


def render_markdown(result: dict[str, object]) -> str:
    """Render absolute scores, controlled deltas, and per-task outcomes."""
    sources = cast(list[dict[str, str]], result["sources"])
    subjects = cast(list[dict[str, object]], result["subjects"])
    comparisons_out = cast(list[dict[str, object]], result["comparisons"])
    omitted = cast(list[dict[str, object]], result["not_comparable"])
    lines = [
        "## Unified evals — deterministic comparisons",
        "",
        "### Sources",
        "",
        "| Branch | Commit |",
        "|---|---|",
    ]
    for source in sources:
        sha = source["sha"] or "workflow checkout"
        lines.append(f"| {_md(source['branch'])} | `{_md(sha)}` |")

    lines.extend(
        [
            "",
            "### Active subjects",
            "",
            "| Branch | Model | Config | Assigned categories | Macro pass/avg | Micro pass/avg | Status |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for subject in subjects:
        macro = cast(dict[str, float | None], subject["macro"])
        micro = cast(dict[str, float | None], subject["micro"])
        status = "incomplete" if subject["incomplete"] else "complete"
        assigned = ", ".join(cast(list[str], subject["assigned_categories"]))
        lines.append(
            f"| {_md(subject['branch'])} | {_md(subject['model'])} | "
            f"{_md(subject['config'])} | {_md(assigned)} | "
            f"{_fmt(macro['pass_at_k'])}/{_fmt(macro['avg_at_k'])} | "
            f"{_fmt(micro['pass_at_k'])}/{_fmt(micro['avg_at_k'])} | {status} |"
        )

    lines.extend(
        [
            "",
            "### Pairwise deltas",
            "",
            "Delta is candidate minus baseline. Different models are never compared.",
            "",
            "| Kind | Comparison | Model | Scope | Δ pass@k | Δ avg@k | Status |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for comparison in comparisons_out:
        baseline = cast(dict[str, str], comparison["baseline"])
        candidate = cast(dict[str, str], comparison["candidate"])
        metrics = cast(dict[str, object], comparison["metrics"])
        category_metrics = cast(
            dict[str, dict[str, dict[str, float | None]]], metrics["categories"]
        )
        scopes = [
            *(
                (category, category_metrics[category])
                for category in cast(list[str], comparison["shared_categories"])
            ),
            ("macro", cast(dict[str, dict[str, float | None]], metrics["macro"])),
            ("micro", cast(dict[str, dict[str, float | None]], metrics["micro"])),
        ]
        label = f"{_identity_label(baseline)} → {_identity_label(candidate)}"
        for scope, values in scopes:
            lines.append(
                f"| {_md(comparison['kind'])} | {_md(label)} | "
                f"{_md(baseline['model'])} | {_md(scope)} | "
                f"{_fmt(values['pass_at_k']['delta'], signed=True)} | "
                f"{_fmt(values['avg_at_k']['delta'], signed=True)} | "
                f"{_md(comparison['status'])} |"
            )

    lines.extend(
        [
            "",
            "### Per-task outcomes",
            "",
            "Wins and losses are from the candidate’s perspective.",
            "",
            "| Kind | Comparison | Model | Category | Wins | Losses | Ties | Missing |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in comparisons_out:
        baseline = cast(dict[str, str], comparison["baseline"])
        candidate = cast(dict[str, str], comparison["candidate"])
        outcomes = cast(dict[str, dict[str, int]], comparison["task_outcomes"])
        label = f"{_identity_label(baseline)} → {_identity_label(candidate)}"
        for category, counts in outcomes.items():
            lines.append(
                f"| {_md(comparison['kind'])} | {_md(label)} | "
                f"{_md(baseline['model'])} | {_md(category)} | {counts['wins']} | "
                f"{counts['losses']} | {counts['ties']} | {counts['missing']} |"
            )

    if omitted:
        lines.extend(
            [
                "",
                "### Not comparable",
                "",
                "These pairs were intentionally omitted; they are not evaluation failures.",
                "",
                "| Kind | Pair | Model | Reason |",
                "|---|---|---|---|",
            ]
        )
        for item in omitted:
            baseline = cast(dict[str, str], item["baseline"])
            candidate = cast(dict[str, str], item["candidate"])
            label = f"{_identity_label(baseline)} ↔ {_identity_label(candidate)}"
            lines.append(
                f"| {_md(item['kind'])} | {_md(label)} | {_md(baseline['model'])} | "
                f"{_md(item['reason'])} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, object], out_dir: Path) -> bool:
    """Write comparison outputs when at least one meaningful pair exists."""
    comparisons_out = cast(list[dict[str, object]], result["comparisons"])
    if not comparisons_out:
        print("No comparable Unified Eval subjects; no comparison artifact written.")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(result)
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("\n" + markdown)
    return True


def main(argv: list[str] | None = None) -> int:
    """Run deterministic comparison reporting for one Unified Evals dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources-json", required=True)
    parser.add_argument("--expected-leaves-json", required=True)
    parser.add_argument("--categories-json", required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")
    try:
        result = compare(
            args.root,
            sources=parse_sources(args.sources_json),
            expected_leaves=parse_expected_leaves(args.expected_leaves_json),
            categories=parse_categories(args.categories_json),
            rollouts=args.rollouts,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    write_outputs(result, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
