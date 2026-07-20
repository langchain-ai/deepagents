"""Combine per-(model x branch x config x category) Harbor summary.json files into
a cross-row comparison (macro + micro overalls), a leaderboard, combined JSON, and
radar input, ranking flat (model, branch, config) rows.

Each leaf directory (one per model x branch x config x category) has a summary.json
written by aggregate_shards.py, which records the model, branch, config, and
category authoritatively (via --model/--config/--category/--branch)
plus dynamic pass@{K}/avg@{K} keys.

The combiner is given the expected leaf grid (EXPECTED_LEAVES, a list of
{model, branch, config, category} quads / EXPECTED_CATEGORIES) so a leaf that failed
to upload is still shown and flagged incomplete, rather than silently ranking on
fewer categories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import cast

from unified_types import LeafKey, RowKey


class _LeafSummaryError(ValueError):
    """Raised when a leaf summary cannot safely participate in aggregation."""


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{field} must be a JSON object"
        raise _LeafSummaryError(msg)
    return cast(dict[str, object], value)


def _require_integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{field} must be an integer >= {minimum}"
        raise _LeafSummaryError(msg)
    return value


def _require_metric(summary: dict[str, object], field: str, *, tasks: int) -> float | None:
    if field not in summary:
        msg = f"{field} is required"
        raise _LeafSummaryError(msg)
    value = summary[field]
    if tasks == 0:
        if value is not None:
            msg = f"{field} must be null when totals.tasks is 0"
            raise _LeafSummaryError(msg)
        return None
    if value is None:
        msg = f"{field} may be null only when totals.tasks is 0"
        raise _LeafSummaryError(msg)
    msg = f"{field} must be a finite number in [0, 1] or null"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _LeafSummaryError(msg)
    try:
        metric = float(value)
    except OverflowError as exc:
        raise _LeafSummaryError(msg) from exc
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise _LeafSummaryError(msg)
    return metric


def read_leaf(leaf_dir: Path, *, expected_rollouts: int | None = None) -> dict:
    text = (leaf_dir / "summary.json").read_text(encoding="utf-8")
    try:
        raw: object = json.loads(text)
    except ValueError as exc:
        # json.loads raises JSONDecodeError (a ValueError) on corrupt JSON, and a
        # plain ValueError when a numeric literal exceeds the int-string-conversion
        # limit. Both are bad leaf data -- normalize to _LeafSummaryError so callers
        # catch one narrow type rather than a broad ValueError.
        msg = f"summary.json is not valid JSON: {exc}"
        raise _LeafSummaryError(msg) from exc
    summary = _require_object(raw, "summary")
    k = _require_integer(summary.get("rollouts_per_task"), "rollouts_per_task", minimum=1)
    if expected_rollouts is not None and k != expected_rollouts:
        msg = f"rollouts_per_task is {k}; expected {expected_rollouts}"
        raise _LeafSummaryError(msg)
    totals = _require_object(summary.get("totals"), "totals")
    tasks = _require_integer(totals.get("tasks"), "totals.tasks", minimum=0)
    if tasks > sys.maxsize:
        msg = f"totals.tasks must be an integer in 0..{sys.maxsize}"
        raise _LeafSummaryError(msg)
    passed = _require_integer(totals.get("passed"), "totals.passed", minimum=0)
    model = summary.get("model")
    category = summary.get("category")
    if model is not None and not isinstance(model, str):
        msg = "model must be a string or null"
        raise _LeafSummaryError(msg)
    if category is not None and not isinstance(category, str):
        msg = "category must be a string or null"
        raise _LeafSummaryError(msg)
    config = summary.get("config")
    if config is not None and not isinstance(config, str):
        msg = "config must be a string or null"
        raise _LeafSummaryError(msg)
    branch = summary.get("branch")
    if branch is not None and not isinstance(branch, str):
        msg = "branch must be a string or null"
        raise _LeafSummaryError(msg)
    source_sha = summary.get("source_sha")
    if source_sha is not None and not isinstance(source_sha, str):
        msg = "source_sha must be a string or null"
        raise _LeafSummaryError(msg)
    if "incomplete" not in summary:
        msg = "incomplete is required"
        raise _LeafSummaryError(msg)
    incomplete = summary["incomplete"]
    if not isinstance(incomplete, bool):
        msg = "incomplete must be a boolean"
        raise _LeafSummaryError(msg)
    return {
        "model": model or "unknown",
        "category": category or "unknown",
        "config": config or "unknown",
        "branch": branch or "current",
        "source_sha": source_sha or "",
        "pass_at_k": _require_metric(summary, f"pass@{k}", tasks=tasks),
        "avg_at_k": _require_metric(summary, f"avg@{k}", tasks=tasks),
        "tasks": tasks,
        "passed": passed,
        "incomplete": incomplete,
    }


def _mean(vals: list[float | None]) -> float | None:
    present = [v for v in vals if v is not None]
    return sum(present) / len(present) if present else None


def combine(
    leaves: list[dict],
    expected_leaves: list[LeafKey | dict[str, str]] | None = None,
    expected_categories: list[str] | None = None,
) -> dict:
    present_cats = {leaf["category"] for leaf in leaves}
    if expected_categories:
        categories = list(expected_categories)
        categories += sorted(present_cats - set(categories))
    else:
        categories = sorted(present_cats)

    # Required (model, branch, config) -> {category} grid from the expected quads,
    # so a missing leaf is flagged without assuming which categories a config ran
    # (tau3 covers conversation only; code configs cover the code categories).
    required_by_row: dict[RowKey, set[str]] = {}
    source_sha_by_row: dict[RowKey, str] = {}
    row_order: list[RowKey] = []
    for quad in expected_leaves or []:
        key = _as_leaf_key(quad)
        row = RowKey(key.model, key.branch, key.config)
        if row not in required_by_row:
            required_by_row[row] = set()
            row_order.append(row)
            source_sha_by_row[row] = (
                quad.get("source_sha", "") if isinstance(quad, dict) else ""
            )
        required_by_row[row].add(key.category)

    by_row: dict[RowKey, list[dict]] = {}
    seen: set[LeafKey] = set()
    for leaf in leaves:
        row = RowKey(leaf["model"], leaf["branch"], leaf["config"])
        identity = LeafKey(leaf["model"], leaf["branch"], leaf["config"], leaf["category"])
        if identity in seen:
            msg = (
                f"Duplicate leaf for model {leaf['model']!r}, branch "
                f"{leaf['branch']!r}, config {leaf['config']!r}, category "
                f"{leaf['category']!r}"
            )
            raise ValueError(msg)
        seen.add(identity)
        by_row.setdefault(row, []).append(leaf)
        if row not in required_by_row:
            required_by_row[row] = set()
            row_order.append(row)
        source_sha_by_row.setdefault(row, leaf.get("source_sha", ""))

    rows_out: list[dict] = []
    for row in row_order:
        model, branch, config = row
        row_leaves = by_row.get(row, [])
        required = required_by_row.get(row, set())
        scored = [leaf for leaf in row_leaves if not required or leaf["category"] in required]
        cats = {
            leaf["category"]: {
                "pass_at_k": leaf["pass_at_k"],
                "avg_at_k": leaf["avg_at_k"],
                "tasks": leaf["tasks"],
                "incomplete": leaf["incomplete"] or leaf["tasks"] == 0,
            }
            for leaf in row_leaves
        }
        missing = [c for c in sorted(required) if c not in cats]
        macro = {
            "pass_at_k": _mean([leaf["pass_at_k"] for leaf in scored]),
            "avg_at_k": _mean([leaf["avg_at_k"] for leaf in scored]),
        }
        total_tasks = sum(leaf["tasks"] for leaf in scored) or 0
        # Validated None metrics have zero tasks, so None-as-zero is neutral in
        # these task-weighted numerators.
        micro_pass = (
            sum((leaf["pass_at_k"] or 0.0) * leaf["tasks"] for leaf in scored) / total_tasks
            if total_tasks
            else None
        )
        micro_avg = (
            sum((leaf["avg_at_k"] or 0.0) * leaf["tasks"] for leaf in scored) / total_tasks
            if total_tasks
            else None
        )
        rows_out.append(
            {
                "model": model,
                "branch": branch,
                "source_sha": source_sha_by_row.get(row, ""),
                "config": config,
                "categories": cats,
                "macro": macro,
                "micro": {"pass_at_k": micro_pass, "avg_at_k": micro_avg},
                "missing_categories": missing,
                "incomplete": (
                    not row_leaves
                    or bool(missing)
                    or any(leaf["incomplete"] or leaf["tasks"] == 0 for leaf in scored)
                ),
            }
        )
    return {"rows": rows_out, "categories": categories}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _incomplete_note(*, has_leaves: bool, missing_categories: list[str]) -> str:
    if not has_leaves:
        return "no leaf summaries found"
    if missing_categories:
        return f"missing categories: {', '.join(missing_categories)}"
    return "a category reported incomplete data"


def render_markdown(combined: dict, k: int) -> str:
    cats = combined["categories"]
    header = (
        ["Model / branch / config"]
        + [f"{c} pass@{k}/avg@{k}" for c in cats]
        + [
            f"Overall macro pass@{k}",
            f"macro avg@{k}",
            f"micro pass@{k}",
            f"micro avg@{k}",
        ]
    )
    ranked = sorted(
        combined["rows"],
        key=lambda r: (
            r["macro"]["pass_at_k"] is None,
            -(r["macro"]["pass_at_k"] or 0.0),
        ),
    )
    rows = []
    for r in ranked:
        label = f"{r['model']} / {r['branch']} / {r['config']}"
        cells = [label + (" ⚠️" if r["incomplete"] else "")]
        for c in cats:
            cat = r["categories"].get(c)
            cells.append(f"{_fmt(cat['pass_at_k'])}/{_fmt(cat['avg_at_k'])}" if cat else "—")
        cells += [
            _fmt(r["macro"]["pass_at_k"]),
            _fmt(r["macro"]["avg_at_k"]),
            _fmt(r["micro"]["pass_at_k"]),
            _fmt(r["micro"]["avg_at_k"]),
        ]
        rows.append(cells)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    md = "\n".join(lines) + "\n"

    incompletes = [r for r in combined["rows"] if r["incomplete"]]
    if incompletes:
        md += "\n> ⚠️ **Ranked on partial data** — treat these rows with caution:\n"
        for r in incompletes:
            miss = r.get("missing_categories") or []
            note = _incomplete_note(has_leaves=bool(r["categories"]), missing_categories=miss)
            md += f"> - `{r['model']} / {r['branch']} / {r['config']}` — {note}\n"
    return md


def radar_results(combined: dict) -> list[dict]:
    out = []
    for r in combined["rows"]:
        scores = {
            c: v["pass_at_k"] for c, v in r["categories"].items() if v.get("pass_at_k") is not None
        }
        out.append({"model": f"{r['model']} / {r['branch']} / {r['config']}", "scores": scores})
    return out


def write_outputs(combined: dict, k: int, out_dir: Path, step_summary_path: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unified_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    # Radar needs >= 3 axes to be meaningful. Emit its input only then; the
    # workflow's radar step keys off this file's existence.
    if len(combined["categories"]) >= 3:
        (out_dir / "radar_results.json").write_text(
            json.dumps(radar_results(combined), indent=2) + "\n"
        )
    md = render_markdown(combined, k)
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write("## Unified evals — cross-model comparison\n\n")
            f.write(md)
    # A machine-visible signal so a partially-covered ranking isn't taken at face value.
    for r in combined["rows"]:
        if r["incomplete"]:
            note = _incomplete_note(
                has_leaves=bool(r["categories"]),
                missing_categories=r.get("missing_categories") or [],
            )
            print(
                f"::warning::{r['model']} / {r['branch']} / {r['config']} "
                f"incomplete ({note}); ranked on partial data."
            )


def _discover_leaves(root: Path, *, expected_rollouts: int | None = None) -> list[dict]:
    leaves: list[dict] = []
    if (root / "summary.json").exists():
        candidates = [root]
    else:
        candidates = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / "summary.json").exists()
        ]
    for leaf_dir in candidates:
        try:
            leaves.append(read_leaf(leaf_dir, expected_rollouts=expected_rollouts))
        except (OSError, UnicodeError, _LeafSummaryError) as exc:
            # Catch only genuine bad-data signals: an unreadable file (OSError /
            # UnicodeError) or a schema/parse violation, which read_leaf always
            # raises as _LeafSummaryError (including corrupt or oversized JSON). A
            # broad ValueError would also swallow an incidental bug inside
            # read_leaf, silently dropping a valid leaf as "malformed".
            print(
                f"::warning::Skipping malformed eval summary at {leaf_dir / 'summary.json'}: {exc}"
            )
    return leaves


def _load_list_env(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    msg = f"{name} must be a JSON list of strings"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(msg) from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(msg)
    return cast(list[str], value) or None


def _load_leaves_env(name: str) -> list[dict[str, str]] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    msg = f"{name} must be a JSON list of {{model, branch, config, category}} objects"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(msg) from exc
    fields = {"model", "branch", "config", "category"}
    if not isinstance(value, list) or not all(
        isinstance(item, dict)
        and fields <= set(item)
        and all(isinstance(item[field], str) for field in fields)
        for item in value
    ):
        raise SystemExit(msg)
    return cast(list[dict[str, str]], value)


def _as_leaf_key(value: LeafKey | dict[str, str]) -> LeafKey:
    """Normalize a typed leaf key or legacy mapping for direct callers."""
    if isinstance(value, LeafKey):
        return value
    return LeafKey(value["model"], value["branch"], value["config"], value["category"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")
    out_dir = args.out_dir or args.root

    leaves = _discover_leaves(args.root, expected_rollouts=args.rollouts)
    expected_leaves = _load_leaves_env("EXPECTED_LEAVES")
    combined = combine(
        leaves,
        expected_leaves,
        _load_list_env("EXPECTED_CATEGORIES"),
    )
    write_outputs(combined, args.rollouts, out_dir, os.environ.get("GITHUB_STEP_SUMMARY"))
    if not leaves:
        print("::error::No usable eval leaf summaries were found; failing the run.")
        return 1
    rows = combined["rows"]
    # Incompleteness is surfaced per row in write_outputs (a ::warning:: plus the
    # ⚠️ markers in the table) and does not fail a run that has usable leaves. A
    # single errored shard no longer nukes the whole cross-model comparison.
    if expected_leaves is not None and rows and all(r["incomplete"] for r in rows):
        print(
            "::warning::Every expected (model, branch, config) row is incomplete; "
            "the scorecard below is ranked on partial data — inspect the per-row "
            "notes above."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
