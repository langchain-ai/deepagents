from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from tabulate import tabulate


def _format_table(
    rows: list[dict[str, object]], headers: list[str]
) -> list[list[object]]:
    """Build tabulate-ready rows from report dicts."""
    return [
        [
            str(r.get("model", "")),
            r.get("passed", 0),
            r.get("failed", 0),
            r.get("skipped", 0),
            r.get("total", 0),
            r.get("correctness", 0.0),
            r.get("solve_rate") or "n/a",
            r.get("step_ratio") or "n/a",
            r.get("tool_call_ratio") or "n/a",
            r.get("median_duration_s", 0.0),
        ]
        for r in rows
    ]


_COLALIGN = (
    "left",
    "right",
    "right",
    "right",
    "right",
    "right",
    "right",
    "right",
    "right",
    "right",
)

_HEADERS = [
    "model",
    "passed",
    "failed",
    "skipped",
    "total",
    "correctness",
    "solve_rate",
    "step_ratio",
    "tool_call_ratio",
    "median_duration_s",
]


_CATEGORIES_JSON = (
    Path(__file__).resolve().parents[2]
    / "libs"
    / "evals"
    / "deepagents_evals"
    / "categories.json"
)


def _load_category_labels() -> dict[str, str]:
    """Load human-readable category labels from `categories.json`.

    Returns:
        Mapping of category name to display label, or empty dict on failure.
    """
    try:
        return json.loads(_CATEGORIES_JSON.read_text(encoding="utf-8"))["labels"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        print(
            f"warning: could not load category labels from {_CATEGORIES_JSON}: {exc}",
            file=sys.stderr,
        )
        return {}


def _build_category_table(rows: list[dict[str, object]]) -> list[str]:
    """Build a per-category scores table from report rows.

    Returns a single-element list containing the rendered Markdown table
    string, or an empty list when no category data is present.

    Args:
        rows: Report row dicts, each expected to contain a `category_scores`
            mapping and a `model` string.
    """
    # Collect all categories across all models (preserving insertion order).
    all_cats: list[str] = list(
        dict.fromkeys(cat for r in rows for cat in (r.get("category_scores") or {}))
    )

    if not all_cats:
        return []

    labels = _load_category_labels()
    headers = ["model", *[labels.get(c, c) for c in all_cats]]
    table_rows: list[list[object]] = []
    for r in rows:
        scores = r.get("category_scores") or {}
        table_rows.append(
            [
                str(r.get("model", "")),
                *[scores.get(c, "—") for c in all_cats],
            ]
        )

    colalign = ("left", *("right" for _ in all_cats))
    return [tabulate(table_rows, headers=headers, tablefmt="github", colalign=colalign)]


def main() -> None:
    """Generate an aggregated report."""
    report_files = sorted(
        glob.glob("evals_artifacts/**/evals_report.json", recursive=True)
    )

    rows: list[dict[str, object]] = []
    for file in report_files:
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        rows.append(payload)

    # --- JSON artifact for offline analysis ---
    summary_json_path = Path("evals_summary.json")
    summary_json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- Table 1: grouped by provider, then correctness desc ---
    by_provider = sorted(
        rows,
        key=lambda r: (
            str(r.get("model", "")).split(":")[0],
            -float(r.get("correctness", 0.0)),
        ),
    )

    lines: list[str] = []
    lines.append("## Evals summary")
    lines.append("")

    table_rows = _format_table(by_provider, _HEADERS)
    if table_rows:
        lines.append(
            tabulate(
                table_rows, headers=_HEADERS, tablefmt="github", colalign=_COLALIGN
            )
        )
    else:
        lines.append("_No eval artifacts found._")

    # --- Table 2: ranked by correctness desc, then solve_rate desc ---
    by_correctness = sorted(
        rows,
        key=lambda r: (
            -float(r.get("correctness", 0.0)),
            -float(r.get("solve_rate") or 0.0),
        ),
    )

    lines.append("")
    lines.append("## Ranked by correctness / solve rate")
    lines.append("")

    ranked_rows = _format_table(by_correctness, _HEADERS)
    if ranked_rows:
        lines.append(
            tabulate(
                ranked_rows, headers=_HEADERS, tablefmt="github", colalign=_COLALIGN
            )
        )
    else:
        lines.append("_No eval artifacts found._")

    # --- Table 3: per-category scores ---
    cat_table = _build_category_table(rows)
    if cat_table:
        lines.append("")
        lines.append("## Per-category correctness")
        lines.append("")
        lines.extend(cat_table)

    # --- Footer for n/a metrics ---
    has_na = any(
        r.get("solve_rate") is None
        and r.get("step_ratio") is None
        and r.get("tool_call_ratio") is None
        for r in rows
    )
    if has_na:
        lines.append("")
        lines.append(
            "> **N/A** — `solve_rate`, `step_ratio`, and `tool_call_ratio` are only"
            " reported for eval categories whose tests specify expected step/tool-call"
            " counts. Categories that use custom assertions (e.g. summarization) do not"
            " currently produce these metrics."
        )

    # --- LangSmith experiment links ---
    experiment_entries: list[
        tuple[str, str, str, str]
    ] = []  # (model, name, url, public_url)
    for r in rows:
        model = str(r.get("model", ""))
        # Prefer rich experiment_links (name + url); fall back to bare experiment_urls
        # for older reports that only have the URL list.
        raw_links = r.get("experiment_links") or []
        if isinstance(raw_links, list) and raw_links:
            for link in raw_links:
                if isinstance(link, dict):
                    name = str(link.get("name", ""))
                    url = str(link.get("url", ""))
                    public_url = str(link.get("public_url", ""))
                    if url:
                        experiment_entries.append((model, name or url, url, public_url))
        else:
            raw_urls = r.get("experiment_urls") or []
            if isinstance(raw_urls, list):
                for url in raw_urls:
                    experiment_entries.append((model, str(url), str(url), ""))
    if experiment_entries:
        lines.append("")
        lines.append("## LangSmith experiments")
        lines.append("")
        for model, name, url, public_url in experiment_entries:
            if public_url:
                lines.append(
                    f"- **{model}**: [{name}]({public_url}) ([internal]({url}))"
                )
            else:
                lines.append(f"- **{model}**: [{name}]({url})")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        Path(summary_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
