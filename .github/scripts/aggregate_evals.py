from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from tabulate import tabulate


def _format_table(rows: list[dict[str, object]], headers: list[str]) -> list[list[object]]:
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


_COLALIGN = ("left", "right", "right", "right", "right", "right", "right", "right", "right", "right")

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


def main() -> None:
    """Generate an aggregated report."""
    report_files = sorted(glob.glob("evals_artifacts/**/evals_report.json", recursive=True))

    rows: list[dict[str, object]] = []
    for file in report_files:
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        rows.append(payload)

    # --- JSON artifact for offline analysis ---
    summary_json_path = Path("evals_summary.json")
    summary_json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- Table 1: grouped by provider, then correctness desc ---
    by_provider = sorted(
        rows,
        key=lambda r: (str(r.get("model", "")).split(":")[0], -float(r.get("correctness", 0.0))),
    )

    lines: list[str] = []
    lines.append("## Evals summary")
    lines.append("")

    table_rows = _format_table(by_provider, _HEADERS)
    if table_rows:
        lines.append(
            tabulate(table_rows, headers=_HEADERS, tablefmt="github", colalign=_COLALIGN)
        )
    else:
        lines.append("_No eval artifacts found._")

    # --- Table 2: ranked by correctness desc, then solve_rate desc ---
    by_correctness = sorted(
        rows,
        key=lambda r: (-float(r.get("correctness", 0.0)), -float(r.get("solve_rate") or 0.0)),
    )

    lines.append("")
    lines.append("## Ranked by correctness / solve rate")
    lines.append("")

    ranked_rows = _format_table(by_correctness, _HEADERS)
    if ranked_rows:
        lines.append(
            tabulate(ranked_rows, headers=_HEADERS, tablefmt="github", colalign=_COLALIGN)
        )
    else:
        lines.append("_No eval artifacts found._")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        Path(summary_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
