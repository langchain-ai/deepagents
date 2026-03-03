from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from tabulate import tabulate


def main() -> None:
    """Generate an aggregated report."""
    report_files = sorted(glob.glob("evals_artifacts/**/evals_report.json", recursive=True))

    rows: list[dict[str, object]] = []
    for file in report_files:
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        rows.append(payload)

    rows.sort(key=lambda r: (str(r.get("model", "")).split(":")[0], -float(r.get("accuracy", 0.0))))

    headers = [
        "model",
        "passed",
        "failed",
        "skipped",
        "total",
        "accuracy",
        "median_duration_s",
    ]

    table_rows: list[list[object]] = [
        [
            str(r.get("model", "")),
            r.get("passed", 0),
            r.get("failed", 0),
            r.get("skipped", 0),
            r.get("total", 0),
            r.get("accuracy", 0.0),
            r.get("median_duration_s", 0.0),
        ]
        for r in rows
    ]

    lines: list[str] = []
    lines.append("## Evals summary")
    lines.append("")

    if table_rows:
        lines.append(
            tabulate(
                table_rows,
                headers=headers,
                tablefmt="github",
                colalign=("left", "right", "right", "right", "right", "right", "right"),
            )
        )
    else:
        lines.append("_No eval artifacts found._")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        Path(summary_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
