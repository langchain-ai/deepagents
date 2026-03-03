from __future__ import annotations

import glob
import json
import os
from pathlib import Path


def main() -> None:
    """Generate an aggregated report."""
    report_files = sorted(glob.glob("evals_artifacts/**/evals_report.json", recursive=True))

    rows: list[dict[str, object]] = []
    for file in report_files:
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        rows.append(payload)

    rows.sort(key=lambda r: str(r.get("model", "")))

    lines: list[str] = []
    lines.append("## Evals summary")
    lines.append("")
    lines.append("| model | passed | failed | skipped | total | accuracy | median_duration_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        model = r.get("model", "")
        passed = r.get("passed", 0)
        failed = r.get("failed", 0)
        skipped = r.get("skipped", 0)
        total = r.get("total", 0)
        accuracy = r.get("accuracy", 0.0)
        median_duration_s =r.get("median_duration_s", 0.0)

        lines.append(
            f"| {model} | {passed} | {failed} | {skipped} | {total} | {accuracy:.2f} | {median_duration_s:.4f} |"
        )

    if not rows:
        lines.append("")
        lines.append("_No eval artifacts found._")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        Path(summary_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
