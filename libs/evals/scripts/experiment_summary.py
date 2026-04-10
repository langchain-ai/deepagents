#!/usr/bin/env python3
"""Generate a markdown summary table from Harbor experiment artifacts.

Reads per-job ``job_summary.json`` files produced by the Harbor workflow and
outputs a GitHub-flavored markdown table suitable for ``$GITHUB_STEP_SUMMARY``.

Usage::

    python scripts/experiment_summary.py all-results/
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from deepagents_harbor.stats import format_ci, min_detectable_effect


def _load_summaries(results_dir: Path) -> list[dict]:
    """Collect all ``job_summary.json`` files from downloaded artifact dirs."""
    summaries: list[dict] = []
    for summary_path in sorted(results_dir.rglob("job_summary.json")):
        data = json.loads(summary_path.read_text())
        if data.get("rewards"):
            summaries.append(data)
    return summaries


def _render_task_table(summaries: list[dict]) -> str:
    """Render per-task reward table."""
    lines = ["## Per-task results", ""]
    lines.append("| Model | Todo mode | Trial | Task | Reward |")
    lines.append("|-------|-----------|-------|------|--------|")

    for s in sorted(summaries, key=lambda x: (x["model"], x["todo_mode"], x["trial"])):
        model = s["model"]
        mode = s["todo_mode"]
        trial = s["trial"]
        for task, reward in sorted(s["rewards"].items()):
            r = f"{reward:.1f}" if isinstance(reward, float) else str(reward)
            lines.append(f"| {model} | {mode} | {trial} | {task} | {r} |")

    return "\n".join(lines)


def _render_aggregate_table(summaries: list[dict]) -> str:
    """Render aggregate scores grouped by model and todo_mode with Wilson CIs."""
    rewards_by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in summaries:
        key = (s["model"], s["todo_mode"])
        for reward in s["rewards"].values():
            rewards_by[key].append(float(reward))

    models = sorted({k[0] for k in rewards_by})
    modes = sorted({k[1] for k in rewards_by})

    lines = ["## Aggregate scores", ""]

    header = "| Model | " + " | ".join(f"`{m}`" for m in modes) + " |"
    sep = "|-------|" + "|".join("------" for _ in modes) + "|"
    lines.append(header)
    lines.append(sep)

    for model in models:
        cells = []
        for mode in modes:
            rewards = rewards_by.get((model, mode), [])
            if not rewards:
                cells.append("—")
                continue
            total = len(rewards)
            successes = sum(1 for r in rewards if r >= 1.0)
            cells.append(format_ci(successes, total))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    if rewards_by:
        n_per_variant = max(len(v) for v in rewards_by.values())
        mde = min_detectable_effect(n_per_variant)
        lines.append("")
        lines.append(f"MDE at n={n_per_variant}: ±{mde * 100:.1f}pp (95% CI)")

    return "\n".join(lines)


def _render_experiment_links(summaries: list[dict]) -> str:
    """Render LangSmith experiment links."""
    seen: set[str] = set()
    links: list[str] = []
    for s in sorted(summaries, key=lambda x: (x["model"], x["todo_mode"], x["trial"])):
        name = s.get("experiment_name", "")
        url = s.get("experiment_url", "")
        if name and name not in seen:
            seen.add(name)
            if url:
                links.append(f"- [{name}]({url})")
            else:
                links.append(f"- {name}")

    if not links:
        return ""
    return "## LangSmith experiments\n\n" + "\n".join(links)


def main() -> int:
    """Entry point."""
    if len(sys.argv) != 2:  # noqa: PLR2004
        print(f"Usage: {sys.argv[0]} <results-dir>", file=sys.stderr)
        return 1

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        return 1

    summaries = _load_summaries(results_dir)
    if not summaries:
        print("# Todo experiment summary\n\nNo results found.")
        return 0

    parts = [
        "# Todo experiment summary",
        "",
        _render_aggregate_table(summaries),
        "",
        _render_task_table(summaries),
        "",
        _render_experiment_links(summaries),
    ]
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
