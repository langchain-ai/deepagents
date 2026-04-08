"""Run the better-harness hill climber and write a report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deepagents_evals.better_harness import hill_climb_prompt_modules


def main() -> None:
    """Parse arguments, run the optimizer, and write artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the Deep Agents better-harness optimizer")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model name to optimize against.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum hill-climbing iterations.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/better-harness/report.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("artifacts/better-harness/report.md"),
        help="Path to write the Markdown report.",
    )
    args = parser.parse_args()

    result = hill_climb_prompt_modules(
        model_name=args.model,
        max_iterations=args.max_iterations,
    )
    result.write(json_path=args.report_json, markdown_path=args.report_md)

    print(result.to_markdown())


if __name__ == "__main__":
    main()
