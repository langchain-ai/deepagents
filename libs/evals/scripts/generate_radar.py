"""Generate radar charts from eval results.

Usage:
    # Toy data (experimentation)
    python scripts/generate_radar.py --toy -o charts/radar.png

    # From evals_summary.json (CI / post-run)
    python scripts/generate_radar.py --summary evals_summary.json -o charts/radar.png

    # From per-category JSON (alternative format with "scores" key)
    python scripts/generate_radar.py --results category_results.json -o charts/radar.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepagents_evals.radar import (
    EVAL_CATEGORIES,
    ModelResult,
    generate_radar,
    load_results_from_summary,
    toy_data,
)


def _load_category_results(path: Path) -> list[ModelResult]:
    """Load per-category results from a JSON file.

    Expected format:

        [
            {
                "model": "anthropic:claude-sonnet-4-6",
                "scores": {"file_operations": 0.92, "memory": 0.83, ...}
            },
            ...
        ]

    Args:
        path: Path to the JSON file.

    Returns:
        List of `ModelResult` objects.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If an entry is missing `model` or `scores`.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ModelResult(model=entry["model"], scores=entry["scores"]) for entry in data]


def main() -> None:
    """Entry point for radar chart generation."""
    parser = argparse.ArgumentParser(description="Generate eval radar charts")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true", help="Use toy data for experimentation")
    source.add_argument("--summary", type=Path, help="Path to evals_summary.json (aggregate only)")
    source.add_argument("--results", type=Path, help="Path to per-category results JSON")

    parser.add_argument(
        "-o", "--output", type=Path, default=Path("charts/radar.png"), help="Output file path"
    )
    parser.add_argument("--title", default="Deep Agents Eval Results", help="Chart title")

    args = parser.parse_args()

    if args.toy:
        results = toy_data()
    elif args.summary:
        try:
            results = load_results_from_summary(args.summary)
        except FileNotFoundError:
            print(f"error: {args.summary} not found", file=sys.stderr)
            sys.exit(1)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"error: could not load {args.summary}: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.results:
        try:
            results = _load_category_results(args.results)
        except FileNotFoundError:
            print(f"error: {args.results} not found", file=sys.stderr)
            sys.exit(1)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"error: could not load {args.results}: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    if not results:
        print("error: no results to plot", file=sys.stderr)
        sys.exit(1)

    # Detect categories from results (use all categories present across models).
    all_cats = set()
    for r in results:
        all_cats.update(r.scores.keys())

    # Preserve EVAL_CATEGORIES ordering for known categories, append unknown ones.
    ordered = [c for c in EVAL_CATEGORIES if c in all_cats]
    ordered.extend(sorted(all_cats - set(ordered)))

    try:
        generate_radar(
            results,
            categories=ordered,
            title=args.title,
            output=args.output,
        )
    except OSError as exc:
        print(f"error: could not save chart to {args.output}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
