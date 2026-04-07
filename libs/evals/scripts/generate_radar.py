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
import os
import sys
from pathlib import Path

from deepagents_evals.radar import (
    ALL_CATEGORIES,
    EVAL_CATEGORIES,
    ModelResult,
    generate_individual_radars,
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


def _no_results_hint(outcome: str) -> str:
    """Return a human-readable hint explaining why there are no results.

    Args:
        outcome: Upstream eval job result string (e.g. "cancelled", "failure").

    Returns:
        Diagnostic message tailored to the outcome.
    """
    outcome = outcome.strip().lower()
    if outcome == "cancelled":
        return (
            "the upstream eval job was cancelled — most likely it hit the "
            "workflow timeout-minutes limit. Check the eval job annotations "
            "for 'exceeded the maximum execution time'."
        )
    if outcome == "failure":
        return (
            "the upstream eval job failed before producing a report. "
            "Check the 'Run Evals' step logs for errors."
        )
    return (
        "the summary file is an empty JSON array — all eval jobs may have "
        "been cancelled (e.g. timeout) or failed before producing a report. "
        "Check the upstream eval job logs for details."
    )


def _clear_stale_outputs(output: Path, individual_dir: Path | None) -> None:
    """Remove stale radar artifacts from a previous run.

    This only removes files created by this script: the aggregate chart output
    and top-level PNG files inside `individual_dir`.

    Args:
        output: Aggregate chart output path.
        individual_dir: Directory containing per-model PNGs, if configured.
    """
    if output.is_file() or output.is_symlink():
        output.unlink()

    if individual_dir is None or not individual_dir.is_dir():
        return

    for path in individual_dir.glob("*.png"):
        if path.is_file() or path.is_symlink():
            path.unlink()


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
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=None,
        help="Directory for per-model radar charts (one PNG each)",
    )
    parser.add_argument(
        "--eval-outcome",
        default=None,
        help="Upstream eval job result (e.g. 'cancelled', 'failure'). "
        "Falls back to EVAL_OUTCOME env var. Used to produce a targeted "
        "diagnostic message when there are no results to chart.",
    )
    parser.add_argument(
        "--keep-zero-scores",
        action="store_true",
        help="Include models with all-zero scores (by default they are "
        "dropped as likely infrastructure failures).",
    )

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

    # Drop models whose scores are all zero — they almost certainly failed
    # (e.g. provider pin mismatch) and would just flatten the chart.
    zero_models: list[ModelResult] = []
    if not args.keep_zero_scores:
        zero_models = [r for r in results if not any(v > 0 for v in r.scores.values())]
        if zero_models:
            names = ", ".join(r.model for r in zero_models)
            print(
                f"note: dropped {len(zero_models)} model(s) with all-zero scores: {names}",
                file=sys.stderr,
            )
            results = [r for r in results if r not in zero_models]

    if not results:
        source = args.summary or args.results or "toy"
        if zero_models:
            msg = (
                f"skipped: all {len(zero_models)} model(s) had all-zero scores and "
                "were dropped. Re-run with --keep-zero-scores to force chart generation."
            )
        else:
            outcome = args.eval_outcome or os.environ.get("EVAL_OUTCOME", "")
            hint = _no_results_hint(outcome)
            msg = f"skipped: no results to plot from {source}\nhint: {hint}"
        _clear_stale_outputs(args.output, args.individual_dir)
        print(msg)
        print(msg, file=sys.stderr)
        sys.exit(0)

    # Detect categories from results (use all categories present across models).
    all_cats = set()
    for r in results:
        all_cats.update(r.scores.keys())

    min_axes = 3
    if len(all_cats) < min_axes:
        msg = f"skipped: radar chart needs >= {min_axes} categories, got {len(all_cats)}"
        _clear_stale_outputs(args.output, args.individual_dir)
        print(msg)
        print(msg, file=sys.stderr)
        sys.exit(0)

    # Preserve EVAL_CATEGORIES ordering for known categories, append unknown ones.
    # Categories in ALL_CATEGORIES but not EVAL_CATEGORIES (e.g. unit_test)
    # are intentionally excluded from radar charts.
    ordered = [c for c in EVAL_CATEGORIES if c in all_cats]
    excluded = set(ALL_CATEGORIES) - set(EVAL_CATEGORIES)
    ordered.extend(sorted(all_cats - set(ordered) - excluded))

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
    except Exception as exc:  # noqa: BLE001  # top-level script should surface chart backend failures cleanly
        print(f"error: chart generation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"saved: {args.output}")

    if args.individual_dir and len(results) > 1:
        try:
            paths = generate_individual_radars(
                results,
                categories=ordered,
                output_dir=args.individual_dir,
                title_prefix=args.title,
            )
        except OSError as exc:
            print(f"error: could not save individual charts: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001  # top-level script should surface chart backend failures cleanly
            print(f"error: individual chart generation failed: {exc}", file=sys.stderr)
            sys.exit(1)
        for p in paths:
            print(f"saved: {p}")


if __name__ == "__main__":
    main()
