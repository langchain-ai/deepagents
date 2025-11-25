#!/usr/bin/env python3
"""Analyze job trials from a jobs directory.

Scans through trial directories, extracts trajectory data and success metrics.
"""

import argparse
import asyncio
from pathlib import Path

from deepagents_harbor.analysis import (
    print_summary,
    scan_jobs_directory,
    write_trial_analysis,
)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze job trials from a jobs directory")
    parser.add_argument(
        "jobs_dir", type=Path, help="Path to the jobs directory (e.g., jobs-terminal-bench/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for detailed analysis files (one per failed trial)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, skip analysis of trials",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format",
    )

    args = parser.parse_args()

    # Scan and analyze all trials
    trials = await scan_jobs_directory(args.jobs_dir)

    # Print human-readable summary
    print_summary(trials)

    # If output directory specified and not summary-only, run deep analysis on failed trials
    if args.output_dir and not args.summary_only:
        failed_trials = [t for t in trials if not t.reward]

        if not failed_trials:
            print("\nNo failed trials to analyze.")
        else:
            print(f"\n{'=' * 80}")
            print("RUNNING DEEP ANALYSIS ON FAILED TRIALS")
            print(f"{'=' * 80}")
            print(f"Analyzing {len(failed_trials)} failed trials...")
            print(f"Output directory: {args.output_dir}")
            print()

            # Analyze each failed trial
            for i, trial in enumerate(failed_trials, 1):
                print(f"[{i}/{len(failed_trials)}] Analyzing {trial.trial_id}...")

                if trial.trial_dir is None:
                    print(f"  Warning: No trial directory found for {trial.trial_id}")
                    continue

                # Run the analysis and write to file
                try:
                    output_file = await write_trial_analysis(
                        trial, trial.trial_dir, args.output_dir
                    )
                    if output_file:
                        print(f"  ✓ Analysis written to: {output_file}")
                    else:
                        print(f"  ✗ Failed to analyze trial")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

            print(f"\n{'=' * 80}")
            print(f"Analysis complete. Results saved to: {args.output_dir}")
            print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
