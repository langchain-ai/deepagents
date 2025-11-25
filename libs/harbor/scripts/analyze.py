#!/usr/bin/env python3
"""Analyze job trials from a jobs directory.

Scans through trial directories, extracts trajectory data and success metrics.
"""

import argparse
import asyncio
from pathlib import Path

from deepagents_harbor.analysis import (
    TrialStatus,
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
        help="Output directory for detailed analysis files (one per failed/pending trial)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, skip detailed LLM analysis of trials",
    )
    parser.add_argument(
        "--analyze-pending",
        action="store_true",
        help="Analyze pending trials in addition to failed trials",
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

    # If output directory specified, run analysis on trials
    if args.output_dir:
        # Determine which trials to analyze based on status
        trials_to_analyze = [
            t for t in trials
            if t.status == TrialStatus.FAILED or (args.analyze_pending and t.status == TrialStatus.PENDING)
        ]

        if not trials_to_analyze:
            status_desc = "failed or pending" if args.analyze_pending else "failed"
            print(f"\nNo {status_desc} trials to analyze.")
        else:
            print(f"\n{'=' * 80}")
            analysis_mode = "SUMMARY" if args.summary_only else "DEEP ANALYSIS"
            trial_types = "FAILED/PENDING" if args.analyze_pending else "FAILED"
            print(f"RUNNING {analysis_mode} ON {trial_types} TRIALS")
            print(f"{'=' * 80}")
            print(f"Processing {len(trials_to_analyze)} trials...")
            print(f"Output directory: {args.output_dir}")
            if args.summary_only:
                print("Mode: Summary only (LLM analysis disabled)")
            if args.analyze_pending:
                print("Mode: Including pending trials")
            print()

            # Analyze each trial
            for i, trial in enumerate(trials_to_analyze, 1):
                status_label = trial.status.value.upper()
                print(f"[{i}/{len(trials_to_analyze)}] Analyzing {trial.trial_id} ({status_label})...")

                if trial.trial_dir is None:
                    print(f"  Warning: No trial directory found for {trial.trial_id}")
                    continue

                # Run the analysis and write to file
                try:
                    output_file = await write_trial_analysis(
                        trial,
                        trial.trial_dir,
                        args.output_dir,
                        summary_only=args.summary_only,
                        analyze_pending=args.analyze_pending,
                    )
                    if output_file:
                        print(f"  ✓ Analysis written to: {output_file}")
                    else:
                        print(f"  ✗ Skipped (no trajectory or already completed)")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

            print(f"\n{'=' * 80}")
            print(f"Analysis complete. Results saved to: {args.output_dir}")
            print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
