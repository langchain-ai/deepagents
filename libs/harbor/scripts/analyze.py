#!/usr/bin/env python3
"""Analyze job trials from a jobs directory.

Scans through trial directories, extracts trajectory data and success metrics.
"""

import argparse
import asyncio
from pathlib import Path

from deepagents_harbor.analysis import print_summary, scan_jobs_directory


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze job trials from a jobs directory")
    parser.add_argument(
        "jobs_dir", type=Path, help="Path to the jobs directory (e.g., jobs-terminal-bench/)"
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


if __name__ == "__main__":
    asyncio.run(main())
