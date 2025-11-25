#!/usr/bin/env python3
"""
Analyze job trials from a jobs directory.

Scans through trial directories, extracts trajectory data and success metrics.
"""
import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Trial:
    """Metadata for a single trial run."""
    trial_id: str
    reward: bool
    trajectory_path: Optional[Path] = None
    reward_path: Optional[Path] = None



async def parse_reward(reward_path: Path) -> bool:
    """Parse the reward file. Returns True if reward is 1, False otherwise."""
    content = reward_path.read_text()
    reward_value = content.strip()
    return reward_value == "1"


async def analyze_trial(trial_dir: Path) -> Optional[Trial]:
    """Analyze a single trial directory."""
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    reward_path = trial_dir / "verifier" / "reward.txt"

    # Check if trajectory exists
    if not trajectory_path.exists():
        return None

    if not reward_path.exists():
        return None

    reward = reward_path.read_text().strip() == "1"
    trial_id = trial_dir.name
    return Trial(
        trial_id=trial_id,
        reward=reward,
        trajectory_path=trajectory_path,
        reward_path=reward_path if reward_path.exists() else None
    )


async def scan_jobs_directory(jobs_dir: Path) -> list[Trial]:
    """Scan the jobs directory and extract all trial metadata."""
    trials = []

    if not jobs_dir.exists():
        print(f"Error: Directory {jobs_dir} does not exist")
        return trials

    # Find all potential trial directories
    # Structure: jobs_dir / date_dir / trial_dir
    trial_dirs = []

    for date_dir in jobs_dir.iterdir():
        if date_dir.is_dir():
            for trial_dir in date_dir.iterdir():
                if trial_dir.is_dir():
                    # Check if this looks like a trial directory
                    if (trial_dir / "agent" / "trajectory.json").exists():
                        trial_dirs.append(trial_dir)

    print(f"Found {len(trial_dirs)} trial directories")

    trials = []
    for trial_dir in trial_dirs:
        trial = await analyze_trial(trial_dir)
        if trial is not None:
            trials.append(trial)
    return trials


def print_summary(trials: list[Trial]):
    """Print a summary of the analyzed trials."""
    print("\n" + "=" * 80)
    print(f"ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total trials: {len(trials)}")

    successful = sum(1 for t in trials if t.reward)
    failed = len(trials) - successful

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if trials:
        success_rate = (successful / len(trials)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print("\n" + "=" * 80)
    print("TRIAL DETAILS")
    print("=" * 80)

    for trial in trials:
        status = "✓ SUCCESS" if trial.reward else "✗ FAILED"
        print(f"\n{status} | {trial.trial_id}")

        if trial.trajectory_path:
            print(f"  Trajectory: {trial.trajectory_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze job trials from a jobs directory"
    )
    parser.add_argument(
        "jobs_dir",
        type=Path,
        help="Path to the jobs directory (e.g., jobs-terminal-bench/)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format"
    )

    args = parser.parse_args()

    # Scan and analyze all trials
    trials = await scan_jobs_directory(args.jobs_dir)
    # Print human-readable summary
    print_summary(trials)


if __name__ == "__main__":
    asyncio.run(main())
