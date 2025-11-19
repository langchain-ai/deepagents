#!/usr/bin/env python3
"""Export Harbor ATIF trajectories to LangSmith for visualization."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harbor_deepagents.utils.langsmith_exporter import LangSmithTrajectoryExporter


def main():
    """Export all trajectories from latest job to LangSmith."""
    jobs_dir = Path("jobs")

    if not jobs_dir.exists():
        print("✗ No jobs directory found. Run a job first!")
        print("  → uv run harness --model openai/gpt-5-mini")
        return 1

    # Find most recent job
    job_dirs = sorted(
        [d for d in jobs_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not job_dirs:
        print("✗ No jobs found in jobs directory")
        return 1

    latest_job = job_dirs[0]
    print(f"📦 Exporting trajectories from: {latest_job.name}")
    print("")

    # Find all trajectory files
    trajectory_files = list(latest_job.rglob("trajectory.json"))

    if not trajectory_files:
        print("✗ No trajectory files found")
        print("  Trajectories are saved to: jobs/<job>/<trial>/agent/trajectory.json")
        return 1

    print(f"Found {len(trajectory_files)} trajectory file(s)")
    print("")

    # Export to LangSmith
    exporter = LangSmithTrajectoryExporter()

    exported_count = 0
    failed_count = 0

    for trajectory_path in trajectory_files:
        trial_name = trajectory_path.parent.parent.name
        try:
            run_id = exporter.export_trajectory(
                trajectory_path,
                project_name="harbor-deepagents",
            )
            print(f"✓ Exported {trial_name}: {run_id[:12]}...")
            exported_count += 1
        except Exception as e:
            print(f"✗ Failed to export {trial_name}: {e}")
            failed_count += 1

    print("")
    print("=" * 60)
    print(f"Export complete! {exported_count} succeeded, {failed_count} failed")
    print("=" * 60)
    print("")

    if exported_count > 0:
        print("View in LangSmith:")
        print("  → https://smith.langchain.com")
        print("  → Project: harbor-deepagents")
    print("")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
