#!/usr/bin/env python3
"""Script to attach harbor_reward feedback to LangSmith traces based on Harbor job results.

This script matches LangSmith traces to Harbor trials using the trial_name metadata field.
The trial_name is derived from the trial directory name (e.g., 'chess-best-move__FvULBbA').
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


def _extract_reward(trial_dir: Path) -> Optional[float]:
    """Extract reward from trial's result.json."""
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None

    try:
        with open(result_path) as f:
            result = json.load(f)
            return result.get("verifier_result", {}).get("rewards", {}).get("reward")
    except Exception as e:
        print(f"  Error reading result.json: {e}")
        return None


def _process_trial(
    client: Client,
    trial_dir: Path,
    project_name: str,
    dry_run: bool = False,
) -> dict:
    """Process a single trial and update its trace."""
    trial_name = trial_dir.name

    # Find the trace by trial_name metadata
    try:
        # Build filter to match trial_name in metadata
        filter_query = f'and(eq(metadata_key, "trial_name"), eq(metadata_value, "{trial_name}"))'

        # Fetch runs matching the filter
        runs = list(
            client.list_runs(
                project_name=project_name,
                filter=filter_query,
                is_root=True,
            )
        )
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch trace: {e}"}

    if not runs:
        return {"status": "error", "message": f"No trace found for trial_name {trial_name}"}

    if len(runs) > 1:
        return {"status": "error", "message": f"Multiple traces found for trial_name {trial_name}"}

    run = runs[0]
    run_id = str(run.id)

    # Check if feedback already exists
    try:
        feedback_list = list(client.list_feedback(run_ids=[run_id]))
        if any(fb.key == "harbor_reward" for fb in feedback_list):
            return {"status": "skipped", "message": "Feedback already exists"}
    except Exception:
        pass  # Continue if feedback check fails

    # Extract reward
    reward = _extract_reward(trial_dir)
    if reward is None:
        return {"status": "error", "message": "No reward found in result.json"}

    if not dry_run:
        client.create_feedback(
            run_id=run_id,
            key="harbor_reward",
            score=reward,
        )
        return {
            "status": "success",
            "message": f"Added harbor_reward feedback: {reward}",
        }
    else:
        return {
            "status": "success",
            "message": f"Would add harbor_reward feedback: {reward}",
        }


def _process_job_folder(
    job_folder: Path,
    project_name: str,
    dry_run: bool = False,
) -> None:
    """Process all trials in a job folder."""
    client = Client()

    print(f"Processing job folder: {job_folder}")
    print(f"LangSmith project: {project_name}")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    print()

    # Find all trial directories
    trial_dirs = [d for d in job_folder.iterdir() if d.is_dir()]
    print(f"Found {len(trial_dirs)} trial directories\n")

    results = {"success": 0, "skipped": 0, "error": 0}

    for i, trial_dir in enumerate(trial_dirs, 1):
        print(f"[{i}/{len(trial_dirs)}] Processing {trial_dir.name}...")

        result = _process_trial(
            trial_dir=trial_dir,
            project_name=project_name,
            client=client,
            dry_run=dry_run,
        )

        status = result["status"]
        message = result["message"]

        if status == "success":
            print(f"  ✓ {message}")
            results["success"] += 1
        elif status == "skipped":
            print(f"  ⊘ {message}")
            results["skipped"] += 1
        else:  # error
            print(f"  ✗ {message}")
            results["error"] += 1

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total trials: {len(trial_dirs)}")
    print(f"Successfully updated: {results['success']}")
    print(f"Skipped (already has feedback): {results['skipped']}")
    print(f"Errors: {results['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Associate rewards with LangSmith traces based on Harbor job results.",
    )
    parser.add_argument(
        "job_folder",
        type=Path,
        help="Path to the job folder (e.g., jobs/terminal-bench/2025-11-26__22-44-45)",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="LangSmith project name to search for traces",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if not args.job_folder.exists():
        print(f"Error: Job folder does not exist: {args.job_folder}")
        return 1

    _process_job_folder(
        job_folder=args.job_folder,
        project_name=args.project_name,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    exit(main())
