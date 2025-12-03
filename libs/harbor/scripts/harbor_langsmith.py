#!/usr/bin/env python3
"""
CLI for LangSmith integration with Harbor.

Provides commands for:
- Creating LangSmith datasets from Harbor tasks
- Creating experiment sessions
- Adding feedback from Harbor job results to LangSmith traces
"""

import argparse
import asyncio
import datetime
import json
import os
import tempfile
from pathlib import Path

import aiohttp
import toml
from dotenv import load_dotenv
from harbor.models.dataset_item import DownloadedDatasetItem
from harbor.registry.client import RegistryClient
from langsmith import Client

from deepagents_harbor.tracing import create_example_id_from_instruction

load_dotenv()

LANGSMITH_API_URL = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
HEADERS = {
    "x-api-key": os.getenv("LANGSMITH_API_KEY"),
}


# ============================================================================
# CREATE DATASET
# ============================================================================


def _read_instruction(task_path: Path) -> str:
    """Read the instruction.md file from a task directory."""
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists():
        return instruction_file.read_text()
    return ""


def _read_task_metadata(task_path: Path) -> dict:
    """Read metadata from task.toml file."""
    task_toml = task_path / "task.toml"
    if task_toml.exists():
        return toml.load(task_toml)
    return {}


def _read_solution(task_path: Path) -> str | None:
    """Read the solution script from a task directory.

    Args:
        task_path: Path to the task directory

    Returns:
        Solution script content if it exists, None otherwise
    """
    solution_file = task_path / "solution" / "solve.sh"
    if solution_file.exists():
        return solution_file.read_text()
    return None


def _scan_downloaded_tasks(downloaded_tasks: list[DownloadedDatasetItem]) -> list:
    """Scan downloaded tasks and extract all task information.

    Args:
        downloaded_tasks: List of DownloadedDatasetItem objects from Harbor

    Returns:
        List of example dictionaries for LangSmith
    """
    examples = []

    for downloaded_task in downloaded_tasks:
        task_path = downloaded_task.downloaded_path

        instruction = _read_instruction(task_path)
        metadata = _read_task_metadata(task_path)
        solution = _read_solution(task_path)
        task_name = downloaded_task.id.name
        task_id = str(downloaded_task.id)

        if instruction:
            # Create deterministic example_id from instruction content
            example_id = create_example_id_from_instruction(instruction)

            # Build outputs dict with reference solution if available
            outputs = {}
            if solution:
                outputs["reference_solution"] = solution

            example = {
                "id": example_id,  # Explicitly set the example ID
                "inputs": {
                    "task_id": task_id,
                    "task_name": task_name,
                    "instruction": instruction,
                    "metadata": metadata.get("metadata", {}),
                },
                "outputs": outputs,
            }
            examples.append(example)

            solution_status = "with solution" if solution else "without solution"
            print(f"Added task: {task_name} (ID: {task_id}, Example ID: {example_id}) [{solution_status}]")

    return examples


def create_dataset(dataset_name: str, version: str = "head", overwrite: bool = False) -> None:
    """Create a LangSmith dataset from Harbor tasks.

    Args:
        dataset_name: Dataset name (used for both Harbor download and LangSmith dataset)
        version: Harbor dataset version (default: 'head')
        overwrite: Whether to overwrite cached remote tasks
    """
    langsmith_client = Client()
    output_dir = Path(tempfile.mkdtemp(prefix="harbor_tasks_"))
    print(f"Using temporary directory: {output_dir}")

    # Download from Harbor registry
    print(f"Downloading dataset '{dataset_name}@{version}' from Harbor registry...")
    registry_client = RegistryClient()
    downloaded_tasks = registry_client.download_dataset(
        name=dataset_name,
        version=version,
        overwrite=overwrite,
        output_dir=output_dir,
    )

    print(f"Downloaded {len(downloaded_tasks)} tasks")
    examples = _scan_downloaded_tasks(downloaded_tasks)

    print(f"\nFound {len(examples)} tasks")

    # Create the dataset
    print(f"\nCreating LangSmith dataset: {dataset_name}")
    dataset = langsmith_client.create_dataset(dataset_name=dataset_name)

    print(f"Dataset created with ID: {dataset.id}")

    # Add examples to the dataset
    print(f"\nAdding {len(examples)} examples to dataset...")
    langsmith_client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"\nSuccessfully created dataset '{dataset_name}' with {len(examples)} examples")
    print(f"Dataset ID: {dataset.id}")


# ============================================================================
# CREATE EXPERIMENT
# ============================================================================


async def _create_experiment_session(
    dataset_id: str, name: str, session: aiohttp.ClientSession
) -> dict:
    """Create a LangSmith experiment session.

    Args:
        dataset_id: LangSmith dataset ID to associate with
        name: Name for the experiment session
        session: aiohttp ClientSession for making requests

    Returns:
        Experiment session dictionary with 'id' field
    """
    async with session.post(
        f"{LANGSMITH_API_URL}/sessions",
        headers=HEADERS,
        json={
            "start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reference_dataset_id": dataset_id,
            "name": name,
        },
    ) as experiment_response:
        if experiment_response.status == 200:
            return await experiment_response.json()
        else:
            raise Exception(
                f"Failed to create experiment: {experiment_response.status} {await experiment_response.text()}"
            )


async def _get_dataset_by_name(dataset_name: str, session: aiohttp.ClientSession) -> dict:
    """Get a LangSmith dataset by name.

    Args:
        dataset_name: Name of the dataset to retrieve
        session: aiohttp ClientSession for making requests

    Returns:
        Dataset dictionary with 'id' field
    """
    async with session.get(
        f"{LANGSMITH_API_URL}/datasets?name={dataset_name}&limit=1",
        headers=HEADERS,
    ) as response:
        if response.status == 200:
            datasets = await response.json()
            if len(datasets) > 0:
                return datasets[0]
            else:
                raise Exception(f"Dataset '{dataset_name}' not found")
        else:
            raise Exception(f"Failed to get dataset: {response.status} {await response.text()}")


async def create_experiment_async(dataset_name: str, experiment_name: str | None = None) -> str:
    """Create a LangSmith experiment session for the given dataset.

    Args:
        dataset_name: Name of the LangSmith dataset to create experiment for
        experiment_name: Optional name for the experiment (auto-generated if not provided)

    Returns:
        The experiment session ID
    """
    async with aiohttp.ClientSession() as session:
        # Get the dataset
        dataset = await _get_dataset_by_name(dataset_name, session)
        dataset_id = dataset["id"]
        print(f"Found dataset '{dataset_name}' with ID: {dataset_id}")

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"harbor-experiment-{timestamp}"

        # Create experiment session
        print(f"Creating experiment session: {experiment_name}")
        experiment_session = await _create_experiment_session(dataset_id, experiment_name, session)
        session_id = experiment_session["id"]
        tenant_id = experiment_session["tenant_id"]

        print(f"✓ Experiment created successfully!")
        print(f"  Session ID: {session_id}")
        print(
            f"  View at: https://smith.langchain.com/o/{tenant_id}/datasets/{dataset_id}/compare?selectedSessions={session_id}"
        )
        print(f"\nTo run Harbor with this experiment, use:")
        print(f"  LANGSMITH_EXPERIMENT={experiment_name} harbor run ...")

        return session_id


def create_experiment(dataset_name: str, experiment_name: str | None = None) -> str:
    """Synchronous wrapper for create_experiment_async."""
    return asyncio.run(create_experiment_async(dataset_name, experiment_name))


# ============================================================================
# ADD FEEDBACK
# ============================================================================


def _extract_reward(trial_dir: Path) -> float:
    """Extract reward from trial's result.json."""
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        # If task completed but no result.json, assume reward 0.0 as default
        # because it was likely due to an exception.
        return 0.0

    with open(result_path) as f:
        result = json.load(f)
        verifier_result = result.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        return rewards.get("reward")


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


def add_feedback(job_folder: Path, project_name: str, dry_run: bool = False) -> None:
    """Add Harbor reward feedback to LangSmith traces.

    Args:
        job_folder: Path to the Harbor job folder
        project_name: LangSmith project name to search for traces
        dry_run: If True, show what would be done without making changes
    """
    print(f"Processing job folder: {job_folder}")
    print(f"LangSmith project: {project_name}")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    print()

    # Find all trial directories
    trial_dirs = [d for d in job_folder.iterdir() if d.is_dir()]
    print(f"Found {len(trial_dirs)} trial directories\n")

    results = {"success": 0, "skipped": 0, "error": 0}
    client = Client()

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


def main() -> None:
    """Main CLI entrypoint with subcommands."""
    parser = argparse.ArgumentParser(
        description="Harbor-LangSmith integration CLI for managing datasets, experiments, and feedback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # ========================================================================
    # create-dataset subcommand
    # ========================================================================
    dataset_parser = subparsers.add_parser(
        "create-dataset",
        help="Create a LangSmith dataset from Harbor tasks",
    )
    dataset_parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (e.g., 'terminal-bench')",
    )
    dataset_parser.add_argument(
        "--version",
        type=str,
        default="head",
        help="Dataset version (default: 'head')",
    )
    dataset_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached remote tasks",
    )

    # ========================================================================
    # create-experiment subcommand
    # ========================================================================
    experiment_parser = subparsers.add_parser(
        "create-experiment",
        help="Create an experiment session for a dataset",
    )
    experiment_parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (must already exist in LangSmith)",
    )
    experiment_parser.add_argument(
        "--name",
        type=str,
        help="Name for the experiment (auto-generated if not provided)",
    )

    # ========================================================================
    # add-feedback subcommand
    # ========================================================================
    feedback_parser = subparsers.add_parser(
        "add-feedback",
        help="Add Harbor reward feedback to LangSmith traces",
    )
    feedback_parser.add_argument(
        "job_folder",
        type=Path,
        help="Path to the job folder (e.g., jobs/terminal-bench/2025-12-02__16-25-40)",
    )
    feedback_parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="LangSmith project name to search for traces",
    )
    feedback_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.command == "create-dataset":
        create_dataset(
            dataset_name=args.dataset_name,
            version=args.version,
            overwrite=args.overwrite,
        )
    elif args.command == "create-experiment":
        create_experiment(
            dataset_name=args.dataset_name,
            experiment_name=args.name,
        )
    elif args.command == "add-feedback":
        if not args.job_folder.exists():
            print(f"Error: Job folder does not exist: {args.job_folder}")
            return 1
        add_feedback(
            job_folder=args.job_folder,
            project_name=args.project_name,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    exit(main())
