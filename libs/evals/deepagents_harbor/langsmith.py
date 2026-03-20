"""LangSmith integration for Harbor: datasets, experiments, and feedback.

Provides functions for:

- Creating deterministic example IDs from task instructions
- Creating and ensuring LangSmith datasets from Harbor tasks
- Creating experiment sessions
- Adding reward feedback from Harbor job results to LangSmith traces
"""

import asyncio
import datetime
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import toml
from harbor.models.dataset_item import DownloadedDatasetItem
from harbor.registry.client import RegistryClientFactory
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

LANGSMITH_API_URL = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
HEADERS = {
    "x-api-key": os.getenv("LANGSMITH_API_KEY"),
}


# ============================================================================
# EXAMPLE IDS
# ============================================================================


def create_example_id_from_instruction(instruction: str, seed: int = 42) -> str:
    """Create a deterministic UUID from an instruction string.

    Normalizes the instruction by stripping whitespace and creating a SHA-256
    hash, then converting to a UUID for LangSmith compatibility.

    Args:
        instruction: The task instruction string to hash.
        seed: Integer seed to avoid collisions with existing examples.

    Returns:
        A UUID string generated from the hash of the normalized instruction.
    """
    normalized = instruction.strip()
    seeded_data = seed.to_bytes(8, byteorder="big") + normalized.encode("utf-8")
    hash_bytes = hashlib.sha256(seeded_data).digest()
    example_uuid = uuid.UUID(bytes=hash_bytes[:16])
    return str(example_uuid)


# ============================================================================
# DATASETS
# ============================================================================


def _read_instruction(task_path: Path) -> str:
    """Read the instruction.md file from a task directory."""
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists():
        return instruction_file.read_text()
    return ""


def _read_task_metadata(task_path: Path) -> dict[str, Any]:
    """Read metadata from task.toml file."""
    task_toml = task_path / "task.toml"
    if task_toml.exists():
        return toml.load(task_toml)
    return {}


def _read_solution(task_path: Path) -> str | None:
    """Read the solution script from a task directory.

    Args:
        task_path: Path to the task directory.

    Returns:
        Solution script content if it exists, None otherwise.
    """
    solution_file = task_path / "solution" / "solve.sh"
    if solution_file.exists():
        return solution_file.read_text()
    return None


def _scan_downloaded_tasks(
    downloaded_tasks: list[DownloadedDatasetItem],
) -> list[dict[str, Any]]:
    """Scan downloaded tasks and extract all task information.

    Args:
        downloaded_tasks: List of `DownloadedDatasetItem` objects from Harbor.

    Returns:
        List of example dictionaries for LangSmith.
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
            example_id = create_example_id_from_instruction(instruction)

            outputs = {}
            if solution:
                outputs["reference_solution"] = solution

            example = {
                "id": example_id,
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
            print(
                f"Added task: {task_name} (ID: {task_id}, Example ID: {example_id}) [{solution_status}]"
            )

    return examples


def create_dataset(dataset_name: str, version: str = "head", overwrite: bool = False) -> None:
    """Create a LangSmith dataset from Harbor tasks.

    Args:
        dataset_name: Dataset name (used for both Harbor download and
            LangSmith dataset).
        version: Harbor dataset version.
        overwrite: Whether to overwrite cached remote tasks.
    """
    langsmith_client = Client()
    output_dir = Path(tempfile.mkdtemp(prefix="harbor_tasks_"))
    print(f"Using temporary directory: {output_dir}")

    print(f"Downloading dataset '{dataset_name}@{version}' from Harbor registry...")
    registry_client = RegistryClientFactory.create()
    downloaded_tasks = registry_client.download_dataset(
        name=dataset_name,
        version=version,
        overwrite=overwrite,
        output_dir=output_dir,
    )

    print(f"Downloaded {len(downloaded_tasks)} tasks")
    examples = _scan_downloaded_tasks(downloaded_tasks)

    print(f"\nFound {len(examples)} tasks")

    print(f"\nCreating LangSmith dataset: {dataset_name}")
    dataset = langsmith_client.create_dataset(dataset_name=dataset_name)

    print(f"Dataset created with ID: {dataset.id}")

    print(f"\nAdding {len(examples)} examples to dataset...")
    langsmith_client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"\nSuccessfully created dataset '{dataset_name}' with {len(examples)} examples")
    print(f"Dataset ID: {dataset.id}")


def ensure_dataset(dataset_name: str, version: str = "head", overwrite: bool = False) -> None:
    """Create the dataset if it does not already exist.

    Args:
        dataset_name: Dataset name to look up in LangSmith.
        version: Harbor dataset version to use when creating the dataset.
        overwrite: Whether to overwrite cached remote tasks when creating
            the dataset.
    """
    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        create_dataset(dataset_name=dataset_name, version=version, overwrite=overwrite)
        return

    print(f"Dataset '{dataset_name}' already exists with ID: {dataset.id}")


# ============================================================================
# EXPERIMENTS
# ============================================================================


async def _create_experiment_session(
    dataset_id: str,
    name: str,
    metadata: dict[str, str],
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    """Create a LangSmith experiment session.

    Args:
        dataset_id: LangSmith dataset ID to associate with.
        name: Name for the experiment session.
        metadata: Metadata to attach to the experiment session.
        session: aiohttp ClientSession for making requests.

    Returns:
        Experiment session dictionary with `id` and `tenant_id` fields.
    """
    async with session.post(
        f"{LANGSMITH_API_URL}/sessions",
        headers=HEADERS,
        json={
            "start_time": datetime.datetime.now(datetime.UTC).isoformat(),
            "reference_dataset_id": dataset_id,
            "name": name,
            "metadata": metadata,
        },
    ) as experiment_response:
        if experiment_response.status == 200:  # noqa: PLR2004
            return await experiment_response.json()
        msg = (
            f"Failed to create experiment: "
            f"{experiment_response.status} {await experiment_response.text()}"
        )
        raise RuntimeError(msg)


async def _get_dataset_by_name(dataset_name: str, session: aiohttp.ClientSession) -> dict[str, Any]:
    """Get a LangSmith dataset by name.

    Args:
        dataset_name: Name of the dataset to retrieve.
        session: aiohttp `ClientSession` for making requests.

    Returns:
        Dataset dictionary with `id` field.

    Raises:
        LookupError: If the dataset is not found.
        RuntimeError: If the API request fails.
    """
    async with session.get(
        f"{LANGSMITH_API_URL}/datasets",
        headers=HEADERS,
        params={"name": dataset_name, "limit": "1"},
    ) as response:
        if response.status == 200:  # noqa: PLR2004
            datasets = await response.json()
            if len(datasets) > 0:
                return datasets[0]
            msg = f"Dataset '{dataset_name}' not found"
            raise LookupError(msg)
        msg = f"Failed to get dataset: {response.status} {await response.text()}"
        raise RuntimeError(msg)


async def create_experiment_async(
    dataset_name: str,
    experiment_name: str | None = None,
    *,
    metadata: dict[str, str] | None = None,
) -> str:
    """Create a LangSmith experiment session for the given dataset.

    Args:
        dataset_name: Name of the LangSmith dataset to create experiment for.
        experiment_name: Optional name for the experiment (auto-generated if
            not provided).
        metadata: Optional metadata to attach to the experiment session.

    Returns:
        The experiment name.
            Diagnostic output is printed to stderr; the returned name is the
            only value intended for stdout capture.
    """
    async with aiohttp.ClientSession() as session:
        dataset = await _get_dataset_by_name(dataset_name, session)
        dataset_id = dataset["id"]
        print(f"Found dataset '{dataset_name}' with ID: {dataset_id}", file=sys.stderr)

        if experiment_name is None:
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"{dataset_name}-{timestamp}"

        experiment_metadata = metadata or {}

        print(f"Creating experiment session: {experiment_name}", file=sys.stderr)
        experiment_session = await _create_experiment_session(
            dataset_id,
            experiment_name,
            experiment_metadata,
            session,
        )
        session_id = experiment_session["id"]
        tenant_id = experiment_session["tenant_id"]

        print("Experiment created successfully!", file=sys.stderr)
        print(f"  Session ID: {session_id}", file=sys.stderr)
        print(
            f"  View at: https://smith.langchain.com/o/{tenant_id}/datasets/{dataset_id}/compare?selectedSessions={session_id}",
            file=sys.stderr,
        )
        print("\nTo run Harbor with this experiment, use:", file=sys.stderr)
        print(f"  LANGSMITH_EXPERIMENT={experiment_name} harbor run ...", file=sys.stderr)

        return experiment_name


def create_experiment(
    dataset_name: str,
    experiment_name: str | None = None,
    *,
    metadata: dict[str, str] | None = None,
) -> str:
    """Synchronous wrapper for `create_experiment_async`."""
    return asyncio.run(
        create_experiment_async(
            dataset_name,
            experiment_name,
            metadata=metadata,
        )
    )


# ============================================================================
# FEEDBACK
# ============================================================================


def _extract_reward(trial_dir: Path) -> tuple[float, str | None]:
    """Extract reward from trial's `result.json`.

    Falls back to `0.0` when the verifier did not produce a usable reward
    (e.g. `verifier_result` is missing, empty, or lacks a `rewards.reward`
    key).

    Args:
        trial_dir: Path to the trial directory.

    Returns:
        A `(reward, comment)` tuple.  `comment` is `None` when the reward
            was extracted normally, or a short explanation when a fallback
            was used.

    Raises:
        FileNotFoundError: If `result.json` does not exist.
        ValueError: If `result.json` contains malformed JSON.
    """
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        msg = f"{result_path} does not exist"
        raise FileNotFoundError(msg)

    try:
        with result_path.open() as f:
            result = json.load(f)
    except json.JSONDecodeError as exc:
        msg = f"malformed JSON in {result_path}: {exc}"
        raise ValueError(msg) from exc

    verifier_result = result.get("verifier_result")
    if not isinstance(verifier_result, dict):
        print(f"  Warning: no verifier_result in {result_path}", file=sys.stderr)
        return 0.0, "no verifier_result — agent likely failed or timed out"

    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict) or "reward" not in rewards:
        print(f"  Warning: no reward key in {result_path}", file=sys.stderr)
        return 0.0, "no reward key in verifier_result"

    raw = rewards["reward"]
    if not isinstance(raw, int | float):
        print(
            f"  Warning: reward is {type(raw).__name__} in {result_path}",
            file=sys.stderr,
        )
        return 0.0, f"reward value is {type(raw).__name__}, expected number"

    return float(raw), None


def _process_trial(
    client: Client,
    trial_dir: Path,
    project_name: str,
    dry_run: bool = False,
) -> dict[str, str]:
    """Process a single trial and update its trace."""
    trial_name = trial_dir.name

    try:
        filter_query = f'and(eq(metadata_key, "trial_name"), eq(metadata_value, "{trial_name}"))'
        runs = list(
            client.list_runs(
                project_name=project_name,
                filter=filter_query,
                is_root=True,
            )
        )
    except Exception as e:  # noqa: BLE001  # LangSmith API; any failure → error status
        return {"status": "error", "message": f"Failed to fetch trace: {e}"}

    if not runs:
        return {
            "status": "error",
            "message": f"No trace found for trial_name {trial_name}",
        }

    if len(runs) > 1:
        return {
            "status": "error",
            "message": f"Multiple traces found for trial_name {trial_name}",
        }

    run = runs[0]
    run_id = str(run.id)

    try:
        feedback_list = list(client.list_feedback(run_ids=[run_id]))
        if any(fb.key == "harbor_reward" for fb in feedback_list):
            return {"status": "skipped", "message": "Feedback already exists"}
    except Exception as exc:  # noqa: BLE001  # dedup check is best-effort
        print(
            f"  Warning: feedback dedup check failed ({type(exc).__name__}: {exc}), proceeding anyway",
            file=sys.stderr,
        )

    try:
        reward, comment = _extract_reward(trial_dir)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}

    status = "fallback" if comment else "success"

    if not dry_run:
        try:
            client.create_feedback(
                run_id=run_id,
                key="harbor_reward",
                score=reward,
                comment=comment,
            )
        except Exception as exc:  # noqa: BLE001  # LangSmith API; any failure → error status
            return {
                "status": "error",
                "message": f"Failed to submit feedback: {exc}",
            }
        return {
            "status": status,
            "message": f"Added harbor_reward feedback: {reward}"
            + (f" ({comment})" if comment else ""),
        }
    return {
        "status": status,
        "message": f"Would add harbor_reward feedback: {reward}"
        + (f" ({comment})" if comment else ""),
    }


def add_feedback(job_folder: Path, project_name: str, dry_run: bool = False) -> None:
    """Add Harbor reward feedback to LangSmith traces.

    Args:
        job_folder: Path to the Harbor job folder.
        project_name: LangSmith project name to search for traces.
        dry_run: If True, show what would be done without making changes.
    """
    print(f"Processing job folder: {job_folder}")
    print(f"LangSmith project: {project_name}")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    print()

    trial_dirs = [d for d in job_folder.iterdir() if d.is_dir()]
    print(f"Found {len(trial_dirs)} trial directories\n")

    results = {"success": 0, "fallback": 0, "skipped": 0, "error": 0}
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
        elif status == "fallback":
            print(f"  ⚠ {message}")
            results["fallback"] += 1
        elif status == "skipped":
            print(f"  ⊘ {message}")
            results["skipped"] += 1
        else:
            print(f"  ✗ {message}")
            results["error"] += 1

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total trials: {len(trial_dirs)}")
    print(f"Successfully updated: {results['success']}")
    print(f"Fallback to 0.0 (no verifier result): {results['fallback']}")
    print(f"Skipped (already has feedback): {results['skipped']}")
    print(f"Errors: {results['error']}")
