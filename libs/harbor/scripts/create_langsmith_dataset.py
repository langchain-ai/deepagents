#!/usr/bin/env python3
"""
Script to create a LangSmith dataset from Harbor tasks.
Downloads tasks from the Harbor registry and creates a LangSmith dataset.
"""

import argparse
import asyncio
import datetime
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

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
        task_name = downloaded_task.id.name
        task_id = str(downloaded_task.id)

        if instruction:
            # Create deterministic example_id from instruction content
            example_id = create_example_id_from_instruction(instruction)

            example = {
                "id": example_id,  # Explicitly set the example ID
                "inputs": {
                    "task_id": task_id,
                    "task_name": task_name,
                    "instruction": instruction,
                    "metadata": metadata.get("metadata", {}),
                },
                "outputs": {},
            }
            examples.append(example)
            print(f"Added task: {task_name} (ID: {task_id}, Example ID: {example_id})")

    return examples


def create_langsmith_dataset(
    dataset_name: str,
    version: str = "head",
    overwrite: bool = False,
) -> None:
    """
    Create a LangSmith dataset from Harbor tasks.

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
            "description": "Harbor experiment run via REST API",
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
            raise Exception(
                f"Failed to get dataset: {response.status} {await response.text()}"
            )


async def create_experiment(dataset_name: str, experiment_name: str | None = None) -> str:
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
        experiment_session = await _create_experiment_session(
            dataset_id, experiment_name, session
        )
        session_id = experiment_session["id"]
        tenant_id = experiment_session["tenant_id"]

        print(f"✓ Experiment created successfully!")
        print(f"  Session ID: {session_id}")
        print(
            f"  View at: https://smith.langchain.com/o/{tenant_id}/datasets/{dataset_id}/compare?selectedSessions={session_id}"
        )

        return session_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a LangSmith dataset or experiment from Harbor tasks."
    )
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., 'terminal-bench')")
    parser.add_argument(
        "--version", type=str, default="head", help="Dataset version (default: 'head')"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached remote tasks")
    parser.add_argument(
        "--create-experiment",
        action="store_true",
        help="Create an experiment session for the dataset (dataset must already exist)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment (auto-generated if not provided)",
    )

    args = parser.parse_args()

    if args.create_experiment:
        # Create experiment for existing dataset
        session_id = asyncio.run(
            create_experiment(args.dataset_name, args.experiment_name)
        )
        print(f"\nExperiment session ID: {session_id}")
        print("Set this as an environment variable when running Harbor:")
        print(f"export LANGSMITH_EXPERIMENT_SESSION_ID={session_id}")
    else:
        # Create dataset from Harbor tasks
        create_langsmith_dataset(
            dataset_name=args.dataset_name,
            version=args.version,
            overwrite=args.overwrite,
        )
