#!/usr/bin/env python3
"""
Script to create a LangSmith dataset from Harbor tasks.
Downloads tasks from the Harbor registry and creates a LangSmith dataset.
"""

import argparse
import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import toml
from harbor.models.dataset_item import DownloadedDatasetItem
from harbor.registry.client import RegistryClient
from langsmith import Client


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


def _create_uuid_from_task_id(task_id: str) -> str:
    """Create a deterministic UUID from a task ID string using hash.

    Args:
        task_id: The task ID string to hash

    Returns:
        A UUID string generated from the hash of the task ID
    """
    # Create SHA-256 hash of the task_id
    hash_bytes = hashlib.sha256(task_id.encode('utf-8')).digest()

    # Use first 16 bytes to create a UUID
    task_uuid = uuid.UUID(bytes=hash_bytes[:16])

    return str(task_uuid)


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
        task_uuid = _create_uuid_from_task_id(task_id)

        if instruction:
            example = {
                "inputs": {
                    "task_id": task_id,
                    "task_uuid": task_uuid,
                    "task_name": task_name,
                    "instruction": instruction,
                    "metadata": metadata.get("metadata", {}),
                },
                "outputs": {},
            }
            examples.append(example)
            print(f"Added task: {task_name} (ID: {task_id}, UUID: {task_uuid})")

    return examples


def create_langsmith_dataset(
    dataset_name: str,
    version: str = "head",
    registry_url: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Create a LangSmith dataset from Harbor tasks.

    Args:
        dataset_name: Dataset name (used for both Harbor download and LangSmith dataset)
        version: Harbor dataset version (default: 'head')
        registry_url: URL of Harbor registry (uses default if not specified)
        overwrite: Whether to overwrite cached remote tasks
        output_dir: Directory to cache downloaded tasks (uses temp dir if not specified)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a LangSmith dataset by downloading tasks from Harbor registry."
    )
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., 'terminal-bench')")
    parser.add_argument(
        "--version", type=str, default="head", help="Dataset version (default: 'head')"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached remote tasks")

    args = parser.parse_args()

    create_langsmith_dataset(
        dataset_name=args.dataset_name,
        version=args.version,
        overwrite=args.overwrite,
    )
