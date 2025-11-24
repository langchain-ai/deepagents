#!/usr/bin/env python3
"""
Script to create a LangSmith dataset from Harbor tasks.
Downloads tasks from the Harbor registry and creates a LangSmith dataset.
"""

import argparse
import tempfile
import toml
from pathlib import Path
from typing import Optional

from langsmith import Client
from harbor.models.dataset_item import DownloadedDatasetItem
from harbor.registry.client import RegistryClient


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


def _extract_task_name(task_path: Path) -> str:
    """Extract the task name from the directory path."""
    return task_path.name


def _scan_downloaded_tasks(downloaded_tasks: list[DownloadedDatasetItem]) -> list:
    """
    Scan downloaded tasks and extract all task information.

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
        task_name = _extract_task_name(task_path)
        task_id = str(downloaded_task.id)

        if instruction:
            example = {
                "inputs": {
                    "task_id": task_id,
                    "task_name": task_name,
                    "instruction": instruction,
                    "metadata": metadata.get("metadata", {}),
                },
                "outputs": {},
            }
            examples.append(example)
            print(f"Added task: {task_name} (ID: {task_id})")

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

    if registry_url:
        registry_client = RegistryClient(url=registry_url)
    else:
        # Use default registry
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
    parser.add_argument(
        "--registry-url", type=str, help="URL of Harbor registry (uses default if not specified)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached remote tasks")

    args = parser.parse_args()

    create_langsmith_dataset(
        dataset_name=args.dataset_name,
        version=args.version,
        registry_url=args.registry_url,
        overwrite=args.overwrite,
    )
