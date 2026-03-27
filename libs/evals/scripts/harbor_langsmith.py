#!/usr/bin/env python3
"""CLI for LangSmith integration with Harbor.

Thin CLI wrapper around `deepagents_harbor.langsmith`. All business logic
lives in that module; this script only handles argument parsing.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from deepagents_harbor.langsmith import (
    add_feedback,
    create_dataset,
    create_experiment,
    ensure_dataset,
)

load_dotenv()


def main() -> int:
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
    # ensure-dataset subcommand
    # ========================================================================
    ensure_dataset_parser = subparsers.add_parser(
        "ensure-dataset",
        help="Ensure a LangSmith dataset exists for Harbor tasks",
    )
    ensure_dataset_parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (e.g., 'terminal-bench')",
    )
    ensure_dataset_parser.add_argument(
        "--version",
        type=str,
        default="head",
        help="Dataset version (default: 'head')",
    )
    ensure_dataset_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached remote tasks when creating the dataset",
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
    experiment_parser.add_argument(
        "--model",
        type=str,
        help="Model identifier used as suffix in auto-generated experiment names (e.g. 'anthropic:claude-sonnet-4-6')",
    )
    experiment_parser.add_argument(
        "--metadata",
        type=str,
        default="{}",
        help="JSON metadata to attach to the experiment session",
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
    elif args.command == "ensure-dataset":
        ensure_dataset(
            dataset_name=args.dataset_name,
            version=args.version,
            overwrite=args.overwrite,
        )
    elif args.command == "create-experiment":
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"Error: --metadata must be valid JSON: {exc}", file=sys.stderr)
            return 1
        if not isinstance(metadata, dict):
            print("Error: --metadata must be a JSON object.", file=sys.stderr)
            return 1
        name = create_experiment(
            dataset_name=args.dataset_name,
            experiment_name=args.name,
            model=args.model,
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
        print(name)
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
    sys.exit(main())
