"""CLI driver for generating DRBench Harbor tasks by id.

Run as `python -m harbor_adapters.drbench.main`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harbor_adapters.drbench import adapter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DRBench Harbor tasks by id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Dataset directory that will contain the generated task(s). "
            "Required unless --populate is given."
        ),
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        metavar="ID",
        help="Task ids to generate, e.g. `DR0001`.",
    )
    parser.add_argument(
        "--populate",
        type=Path,
        metavar="DATASET_DIR",
        help=(
            "Populate each generated DRBench task's environment/files/ from the pinned "
            "upstream tree, plus the single-sourced build and verifier files (the "
            "per-task corpus is git-ignored). Run before `harbor run --path DATASET_DIR`. "
            "Mutually exclusive with --task-ids/--limit."
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        metavar="TARBALL",
        help=(
            "Optional pre-downloaded upstream tarball for --populate, so a rerun can "
            "skip the download."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "When set and `--task-ids` is omitted, generate the first N vendored tasks "
            "in id order."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate every vendored task. Mutually exclusive with --task-ids/--limit.",
    )
    return parser


def _resolve_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        if args.all or args.limit is not None:
            msg = "`--task-ids` is mutually exclusive with `--all`/`--limit`"
            raise ValueError(msg)
        return list(args.task_ids)
    available = adapter.available_task_ids()
    if args.all:
        if args.limit is not None:
            msg = "`--all` is mutually exclusive with `--limit`"
            raise ValueError(msg)
        return available
    if args.limit is not None:
        return available[: args.limit]
    msg = "One of `--task-ids`, `--limit`, or `--all` must be provided"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    """Generate one or more DRBench Harbor tasks, or populate a generated dataset.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]` when `None`.

    Raises:
        ValueError: If the selected flags are mutually exclusive or none identify
            which tasks to generate.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.populate is not None:
        if args.task_ids or args.limit is not None or args.all:
            msg = "`--populate` is mutually exclusive with `--task-ids`/`--limit`/`--all`"
            raise ValueError(msg)
        count = adapter.populate_corpus(args.populate, archive=args.archive)
        print(f"Populated corpus for {count} DRBench task(s) in {args.populate}")
        return

    if args.output_dir is None:
        msg = "`--output-dir` is required unless `--populate` is given"
        raise ValueError(msg)
    task_ids = _resolve_task_ids(args)

    for task_id in task_ids:
        adapter.generate_task(output_dir=args.output_dir, task_id=task_id)
    print(f"Generated {len(task_ids)} DRBench task(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
