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
            "Lay down each generated DRBench task's single-sourced, git-ignored files: "
            "the `main` service's build inputs and the verifier. App mode needs no "
            "document corpus on disk — the per-task image serves the documents. Run "
            "before `harbor run --path DATASET_DIR`. Mutually exclusive with "
            "--task-ids/--limit/--all."
        ),
    )
    parser.add_argument(
        "--refresh-digests",
        action="store_true",
        help=(
            "Re-resolve every task's upstream image tag to an immutable digest and "
            "rewrite vendor/image_digests.json. Requires network. Run this when "
            "upstream republishes images; it is the only step that is not offline."
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

    if args.refresh_digests:
        if args.populate is not None or args.task_ids or args.limit is not None or args.all:
            msg = "`--refresh-digests` is mutually exclusive with the other modes"
            raise ValueError(msg)
        count = adapter.refresh_image_digests()
        print(f"Refreshed {count} DRBench image digest(s)")
        return

    if args.populate is not None:
        if args.task_ids or args.limit is not None or args.all:
            msg = "`--populate` is mutually exclusive with `--task-ids`/`--limit`/`--all`"
            raise ValueError(msg)
        count = adapter.populate_corpus(args.populate)
        print(f"Populated {count} DRBench task(s) in {args.populate}")
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
