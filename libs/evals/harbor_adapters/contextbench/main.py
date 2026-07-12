"""CLI driver for generating Context-Bench Harbor tasks by id.

Run as `python -m harbor_adapters.contextbench.main`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harbor_adapters.contextbench import adapter

_DEFAULT_SUITE = "cloud"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Context-Bench Harbor tasks by id.",
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
        help="Task ids to generate, e.g. `cb-cloud-1`.",
    )
    parser.add_argument(
        "--populate",
        type=Path,
        metavar="DATASET_DIR",
        help=(
            "Populate each generated Context-Bench task's environment/files/ from "
            "the single vendored corpus (the per-task corpus is git-ignored). Run "
            "before `harbor run --path DATASET_DIR`. Mutually exclusive with "
            "--task-ids/--limit."
        ),
    )
    parser.add_argument(
        "--stamp-tiers",
        type=Path,
        metavar="DATASET_DIR",
        help=(
            "Overwrite each task's `difficulty` in DATASET_DIR with its calibrated "
            "tier from --calibration. Mutually exclusive with --task-ids/--limit/--populate."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        metavar="CALIBRATION_JSON",
        help="Calibration record (JSON) read by --stamp-tiers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            f"When set and `--task-ids` is omitted, generate the first N tasks "
            f"of the `{_DEFAULT_SUITE}` suite."
        ),
    )
    return parser


def _resolve_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        return list(args.task_ids)
    if args.limit is not None:
        return [f"cb-{_DEFAULT_SUITE}-{i}" for i in range(args.limit)]
    msg = "Either `--task-ids` or `--limit` must be provided"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    """Generate one or more Context-Bench Harbor tasks by id.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]` when `None`.

    Raises:
        ValueError: If neither `--task-ids` nor `--limit` is provided.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.stamp_tiers is not None:
        if args.task_ids or args.limit is not None or args.populate is not None:
            msg = "`--stamp-tiers` is mutually exclusive with --task-ids/--limit/--populate"
            raise ValueError(msg)
        if args.calibration is None:
            msg = "`--stamp-tiers` requires `--calibration`"
            raise ValueError(msg)
        count = adapter.stamp_calibrated_tiers(args.stamp_tiers, args.calibration)
        print(f"Stamped calibrated tiers for {count} task(s) in {args.stamp_tiers}")
        return

    if args.populate is not None:
        if args.task_ids or args.limit is not None:
            msg = "`--populate` is mutually exclusive with `--task-ids`/`--limit`"
            raise ValueError(msg)
        count = adapter.populate_corpus(args.populate)
        print(f"Populated corpus for {count} Context-Bench task(s) in {args.populate}")
        return

    if args.output_dir is None:
        msg = "`--output-dir` is required unless `--populate` is given"
        raise ValueError(msg)
    task_ids = _resolve_task_ids(args)

    for task_id in task_ids:
        adapter.record_for_task_id(task_id)  # validates the id up front
        suite, line_index = adapter.parse_task_id(task_id)
        adapter.generate_task(
            source_jsonl=adapter.vendor_dir() / f"filesystem_{suite}.jsonl",
            source_files_dir=adapter.vendor_dir() / "files",
            output_dir=args.output_dir,
            task_id=task_id,
            line_index=line_index,
        )


if __name__ == "__main__":
    main()
