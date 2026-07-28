"""CLI driver for generating LoHoSearch Harbor tasks.

Run as `python -m harbor_adapters.lohosearch.main`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harbor_adapters.lohosearch import adapter

_CACHE_DIRNAME = ".cache"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LoHoSearch Harbor tasks from a dataset manifest.",
    )
    parser.add_argument(
        "--populate",
        type=Path,
        metavar="DATASET_DIR",
        help=(
            "Resolve DATASET_DIR/manifest.json against the latest upstream LoHoSearch "
            "release and generate every listed task. Generated tasks are git-ignored, "
            "so run this before `harbor run --path DATASET_DIR`."
        ),
    )
    parser.add_argument(
        "--select",
        nargs="+",
        type=int,
        metavar="ROW",
        help=(
            "Print the durable `question_sha256` for the given zero-based row indices "
            "so they can be pasted into a manifest. Development helper."
        ),
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help=(
            "With --select, also print each question's first line. Off by default so "
            "benchmark plaintext is not written to shared terminals or CI logs."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        metavar="DIR",
        help="With --select, directory for the downloaded CSV. Defaults to "
        "./.cache/lohosearch. --populate always caches under DATASET_DIR/.cache.",
    )
    return parser


def _run_select(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir or Path(_CACHE_DIRNAME) / "lohosearch"
    rows = adapter.fetch_rows(cache_dir)
    for index in args.select:
        if not 0 <= index < len(rows):
            print(f"row {index}: out of range (0-{len(rows) - 1})")
            continue
        row = rows[index]
        print(f"row {index}: {row.question_sha256}")
        if args.show_text:
            print(f"    {row.question.splitlines()[0][:120]}")


def main(argv: list[str] | None = None) -> None:
    """Generate LoHoSearch Harbor tasks, or report row identities for selection.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]` when `None`.

    Raises:
        ValueError: If neither or both of `--populate` and `--select` are given.
    """
    args = _build_parser().parse_args(argv)

    if bool(args.populate) == bool(args.select):
        msg = "Exactly one of `--populate` or `--select` must be provided"
        raise ValueError(msg)

    if args.select:
        _run_select(args)
        return

    count = adapter.populate_tasks(args.populate)
    print(f"Generated {count} LoHoSearch task(s) in {args.populate}")


if __name__ == "__main__":
    main()
