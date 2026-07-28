"""CLI driver for the OOLONG-synth Harbor dataset.

Run as ``python -m harbor_adapters.oolong.main``.

``--populate <dataset_dir>`` reads the bucket recorded in ``<dataset_dir>/bucket.toml``
(``subset`` / ``context_len`` / ``n_examples``), fetches the matching rows from
HuggingFace, and writes the task directories into ``<dataset_dir>``. The task dirs
are git-ignored, so this runs before ``harbor run --path <dataset_dir>``.

The bucket lives in the committed ``bucket.toml`` (not a runtime default) so the
directory always reproduces the same dataset. To evaluate another bucket, add a
new dataset directory with its own ``bucket.toml``.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from harbor_adapters.oolong.generate_oolong_tasks import generate_dataset

_BUCKET_FILE = "bucket.toml"


def _read_bucket(dataset_dir: Path) -> tuple[str, int, int | None]:
    """Read ``(subset, context_len, n_examples)`` from ``<dataset_dir>/bucket.toml``.

    Args:
        dataset_dir: Dataset directory that must contain ``bucket.toml``.

    Returns:
        ``(subset, context_len, n_examples)`` where ``n_examples`` is ``None`` for
        the full bucket.

    Raises:
        FileNotFoundError: If ``bucket.toml`` is missing.
        ValueError: If required keys are missing or have the wrong type.
    """
    bucket_path = dataset_dir / _BUCKET_FILE
    if not bucket_path.is_file():
        msg = f"No {_BUCKET_FILE} in {dataset_dir}; cannot determine the OOLONG bucket to fetch."
        raise FileNotFoundError(msg)

    data = tomllib.loads(bucket_path.read_text(encoding="utf-8"))
    subset = data.get("subset")
    context_len = data.get("context_len")
    n_examples = data.get("n_examples", 0)
    if not isinstance(subset, str) or not subset:
        msg = f"{bucket_path}: `subset` must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(context_len, int) or isinstance(context_len, bool):
        msg = f"{bucket_path}: `context_len` must be an integer"
        raise ValueError(msg)
    if not isinstance(n_examples, int) or isinstance(n_examples, bool):
        msg = f"{bucket_path}: `n_examples` must be an integer (0 = full bucket)"
        raise ValueError(msg)
    return subset, context_len, (None if n_examples == 0 else n_examples)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate OOLONG-synth Harbor tasks from the bucket in <dir>/bucket.toml.",
    )
    parser.add_argument(
        "--populate",
        type=Path,
        metavar="DATASET_DIR",
        required=True,
        help=(
            "Dataset directory to (re)generate task dirs into. The bucket is read "
            "from its bucket.toml and fetched from the HuggingFace datasets-server. "
            "Run before `harbor run --path DATASET_DIR`."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Populate the OOLONG dataset directory from the bucket in its ``bucket.toml``.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.
    """
    args = _build_parser().parse_args(argv)
    subset, context_len, n_examples = _read_bucket(args.populate)
    count = generate_dataset(
        out_dir=args.populate,
        dataset=subset,
        context_len=context_len,
        n_examples=n_examples,
    )
    print(f"Populated {count} OOLONG task(s) ({subset} @ {context_len}) in {args.populate}")


if __name__ == "__main__":
    main()
