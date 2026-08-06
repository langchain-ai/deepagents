#!/usr/bin/env python3
"""Materialize a local Harbor dataset's git-ignored task content before a run.

Local datasets under ``libs/evals/datasets/`` do not commit everything Harbor
needs. Context-Bench git-ignores its shared corpus and invariant verifier files;
DRBench git-ignores its invariant verifier files. Each has its own adapter, so
this maps a dataset path to the module that populates it and invokes that
module's ``--populate``.

The mapping is an explicit allowlist: an unknown dataset path is an error, never
a guess, so a typo cannot silently run the wrong adapter or leave a dataset
empty (which Harbor would report as "0 tasks" rather than a failure).

Usage:
    python3 prepare_local_dataset.py datasets/drbench-evals

Run from ``libs/evals``; the path is interpreted relative to the current
directory, matching how the workflow passes it to ``harbor run --path``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Dataset path (relative to libs/evals) -> adapter module exposing --populate.
ADAPTERS = {
    "datasets/context-retrieval-evals": "harbor_adapters.contextbench.main",
    "datasets/drbench-evals": "harbor_adapters.drbench.main",
}

# Mirrors the workflow's own shell guard so this is safe to call directly too.
_SAFE_PATH_RE = re.compile(r"^datasets/[A-Za-z0-9._/-]+$")


def resolve_adapter(dataset_path: str) -> str:
    """Return the adapter module registered for a local dataset path.

    Args:
        dataset_path: Dataset path relative to `libs/evals`, e.g.
            `datasets/drbench-evals`.

    Returns:
        The dotted module path of the adapter's CLI.

    Raises:
        ValueError: If the path is malformed or has no registered adapter.
    """
    normalized = dataset_path.rstrip("/")
    # Traversal sequences are rejected before any canonicalization, so a path
    # cannot be normalized into something that passes.
    if not _SAFE_PATH_RE.match(normalized) or ".." in normalized or "~" in normalized:
        msg = f"Invalid local Harbor dataset path: {dataset_path!r}"
        raise ValueError(msg)
    adapter = ADAPTERS.get(normalized)
    if adapter is None:
        msg = (
            f"No populate adapter registered for {normalized!r}. "
            f"Known datasets: {sorted(ADAPTERS)}. Add an entry to ADAPTERS in "
            f"{Path(__file__).name} when introducing a new local dataset."
        )
        raise ValueError(msg)
    return adapter


def main(argv: list[str] | None = None) -> int:
    """Populate one local Harbor dataset via its registered adapter.

    Args:
        argv: Command-line arguments, excluding the program name.

    Returns:
        The adapter's exit code, or 1 if the dataset path is not registered.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        help="Local dataset path relative to libs/evals, e.g. datasets/drbench-evals",
    )
    args = parser.parse_args(argv)

    try:
        adapter = resolve_adapter(args.dataset_path)
    except ValueError as exc:
        print(f"::error::{exc}")
        return 1

    # Module name comes from the ADAPTERS allowlist, never from the argument.
    # List form with no shell, so the path cannot be interpreted as a command.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", adapter, "--populate", args.dataset_path],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
