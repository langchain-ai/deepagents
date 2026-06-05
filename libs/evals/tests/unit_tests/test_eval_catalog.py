"""Drift test: `EVAL_CATALOG.md` must match the eval test files on disk."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_EVALS_DIR = Path(__file__).resolve().parents[2]
_SCRIPT = _EVALS_DIR / "scripts" / "generate_eval_catalog.py"


def test_eval_catalog_up_to_date() -> None:
    """Fail if `EVAL_CATALOG.md` is stale or missing.

    Regenerate with: `make eval-catalog` from `libs/evals/`.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"EVAL_CATALOG.md is out of date. "
        f"Run `make eval-catalog` from libs/evals/ to regenerate.\n\n"
        f"{result.stdout}"
    )
