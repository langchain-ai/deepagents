"""Drift test: `MODEL_GROUPS.md` must match the canonical model registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_EVALS_DIR = Path(__file__).resolve().parents[2]
_SCRIPT = _EVALS_DIR / "scripts" / "generate_model_groups.py"


def test_model_groups_up_to_date() -> None:
    """Fail if `MODEL_GROUPS.md` is stale or missing.

    Regenerate with: `make model-groups` from `libs/evals/`.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"MODEL_GROUPS.md is out of date.\n"
        f"Run `make model-groups` from libs/evals/ to regenerate.\n"
        f"{result.stdout}{result.stderr}"
    )
