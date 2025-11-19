"""Bootstrap wrapper so console scripts can import from src without editable installs."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _maybe_add_src_path() -> None:
    """Ensure the repo's src directory is importable when developing locally."""

    candidate_roots = [
        Path.cwd(),
        Path(os.environ.get("HARBOR_DEEPAGENTS_ROOT", "")),
    ]

    for root in candidate_roots:
        if not root:
            continue
        src_dir = root / "src"
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))
            return


_maybe_add_src_path()

from harbor_deepagents.cli import main as _main
from harbor_deepagents.cli import run_tb_daytona as _run_tb_daytona
from harbor_deepagents.cli import run_tb_docker as _run_tb_docker


def main() -> None:
    _main()


def tb_docker() -> None:
    _run_tb_docker()


def tb_daytona() -> None:
    _run_tb_daytona()
