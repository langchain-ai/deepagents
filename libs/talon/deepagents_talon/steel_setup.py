"""Explicit one-time setup: `python -m deepagents_talon.steel_setup`."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from deepagents_talon.steel import STEEL_REVISION


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        msg = f"Steel setup requires {name} on PATH"
        raise RuntimeError(msg)
    return executable


def setup(directory: Path) -> None:
    """Prepare pinned Steel without overwriting an existing installation.

    Args:
        directory: New installation directory, separate from browser profiles.
    """
    node, npm, git = (_executable(name) for name in ("node", "npm", "git"))
    version = subprocess.check_output([node, "--version"], text=True)  # noqa: S603  # Resolved executable.
    if not version.startswith("v24."):
        msg = "Steel setup requires Node.js 24 (the tested DuckDB native-module ABI)"
        raise RuntimeError(msg)
    directory = directory.expanduser().absolute()
    if directory.exists():
        msg = "Steel setup destination already exists; choose a new directory"
        raise FileExistsError(msg)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".steel-setup-") as temporary:
        source = Path(temporary) / "source"
        _build(source, git, npm)
        (source / ".talon-prepared.json").write_text(
            json.dumps({"revision": STEEL_REVISION, "node": str(Path(node).resolve())})
        )
        source.rename(directory)


def _build(source: Path, git: str, npm: str) -> None:
    commands = [
        [
            git,
            "clone",
            "--no-checkout",
            "https://github.com/steel-dev/steel-browser.git",
            str(source),
        ],
        [git, "checkout", "--detach", STEEL_REVISION],
        [npm, "ci", "--ignore-scripts", "--workspace", "api", "--include-workspace-root"],
        [npm, "rebuild", "duckdb"],
        [npm, "run", "build", "--workspace", "api"],
    ]
    for command in commands:
        subprocess.run(  # noqa: S603  # Fixed argv; setup is explicitly operator-invoked.
            command,
            cwd=source if source.exists() else source.parent,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("~/.deepagents/steel"))
    setup(parser.parse_args().directory)
