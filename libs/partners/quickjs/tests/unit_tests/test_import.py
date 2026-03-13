from __future__ import annotations

from pathlib import Path

import langchain_quickjs


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    version_line = next(
        line
        for line in pyproject.read_text().splitlines()
        if line.startswith("version = ")
    )
    assert langchain_quickjs.__version__ == version_line.split('"')[1]
