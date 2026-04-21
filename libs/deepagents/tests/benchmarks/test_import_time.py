"""Import-time benchmarks for the deepagents package.

Each test spawns a fresh subprocess so Python's module cache is cold.
CodSpeed tracks these in walltime mode; local runs produce pytest-benchmark
tables.

Run locally:  uv run --group test pytest ./tests -m benchmark -k import
Run with CodSpeed: uv run --group test pytest ./tests -m benchmark --codspeed
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


def _run_import(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.benchmark
class TestImportTime:
    """Cold-process import-time benchmarks."""

    def test_import_create_deep_agent(self, benchmark: BenchmarkFixture) -> None:
        """Time `from deepagents import create_deep_agent` in a fresh process."""

        @benchmark  # type: ignore[misc]
        def _() -> None:
            _run_import("from deepagents import create_deep_agent")

    def test_import_deepagents_package(self, benchmark: BenchmarkFixture) -> None:
        """Time `import deepagents` in a fresh process (exercises __init__.py)."""

        @benchmark  # type: ignore[misc]
        def _() -> None:
            _run_import("import deepagents")
