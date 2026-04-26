"""Wall-time benchmarks for Monty-backed REPL evaluation.

Run locally: `uv run --group test pytest ./tests -m benchmark`
Run with CodSpeed: `uv run --group test pytest ./tests -m benchmark --codspeed`

These tests measure wall time for representative Monty programs. Regression
tracking is handled by CodSpeed in CI. Local runs produce pytest-benchmark
output for human inspection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain_monty.middleware import MontyMiddleware

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def _echo(value: str) -> str:
    return value


def _echo_middleware() -> MontyMiddleware:
    """Create a fresh middleware with a simple `echo` function."""
    return MontyMiddleware(ptc=[_echo])


def _echo_program(*, line_count: int) -> str:
    """Build a multiline Monty program with a fixed number of calls."""
    return "\n".join(['print(echo("hello"))' for _ in range(line_count)])


@pytest.mark.benchmark
class TestMontyBenchmark:
    """Wall-time benchmarks for Monty execution."""

    def test_simple_echo_program(self, benchmark: BenchmarkFixture) -> None:
        """Baseline single-line program."""
        program = 'print(echo("hello"))'
        middleware = _echo_middleware()
        middleware._run_monty(program, timeout=None)

        @benchmark  # type: ignore[misc]
        def _() -> None:
            middleware = _echo_middleware()
            middleware._run_monty(program, timeout=None)

    def test_thousand_line_echo_program(self, benchmark: BenchmarkFixture) -> None:
        """Large multiline program with repeated foreign function calls."""
        program = _echo_program(line_count=1000)
        middleware = _echo_middleware()
        middleware._run_monty(program, timeout=None)

        @benchmark  # type: ignore[misc]
        def _() -> None:
            middleware = _echo_middleware()
            middleware._run_monty(program, timeout=None)
