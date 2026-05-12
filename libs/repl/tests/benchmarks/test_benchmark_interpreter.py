"""Wall-time benchmarks for REPL program evaluation.

Run locally: `uv run --group test pytest ./tests -m benchmark`
Run with CodSpeed: `uv run --group test pytest ./tests -m benchmark --codspeed`

These tests measure wall time for `Interpreter.evaluate()` on representative
programs. Regression detection is handled by CodSpeed in CI. Local runs produce
pytest-benchmark tables for human inspection.
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from langchain_repl.interpreter import Interpreter


def _echo_interpreter() -> Interpreter:
    """Create a fresh interpreter with a simple `echo` function."""
    return Interpreter(functions={"echo": lambda value: value})


def _echo_program(*, line_count: int) -> str:
    """Build a multiline echo program with a fixed number of statements."""
    return "\n".join(['echo("hello")' for _ in range(line_count)])


@pytest.mark.benchmark
class TestInterpreterEvaluateBenchmark:
    """Wall-time benchmarks for `Interpreter.evaluate()`."""

    def test_simple_echo_program(self, benchmark: BenchmarkFixture) -> None:
        """Baseline single-line program."""
        program = "echo(42)"
        interpreter = _echo_interpreter()
        interpreter.evaluate(program)

        @benchmark  # type: ignore[misc]
        def _() -> None:
            interpreter = _echo_interpreter()
            interpreter.evaluate(program)

    def test_thousand_line_echo_program(self, benchmark: BenchmarkFixture) -> None:
        """Large multiline program with repeated function calls."""
        program = _echo_program(line_count=1000)
        interpreter = _echo_interpreter()
        interpreter.evaluate(program)

        @benchmark  # type: ignore[misc]
        def _() -> None:
            interpreter = _echo_interpreter()
            interpreter.evaluate(program)
