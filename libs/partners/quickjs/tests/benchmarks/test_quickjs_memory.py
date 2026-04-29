"""Memory benchmarks for QuickJS REPL middleware.

Run locally:  `make benchmark`
Run with CodSpeed:  `uv run --group test pytest ./tests -m benchmark --codspeed`

These tests exercise memory-targeted workloads for QuickJS eval execution under
different thread counts and tool shapes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from langchain_quickjs import REPLMiddleware
from tests.benchmarks._common import (
    CONSOLE_LOG_CODE,
    PTC_ONLY_CODE,
    assert_eval_succeeded,
    echo_payload,
    invoke_payload,
    make_agent,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.memory_benchmark
class TestQuickJSMemoryBenchmarks:
    """Benchmarks that compare Python heap pressure under common REPL workloads."""

    def _run_concurrent_memory_workload(
        self,
        *,
        thread_count: int,
        code: str,
        use_ptc: bool,
    ) -> None:
        def _worker(index: int) -> None:
            middleware = REPLMiddleware(
                capture_console=True,
                ptc=[echo_payload] if use_ptc else None,
            )
            agent = make_agent(code=code, middleware=middleware, repeats=1)
            result = agent.invoke(
                invoke_payload(),
                config={"configurable": {"thread_id": f"memory-bench-{index}"}},
            )
            assert_eval_succeeded(result)

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            list(executor.map(_worker, range(thread_count)))

    @pytest.mark.parametrize(
        "thread_count", [1, 8, 32, 64], ids=lambda n: f"{n}_threads"
    )
    @pytest.mark.parametrize(
        ("scenario", "code", "use_ptc"),
        [
            ("console_log", CONSOLE_LOG_CODE, False),
            ("ptc_tools", PTC_ONLY_CODE, True),
        ],
        ids=["console_log", "ptc_tools"],
    )
    def test_repl_memory_peak(
        self,
        benchmark: BenchmarkFixture,
        thread_count: int,
        scenario: str,
        code: str,
        use_ptc: bool,
    ) -> None:
        """Measure memory-instrumented workload cost for each scenario."""

        @benchmark
        def _() -> None:
            self._run_concurrent_memory_workload(
                thread_count=thread_count,
                code=code,
                use_ptc=use_ptc,
            )

        benchmark.extra_info["scenario"] = scenario
        benchmark.extra_info["thread_count"] = thread_count
