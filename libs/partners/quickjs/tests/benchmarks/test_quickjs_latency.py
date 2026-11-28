"""Wall-time benchmarks for the costs an agent pays on every QuickJS eval.

Run locally:  `make benchmark`
Run with CodSpeed:  `make bench`

Each round builds a fresh agent in `setup`, so only the measured invoke is timed.
The workloads cover the per-leaf JS-to-Python value walk, the per-call PTC
bridge under `ainvoke`, and the per-turn floor with a checkpointer, where the
REPL is restored from a snapshot and a delta is taken against the prior one.
"""

from __future__ import annotations

import asyncio
from itertools import count
from typing import TYPE_CHECKING, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langchain_quickjs import CodeInterpreterMiddleware
from tests.benchmarks._common import (
    PTC_ONLY_CODE,
    assert_eval_succeeded,
    echo_payload,
    eval_tool_message,
    invoke_payload,
    make_agent,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_benchmark.fixture import BenchmarkFixture

ROW_OBJECTS_CODE = (
    "Array.from({length: 1000}, (_, i) => ({id: i, name: `row-${i}`, score: i / 7}));"
)
TURN_COUNTER_CODE = "globalThis.turns = (globalThis.turns ?? 0) + 1; turns;"
# Roughly the GC-tracked heap of a long-lived server process.
LIVE_HEAP_OBJECTS = 1_000_000


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """One event loop reused across rounds, as a server would hold."""
    event_loop = asyncio.new_event_loop()
    yield event_loop
    event_loop.close()


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.benchmark
class TestQuickJSLatencyBenchmarks:
    """Benchmarks that track per-eval latency on hot agent paths."""

    def test_eval_returns_1k_row_objects(self, benchmark: BenchmarkFixture) -> None:
        """Measure one invoke whose eval returns 1,000 `{id, name, score}` rows."""

        def setup() -> tuple[tuple[Any], dict[str, Any]]:
            agent = make_agent(
                code=ROW_OBJECTS_CODE,
                middleware=CodeInterpreterMiddleware(),
                repeats=1,
            )
            return (agent,), {}

        def run(agent: Any) -> None:
            result = agent.invoke(invoke_payload(), config=_config("rows-bench"))
            assert "row-0" in eval_tool_message(result).content

        benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)
        benchmark.extra_info["workload"] = "eval_result_1k_row_objects"

    def test_ainvoke_100_serial_ptc_calls(
        self,
        benchmark: BenchmarkFixture,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Measure one `ainvoke` whose eval awaits 100 PTC tool calls in series."""

        def setup() -> tuple[tuple[Any], dict[str, Any]]:
            agent = make_agent(
                code=PTC_ONLY_CODE,
                middleware=CodeInterpreterMiddleware(ptc=[echo_payload]),
                repeats=1,
            )
            return (agent,), {}

        def run(agent: Any) -> None:
            result = loop.run_until_complete(
                agent.ainvoke(invoke_payload(), config=_config("ptc-bench"))
            )
            assert eval_tool_message(result).content == "<result>100</result>"

        benchmark.pedantic(run, setup=setup, rounds=10, warmup_rounds=1)
        benchmark.extra_info["workload"] = "ainvoke_100_serial_ptc_calls"
        benchmark.extra_info["ptc_calls"] = 100

    @pytest.mark.parametrize(
        "live_objects",
        [0, LIVE_HEAP_OBJECTS],
        ids=["small_heap", "1m_live_objects"],
    )
    def test_turn_floor_with_checkpointer(
        self,
        benchmark: BenchmarkFixture,
        loop: asyncio.AbstractEventLoop,
        live_objects: int,
    ) -> None:
        """Measure a second `ainvoke` turn that restores and re-snapshots the REPL.

        Evicting the REPL after each turn runs a full `gc.collect()`, so the
        floor grows with the size of the live Python heap.
        """
        heap = [[index] for index in range(live_objects)]
        thread_ids = (f"floor-bench-{index}" for index in count())

        def setup() -> tuple[tuple[Any, str], dict[str, Any]]:
            agent = make_agent(
                code=TURN_COUNTER_CODE,
                middleware=CodeInterpreterMiddleware(),
                repeats=2,
                checkpointer=InMemorySaver(),
            )
            thread_id = next(thread_ids)
            result = loop.run_until_complete(
                agent.ainvoke(invoke_payload(), config=_config(thread_id))
            )
            assert_eval_succeeded(result)
            return (agent, thread_id), {}

        def run(agent: Any, thread_id: str) -> None:
            result = loop.run_until_complete(
                agent.ainvoke(invoke_payload(), config=_config(thread_id))
            )
            # `2` proves the second turn restored the first turn's REPL state.
            assert eval_tool_message(result).content == "<result>2</result>"

        benchmark.pedantic(run, setup=setup, rounds=10, warmup_rounds=1)
        benchmark.extra_info["workload"] = "turn_floor_with_checkpointer"
        benchmark.extra_info["live_objects"] = len(heap)
