"""Wall-time benchmarks for `create_deep_agent` construction.

Run with `make benchmark`
(executes: `uv run --group test pytest ./tests -m benchmark`)

These tests measure the wall time of building a `CompiledStateGraph` via
`create_deep_agent` under various configurations. They do NOT invoke the graph —
they only time the construction phase (middleware wiring, tool registration,
subagent compilation, etc.).

Each scenario runs multiple iterations and reports min / mean / max / stddev.
A global budget assert (default 10 s per scenario) catches severe regressions in
CI without being flaky on slower machines.
"""

import concurrent.futures
import cProfile
import pstats
import statistics
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from deepagents.graph import create_deep_agent
from deepagents.middleware import subagents as subagents_mod
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from deepagents.middleware.subagents import SubAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_model() -> GenericFakeChatModel:
    """Create a fresh fake model for benchmarking.

    `bind_tools` returns self (no-op), so tool schema serialization cost is
    excluded. This is acceptable because we measure graph assembly, not
    model-level tool binding.
    """
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


@tool(description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


@tool(description="Echo input")
def echo(text: str) -> str:
    """Return text unchanged."""
    return text


def _make_tool(idx: int) -> BaseTool:
    """Create a named tool for scaling tests."""

    @tool(description=f"Tool {idx}")
    def dynamic_tool(x: str) -> str:
        return f"tool_{idx}({x})"

    dynamic_tool.name = f"tool_{idx}"
    return dynamic_tool


def _time_create(iterations: int, **kwargs: Any) -> list[float]:
    """Run `create_deep_agent` `iterations` times, return list of wall-time durations."""
    durations: list[float] = []
    for i in range(iterations):
        start = time.perf_counter()
        result = create_deep_agent(**kwargs)
        durations.append(time.perf_counter() - start)
        if i == 0:
            assert result is not None, "create_deep_agent returned None"
            assert isinstance(result, CompiledStateGraph), f"Expected CompiledStateGraph, got {type(result).__name__}"
    return durations


def _report(label: str, durations: list[float]) -> None:
    """Print a human-readable summary line (visible in `pytest -s` output)."""
    n = len(durations)
    mn = min(durations)
    mx = max(durations)
    avg = statistics.mean(durations)
    sd = statistics.stdev(durations) if n > 1 else 0.0
    print(  # noqa: T201
        f"\n  [{label}] n={n}  min={mn:.4f}s  mean={avg:.4f}s  max={mx:.4f}s  stddev={sd:.4f}s"
    )


def _report_phase(label: str, phase_times: dict[str, list[float]]) -> None:
    """Print phase-level breakdown (visible in `pytest -s` output)."""
    print(f"\n  [{label}] phase breakdown:")  # noqa: T201
    for phase, durations in phase_times.items():
        avg = statistics.mean(durations)
        print(f"    {phase:30s}  mean={avg:.4f}s")  # noqa: T201


# Budget per-scenario — generous enough to never flake, tight enough to
# catch order-of-magnitude regressions.
MAX_MEAN_SECONDS = 10.0
ITERATIONS = 5


# ---------------------------------------------------------------------------
# Scenario benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCreateDeepAgentBenchmark:
    """Wall-time benchmarks for `create_deep_agent` graph construction."""

    def test_bare_minimum(self) -> None:
        """No user-supplied tools, subagents, or middleware — baseline construction cost.

        Note: the default general-purpose subagent is still compiled internally.
        """
        model = _fake_model()
        durations = _time_create(ITERATIONS, model=model)
        _report("bare_minimum", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_tools(self) -> None:
        """Three user-supplied tools."""
        model = _fake_model()
        durations = _time_create(ITERATIONS, model=model, tools=[add, multiply, echo])
        _report("with_tools", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_system_prompt_string(self) -> None:
        """Custom string system prompt concatenation."""
        model = _fake_model()
        prompt = "You are a helpful math tutor.\n" * 50
        durations = _time_create(ITERATIONS, model=model, system_prompt=prompt)
        _report("system_prompt_string", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_one_subagent(self) -> None:
        """Single custom subagent spec."""
        model = _fake_model()
        sub: SubAgent = {
            "name": "math_agent",
            "description": "Handles math questions",
            "system_prompt": "You are a math expert.",
            "tools": [add, multiply],
        }
        durations = _time_create(ITERATIONS, model=model, subagents=[sub])
        _report("one_subagent", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_multiple_subagents(self) -> None:
        """Five custom subagents (six total including the default general-purpose subagent).

        Tests scaling of `SubAgentMiddleware` wiring.
        """
        model = _fake_model()
        subs: list[SubAgent] = [
            {
                "name": f"agent_{i}",
                "description": f"Subagent number {i}",
                "system_prompt": f"You are agent {i}.",
                "tools": [echo],
            }
            for i in range(5)
        ]
        durations = _time_create(ITERATIONS, model=model, subagents=subs)
        _report("five_subagents", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_interrupt_on(self) -> None:
        """`HumanInTheLoopMiddleware` via `interrupt_on` config."""
        model = _fake_model()
        durations = _time_create(
            ITERATIONS,
            model=model,
            tools=[echo],
            interrupt_on={"echo": True},
        )
        _report("interrupt_on", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_skills(self) -> None:
        """`SkillsMiddleware` activated."""
        model = _fake_model()
        durations = _time_create(
            ITERATIONS,
            model=model,
            skills=["/skills/user/"],
        )
        _report("with_skills", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_memory(self) -> None:
        """`MemoryMiddleware` activated."""
        model = _fake_model()
        durations = _time_create(
            ITERATIONS,
            model=model,
            memory=["/memory/AGENTS.md"],
        )
        _report("with_memory", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_with_string_model_resolution(self) -> None:
        """String model name resolved via `resolve_model`."""
        fake = _fake_model()
        with patch("deepagents.graph.resolve_model", return_value=fake):
            durations = _time_create(
                ITERATIONS,
                model="claude-sonnet-4-6",
                tools=[add],
            )
        _report("string_model_resolution", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_full_featured(self) -> None:
        """Most optional features enabled simultaneously — near worst-case construction."""
        model = _fake_model()
        subs: list[SubAgent] = [
            {
                "name": f"sub_{i}",
                "description": f"Subagent {i}",
                "system_prompt": f"You are subagent {i}.",
                "tools": [add, multiply],
            }
            for i in range(3)
        ]
        durations = _time_create(
            ITERATIONS,
            model=model,
            tools=[add, multiply, echo],
            system_prompt="You are a comprehensive assistant.",
            subagents=subs,
            skills=["/skills/user/", "/skills/project/"],
            memory=["/memory/AGENTS.md"],
            interrupt_on={"echo": True, "add": True},
            debug=True,
            name="full_featured_agent",
        )
        _report("full_featured", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS

    def test_repeated_construction(self) -> None:
        """20 back-to-back constructions — detects per-invocation cost degradation.

        Catches growing caches, import-time side effects, or other accumulating
        overhead across repeated constructions.
        """
        model = _fake_model()
        durations = _time_create(20, model=model, tools=[add])
        _report("repeated_x20", durations)
        assert statistics.mean(durations) < MAX_MEAN_SECONDS
        # Check that later iterations aren't significantly slower than early ones
        first_half = statistics.mean(durations[:10])
        second_half = statistics.mean(durations[10:])
        # Allow up to 3x degradation before flagging
        assert second_half < first_half * 3, f"Construction slowing over time: first_half={first_half:.4f}s, second_half={second_half:.4f}s"

    def test_scaling_tools(self) -> None:
        """Measure construction time as tool count increases (1, 5, 10, 20)."""
        model = _fake_model()
        counts = [1, 5, 10, 20]
        means: dict[int, float] = {}
        for count in counts:
            tools = [_make_tool(i) for i in range(count)]
            durations = _time_create(ITERATIONS, model=model, tools=tools)
            avg = statistics.mean(durations)
            means[count] = avg
            _report(f"scaling_tools_{count}", durations)

        # Construction should not scale worse than linearly with a generous
        # constant factor.  Going from 1 tool to 20 tools should not be
        # more than 20x slower.
        assert means[1] > 0, "1-tool construction measured 0.0s — perf_counter resolution too low or construction short-circuited"
        ratio = means[20] / means[1]
        assert ratio < 20, f"Tool scaling ratio {ratio:.1f}x exceeds 20x threshold (1 tool: {means[1]:.4f}s, 20 tools: {means[20]:.4f}s)"

    def test_scaling_subagents(self) -> None:
        """Measure construction time as subagent count increases (1, 3, 5, 10)."""
        model = _fake_model()
        counts = [1, 3, 5, 10]
        means: dict[int, float] = {}
        for count in counts:
            subs: list[SubAgent] = [
                {
                    "name": f"sub_{i}",
                    "description": f"Subagent {i}",
                    "system_prompt": f"You are subagent {i}.",
                    "tools": [echo],
                }
                for i in range(count)
            ]
            durations = _time_create(ITERATIONS, model=model, subagents=subs)
            avg = statistics.mean(durations)
            means[count] = avg
            _report(f"scaling_subagents_{count}", durations)

        assert means[1] > 0, "1-subagent construction measured 0.0s — perf_counter resolution too low or construction short-circuited"
        ratio = means[10] / means[1]
        assert ratio < 20, f"Subagent scaling ratio {ratio:.1f}x exceeds 20x threshold (1 sub: {means[1]:.4f}s, 10 subs: {means[10]:.4f}s)"


# ---------------------------------------------------------------------------
# Phase-level profiling
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCreateDeepAgentProfiling:
    """Break down where wall time is spent inside `create_deep_agent`.

    Instruments key internal functions to attribute time to discrete phases:
    middleware instantiation, subagent graph compilation, and final graph
    compilation.

    Note: patches target call-site modules (`deepagents.graph`,
    `deepagents.middleware.subagents`) rather than definition-site modules,
    because `from X import Y` creates local bindings.
    """

    @staticmethod
    def _profile_phases(iterations: int, **kwargs: Any) -> dict[str, list[float]]:
        """Run `create_deep_agent` with phase-level instrumentation.

        Wraps `SubAgentMiddleware.__init__`, `create_summarization_middleware`
        (at the `deepagents.graph` call-site), and `create_agent` (at both
        `deepagents.graph` and `deepagents.middleware.subagents` call-sites)
        to measure their contribution.

        Returns:
            Mapping of phase name to list of per-iteration durations.
        """
        phases: dict[str, list[float]] = {
            "total": [],
            "SubAgentMiddleware.__init__": [],
            "create_summarization_middleware": [],
            "create_agent (final compile)": [],
            "other": [],
        }

        orig_subagent_init = subagents_mod.SubAgentMiddleware.__init__
        # Resolve originals from the call-site modules for accurate wrapping
        import deepagents.graph as graph_mod  # noqa: PLC0415

        orig_create_summ_fn = graph_mod.create_summarization_middleware
        orig_create_agent_fn = graph_mod.create_agent

        for _ in range(iterations):
            # Mutable containers shared with closures — bound via default
            # args to satisfy B023 (function-definition-in-loop).
            accum: dict[str, float] = {
                "subagent_init": 0.0,
                "create_summ": 0.0,
                "create_agent": 0.0,
            }
            depth: dict[str, int] = {
                "subagent_init": 0,
                "create_agent": 0,
            }

            def patched_subagent_init(self_inner, *a: Any, _acc=accum, _dep=depth, **kw: Any):
                _dep["subagent_init"] += 1
                t0 = time.perf_counter()
                result = orig_subagent_init(self_inner, *a, **kw)
                _acc["subagent_init"] += time.perf_counter() - t0
                _dep["subagent_init"] -= 1
                return result

            def patched_create_summ(*a: Any, _acc=accum, _orig=orig_create_summ_fn, **kw: Any):
                t0 = time.perf_counter()
                result = _orig(*a, **kw)
                _acc["create_summ"] += time.perf_counter() - t0
                return result

            def patched_create_agent(*a: Any, _acc=accum, _dep=depth, _orig_g=orig_create_agent_fn, **kw: Any):
                # Only count the outermost create_agent call (the final
                # graph compile). Inner calls from SubAgentMiddleware are
                # already captured in subagent_init.
                _dep["create_agent"] += 1
                t0 = time.perf_counter()
                result = _orig_g(*a, **kw)
                elapsed = time.perf_counter() - t0
                _dep["create_agent"] -= 1
                if _dep["subagent_init"] == 0:
                    _acc["create_agent"] += elapsed
                return result

            with (
                patch.object(subagents_mod.SubAgentMiddleware, "__init__", patched_subagent_init),
                patch("deepagents.graph.create_summarization_middleware", patched_create_summ),
                patch("deepagents.graph.create_agent", patched_create_agent),
                patch("deepagents.middleware.subagents.create_agent", patched_create_agent),
            ):
                total_start = time.perf_counter()
                create_deep_agent(**kwargs)
                total_elapsed = time.perf_counter() - total_start

            phases["total"].append(total_elapsed)
            phases["SubAgentMiddleware.__init__"].append(accum["subagent_init"])
            phases["create_summarization_middleware"].append(accum["create_summ"])
            phases["create_agent (final compile)"].append(accum["create_agent"])
            phases["other"].append(total_elapsed - accum["subagent_init"] - accum["create_summ"] - accum["create_agent"])

        return phases

    def test_profile_bare_minimum(self) -> None:
        """Phase breakdown for baseline (no user-supplied subagents)."""
        model = _fake_model()
        phases = self._profile_phases(ITERATIONS, model=model)
        _report("profile_bare", phases["total"])
        _report_phase("profile_bare", phases)
        assert statistics.mean(phases["total"]) < MAX_MEAN_SECONDS

    def test_profile_full_featured(self) -> None:
        """Phase breakdown for full-featured config (3 subagents + all options)."""
        model = _fake_model()
        subs: list[SubAgent] = [
            {
                "name": f"sub_{i}",
                "description": f"Subagent {i}",
                "system_prompt": f"You are subagent {i}.",
                "tools": [add, multiply],
            }
            for i in range(3)
        ]
        phases = self._profile_phases(
            ITERATIONS,
            model=model,
            tools=[add, multiply, echo],
            system_prompt="You are a comprehensive assistant.",
            subagents=subs,
            skills=["/skills/user/", "/skills/project/"],
            memory=["/memory/AGENTS.md"],
            interrupt_on={"echo": True, "add": True},
            name="profiled_agent",
        )
        _report("profile_full", phases["total"])
        _report_phase("profile_full", phases)
        assert statistics.mean(phases["total"]) < MAX_MEAN_SECONDS

        # Verify instrumentation fired for subagent init
        assert any(t > 0 for t in phases["SubAgentMiddleware.__init__"]), (
            "SubAgentMiddleware.__init__ patch never fired — has the internal API been renamed?"
        )

        total_mean = statistics.mean(phases["total"])
        assert total_mean > 0, "Total construction time measured 0.0s"
        subagent_pct = statistics.mean(phases["SubAgentMiddleware.__init__"]) / total_mean
        print(f"\n  SubAgentMiddleware.__init__ accounts for {subagent_pct:.0%} of total")  # noqa: T201

    def test_cprofile_top_functions(self) -> None:
        """Top-20 cumulative functions via cProfile for full-featured construction."""
        model = _fake_model()
        subs: list[SubAgent] = [
            {
                "name": f"sub_{i}",
                "description": f"Subagent {i}",
                "system_prompt": f"You are subagent {i}.",
                "tools": [add, multiply],
            }
            for i in range(3)
        ]
        profiler = cProfile.Profile()
        profiler.enable()
        result = create_deep_agent(
            model=model,
            tools=[add, multiply, echo],
            subagents=subs,
            skills=["/skills/user/"],
            name="cprofile_agent",
        )
        profiler.disable()

        assert isinstance(result, CompiledStateGraph)

        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\n  [cprofile] top 20 by cumulative time:")  # noqa: T201
        stats.print_stats(20)


# ---------------------------------------------------------------------------
# Concurrent construction
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCreateDeepAgentConcurrency:
    """Benchmark parallel `create_deep_agent` calls.

    Simulates a server constructing agents for multiple concurrent requests.

    CPU-bound construction limits max concurrency and inflates TTFT under load.
    """

    @staticmethod
    def _build_once() -> float:
        """Construct one full-featured agent, return wall time."""
        model = _fake_model()
        subs: list[SubAgent] = [
            {
                "name": f"sub_{i}",
                "description": f"Subagent {i}",
                "system_prompt": f"You are subagent {i}.",
                "tools": [add, multiply],
            }
            for i in range(3)
        ]
        start = time.perf_counter()
        create_deep_agent(
            model=model,
            tools=[add, multiply, echo],
            subagents=subs,
            name="concurrent_agent",
        )
        return time.perf_counter() - start

    def test_serial_baseline(self) -> None:
        """Serial construction of 10 agents — baseline for comparison."""
        durations = [self._build_once() for _ in range(10)]
        total = sum(durations)
        _report("serial_10x", durations)
        print(f"\n  serial total: {total:.4f}s  throughput: {10 / total:.1f} agents/s")  # noqa: T201

    def test_concurrent_threads(self) -> None:
        """Concurrent construction with `ThreadPoolExecutor` (GIL contention test).

        Measures how much throughput degrades when multiple threads construct
        agents simultaneously. Pure-CPU work under the GIL means threads
        effectively serialize — the degradation ratio quantifies GIL
        contention overhead.
        """
        workers = 4
        tasks = 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            total_start = time.perf_counter()
            futures = [pool.submit(self._build_once) for _ in range(tasks)]
            durations = [f.result() for f in concurrent.futures.as_completed(futures)]
            wall_time = time.perf_counter() - total_start

        _report(f"concurrent_{workers}w_{tasks}t", durations)
        throughput = tasks / wall_time
        print(  # noqa: T201
            f"\n  wall_time={wall_time:.4f}s  throughput={throughput:.1f} agents/s  workers={workers}  tasks={tasks}"
        )

        # With pure GIL-bound work, wall_time should be roughly equal to
        # serial (no speedup). Assert it doesn't get worse than 2x serial.
        serial_estimate = statistics.mean(durations) * tasks
        degradation = wall_time / serial_estimate
        print(f"  degradation ratio: {degradation:.2f}x (1.0 = no overhead)")  # noqa: T201
        assert degradation < 2.0, (
            f"GIL contention overhead {degradation:.2f}x exceeds 2x threshold (wall_time={wall_time:.4f}s, serial_estimate={serial_estimate:.4f}s)"
        )

    def test_concurrent_scaling(self) -> None:
        """Measure throughput at 1, 2, 4, 8 concurrent workers.

        Establishes a scaling curve to quantify GIL contention at increasing
        concurrency levels.
        """
        tasks_per_level = 8
        levels = [1, 2, 4, 8]
        results: dict[int, float] = {}

        for workers in levels:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                start = time.perf_counter()
                futures = [pool.submit(self._build_once) for _ in range(tasks_per_level)]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
                wall = time.perf_counter() - start

            throughput = tasks_per_level / wall
            results[workers] = throughput
            print(  # noqa: T201
                f"\n  [{workers} workers] wall={wall:.4f}s  throughput={throughput:.1f} agents/s"
            )

        # Throughput shouldn't collapse — even with GIL, 8 workers should
        # achieve at least 25% of single-worker throughput.
        assert results[1] > 0, "1-worker throughput measured 0.0 — construction may be short-circuited"
        ratio = results[8] / results[1]
        print(f"\n  8-worker efficiency: {ratio:.0%} of 1-worker throughput")  # noqa: T201
        assert ratio > 0.25, f"Severe throughput collapse at 8 workers: {results[8]:.1f} vs {results[1]:.1f} agents/s ({ratio:.0%})"
