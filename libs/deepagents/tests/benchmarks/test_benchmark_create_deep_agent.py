"""Wall-time benchmarks for `create_deep_agent` graph construction.

Run locally:  `make benchmark`
Run with CodSpeed:  `uv run --group test pytest ./tests -m benchmark --codspeed`

These tests measure the wall time of building a `CompiledStateGraph` via
`create_deep_agent` under various configurations. They do NOT invoke the
graph -- they only time the construction phase (middleware wiring, tool
registration, subagent compilation, etc.).

Regression detection is handled by CodSpeed in CI. Local runs produce
pytest-benchmark tables (min/max/mean/stddev) for human inspection.
"""

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from pytest_benchmark.fixture import BenchmarkFixture

from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from deepagents.middleware.subagents import SubAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_model() -> GenericFakeChatModel:
    """Create a fresh fake model for benchmarking.

    `bind_tools` returns self (no-op), so model-level tool binding cost is
    excluded. We measure graph assembly, not model-level tool binding.
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


def _build_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build default kwargs for `create_deep_agent`, merged with overrides."""
    kwargs: dict[str, Any] = {"model": _fake_model()}
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Scenario benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCreateDeepAgentBenchmark:
    """Wall-time benchmarks for `create_deep_agent` graph construction."""

    def test_filesystem_init(self, benchmark: BenchmarkFixture) -> None:
        """Measure the cost of repeated `FilesystemMiddleware` setup."""

        @benchmark  # type: ignore[misc]
        def _() -> None:
            FilesystemMiddleware()

    def test_create_deep_agent_minimal(self, benchmark: BenchmarkFixture) -> None:
        """Baseline: no user-supplied tools, subagents, or middleware."""
        kwargs = _build_kwargs()

        @benchmark  # type: ignore[misc]
        def _() -> None:
            create_deep_agent(**kwargs)

    def test_with_tools(self, benchmark: BenchmarkFixture) -> None:
        """Three user-supplied tools."""
        kwargs = _build_kwargs(tools=[add, multiply, echo])

        @benchmark
        def _() -> None:
            create_deep_agent(**kwargs)

    def test_with_one_subagent(self, benchmark: BenchmarkFixture) -> None:
        """Single custom subagent spec."""
        sub: SubAgent = {
            "name": "math_agent",
            "description": "Handles math questions",
            "system_prompt": "You are a math expert.",
            "tools": [add, multiply],
        }
        kwargs = _build_kwargs(subagents=[sub])

        @benchmark
        def _() -> None:
            create_deep_agent(**kwargs)

    def test_with_multiple_subagents(self, benchmark: BenchmarkFixture) -> None:
        """Five custom subagents (six total including the default)."""
        subs: list[SubAgent] = [
            {
                "name": f"agent_{i}",
                "description": f"Subagent number {i}",
                "system_prompt": f"You are agent {i}.",
                "tools": [echo],
            }
            for i in range(5)
        ]
        kwargs = _build_kwargs(subagents=subs)

        @benchmark
        def _() -> None:
            create_deep_agent(**kwargs)

    def test_full_featured(self, benchmark: BenchmarkFixture) -> None:
        """Most optional features enabled -- near worst-case construction."""
        subs: list[SubAgent] = [
            {
                "name": f"sub_{i}",
                "description": f"Subagent {i}",
                "system_prompt": f"You are subagent {i}.",
                "tools": [add, multiply],
            }
            for i in range(3)
        ]
        kwargs = _build_kwargs(
            tools=[add, multiply, echo],
            system_prompt="You are a comprehensive assistant.",
            subagents=subs,
            skills=["/skills/user/", "/skills/project/"],
            memory=["/memory/AGENTS.md"],
            interrupt_on={"echo": True, "add": True},
            debug=True,
            name="full_featured_agent",
        )

        @benchmark
        def _() -> None:
            create_deep_agent(**kwargs)

    def test_with_string_model_resolution(self, benchmark: BenchmarkFixture) -> None:
        """String model name resolved via `resolve_model`."""
        fake = _fake_model()
        with patch("deepagents.graph.resolve_model", return_value=fake):
            kwargs = _build_kwargs(model="claude-sonnet-4-6", tools=[add])

            @benchmark
            def _() -> None:
                create_deep_agent(**kwargs)


# ---------------------------------------------------------------------------
# Scaling benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCreateDeepAgentScaling:
    """Measure construction time as tool/subagent count increases."""

    @pytest.mark.parametrize("tool_count", [1, 5, 10, 20], ids=lambda n: f"{n}_tools")
    def test_scaling_tools(self, benchmark: BenchmarkFixture, tool_count: int) -> None:
        """Construction time vs tool count."""
        tools = [_make_tool(i) for i in range(tool_count)]
        kwargs = _build_kwargs(tools=tools)

        @benchmark
        def _() -> None:
            create_deep_agent(**kwargs)

    @pytest.mark.parametrize("sub_count", [1, 3, 5, 10], ids=lambda n: f"{n}_subagents")
    def test_scaling_subagents(self, benchmark: BenchmarkFixture, sub_count: int) -> None:
        """Construction time vs subagent count."""
        subs: list[SubAgent] = [
            {
                "name": f"sub_{i}",
                "description": f"Subagent {i}",
                "system_prompt": f"You are subagent {i}.",
                "tools": [echo],
            }
            for i in range(sub_count)
        ]
        kwargs = _build_kwargs(subagents=subs)

        @benchmark
        def _() -> None:
            create_deep_agent(**kwargs)
