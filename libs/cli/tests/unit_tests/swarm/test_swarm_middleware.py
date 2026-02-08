"""Unit tests for SwarmMiddleware."""

import json
from typing import cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

from deepagents_cli.swarm.middleware import SwarmMiddleware


class MockSubagent:
    """Mock subagent for testing."""

    async def ainvoke(self, _state: dict) -> dict:
        return {"messages": [AIMessage(content="Task completed")]}


@pytest.fixture
def subagent_graphs():
    return {"general-purpose": cast("Runnable", MockSubagent())}


@pytest.fixture
def middleware(subagent_graphs):
    return SwarmMiddleware(subagent_graphs=subagent_graphs)


class TestSwarmMiddleware:
    def test_has_one_tool(self, middleware):
        """Middleware should provide only swarm_execute."""
        assert len(middleware.tools) == 1
        tool_names = {tool.name for tool in middleware.tools}
        assert tool_names == {"swarm_execute"}

    def test_tool_is_structured_tool(self, middleware):
        """Tool should be a StructuredTool instance."""
        assert isinstance(middleware.tools[0], StructuredTool)

    def test_tool_description(self, middleware):
        """Tool should have a meaningful description."""
        tool = middleware.tools[0]
        assert "parallel" in tool.description.lower()

    def test_requires_subagent_config(self):
        """Middleware should require either subagent_graphs or factory."""
        with pytest.raises(ValueError, match="subagent_graphs or subagent_factory"):
            SwarmMiddleware()

    def test_accepts_subagent_graphs(self, subagent_graphs):
        """Middleware should accept pre-built subagent_graphs."""
        middleware = SwarmMiddleware(subagent_graphs=subagent_graphs)
        assert middleware.subagent_graphs is subagent_graphs

    def test_accepts_subagent_factory(self, subagent_graphs):
        """Middleware should accept a factory function."""
        middleware = SwarmMiddleware(subagent_factory=lambda: subagent_graphs)

        assert middleware._subagent_graphs is None
        result = middleware.subagent_graphs
        assert result is subagent_graphs

    def test_lazy_initialization(self, subagent_graphs):
        """Factory should only be called when subagent_graphs is accessed."""
        call_count = 0

        def counting_factory() -> dict[str, Runnable]:
            nonlocal call_count
            call_count += 1
            return subagent_graphs

        middleware = SwarmMiddleware(subagent_factory=counting_factory)

        assert call_count == 0
        _ = middleware.subagent_graphs
        assert call_count == 1
        _ = middleware.subagent_graphs
        assert call_count == 1


class TestSwarmMiddlewareConfiguration:
    def test_default_concurrency(self, subagent_graphs):
        middleware = SwarmMiddleware(subagent_graphs=subagent_graphs)
        assert middleware.default_concurrency == 10

    def test_custom_concurrency(self, subagent_graphs):
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            default_concurrency=20,
        )
        assert middleware.default_concurrency == 20

    def test_max_concurrency(self, subagent_graphs):
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            max_concurrency=100,
        )
        assert middleware.max_concurrency == 100

    def test_timeout_seconds(self, subagent_graphs):
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            timeout_seconds=600.0,
        )
        assert middleware.timeout_seconds == 600.0


class TestSwarmMiddlewareParallelism:
    @pytest.mark.asyncio
    async def test_swarm_execute_uses_num_parallel(self, subagent_graphs, tmp_path):
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            max_concurrency=50,
        )
        task_file = tmp_path / "tasks.jsonl"
        task_file.write_text('{"id": "1", "description": "Do one task"}\n')
        output_dir = tmp_path / "batch"

        tool = next(tool for tool in middleware.tools if tool.name == "swarm_execute")
        await tool.ainvoke(
            {
                "source": str(task_file),
                "num_parallel": 7,
                "output_dir": str(output_dir),
            }
        )

        summary = json.loads((output_dir / "summary.json").read_text())
        assert summary["concurrency"] == 7

    @pytest.mark.asyncio
    async def test_swarm_execute_clamps_num_parallel_to_max(
        self,
        subagent_graphs,
        tmp_path,
    ):
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            max_concurrency=8,
        )
        task_file = tmp_path / "tasks.jsonl"
        task_file.write_text('{"id": "1", "description": "Do one task"}\n')
        output_dir = tmp_path / "batch"

        tool = next(tool for tool in middleware.tools if tool.name == "swarm_execute")
        await tool.ainvoke(
            {
                "source": str(task_file),
                "num_parallel": 100,
                "output_dir": str(output_dir),
            }
        )

        summary = json.loads((output_dir / "summary.json").read_text())
        assert summary["concurrency"] == 8
