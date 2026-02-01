"""Unit tests for SwarmMiddleware."""

import pytest
from langchain_core.messages import AIMessage

from deepagents_cli.swarm.middleware import SwarmMiddleware


class MockSubagent:
    """Mock subagent for testing."""

    async def ainvoke(self, state: dict) -> dict:
        return {"messages": [AIMessage(content="Task completed")]}


@pytest.fixture
def subagent_graphs():
    return {"general-purpose": MockSubagent()}


@pytest.fixture
def middleware(subagent_graphs):
    return SwarmMiddleware(subagent_graphs=subagent_graphs)


class TestSwarmMiddleware:
    def test_has_one_tool(self, middleware):
        """Middleware should provide swarm_execute tool."""
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "swarm_execute"

    def test_tool_is_structured_tool(self, middleware):
        """Tool should be a StructuredTool instance."""
        from langchain_core.tools import StructuredTool

        assert isinstance(middleware.tools[0], StructuredTool)

    def test_tool_description(self, middleware):
        """Tool should have a meaningful description."""
        tool = middleware.tools[0]
        assert "batch" in tool.description.lower() or "parallel" in tool.description.lower()

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

        # Factory should be called lazily
        assert middleware._subagent_graphs is None

        # Accessing subagent_graphs should trigger factory
        result = middleware.subagent_graphs
        assert result is subagent_graphs

    def test_lazy_initialization(self, subagent_graphs):
        """Factory should only be called when subagent_graphs is accessed."""
        call_count = 0

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return subagent_graphs

        middleware = SwarmMiddleware(subagent_factory=counting_factory)

        # Not called yet
        assert call_count == 0

        # First access calls factory
        _ = middleware.subagent_graphs
        assert call_count == 1

        # Second access doesn't call factory again
        _ = middleware.subagent_graphs
        assert call_count == 1


class TestSwarmMiddlewareConfiguration:
    def test_default_concurrency(self, subagent_graphs):
        """Should use default concurrency of 10."""
        middleware = SwarmMiddleware(subagent_graphs=subagent_graphs)
        assert middleware.default_concurrency == 10

    def test_custom_concurrency(self, subagent_graphs):
        """Should accept custom default concurrency."""
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            default_concurrency=20,
        )
        assert middleware.default_concurrency == 20

    def test_max_concurrency(self, subagent_graphs):
        """Should accept max concurrency limit."""
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            max_concurrency=100,
        )
        assert middleware.max_concurrency == 100

    def test_timeout_seconds(self, subagent_graphs):
        """Should accept custom timeout."""
        middleware = SwarmMiddleware(
            subagent_graphs=subagent_graphs,
            timeout_seconds=600.0,
        )
        assert middleware.timeout_seconds == 600.0
