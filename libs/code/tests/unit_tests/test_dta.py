from typing import Any
from unittest.mock import patch, MagicMock
import pytest
from deepagents_code.dta.indexer import HybridToolIndexer, ToolCandidate
from deepagents_code.dta.gating import ToolNamespaceRegistry
from deepagents_code.dta.selector import ToolSelectorNode
from deepagents_code.dta.middleware import DynamicToolAllocationMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

@pytest.fixture
def mock_registry() -> ToolNamespaceRegistry:
    return ToolNamespaceRegistry()

@pytest.fixture
def indexer(mock_registry: ToolNamespaceRegistry) -> HybridToolIndexer:
    idx = HybridToolIndexer(registry=mock_registry)
    idx.add_tool(ToolCandidate(
        name="git_commit",
        description="Commit changes to git",
        schema={"name": "git_commit"},
        namespace="git"
    ))
    idx.add_tool(ToolCandidate(
        name="web_search",
        description="Search the web for information",
        schema={"name": "web_search"},
        namespace="builtin"
    ))
    idx.add_tool(ToolCandidate(
        name="db_query",
        description="Query the database",
        schema={"name": "db_query"},
        namespace="postgres"
    ))
    return idx

def test_indexer_search(indexer: HybridToolIndexer) -> None:
    # Test namespace gating
    namespaces = {"builtin", "git"}
    results = indexer.search("commit changes", namespaces=namespaces, top_k=2)
    names = [r["name"] for r in results]
    assert "git_commit" in names
    assert "db_query" not in names

def test_registry_namespaces() -> None:
    registry = ToolNamespaceRegistry()
    assert registry.classify_tool({"name": "git_commit"}) == "git"
    assert registry.classify_tool({"name": "postgres_query"}) == "database"

def test_selector_node() -> None:
    selector = ToolSelectorNode()
    candidates = [{"name": "tool1"}, {"name": "tool2"}]
    selected = selector.select(messages=[], candidates=candidates, budget=1)
    assert len(selected) == 1
    assert "tool1" in selected

class MockLLM:
    def __init__(self, structured_output_class: Any):
        self.structured_output_class = structured_output_class

    def with_structured_output(self, schema: Any) -> "MockLLM":
        return MockLLM(schema)

    def invoke(self, messages: list[Any], config: dict[str, Any] = None) -> Any:
        from deepagents_code.dta.gating import ActiveNamespacesResult
        from deepagents_code.dta.selector import ToolSelectionResult

        if self.structured_output_class.__name__ == "ActiveNamespacesResult":
            return self.structured_output_class(active_namespaces=["git", "builtin"])
        elif self.structured_output_class.__name__ == "ToolSelectionResult":
            return self.structured_output_class(
                selected_tools=["git_commit"],
                rationale="Selected git commit"
            )

    async def ainvoke(self, messages: list[Any], config: dict[str, Any] = None) -> Any:
        return self.invoke(messages, config)

@patch("deepagents_code.config.create_model")
def test_middleware_override(mock_create_model: MagicMock, indexer: HybridToolIndexer) -> None:
    mock_model_res = MagicMock()
    mock_model_res.model = MockLLM(None)
    mock_create_model.return_value = mock_model_res

    selector = ToolSelectorNode()
    middleware = DynamicToolAllocationMiddleware(indexer=indexer, selector_node=selector, max_tools_budget=7)

    class DummyTool:
        def __init__(self, name: str, description: str = ""):
            self.name = name
            self.description = description

    class DummyRequest:
        def __init__(self, tools: list[Any], messages: list[Any]):
            self.tools = tools
            self.messages = messages
            self.state = {}

        def override(self, tools: list[Any]) -> "DummyRequest":
            self.tools = tools
            return self

    req = DummyRequest(
        tools=[
            DummyTool("git_commit", "Commit changes to git"),
            DummyTool("web_search", "Search the web for information"),
            DummyTool("unknown_tool", "Does something unknown")
        ],
        messages=[HumanMessage(content="commit changes")]
    )

    def dummy_handler(request: Any) -> Any:
        return request

    # Using wrap_model_call
    res = middleware.wrap_model_call(req, dummy_handler)  # type: ignore

    allocated_names = [t.name for t in res.tools]
    assert "git_commit" in allocated_names
    assert "unknown_tool" not in allocated_names

def test_temporal_continuation() -> None:
    middleware = DynamicToolAllocationMiddleware(indexer=None, selector_node=None)  # type: ignore

    # Simulate a tool call in the last 3 turns
    messages = [
        AIMessage(content="", tool_calls=[{"name": "my_db_tool", "args": {}, "id": "1"}])
    ]

    recent = middleware._extract_recent_tools(messages)
    assert "my_db_tool" in recent
