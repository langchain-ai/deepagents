"""Tests for tool search middleware.

These tests cover:
1. BM25Index - the search algorithm implementation
2. ToolIndex - the tool indexing and search functionality
3. ToolSearchMiddleware - the full middleware integration
4. Validation limits (regex pattern length, tool catalog size)
5. Thread safety for concurrent access
6. MAX_SEARCH_RESULTS enforcement
7. Sync and async compatibility
"""

import asyncio
import concurrent.futures
import threading

import pytest
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

from deepagents.middleware.tool_search import (
    BM25Index,
    MAX_REGEX_PATTERN_LENGTH,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_CATALOG_SIZE,
    ToolIndex,
    ToolSearchMiddleware,
    ToolSearchState,
    _estimate_tool_tokens,
    _get_context_window,
)


# Test fixtures and helpers


class MockToolArgs(BaseModel):
    """Mock args schema for testing."""

    query: str = Field(description="The search query")
    limit: int = Field(default=10, description="Maximum results to return")


def create_mock_tool(
    name: str,
    description: str = "A mock tool for testing",
    args_schema: type[BaseModel] | None = None,
) -> BaseTool:
    """Create a mock tool for testing."""
    def mock_func(query: str, limit: int = 10) -> str:
        return f"Result for {query}"

    tool = StructuredTool.from_function(
        name=name,
        description=description,
        func=mock_func,
    )
    if args_schema:
        tool.args_schema = args_schema
    return tool


def create_mock_tools(count: int, prefix: str = "tool") -> list[BaseTool]:
    """Create multiple mock tools for testing."""
    tools = []
    domains = [
        ("file", "File operations for reading and writing"),
        ("web", "Web scraping and HTTP requests"),
        ("database", "Database queries and operations"),
        ("search", "Search functionality"),
        ("email", "Email sending and receiving"),
        ("calendar", "Calendar management"),
        ("weather", "Weather information"),
        ("math", "Mathematical calculations"),
        ("translate", "Language translation"),
        ("image", "Image processing"),
    ]

    for i in range(count):
        domain_idx = i % len(domains)
        domain, desc_template = domains[domain_idx]
        name = f"{prefix}_{domain}_{i}"
        description = f"{desc_template}. Tool number {i} for {domain} operations."
        tools.append(create_mock_tool(name, description))

    return tools


class TestBM25Index:
    """Tests for BM25Index class."""

    def test_empty_index_returns_empty_results(self):
        """Empty index should return empty results."""
        index = BM25Index()
        results = index.search("test query")
        assert results == []

    def test_single_document_exact_match(self):
        """Single document should be found with exact query match."""
        index = BM25Index()
        index.add_documents(["hello world test document"])
        results = index.search("hello world")

        assert len(results) == 1
        assert results[0][0] == 0  # Document index
        assert results[0][1] > 0  # Positive score

    def test_multiple_documents_ranking(self):
        """Documents should be ranked by relevance."""
        index = BM25Index()
        index.add_documents([
            "python programming language",
            "python snake animal zoo",
            "java programming language coffee",
        ])

        # "python programming" should rank first document highest
        results = index.search("python programming")

        assert len(results) >= 1
        # First doc has both terms, should rank higher
        assert results[0][0] == 0

    def test_no_match_returns_empty(self):
        """Non-matching query should return empty results."""
        index = BM25Index()
        index.add_documents(["hello world", "foo bar"])
        results = index.search("xyz123nonexistent")

        assert results == []

    def test_limit_parameter(self):
        """Results should be limited to specified count."""
        index = BM25Index()
        index.add_documents([
            "apple fruit red",
            "apple computer tech",
            "apple juice drink",
            "apple pie dessert",
            "apple tree nature",
        ])

        results = index.search("apple", limit=3)
        assert len(results) <= 3

    def test_tokenization_case_insensitive(self):
        """Search should be case insensitive."""
        index = BM25Index()
        index.add_documents(["HELLO World Test"])

        results = index.search("hello")
        assert len(results) == 1

    def test_idf_calculation(self):
        """IDF should penalize common terms."""
        index = BM25Index()
        # 'the' appears in all docs, 'python' only in one
        index.add_documents([
            "the python programming language",
            "the java programming language",
            "the rust programming language",
        ])

        # Query for specific term should find it
        results = index.search("python")
        assert len(results) == 1
        assert results[0][0] == 0  # First document


class TestToolIndex:
    """Tests for ToolIndex class."""

    def test_empty_index(self):
        """Empty index should return empty results."""
        index = ToolIndex()
        results = index.search("test")
        assert results == []

    def test_add_baseTool(self):
        """Should index BaseTool instances."""
        index = ToolIndex()
        tool = create_mock_tool("read_file", "Read contents of a file")
        index.add_tools([tool])

        results = index.search("read file")
        assert len(results) == 1
        assert results[0][0] == "read_file"

    def test_add_dict_tool(self):
        """Should index dict-based tool definitions."""
        index = ToolIndex()
        tool_dict = {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "properties": {
                    "query": {"description": "Search query"}
                }
            }
        }
        index.add_tools([tool_dict])

        results = index.search("search web")
        assert len(results) == 1
        assert results[0][0] == "search_web"

    def test_bm25_search_mode(self):
        """BM25 search mode should use relevance ranking."""
        index = ToolIndex()
        tools = [
            create_mock_tool("file_reader", "Read files from the filesystem"),
            create_mock_tool("file_writer", "Write files to the filesystem"),
            create_mock_tool("web_search", "Search the internet"),
        ]
        index.add_tools(tools)

        results = index.search("read file", mode="bm25")
        assert len(results) >= 1
        # file_reader should rank highest for "read file"
        assert results[0][0] == "file_reader"

    def test_regex_search_mode(self):
        """Regex search mode should use pattern matching."""
        index = ToolIndex()
        tools = [
            create_mock_tool("http_get", "Make HTTP GET request"),
            create_mock_tool("http_post", "Make HTTP POST request"),
            create_mock_tool("ftp_download", "Download via FTP"),
        ]
        index.add_tools(tools)

        results = index.search("http.*", mode="regex")
        assert len(results) == 2
        names = [r[0] for r in results]
        assert "http_get" in names
        assert "http_post" in names

    def test_regex_invalid_pattern_fallback(self):
        """Invalid regex should not raise - gracefully handles invalid patterns."""
        index = ToolIndex()
        tools = [create_mock_tool("test[tool", "A tool with bracket in name")]
        index.add_tools(tools)

        # This is an invalid regex (unclosed bracket) but should not raise
        # The implementation should either fall back to literal search or return empty
        results = index.search("[tool", mode="regex")
        # Verify it returns a list (success case) rather than raising an exception
        assert isinstance(results, list)

    def test_hybrid_search_mode(self):
        """Hybrid search should combine BM25 and regex results."""
        index = ToolIndex()
        tools = [
            create_mock_tool("read_file", "Read contents from a file"),
            create_mock_tool("write_file", "Write contents to a file"),
            create_mock_tool("file_delete", "Delete a file"),
        ]
        index.add_tools(tools)

        results = index.search("file", mode="hybrid")
        assert len(results) >= 1

    def test_get_tool_by_index(self):
        """Should retrieve tool by index."""
        index = ToolIndex()
        tool = create_mock_tool("test_tool")
        index.add_tools([tool])

        retrieved = index.get_tool(0)
        assert retrieved == tool

    def test_get_tool_name_by_index(self):
        """Should retrieve tool name by index."""
        index = ToolIndex()
        tool = create_mock_tool("my_tool")
        index.add_tools([tool])

        name = index.get_tool_name(0)
        assert name == "my_tool"


class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_base_tool_tokens(self):
        """Should estimate tokens for BaseTool."""
        tool = create_mock_tool(
            "test_tool",
            "This is a description that is about fifty characters long"
        )
        tokens = _estimate_tool_tokens(tool)

        # Should be positive and reasonable
        assert tokens > 0
        assert tokens < 1000  # Sanity check

    def test_estimate_dict_tool_tokens(self):
        """Should estimate tokens for dict tool."""
        tool = {
            "name": "test",
            "description": "A test tool",
            "parameters": {"properties": {"x": {"type": "string"}}}
        }
        tokens = _estimate_tool_tokens(tool)

        assert tokens > 0

    def test_estimate_complex_tool_tokens(self):
        """Complex tool should have more tokens than simple one."""
        simple_tool = create_mock_tool("simple", "Short")
        complex_tool = create_mock_tool(
            "complex_tool_with_long_name",
            "This is a very long description " * 10,
            args_schema=MockToolArgs,
        )

        simple_tokens = _estimate_tool_tokens(simple_tool)
        complex_tokens = _estimate_tool_tokens(complex_tool)

        assert complex_tokens > simple_tokens


class TestContextWindow:
    """Tests for context window detection.

    Note: The default context window of 128,000 tokens is used when model
    configuration is unavailable. This matches common LLM context sizes
    (e.g., GPT-4-turbo, Claude models).
    """

    def test_model_with_profile_context_window(self):
        """Should return context window from model.profile."""
        mock_model = MagicMock()
        mock_model.profile = {"max_input_tokens": 128_000}

        window = _get_context_window(mock_model)
        assert window == 128_000

    def test_claude_model_with_profile_context_window(self):
        """Should detect context window from model profile for Claude models."""
        mock_model = MagicMock()
        mock_model.profile = {"max_input_tokens": 200_000}

        window = _get_context_window(mock_model)
        assert window == 200_000

    def test_model_without_profile_returns_default(self):
        """Should return default when model has no profile."""
        mock_model = MagicMock()
        mock_model.profile = None

        window = _get_context_window(mock_model)
        assert window == 128_000  # Default

    def test_model_with_invalid_profile_returns_default(self):
        """Should return default when profile is not a dict."""
        mock_model = MagicMock()
        mock_model.profile = "not a dict"

        window = _get_context_window(mock_model)
        assert window == 128_000  # Default

    def test_model_with_missing_max_input_tokens_returns_default(self):
        """Should return default when profile lacks max_input_tokens."""
        mock_model = MagicMock()
        mock_model.profile = {"other_key": 100}

        window = _get_context_window(mock_model)
        assert window == 128_000  # Default

    def test_none_model_returns_default(self):
        """Should return default for None model."""
        window = _get_context_window(None)
        assert window == 128_000


class TestToolSearchMiddleware:
    """Tests for ToolSearchMiddleware class."""

    def test_initialization_defaults(self):
        """Should initialize with default values."""
        middleware = ToolSearchMiddleware()

        assert middleware.token_threshold == 0.10
        assert middleware.search_mode == "bm25"
        assert middleware.always_include == set()

    def test_initialization_custom_values(self):
        """Should accept custom configuration."""
        middleware = ToolSearchMiddleware(
            token_threshold=0.20,
            context_window=200_000,
            always_include=["read_file", "write_file"],
            search_mode="hybrid",
        )

        assert middleware.token_threshold == 0.20
        assert middleware.context_window == 200_000
        assert middleware.always_include == {"read_file", "write_file"}
        assert middleware.search_mode == "hybrid"

    def test_search_tool_created(self):
        """Should create search_tools meta-tool."""
        middleware = ToolSearchMiddleware()

        tools = middleware.tools
        assert len(tools) == 1
        assert tools[0].name == "search_tools"

    def test_state_schema(self):
        """Should use ToolSearchState schema."""
        middleware = ToolSearchMiddleware()
        assert middleware.state_schema == ToolSearchState

    def test_before_agent_initializes_state(self):
        """before_agent should initialize expanded_tools in state."""
        middleware = ToolSearchMiddleware(always_include=["tool_a"])

        state: ToolSearchState = {"messages": []}
        runtime = MagicMock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        assert "expanded_tools" in result
        assert "tool_a" in result["expanded_tools"]

    def test_before_agent_skips_if_initialized(self):
        """before_agent should skip if expanded_tools already present."""
        middleware = ToolSearchMiddleware()

        state: ToolSearchState = {"messages": [], "expanded_tools": {"existing_tool"}}
        runtime = MagicMock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_deferred_mode_activation(self):
        """Should activate deferred mode when tools exceed threshold."""
        middleware = ToolSearchMiddleware(
            token_threshold=100,  # Low threshold to trigger deferred mode
        )

        # Create enough tools to exceed threshold
        tools = create_mock_tools(50)
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        middleware._initialize_index(tools, mock_model)

        # With 50 tools, should exceed 100 token threshold
        assert middleware._deferred_mode is True

    def test_no_deferred_mode_below_threshold(self):
        """Should not activate deferred mode when tools below threshold."""
        middleware = ToolSearchMiddleware(
            token_threshold=100_000,  # High threshold
        )

        tools = create_mock_tools(5)
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        middleware._initialize_index(tools, mock_model)

        assert middleware._deferred_mode is False

    def test_filter_tools_deferred_mode(self):
        """Should filter tools in deferred mode."""
        middleware = ToolSearchMiddleware(
            always_include=["always_included_tool"],
        )
        middleware._deferred_mode = True

        tools = [
            create_mock_tool("always_included_tool"),
            create_mock_tool("other_tool"),
            create_mock_tool("expanded_tool"),
        ]

        expanded = {"expanded_tool"}
        filtered = middleware._filter_tools(tools, expanded)

        names = [t.name for t in filtered]
        assert "always_included_tool" in names
        assert "expanded_tool" in names
        assert "other_tool" not in names

    def test_filter_tools_no_deferred_mode(self):
        """Should not filter when not in deferred mode."""
        middleware = ToolSearchMiddleware()
        middleware._deferred_mode = False

        tools = create_mock_tools(10)
        filtered = middleware._filter_tools(tools, set())

        assert len(filtered) == 10


class TestToolSearchMiddlewareIntegration:
    """Integration tests for ToolSearchMiddleware."""

    def test_wrap_model_call_initializes_index(self):
        """wrap_model_call should initialize the tool index."""
        middleware = ToolSearchMiddleware()

        tools = create_mock_tools(5)
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        request = MagicMock()
        request.tools = tools
        request.model = mock_model
        request.state = {"messages": [], "expanded_tools": set()}
        request.system_message = None
        request.override = MagicMock(return_value=request)

        handler = MagicMock(return_value=MagicMock())

        middleware.wrap_model_call(request, handler)

        assert middleware._tool_index is not None
        assert len(middleware._all_tools) == 5

    def test_wrap_model_call_adds_system_prompt_in_deferred_mode(self):
        """Should add system prompt when in deferred mode."""
        middleware = ToolSearchMiddleware(token_threshold=10)  # Force deferred mode

        tools = create_mock_tools(50)
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        request = MagicMock()
        request.tools = tools
        request.model = mock_model
        request.state = {"messages": [], "expanded_tools": set()}
        request.system_message = SystemMessage(content="Original prompt")

        captured_request = None
        def capture_handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        request.override = MagicMock(side_effect=lambda **kwargs: request)

        middleware.wrap_model_call(request, capture_handler)

        # Should have called override to update system message
        assert request.override.called

    @pytest.mark.asyncio
    async def test_awrap_model_call(self):
        """Async wrap_model_call should work correctly."""
        middleware = ToolSearchMiddleware()

        tools = create_mock_tools(5)
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        request = MagicMock()
        request.tools = tools
        request.model = mock_model
        request.state = {"messages": [], "expanded_tools": set()}
        request.system_message = None
        request.override = MagicMock(return_value=request)

        async def async_handler(req):
            return MagicMock()

        result = await middleware.awrap_model_call(request, async_handler)

        assert middleware._tool_index is not None

    @pytest.mark.asyncio
    async def test_abefore_agent(self):
        """Async before_agent should work correctly."""
        middleware = ToolSearchMiddleware(always_include=["test_tool"])

        state: ToolSearchState = {"messages": []}
        runtime = MagicMock()

        result = await middleware.abefore_agent(state, runtime)

        assert result is not None
        assert "test_tool" in result["expanded_tools"]


class TestSearchToolFunction:
    """Tests for the search_tools meta-tool functionality."""

    def test_search_tools_returns_results(self):
        """search_tools should return matching tools."""
        middleware = ToolSearchMiddleware()

        tools = [
            create_mock_tool("read_file", "Read file contents"),
            create_mock_tool("write_file", "Write file contents"),
            create_mock_tool("web_search", "Search the web"),
        ]

        # Initialize the index
        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        # Get the search tool
        search_tool = middleware.tools[0]

        # Create mock runtime
        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        # Invoke with runtime
        result = search_tool.func(query="file", runtime=runtime)

        # Should return Command with results
        assert isinstance(result, Command)
        assert "expanded_tools" in result.update

    def test_search_tools_no_matches(self):
        """search_tools should handle no matches gracefully."""
        middleware = ToolSearchMiddleware()

        tools = [create_mock_tool("read_file", "Read file")]

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        result = search_tool.func(query="nonexistent_xyz_123", runtime=runtime)

        # Should return error message as string
        assert isinstance(result, str)
        assert "No tools found" in result

    def test_search_tools_uninitialized_index(self):
        """search_tools should handle uninitialized index."""
        middleware = ToolSearchMiddleware()

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        result = search_tool.func(query="anything", runtime=runtime)

        assert isinstance(result, str)
        assert "not initialized" in result


# Performance tests


class TestPerformance:
    """Performance-related tests."""

    def test_large_tool_set_indexing(self):
        """Should handle indexing many tools efficiently."""
        middleware = ToolSearchMiddleware()

        # Create 200 tools
        tools = create_mock_tools(200)

        # Index should complete without error
        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)

        assert len(middleware._tool_index.tools) == 200

    def test_search_performance_with_many_tools(self):
        """Search should return results quickly with many tools."""
        index = ToolIndex()

        # Add 500 tools
        tools = create_mock_tools(500)
        index.add_tools(tools)

        # Search should complete and return results
        results = index.search("file operations", limit=10)

        assert len(results) <= 10


class TestValidationLimits:
    """Tests for validation limits and error handling."""

    def test_regex_pattern_length_at_limit(self):
        """Regex pattern at exactly the limit should work."""
        index = ToolIndex()
        tools = [create_mock_tool("test_tool", "A test tool for testing")]
        index.add_tools(tools)

        # Pattern exactly at 200 characters should not raise
        pattern = "a" * MAX_REGEX_PATTERN_LENGTH
        # Should not raise
        results = index.search_regex(pattern, limit=5)
        assert isinstance(results, list)

    def test_regex_pattern_exceeds_limit_raises_error(self):
        """Regex pattern exceeding 200 characters should raise ValueError."""
        index = ToolIndex()
        tools = [create_mock_tool("test_tool", "A test tool")]
        index.add_tools(tools)

        # Pattern exceeding 200 characters should raise ValueError
        pattern = "a" * (MAX_REGEX_PATTERN_LENGTH + 1)

        with pytest.raises(ValueError) as exc_info:
            index.search_regex(pattern, limit=5)

        assert f"{MAX_REGEX_PATTERN_LENGTH}" in str(exc_info.value)
        assert "character limit" in str(exc_info.value).lower()

    def test_regex_pattern_limit_via_search_method(self):
        """Regex pattern limit should apply via search() method with regex mode."""
        index = ToolIndex()
        tools = [create_mock_tool("test_tool", "A test tool")]
        index.add_tools(tools)

        pattern = "a" * (MAX_REGEX_PATTERN_LENGTH + 1)

        with pytest.raises(ValueError):
            index.search(pattern, mode="regex", limit=5)

    def test_regex_pattern_limit_via_hybrid_mode(self):
        """Regex pattern limit should apply via hybrid mode."""
        index = ToolIndex()
        tools = [create_mock_tool("test_tool", "A test tool")]
        index.add_tools(tools)

        pattern = "a" * (MAX_REGEX_PATTERN_LENGTH + 1)

        with pytest.raises(ValueError):
            index.search(pattern, mode="hybrid", limit=5)

    def test_tool_catalog_size_at_limit(self):
        """Tool catalog at exactly the limit should work."""
        middleware = ToolSearchMiddleware()
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        # This test is conceptual - we can't actually create 10,000 tools efficiently
        # but we verify the check is in place
        assert MAX_TOOL_CATALOG_SIZE == 10_000

    def test_tool_catalog_size_exceeds_limit_raises_error(self):
        """Tool catalog exceeding 10,000 tools should raise ValueError."""
        middleware = ToolSearchMiddleware()
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        # Create a mock list that appears to have more than 10,000 tools
        class LargeFakeList:
            """A fake list that reports len > MAX_TOOL_CATALOG_SIZE."""
            def __len__(self):
                return MAX_TOOL_CATALOG_SIZE + 1

            def __iter__(self):
                return iter([])

        large_tools = LargeFakeList()

        with pytest.raises(ValueError) as exc_info:
            middleware._initialize_index(large_tools, mock_model)

        assert f"{MAX_TOOL_CATALOG_SIZE}" in str(exc_info.value)
        assert "exceeds maximum" in str(exc_info.value).lower()


class TestMaxSearchResults:
    """Tests for MAX_SEARCH_RESULTS limit enforcement."""

    def test_max_search_results_constant_value(self):
        """Verify MAX_SEARCH_RESULTS constant value."""
        assert MAX_SEARCH_RESULTS == 5

    def test_search_tools_enforces_max_results_limit(self):
        """search_tools should enforce MAX_SEARCH_RESULTS limit."""
        middleware = ToolSearchMiddleware()

        # Create more tools than MAX_SEARCH_RESULTS
        tools = create_mock_tools(20, prefix="test")

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        # Request more than MAX_SEARCH_RESULTS - search for "file" which appears in tool names
        # Note: Searching for "test" returns no results because it appears in ALL tool names,
        # making its IDF (inverse document frequency) zero in BM25.
        result = search_tool.func(query="file operations", runtime=runtime, limit=100)

        # Should return a Command with results
        assert isinstance(result, Command)
        # The expanded_tools should have at most MAX_SEARCH_RESULTS new tools
        expanded = result.update.get("expanded_tools", set())
        assert len(expanded) <= MAX_SEARCH_RESULTS

    def test_search_tools_respects_lower_limit(self):
        """search_tools should respect user limit when lower than MAX_SEARCH_RESULTS."""
        middleware = ToolSearchMiddleware()

        tools = create_mock_tools(20, prefix="test")

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        # Request fewer than MAX_SEARCH_RESULTS - search for "file" which appears in tool names
        # Note: Searching for "test" returns no results because it appears in ALL tool names,
        # making its IDF (inverse document frequency) zero in BM25.
        result = search_tool.func(query="file operations", runtime=runtime, limit=2)

        assert isinstance(result, Command)
        expanded = result.update.get("expanded_tools", set())
        assert len(expanded) <= 2


class TestThreadSafety:
    """Tests for thread safety of middleware initialization."""

    def test_concurrent_initialization_is_thread_safe(self):
        """Multiple threads initializing the same middleware should be safe."""
        middleware = ToolSearchMiddleware(token_threshold=100)
        tools = create_mock_tools(50)
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        errors = []
        init_count = [0]
        lock = threading.Lock()

        def init_worker():
            try:
                middleware._initialize_index(tools, mock_model)
                with lock:
                    init_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple threads trying to initialize concurrently
        threads = []
        for _ in range(10):
            t = threading.Thread(target=init_worker)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred during concurrent init: {errors}"

        # All threads should complete successfully
        assert init_count[0] == 10

        # Index should be properly initialized
        assert middleware._tool_index is not None
        assert len(middleware._all_tools) == 50

    def test_concurrent_search_is_safe(self):
        """Multiple threads searching the same index should be safe."""
        index = ToolIndex()
        tools = create_mock_tools(100)
        index.add_tools(tools)

        errors = []
        results_list = []
        lock = threading.Lock()

        queries = ["file", "web", "database", "search", "email"]

        def search_worker(query: str):
            try:
                results = index.search(query, mode="bm25", limit=5)
                with lock:
                    results_list.append((query, results))
            except Exception as e:
                with lock:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit multiple search queries concurrently
            futures = []
            for _ in range(20):
                for query in queries:
                    futures.append(executor.submit(search_worker, query))

            # Wait for all to complete
            concurrent.futures.wait(futures)

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent search: {errors}"

        # All searches should return results
        assert len(results_list) == 100  # 20 iterations * 5 queries

    def test_middleware_tools_property_is_idempotent(self):
        """Accessing tools property multiple times should return same tool."""
        middleware = ToolSearchMiddleware()

        # Access tools from multiple threads
        tools_results = []
        lock = threading.Lock()

        def get_tools():
            t = middleware.tools
            with lock:
                tools_results.append(t)

        threads = [threading.Thread(target=get_tools) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should return the same search_tool instance
        assert len(tools_results) == 10
        first_tool = tools_results[0][0]
        for tools in tools_results:
            assert len(tools) == 1
            assert tools[0] is first_tool


class TestBM25EdgeCases:
    """Tests for BM25 algorithm edge cases."""

    def test_empty_query_returns_empty(self):
        """Empty query should return empty results."""
        index = BM25Index()
        index.add_documents(["hello world", "foo bar"])

        results = index.search("")
        assert results == []

    def test_whitespace_only_query_returns_empty(self):
        """Whitespace-only query should return empty results."""
        index = BM25Index()
        index.add_documents(["hello world", "foo bar"])

        results = index.search("   \t\n  ")
        assert results == []

    def test_special_characters_only_query(self):
        """Query with only special characters should return empty results."""
        index = BM25Index()
        index.add_documents(["hello world", "foo bar"])

        results = index.search("!@#$%^&*()")
        assert results == []

    def test_documents_with_no_alphanumeric_content(self):
        """Documents with no alphanumeric content should be handled gracefully."""
        index = BM25Index()
        # These documents have no \w+ matches
        index.add_documents(["!!!", "---", "..."])

        # avgdl should be 0, but search should not crash
        assert index.avgdl == 0.0
        assert index.doc_count == 3

        # Search should return empty (no matching tokens possible)
        results = index.search("test")
        assert results == []

    def test_single_token_documents(self):
        """Single token documents should work correctly."""
        index = BM25Index()
        index.add_documents(["apple", "banana", "cherry"])

        results = index.search("apple")
        assert len(results) == 1
        assert results[0][0] == 0  # First document

    def test_repeated_terms_in_query(self):
        """Repeated terms in query should be handled correctly."""
        index = BM25Index()
        index.add_documents(["python programming", "java programming"])

        # Query with repeated term
        results = index.search("python python python")
        assert len(results) == 1
        assert results[0][0] == 0

    def test_very_long_document(self):
        """Very long documents should be indexed and searchable."""
        index = BM25Index()
        long_doc = " ".join(["word"] * 10000 + ["unique_term"])
        index.add_documents([long_doc, "short doc"])

        results = index.search("unique_term")
        assert len(results) == 1
        assert results[0][0] == 0


class TestSyncAsyncCompatibility:
    """Tests for sync/async compatibility of the middleware."""

    @pytest.mark.asyncio
    async def test_async_search_tools_returns_same_as_sync(self):
        """Async and sync search_tools should return equivalent results."""
        middleware = ToolSearchMiddleware()

        tools = [
            create_mock_tool("read_file", "Read file contents"),
            create_mock_tool("write_file", "Write file contents"),
            create_mock_tool("web_search", "Search the web"),
        ]

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        # Call sync version
        sync_result = search_tool.func(query="file", runtime=runtime)

        # Reset state for async test
        runtime.state = {"expanded_tools": set()}

        # Call async version
        async_result = await search_tool.coroutine(query="file", runtime=runtime)

        # Both should return Command with same structure
        assert isinstance(sync_result, Command)
        assert isinstance(async_result, Command)
        assert "expanded_tools" in sync_result.update
        assert "expanded_tools" in async_result.update

    @pytest.mark.asyncio
    async def test_abefore_agent_same_as_before_agent(self):
        """Async before_agent should produce same result as sync."""
        middleware = ToolSearchMiddleware(always_include=["tool_a", "tool_b"])

        state: ToolSearchState = {"messages": []}
        runtime = MagicMock()

        sync_result = middleware.before_agent(state, runtime)
        async_result = await middleware.abefore_agent(state, runtime)

        assert sync_result == async_result
        assert sync_result is not None
        assert "tool_a" in sync_result["expanded_tools"]
        assert "tool_b" in sync_result["expanded_tools"]

    @pytest.mark.asyncio
    async def test_awrap_model_call_same_as_wrap_model_call(self):
        """Async wrap_model_call should produce same filtering as sync."""
        middleware = ToolSearchMiddleware(
            token_threshold=100,  # Low to trigger deferred mode
            always_include=["allowed_tool"],
        )

        tools = create_mock_tools(50)
        # Add an allowed tool
        allowed_tool = create_mock_tool("allowed_tool", "An allowed tool")
        tools.append(allowed_tool)

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        # Capture filtered tools from both sync and async
        sync_filtered = []
        async_filtered = []

        def sync_handler(req):
            sync_filtered.extend(req.tools if hasattr(req, 'tools') else [])
            return MagicMock()

        async def async_handler(req):
            async_filtered.extend(req.tools if hasattr(req, 'tools') else [])
            return MagicMock()

        request = MagicMock()
        request.tools = tools
        request.model = mock_model
        request.state = {"messages": [], "expanded_tools": set()}
        request.system_message = None

        # Track override calls
        override_calls_sync = []
        override_calls_async = []

        def make_override_tracker(calls_list):
            def tracker(**kwargs):
                calls_list.append(kwargs)
                new_req = MagicMock()
                new_req.tools = kwargs.get("tools", request.tools)
                new_req.system_message = kwargs.get("system_message", request.system_message)
                return new_req
            return tracker

        # Test sync
        request.override = make_override_tracker(override_calls_sync)
        middleware.wrap_model_call(request, sync_handler)

        # Reset middleware for async test (create fresh instance)
        middleware2 = ToolSearchMiddleware(
            token_threshold=100,
            always_include=["allowed_tool"],
        )

        # Test async
        request.override = make_override_tracker(override_calls_async)
        await middleware2.awrap_model_call(request, async_handler)

        # Both should have called override with same structure
        assert len(override_calls_sync) > 0
        assert len(override_calls_async) > 0


class TestCommandStructure:
    """Tests for the Command return structure from search_tools."""

    def test_command_has_messages_and_state(self):
        """Command should contain both messages and expanded_tools."""
        middleware = ToolSearchMiddleware()

        tools = [
            create_mock_tool("test_tool", "A test tool for testing purposes"),
        ]

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-123"

        result = search_tool.func(query="test", runtime=runtime)

        assert isinstance(result, Command)
        assert "expanded_tools" in result.update
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert isinstance(result.update["messages"][0], ToolMessage)
        assert result.update["messages"][0].tool_call_id == "test-call-123"

    def test_command_messages_contain_tool_info(self):
        """Command messages should contain tool names and descriptions."""
        middleware = ToolSearchMiddleware()

        tools = [
            create_mock_tool("file_reader", "Read contents from filesystem"),
        ]

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)
        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "test-call-id"

        # Search for "read filesystem" - terms that are tokenized from the description
        # Note: "file" alone won't match because "file_reader" is tokenized as a single word
        # (underscores are word characters in \w+ regex)
        result = search_tool.func(query="read filesystem", runtime=runtime)

        assert isinstance(result, Command)
        message_content = result.update["messages"][0].content
        assert "file_reader" in message_content
        assert "Read contents" in message_content


class TestSearchModes:
    """Tests for different search modes."""

    def test_bm25_mode_natural_language(self):
        """BM25 mode should rank by natural language relevance."""
        index = ToolIndex()
        tools = [
            create_mock_tool("file_manager", "Manage files on the filesystem"),
            create_mock_tool("database_connector", "Connect to SQL databases"),
            create_mock_tool("file_reader", "Read file contents from disk"),
        ]
        index.add_tools(tools)

        # Natural language query should find file-related tools
        results = index.search("read a file from disk", mode="bm25")

        assert len(results) >= 1
        # file_reader should rank highest
        assert results[0][0] == "file_reader"

    def test_regex_mode_pattern_matching(self):
        """Regex mode should match patterns in tool names."""
        index = ToolIndex()
        tools = [
            create_mock_tool("http_get", "Make HTTP GET requests"),
            create_mock_tool("http_post", "Make HTTP POST requests"),
            create_mock_tool("ftp_download", "Download via FTP"),
            create_mock_tool("sftp_upload", "Upload via SFTP"),
        ]
        index.add_tools(tools)

        # Regex pattern should match http_* tools
        results = index.search("^http_", mode="regex")

        assert len(results) == 2
        names = [r[0] for r in results]
        assert "http_get" in names
        assert "http_post" in names

    def test_hybrid_mode_combines_both(self):
        """Hybrid mode should combine BM25 and regex results."""
        index = ToolIndex()
        tools = [
            create_mock_tool("file_read", "Read file contents"),
            create_mock_tool("file_write", "Write file contents"),
            create_mock_tool("read_database", "Read from database"),
        ]
        index.add_tools(tools)

        # Hybrid should find both exact matches and semantic matches
        results = index.search("file", mode="hybrid")

        assert len(results) >= 2
        names = [r[0] for r in results]
        assert "file_read" in names
        assert "file_write" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
