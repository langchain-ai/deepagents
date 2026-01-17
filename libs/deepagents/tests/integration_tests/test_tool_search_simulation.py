"""Integration tests simulating real-world tool search scenarios.

This module provides comprehensive simulation tests for the ToolSearchMiddleware
with realistic tool catalogs similar to what would be seen with MCP servers.

These tests verify:
1. Performance with large tool catalogs (100+ tools)
2. Token threshold calculations and deferred mode activation
3. Search accuracy across different domains
4. Concurrent usage patterns
5. Memory efficiency with large indexes
"""

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepagents.middleware.tool_search import (
    BM25Index,
    ToolIndex,
    ToolSearchMiddleware,
    ToolSearchState,
    _estimate_tool_tokens,
)


# =============================================================================
# Realistic Tool Generators
# =============================================================================


class FilePathArg(BaseModel):
    """File path argument schema."""

    path: str = Field(description="Absolute path to the file")


class SearchQueryArg(BaseModel):
    """Search query argument schema."""

    query: str = Field(description="Search query string")
    limit: int = Field(default=10, description="Maximum results to return")


class DatabaseQueryArg(BaseModel):
    """Database query argument schema."""

    query: str = Field(description="SQL query to execute")
    database: str = Field(default="default", description="Target database name")


def generate_mcp_github_tools() -> list[BaseTool]:
    """Generate tools similar to GitHub MCP server (~35 tools)."""
    tools = []

    # Repository operations
    repo_ops = [
        ("github_list_repos", "List repositories for the authenticated user or organization"),
        ("github_get_repo", "Get details of a specific repository"),
        ("github_create_repo", "Create a new repository"),
        ("github_delete_repo", "Delete a repository"),
        ("github_fork_repo", "Fork an existing repository"),
        ("github_clone_repo", "Clone a repository to local storage"),
    ]

    # Issue operations
    issue_ops = [
        ("github_list_issues", "List issues in a repository with optional filters"),
        ("github_create_issue", "Create a new issue in a repository"),
        ("github_update_issue", "Update an existing issue"),
        ("github_close_issue", "Close an issue"),
        ("github_get_issue", "Get details of a specific issue"),
        ("github_add_issue_comment", "Add a comment to an issue"),
        ("github_list_issue_comments", "List comments on an issue"),
    ]

    # Pull request operations
    pr_ops = [
        ("github_list_prs", "List pull requests in a repository"),
        ("github_create_pr", "Create a new pull request"),
        ("github_merge_pr", "Merge a pull request"),
        ("github_close_pr", "Close a pull request without merging"),
        ("github_get_pr", "Get details of a specific pull request"),
        ("github_review_pr", "Submit a review for a pull request"),
        ("github_list_pr_commits", "List commits in a pull request"),
    ]

    # Branch operations
    branch_ops = [
        ("github_list_branches", "List branches in a repository"),
        ("github_create_branch", "Create a new branch"),
        ("github_delete_branch", "Delete a branch"),
        ("github_get_branch", "Get details of a specific branch"),
    ]

    # Commit operations
    commit_ops = [
        ("github_list_commits", "List commits in a repository"),
        ("github_get_commit", "Get details of a specific commit"),
        ("github_compare_commits", "Compare two commits"),
    ]

    # Actions operations
    actions_ops = [
        ("github_list_workflows", "List GitHub Actions workflows"),
        ("github_trigger_workflow", "Manually trigger a workflow"),
        ("github_list_workflow_runs", "List recent workflow runs"),
        ("github_get_workflow_run", "Get details of a workflow run"),
    ]

    all_ops = repo_ops + issue_ops + pr_ops + branch_ops + commit_ops + actions_ops

    for name, description in all_ops:
        def make_func(n: str):
            def func(**kwargs: Any) -> str:
                return f"Executed {n}"
            return func

        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=make_func(name),
        )
        tools.append(tool)

    return tools


def generate_mcp_slack_tools() -> list[BaseTool]:
    """Generate tools similar to Slack MCP server (~11 tools)."""
    tools = []

    ops = [
        ("slack_send_message", "Send a message to a Slack channel or user"),
        ("slack_list_channels", "List all channels in the workspace"),
        ("slack_get_channel_info", "Get detailed information about a channel"),
        ("slack_list_users", "List all users in the workspace"),
        ("slack_get_user_info", "Get detailed information about a user"),
        ("slack_post_thread_reply", "Reply to a message thread"),
        ("slack_add_reaction", "Add an emoji reaction to a message"),
        ("slack_remove_reaction", "Remove an emoji reaction from a message"),
        ("slack_upload_file", "Upload a file to a channel"),
        ("slack_search_messages", "Search for messages in the workspace"),
        ("slack_set_status", "Set your Slack status"),
    ]

    for name, description in ops:
        def make_func(n: str):
            def func(**kwargs: Any) -> str:
                return f"Executed {n}"
            return func

        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=make_func(name),
        )
        tools.append(tool)

    return tools


def generate_mcp_database_tools() -> list[BaseTool]:
    """Generate tools for database operations (~15 tools)."""
    tools = []

    ops = [
        ("db_execute_query", "Execute a SQL query on the database"),
        ("db_list_tables", "List all tables in the database"),
        ("db_describe_table", "Get schema information for a table"),
        ("db_insert_rows", "Insert one or more rows into a table"),
        ("db_update_rows", "Update rows matching a condition"),
        ("db_delete_rows", "Delete rows matching a condition"),
        ("db_create_table", "Create a new table with specified schema"),
        ("db_drop_table", "Drop an existing table"),
        ("db_create_index", "Create an index on a table column"),
        ("db_drop_index", "Drop an existing index"),
        ("db_backup", "Create a backup of the database"),
        ("db_restore", "Restore database from a backup"),
        ("db_get_stats", "Get database statistics and performance metrics"),
        ("db_vacuum", "Optimize database storage"),
        ("db_explain_query", "Get query execution plan"),
    ]

    for name, description in ops:
        def make_func(n: str):
            def func(**kwargs: Any) -> str:
                return f"Executed {n}"
            return func

        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=make_func(name),
        )
        tools.append(tool)

    return tools


def generate_mcp_filesystem_tools() -> list[BaseTool]:
    """Generate filesystem tools (~10 tools)."""
    tools = []

    ops = [
        ("fs_read_file", "Read contents of a file from the filesystem"),
        ("fs_write_file", "Write content to a file on the filesystem"),
        ("fs_list_directory", "List contents of a directory"),
        ("fs_create_directory", "Create a new directory"),
        ("fs_delete_file", "Delete a file from the filesystem"),
        ("fs_delete_directory", "Delete a directory and its contents"),
        ("fs_copy_file", "Copy a file to a new location"),
        ("fs_move_file", "Move or rename a file"),
        ("fs_get_file_info", "Get metadata about a file"),
        ("fs_search_files", "Search for files matching a pattern"),
    ]

    for name, description in ops:
        def make_func(n: str):
            def func(**kwargs: Any) -> str:
                return f"Executed {n}"
            return func

        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=make_func(name),
        )
        tools.append(tool)

    return tools


def generate_generic_tools(count: int, domain: str = "generic") -> list[BaseTool]:
    """Generate generic tools for a domain."""
    tools = []
    actions = ["get", "list", "create", "update", "delete", "search", "export", "import"]

    for i in range(count):
        action = actions[i % len(actions)]
        name = f"{domain}_{action}_{i}"
        description = f"{action.capitalize()} operation #{i} for {domain} domain"

        def make_func(n: str):
            def func(**kwargs: Any) -> str:
                return f"Executed {n}"
            return func

        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=make_func(name),
        )
        tools.append(tool)

    return tools


# =============================================================================
# Simulation Tests
# =============================================================================


class TestLargeToolCatalog:
    """Tests for handling large tool catalogs."""

    def test_github_mcp_tools_indexing(self):
        """Should efficiently index GitHub MCP tools."""
        tools = generate_mcp_github_tools()
        index = ToolIndex()

        start = time.perf_counter()
        index.add_tools(tools)
        elapsed = time.perf_counter() - start

        assert len(index.tools) == len(tools)
        assert elapsed < 1.0  # Should complete in under 1 second

    def test_combined_mcp_tools_search(self):
        """Should search across combined MCP server tools."""
        # Simulate multiple MCP servers
        all_tools = (
            generate_mcp_github_tools()
            + generate_mcp_slack_tools()
            + generate_mcp_database_tools()
            + generate_mcp_filesystem_tools()
        )

        index = ToolIndex()
        index.add_tools(all_tools)

        # Test search accuracy
        results = index.search("create pull request", mode="bm25", limit=5)
        assert len(results) > 0
        assert any("pr" in r[0].lower() or "pull" in r[0].lower() for r in results)

        results = index.search("send message slack", mode="bm25", limit=5)
        assert len(results) > 0
        assert any("slack" in r[0].lower() for r in results)

        results = index.search("execute sql query database", mode="bm25", limit=5)
        assert len(results) > 0
        assert any("db" in r[0].lower() or "query" in r[0].lower() for r in results)

    def test_100_plus_tools_performance(self):
        """Should handle 100+ tools with acceptable performance."""
        tools = (
            generate_mcp_github_tools()  # ~35
            + generate_mcp_slack_tools()  # ~11
            + generate_mcp_database_tools()  # ~15
            + generate_mcp_filesystem_tools()  # ~10
            + generate_generic_tools(50, "api")  # 50
        )

        assert len(tools) > 100

        # Use a lower threshold (3%) to ensure deferred mode activates
        # Total tokens for ~117 tools is ~5883, threshold at 3% of 128k = 3840
        middleware = ToolSearchMiddleware(
            token_threshold=0.03,  # 3% threshold
            context_window=128_000,
        )

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        start = time.perf_counter()
        middleware._initialize_index(tools, mock_model)
        init_time = time.perf_counter() - start

        # Verify deferred mode activated with this many tools
        assert middleware._deferred_mode is True
        assert init_time < 2.0  # Should complete in under 2 seconds

        # Test search performance
        start = time.perf_counter()
        for _ in range(100):
            middleware._tool_index.search("create file", mode="bm25", limit=5)
        search_time = time.perf_counter() - start

        assert search_time < 1.0  # 100 searches in under 1 second

    def test_500_tools_stress_test(self):
        """Stress test with 500 tools."""
        tools = generate_generic_tools(500, "stress")

        index = ToolIndex()
        start = time.perf_counter()
        index.add_tools(tools)
        index_time = time.perf_counter() - start

        assert index_time < 5.0  # Index 500 tools in under 5 seconds

        # Search should still be fast
        start = time.perf_counter()
        results = index.search("get operation", mode="bm25", limit=5)
        search_time = time.perf_counter() - start

        assert len(results) > 0
        assert search_time < 0.1  # Single search under 100ms


class TestDeferredModeActivation:
    """Tests for deferred mode activation based on token threshold."""

    def test_threshold_percentage_calculation(self):
        """Should correctly calculate percentage-based threshold."""
        tools = generate_mcp_github_tools() + generate_mcp_slack_tools()

        middleware = ToolSearchMiddleware(
            token_threshold=0.10,  # 10%
            context_window=128_000,
        )

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        middleware._initialize_index(tools, mock_model)

        # ~46 tools shouldn't exceed 10% of 128k context
        total_tokens = sum(_estimate_tool_tokens(t) for t in tools)
        threshold = 128_000 * 0.10

        expected_deferred = total_tokens > threshold
        assert middleware._deferred_mode == expected_deferred

    def test_absolute_token_threshold(self):
        """Should correctly use absolute token threshold."""
        tools = generate_mcp_github_tools()

        # Very low threshold to force deferred mode
        middleware = ToolSearchMiddleware(
            token_threshold=100,  # Absolute: 100 tokens
        )

        mock_model = MagicMock()
        middleware._initialize_index(tools, mock_model)

        # Should definitely be in deferred mode with 35 tools > 100 tokens
        assert middleware._deferred_mode is True

    def test_no_deferred_mode_with_few_tools(self):
        """Should not activate deferred mode with few tools."""
        tools = [
            StructuredTool.from_function(
                name="simple_tool",
                description="A simple tool",
                func=lambda: "done",
            )
        ]

        middleware = ToolSearchMiddleware(
            token_threshold=0.10,
            context_window=128_000,
        )

        mock_model = MagicMock()
        middleware._initialize_index(tools, mock_model)

        # Single tool shouldn't trigger deferred mode
        assert middleware._deferred_mode is False


class TestSearchAccuracy:
    """Tests for search accuracy across different query types."""

    def test_exact_name_match(self):
        """Should find exact tool name matches."""
        tools = generate_mcp_github_tools()
        index = ToolIndex()
        index.add_tools(tools)

        results = index.search("github_create_pr", mode="bm25", limit=5)

        assert len(results) > 0
        assert results[0][0] == "github_create_pr"

    def test_semantic_search(self):
        """Should find tools based on semantic meaning."""
        tools = generate_mcp_github_tools() + generate_mcp_slack_tools()
        index = ToolIndex()
        index.add_tools(tools)

        # Search with natural language, not exact tool names
        results = index.search("send a notification to team", mode="bm25", limit=5)

        # Should find slack_send_message as relevant
        tool_names = [r[0] for r in results]
        assert any("slack" in name or "message" in name for name in tool_names)

    def test_cross_domain_search(self):
        """Should correctly rank tools from relevant domains."""
        tools = (
            generate_mcp_github_tools()
            + generate_mcp_database_tools()
            + generate_mcp_filesystem_tools()
        )
        index = ToolIndex()
        index.add_tools(tools)

        # Query about files should prioritize filesystem tools
        results = index.search("read file contents", mode="bm25", limit=5)
        top_result = results[0][0]
        assert "fs_" in top_result or "file" in top_result.lower()

        # Query about code should prioritize github tools
        results = index.search("merge code changes", mode="bm25", limit=5)
        tool_names = [r[0] for r in results]
        assert any("github" in name.lower() or "merge" in name.lower() for name in tool_names)


class TestConcurrentUsage:
    """Tests for concurrent usage scenarios."""

    def test_concurrent_searches(self):
        """Should handle concurrent searches safely."""
        import concurrent.futures

        tools = generate_mcp_github_tools() + generate_mcp_slack_tools()
        index = ToolIndex()
        index.add_tools(tools)

        queries = [
            "create issue",
            "send message",
            "list repositories",
            "merge pull request",
            "upload file",
        ]

        def search_query(query: str) -> list:
            return index.search(query, mode="bm25", limit=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(20):  # 20 concurrent searches
                for query in queries:
                    futures.append(executor.submit(search_query, query))

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All searches should complete successfully
        assert len(results) == 100
        assert all(len(r) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_async_middleware_usage(self):
        """Should work correctly in async context."""
        tools = generate_mcp_github_tools()

        middleware = ToolSearchMiddleware(
            token_threshold=100,  # Force deferred mode
            always_include=["github_list_repos"],
        )

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        state: ToolSearchState = {"messages": []}
        runtime = MagicMock()

        # Test async before_agent
        result = await middleware.abefore_agent(state, runtime)

        assert result is not None
        assert "github_list_repos" in result["expanded_tools"]

        # Test async wrap_model_call
        request = MagicMock()
        request.tools = tools
        request.model = mock_model
        request.state = {"messages": [], "expanded_tools": set()}
        request.system_message = None
        request.override = MagicMock(return_value=request)

        async def async_handler(req):
            return MagicMock()

        await middleware.awrap_model_call(request, async_handler)

        assert middleware._tool_index is not None


class TestTokenEstimation:
    """Tests for token estimation accuracy."""

    def test_estimation_consistency(self):
        """Token estimates should be consistent for same tool."""
        tool = StructuredTool.from_function(
            name="test_tool",
            description="A test tool for testing token estimation",
            func=lambda x: x,
        )

        estimates = [_estimate_tool_tokens(tool) for _ in range(10)]

        # All estimates should be identical
        assert len(set(estimates)) == 1

    def test_estimation_scales_with_complexity(self):
        """More complex tools should have higher token estimates."""
        simple_tool = StructuredTool.from_function(
            name="simple",
            description="Simple",
            func=lambda: None,
        )

        class ComplexArgs(BaseModel):
            query: str = Field(description="The search query to execute")
            filters: str = Field(description="Additional filters to apply")
            limit: int = Field(description="Maximum number of results")
            offset: int = Field(description="Pagination offset")

        complex_tool = StructuredTool.from_function(
            name="complex_tool_with_long_name",
            description="A very complex tool that does many things including "
            "searching, filtering, sorting, and paginating results from multiple "
            "data sources with support for advanced query syntax.",
            func=lambda **kwargs: None,
            args_schema=ComplexArgs,
        )

        simple_tokens = _estimate_tool_tokens(simple_tool)
        complex_tokens = _estimate_tool_tokens(complex_tool)

        assert complex_tokens > simple_tokens


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""

    def test_agent_workflow_simulation(self):
        """Simulate a typical agent workflow with tool search."""
        # Setup: Multiple MCP servers worth of tools
        all_tools = (
            generate_mcp_github_tools()
            + generate_mcp_slack_tools()
            + generate_mcp_database_tools()
        )

        middleware = ToolSearchMiddleware(
            token_threshold=0.05,
            always_include=["fs_read_file", "fs_write_file"],
            search_mode="hybrid",
        )

        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4"

        # Initialize
        middleware._initialize_index(all_tools, mock_model)

        # Build tool name mapping
        for tool in all_tools:
            middleware._tool_name_to_tool[tool.name] = tool

        # Simulate workflow: Agent needs to create a GitHub PR
        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set(middleware.always_include)}
        runtime.tool_call_id = "call-001"

        # Step 1: Search for GitHub tools
        result = search_tool.func(query="create pull request github", runtime=runtime)

        assert isinstance(result, Command)
        assert "github_create_pr" in result.update["expanded_tools"]

        # Step 2: Update state and search for slack notification
        runtime.state = {"expanded_tools": result.update["expanded_tools"]}
        runtime.tool_call_id = "call-002"

        result2 = search_tool.func(query="send slack message notification", runtime=runtime)

        assert isinstance(result2, Command)
        assert "slack_send_message" in result2.update["expanded_tools"]
        # Previous tools should still be expanded
        assert "github_create_pr" in result2.update["expanded_tools"]

    def test_incremental_tool_expansion(self):
        """Test that tools are incrementally expanded across searches."""
        tools = generate_mcp_github_tools()[:10] + generate_mcp_slack_tools()[:5]

        middleware = ToolSearchMiddleware(token_threshold=100)

        for tool in tools:
            middleware._tool_name_to_tool[tool.name] = tool

        middleware._tool_index = ToolIndex()
        middleware._tool_index.add_tools(tools)

        search_tool = middleware.tools[0]

        runtime = MagicMock()
        runtime.state = {"expanded_tools": set()}
        runtime.tool_call_id = "call-1"

        # First search - use terms from tool descriptions that differentiate tools
        # Note: Searching for "github" alone returns empty because it appears in
        # all GitHub tool names, giving it zero IDF in BM25.
        result1 = search_tool.func(query="list repositories", runtime=runtime)
        assert isinstance(result1, Command), f"Expected Command, got {type(result1)}: {result1}"
        expanded_after_1 = result1.update["expanded_tools"]

        # Second search with accumulated state
        runtime.state = {"expanded_tools": expanded_after_1}
        runtime.tool_call_id = "call-2"

        result2 = search_tool.func(query="send message channel", runtime=runtime)
        assert isinstance(result2, Command), f"Expected Command, got {type(result2)}: {result2}"
        expanded_after_2 = result2.update["expanded_tools"]

        # Should have tools from both searches
        assert len(expanded_after_2) >= len(expanded_after_1)
        # Original github tools should still be there
        for tool in expanded_after_1:
            assert tool in expanded_after_2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
