import pytest
import asyncio
from typing import Any
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from deepagents_code.dta.indexer import HybridToolIndexer, ToolCandidate
from deepagents_code.dta.gating import ToolNamespaceRegistry, ToolNamespaceRouterNode
from deepagents_code.dta.selector import ToolSelectorNode, ToolSelectionResult
from deepagents_code.dta.gating import ActiveNamespacesResult
from deepagents_code.dta.middleware import DynamicToolAllocationMiddleware

# ---------------------------------------------------------
# Define the 10 Mock MCP Servers and their 15 realistic tools
# ---------------------------------------------------------

MOCK_SERVERS = {
    "git": [
        ("git_status", "Get the status of the git repository, showing modified and untracked files"),
        ("git_diff", "Show changes between commits, commit and working tree, etc"),
        ("git_commit", "Record changes to the repository with a commit message"),
        ("git_log", "Show commit logs for the repository"),
        ("git_show", "Show various types of objects (commits, tags, etc)"),
        ("git_add", "Add file contents to the index (stage files)"),
        ("git_push", "Update remote refs along with associated objects"),
        ("git_pull", "Fetch from and integrate with another repository or a local branch"),
        ("git_clone", "Clone a repository into a new directory"),
        ("git_branch", "List, create, or delete branches"),
        ("git_checkout", "Switch branches or restore working tree files"),
        ("git_merge", "Join two or more development histories together"),
        ("git_rebase", "Reapply commits on top of another base tip"),
        ("git_stash_push", "Save your local modifications to a new stash entry"),
        ("git_stash_pop", "Remove a single stashed state from the stash list and apply it"),
    ],
    "postgres": [
        ("postgres_query", "Execute a select query on the PostgreSQL database"),
        ("postgres_insert", "Insert rows into a PostgreSQL database table"),
        ("postgres_update", "Update rows in a PostgreSQL database table"),
        ("postgres_delete", "Delete rows from a PostgreSQL database table"),
        ("postgres_list_tables", "List all tables in the PostgreSQL database"),
        ("postgres_describe_table", "Describe the schema and columns of a PostgreSQL table"),
        ("postgres_create_table", "Create a new table in the PostgreSQL database"),
        ("postgres_drop_table", "Drop an existing table from the PostgreSQL database"),
        ("postgres_add_index", "Create an index on a PostgreSQL table column"),
        ("postgres_get_schemas", "List all database schemas in PostgreSQL"),
        ("postgres_backup", "Create a backup of the PostgreSQL database"),
        ("postgres_restore", "Restore a PostgreSQL database from a backup"),
        ("postgres_explain_query", "Explain the execution plan of a SQL query in PostgreSQL"),
        ("postgres_list_indexes", "List all indexes in the PostgreSQL database"),
        ("postgres_query_raw", "Execute a raw SQL query on the PostgreSQL database"),
    ],
    "aws": [
        ("aws_ec2_list", "List EC2 instances in the AWS account"),
        ("aws_ec2_start", "Start a stopped AWS EC2 instance"),
        ("aws_ec2_stop", "Stop a running AWS EC2 instance"),
        ("aws_s3_list_buckets", "List all S3 buckets in the AWS account"),
        ("aws_s3_upload", "Upload a file to an AWS S3 bucket"),
        ("aws_s3_download", "Download a file from an AWS S3 bucket"),
        ("aws_s3_delete", "Delete an object from an AWS S3 bucket"),
        ("aws_rds_list", "List RDS database instances in AWS"),
        ("aws_rds_start", "Start an AWS RDS database instance"),
        ("aws_rds_stop", "Stop an AWS RDS database instance"),
        ("aws_lambda_list", "List AWS Lambda functions"),
        ("aws_lambda_invoke", "Invoke an AWS Lambda function"),
        ("aws_iam_list_users", "List IAM users in the AWS account"),
        ("aws_dynamodb_list_tables", "List DynamoDB tables in AWS"),
        ("aws_cloudwatch_get_logs", "Retrieve logs from AWS CloudWatch"),
    ],
    "docker": [
        ("docker_ps", "List running Docker containers"),
        ("docker_images", "List local Docker images"),
        ("docker_run", "Run a command in a new Docker container"),
        ("docker_stop", "Stop a running Docker container"),
        ("docker_start", "Start a stopped Docker container"),
        ("docker_rm", "Remove a Docker container"),
        ("docker_rmi", "Remove a Docker image"),
        ("docker_logs", "Fetch logs of a Docker container"),
        ("docker_exec", "Run a command in a running Docker container"),
        ("docker_build", "Build a Docker image from a Dockerfile"),
        ("docker_volume_list", "List Docker volumes"),
        ("docker_network_list", "List Docker networks"),
        ("docker_inspect_container", "Return low-level information on a Docker container"),
        ("docker_stats", "Display a live stream of container resource usage statistics"),
        ("docker_compose_up", "Create and start containers using Docker Compose"),
    ],
    "filesystem": [
        ("fs_read_file", "Read the contents of a file from filesystem"),
        ("fs_write_file", "Write contents to a file on the filesystem"),
        ("fs_list_directory", "List contents of a directory on filesystem"),
        ("fs_create_directory", "Create a new directory on filesystem"),
        ("fs_delete_file", "Delete a file from the filesystem"),
        ("fs_delete_directory", "Delete a directory from the filesystem"),
        ("fs_search_files", "Search files matching a pattern"),
        ("fs_get_file_info", "Retrieve file metadata and statistics"),
        ("fs_copy_file", "Copy a file to another location"),
        ("fs_move_file", "Move a file to another location"),
        ("fs_chmod", "Change file permissions on filesystem"),
        ("fs_grep_search", "Grep for text inside files in a directory"),
        ("fs_find_by_extension", "Find files recursively by extension"),
        ("fs_read_binary_file", "Read raw binary contents of a file"),
        ("fs_get_disk_usage", "Retrieve disk space usage statistics"),
    ],
    "github": [
        ("github_create_issue", "Create a new issue on a GitHub repository"),
        ("github_get_issue", "Retrieve a specific GitHub issue"),
        ("github_list_issues", "List issues in a GitHub repository"),
        ("github_close_issue", "Close a GitHub issue"),
        ("github_create_pull_request", "Create a pull request on GitHub"),
        ("github_get_pull_request", "Retrieve details of a pull request"),
        ("github_list_pull_requests", "List pull requests in a GitHub repository"),
        ("github_merge_pull_request", "Merge a GitHub pull request"),
        ("github_create_repo", "Create a new GitHub repository"),
        ("github_search_repos", "Search GitHub repositories"),
        ("github_get_repo_contents", "Get repository directory contents or file"),
        ("github_add_collaborator", "Add a collaborator to a GitHub repository"),
        ("github_list_commits", "List commits on a GitHub branch"),
        ("github_get_user_info", "Get details of a GitHub user"),
        ("github_create_release", "Create a new release on GitHub"),
    ],
    "slack": [
        ("slack_post_message", "Post a message to a Slack channel"),
        ("slack_read_messages", "Read messages from a Slack channel"),
        ("slack_list_channels", "List available public Slack channels"),
        ("slack_create_channel", "Create a new Slack channel"),
        ("slack_invite_user", "Invite a user to a Slack channel"),
        ("slack_add_reaction", "Add a reaction to a Slack message"),
        ("slack_remove_reaction", "Remove a reaction from a Slack message"),
        ("slack_search_messages", "Search for messages matching query on Slack"),
        ("slack_get_users", "Retrieve list of all users in the workspace"),
        ("slack_get_user_presence", "Get active/away presence status of a user"),
        ("slack_update_status", "Update user profile status custom message"),
        ("slack_set_topic", "Set the topic of a Slack channel"),
        ("slack_archive_channel", "Archive an existing Slack channel"),
        ("slack_unarchive_channel", "Unarchive a previously archived Slack channel"),
        ("slack_list_groups", "List private Slack groups"),
    ],
    "sqlite": [
        ("sqlite_execute", "Execute a write statement (insert, update, delete) in SQLite"),
        ("sqlite_query", "Query rows using SQL select statement in SQLite"),
        ("sqlite_insert", "Insert records into SQLite table"),
        ("sqlite_update", "Update records in SQLite table"),
        ("sqlite_delete", "Delete records from SQLite table"),
        ("sqlite_list_tables", "List all tables in SQLite database file"),
        ("sqlite_describe_table", "Show schema and table structures for SQLite"),
        ("sqlite_create_table", "Create a new SQLite table"),
        ("sqlite_drop_table", "Delete SQLite table schema and data"),
        ("sqlite_backup", "Backup SQLite database to another file"),
        ("sqlite_get_schema", "Get the SQLite database schema"),
        ("sqlite_list_indexes", "List all active indexes in SQLite"),
        ("sqlite_add_index", "Create an index on SQLite table column"),
        ("sqlite_vacuum", "Compact the SQLite database file"),
        ("sqlite_import_csv", "Import data from a CSV file into SQLite"),
    ],
    "google_maps": [
        ("maps_search_places", "Search places using Google Maps API"),
        ("maps_get_place_details", "Retrieve details of a place via place ID"),
        ("maps_get_directions", "Get driving directions between locations"),
        ("maps_get_distance_matrix", "Retrieve travel distance matrix for locations"),
        ("maps_geocode", "Convert addresses into geographic coordinates"),
        ("maps_reverse_geocode", "Convert coordinates into human-readable addresses"),
        ("maps_elevation", "Get elevation data for coordinates"),
        ("maps_timezone", "Retrieve timezone offset for geographical coordinates"),
        ("maps_static_map_url", "Generate static map image URL"),
        ("maps_streetview_url", "Retrieve a streetview image URL"),
        ("maps_find_nearby", "Find places nearby a location coordinate"),
        ("maps_autocomplete", "Autocompletion suggestions for place searches"),
        ("maps_get_reviews", "Get user reviews of a place"),
        ("maps_get_local_businesses", "List local businesses registered nearby"),
        ("maps_validate_address", "Validate postal address components"),
    ],
    "browser": [
        ("browser_navigate", "Navigate browser page to the given URL"),
        ("browser_click", "Click element on browser page using selector"),
        ("browser_type", "Type text into selector element on browser page"),
        ("browser_press_key", "Press keyboard key on page element"),
        ("browser_screenshot", "Capture browser window page screenshot"),
        ("browser_get_html", "Get current DOM HTML of the browser page"),
        ("browser_get_text", "Get text content of browser page"),
        ("browser_scroll", "Scroll browser page down or up"),
        ("browser_back", "Go back in browser session history"),
        ("browser_forward", "Go forward in browser session history"),
        ("browser_refresh", "Reload current browser page"),
        ("browser_get_cookies", "Get current cookies for browser page"),
        ("browser_clear_cookies", "Clear all browser session cookies"),
        ("browser_execute_script", "Execute custom Javascript on browser page"),
        ("browser_get_active_element", "Retrieve details of active focused page element"),
    ]
}

class DummyTool:
    def __init__(self, name: str, description: str, metadata: dict[str, Any]):
        self.name = name
        self.description = description
        self.metadata = metadata

def build_all_tools() -> list[DummyTool]:
    tools = []
    for server_name, tool_list in MOCK_SERVERS.items():
        for name, desc in tool_list:
            tools.append(DummyTool(
                name=name,
                description=desc,
                metadata={"server_name": server_name}
            ))
    return tools

# ---------------------------------------------------------
# Mock LLM for structured output testing (Zero-Shot Router & Selector)
# ---------------------------------------------------------

class MockLLM:
    def __init__(self, structured_output_class: Any):
        self.structured_output_class = structured_output_class

    def with_structured_output(self, schema: Any) -> "MockLLM":
        return MockLLM(schema)

    def invoke(self, messages: list[Any], config: dict[str, Any] = None) -> Any:
        human_content = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                human_content = getattr(msg, "content", "").lower()
                break

        if self.structured_output_class.__name__ == "ActiveNamespacesResult":
            # Direct logic simulating how router should activate namespaces
            active = ["builtin"]
            if "git" in human_content or "commit" in human_content:
                active.append("mcp:git")
            if "database" in human_content or "sql" in human_content or "postgres" in human_content:
                active.append("mcp:postgres")
                active.append("mcp:sqlite")
            if "browser" in human_content or "click" in human_content or "navigate" in human_content:
                active.append("mcp:browser")
            if "file" in human_content or "directory" in human_content:
                active.append("mcp:filesystem")
            if "aws" in human_content or "ec2" in human_content or "s3" in human_content:
                active.append("mcp:aws")
            if "docker" in human_content or "container" in human_content:
                active.append("mcp:docker")
            if "slack" in human_content or "channel" in human_content:
                active.append("mcp:slack")
            if "github" in human_content or "pr" in human_content:
                active.append("mcp:github")
            if "map" in human_content or "places" in human_content:
                active.append("mcp:google_maps")

            return self.structured_output_class(active_namespaces=active)

        elif self.structured_output_class.__name__ == "ToolSelectionResult":
            # Selector logic to pick the best tools
            selected = []
            if "git" in human_content:
                selected = ["git_status", "git_commit", "git_diff"]
            elif "database" in human_content or "postgres" in human_content:
                selected = ["postgres_query", "postgres_insert"]
            elif "browser" in human_content:
                selected = ["browser_navigate", "browser_click"]
            else:
                selected = ["fs_read_file"]
                
            return self.structured_output_class(
                selected_tools=selected,
                rationale="Selected relevant tools matching query."
            )

    async def ainvoke(self, messages: list[Any], config: dict[str, Any] = None) -> Any:
        return self.invoke(messages, config)

# ---------------------------------------------------------
# Test Cases
# ---------------------------------------------------------

@pytest.fixture
def indexer() -> HybridToolIndexer:
    registry = ToolNamespaceRegistry()
    idx = HybridToolIndexer(registry=registry)
    # Sync all 150 tools into indexer
    all_tools = build_all_tools()
    idx.sync_tools(all_tools)
    return idx

@pytest.mark.asyncio
@patch("deepagents_code.dta.gating.create_model")
@patch("deepagents_code.dta.selector.create_model")
async def test_dta_e2e_stress(mock_selector_create_model: MagicMock, mock_gating_create_model: MagicMock, indexer: HybridToolIndexer) -> None:
    # 1. Setup mock models
    mock_gating_model_res = MagicMock()
    mock_gating_model_res.model = MockLLM(ActiveNamespacesResult)
    mock_gating_create_model.return_value = mock_gating_model_res

    mock_selector_model_res = MagicMock()
    mock_selector_model_res.model = MockLLM(ToolSelectionResult)
    mock_selector_create_model.return_value = mock_selector_model_res

    # Initialize middleware
    selector_node = ToolSelectorNode()
    router_node = ToolNamespaceRouterNode()
    middleware = DynamicToolAllocationMiddleware(
        indexer=indexer,
        selector_node=selector_node,
        router_node=router_node,
        max_tools_budget=10
    )

    class DummyRequest:
        def __init__(self, tools: list[Any], messages: list[Any]):
            self.tools = tools
            self.messages = messages
            self.state = {}
            
        def override(self, tools: list[Any]) -> "DummyRequest":
            self.tools = tools
            return self

    # Test Case A: Database task
    req_db = DummyRequest(
        tools=build_all_tools(),
        messages=[HumanMessage(content="Query the postgres database for all users.")]
    )
    
    res_db = await middleware._aprocess_request(req_db)
    tool_names_db = [t.name for t in res_db.tools]
    
    # We expect postgres_query to be selected
    assert "postgres_query" in tool_names_db
    # We expect unrelated tools (like git_commit or aws_ec2_list) NOT to be in the allocated toolset (budget limit)
    assert "aws_ec2_list" not in tool_names_db

    # Test Case B: Git task
    req_git = DummyRequest(
        tools=build_all_tools(),
        messages=[HumanMessage(content="Commit the current changes to git.")]
    )
    
    res_git = await middleware._aprocess_request(req_git)
    tool_names_git = [t.name for t in res_git.tools]
    
    assert "git_commit" in tool_names_git
    assert "postgres_query" not in tool_names_git

@pytest.mark.asyncio
@patch("deepagents_code.dta.gating.create_model")
async def test_gating_fast_path(mock_gating_create_model: MagicMock) -> None:
    """Verify the router bypasses LLM entirely when total namespaces <= 3."""
    router_node = ToolNamespaceRouterNode()
    
    active = await router_node.aroute(
        query="Run git status",
        available_namespaces={"mcp:git", "builtin"}
    )
    
    mock_gating_create_model.assert_not_called()
    assert "mcp:git" in active
    assert "builtin" in active


@pytest.mark.asyncio
@patch("deepagents_code.dta.gating.create_model")
async def test_gating_fail_open(mock_gating_create_model: MagicMock) -> None:
    """If the LLM throws an exception, gating must fail open to the full available set."""
    mock_gating_create_model.side_effect = RuntimeError("API key invalid")
    router_node = ToolNamespaceRouterNode()
    
    active = await router_node.aroute(
        query="Deploy container to docker",
        available_namespaces={"mcp:docker", "mcp:git", "mcp:aws", "builtin"}
    )
    
    assert active == {"mcp:docker", "mcp:git", "mcp:aws", "builtin"}


@pytest.mark.asyncio
@patch("deepagents_code.dta.gating.create_model")
@patch("deepagents_code.dta.selector.create_model")
async def test_multi_domain_task_switch(
    mock_selector_create_model: MagicMock,
    mock_gating_create_model: MagicMock,
    indexer: HybridToolIndexer,
) -> None:
    """Verify that a semantic domain switch triggers a cold reset, evicting stale tools."""
    mock_gating_model_res = MagicMock()
    mock_gating_model_res.model = MockLLM(ActiveNamespacesResult)
    mock_gating_create_model.return_value = mock_gating_model_res

    mock_selector_model_res = MagicMock()
    mock_selector_model_res.model = MockLLM(ToolSelectionResult)
    mock_selector_create_model.return_value = mock_selector_model_res

    selector_node = ToolSelectorNode()
    router_node = ToolNamespaceRouterNode()
    middleware = DynamicToolAllocationMiddleware(
        indexer=indexer,
        selector_node=selector_node,
        router_node=router_node,
        max_tools_budget=10
    )

    class DummyRequest:
        def __init__(self, tools: list[Any], messages: list[Any]):
            self.tools = tools
            self.messages = messages
            self.state = {}

        def override(self, tools: list[Any]) -> "DummyRequest":
            self.tools = tools
            return self

    # First turn: git task — set last query to "commit"
    req_git = DummyRequest(
        tools=build_all_tools(),
        messages=[HumanMessage(content="Commit changes to git.")]
    )
    res_git = await middleware._aprocess_request(req_git)
    assert "git_commit" in [t.name for t in res_git.tools]

    # Simulate the middleware remembering the prior query
    assert middleware._last_user_query == "Commit changes to git."

    # Second turn: completely different domain (browser) — should trigger Cold Reset
    req_browser = DummyRequest(
        tools=build_all_tools(),
        messages=[HumanMessage(content="Navigate browser to the login page and click the submit button.")]
    )
    res_browser = await middleware._aprocess_request(req_browser)
    browser_tool_names = [t.name for t in res_browser.tools]

    # browser_navigate should appear
    assert "browser_navigate" in browser_tool_names
    # git_commit should NOT persist from the prior turn (cold reset evicts history)
    assert "git_commit" not in browser_tool_names


@pytest.mark.asyncio
@patch("deepagents_code.dta.gating.create_model")
@patch("deepagents_code.dta.selector.create_model")
async def test_temporal_tool_continuation(
    mock_selector_create_model: MagicMock,
    mock_gating_create_model: MagicMock,
    indexer: HybridToolIndexer,
) -> None:
    """Tools used in the recent conversation history are preserved across turns (temporal boost)."""
    mock_gating_model_res = MagicMock()
    mock_gating_model_res.model = MockLLM(ActiveNamespacesResult)
    mock_gating_create_model.return_value = mock_gating_model_res

    mock_selector_model_res = MagicMock()
    mock_selector_model_res.model = MockLLM(ToolSelectionResult)
    mock_selector_create_model.return_value = mock_selector_model_res

    selector_node = ToolSelectorNode()
    router_node = ToolNamespaceRouterNode()
    middleware = DynamicToolAllocationMiddleware(
        indexer=indexer,
        selector_node=selector_node,
        router_node=router_node,
        max_tools_budget=10
    )

    class DummyRequest:
        def __init__(self, tools: list[Any], messages: list[Any]):
            self.tools = tools
            self.messages = messages
            self.state = {}

        def override(self, tools: list[Any]) -> "DummyRequest":
            self.tools = tools
            return self

    # Simulate a tool output turn — the AI just used postgres_query
    messages = [
        HumanMessage(content="Check the database for all users."),
        AIMessage(content="", tool_calls=[{"name": "postgres_query", "args": {}, "id": "tc1"}]),
    ]
    req = DummyRequest(tools=build_all_tools(), messages=messages)
    res = await middleware._aprocess_request(req)
    
    tool_names = [t.name for t in res.tools]
    # postgres_query should remain available as a temporal tool even if the allocator
    # would have filtered it away (temporal continuation)
    assert "postgres_query" in tool_names


@pytest.mark.asyncio
@patch("deepagents_code.dta.gating.create_model")
@patch("deepagents_code.dta.selector.create_model")
async def test_smart_cache_on_tool_output(
    mock_selector_create_model: MagicMock,
    mock_gating_create_model: MagicMock,
    indexer: HybridToolIndexer,
) -> None:
    """Verify the smart cache returns the previous toolset on tool-output turns without retrieval."""
    mock_gating_model_res = MagicMock()
    mock_gating_model_res.model = MockLLM(ActiveNamespacesResult)
    mock_gating_create_model.return_value = mock_gating_model_res

    mock_selector_model_res = MagicMock()
    mock_selector_model_res.model = MockLLM(ToolSelectionResult)
    mock_selector_create_model.return_value = mock_selector_model_res

    selector_node = ToolSelectorNode()
    router_node = ToolNamespaceRouterNode()
    middleware = DynamicToolAllocationMiddleware(
        indexer=indexer,
        selector_node=selector_node,
        router_node=router_node,
        max_tools_budget=10
    )

    class DummyTool:
        def __init__(self, name: str, description: str = ""):
            self.name = name
            self.description = description
            self.metadata: dict[str, Any] = {}

    class DummyToolMessage:
        type = "tool"
        name = "git_status"

    class DummyRequest:
        def __init__(self, tools: list[Any], messages: list[Any]):
            self.tools = tools
            self.messages = messages
            self.state = {}

        def override(self, tools: list[Any]) -> "DummyRequest":
            self.tools = tools
            return self

    # First turn — prime the cache
    req1 = DummyRequest(
        tools=build_all_tools(),
        messages=[HumanMessage(content="Check git status.")]
    )
    res1 = await middleware._aprocess_request(req1)
    cached = middleware._cached_toolset.copy()
    assert len(cached) > 0

    # Second turn — tool output message, should hit cache
    req2 = DummyRequest(
        tools=build_all_tools(),
        messages=[
            HumanMessage(content="Check git status."),
            DummyToolMessage(),  # type: ignore[list-item]
        ]
    )
    initial_call_count = mock_gating_create_model.call_count
    res2 = await middleware._aprocess_request(req2)

    # Router should NOT have been called again (cache hit)
    assert mock_gating_create_model.call_count == initial_call_count
    # The toolset should be unchanged
    assert {t.name for t in res2.tools} == {t.name for t in res1.tools}


def test_tool_deduplication(indexer: HybridToolIndexer) -> None:
    """Adding the same tool twice (by description+schema) should not duplicate it in the index."""
    before_count = len(indexer.tools)
    # Re-add a tool that is already indexed by sync_tools
    dupe = DummyTool(
        name="git_commit",
        description="Record changes to the repository with a commit message",
        metadata={"server_name": "git"}
    )
    indexer.sync_tools([dupe])
    after_count = len(indexer.tools)
    assert after_count == before_count, "Deduplication failed: tool count changed after re-adding an identical tool"


def test_indexer_namespace_gating_150_tools(indexer: HybridToolIndexer) -> None:
    """With 150 tools loaded, namespace gating should eliminate > 80% of candidates."""
    # Only allow the git namespace
    results = indexer.search("commit changes to repository", namespaces={"mcp:git"}, top_k=50)
    names = {r["name"] for r in results}

    # All returned tools must belong to the git server
    assert all(n.startswith("git_") for n in names), f"Non-git tool leaked through: {names - {n for n in names if n.startswith('git_')}}"
    # We should get at most 15 tools (the full git server) and no more
    assert len(results) <= 15


def test_selector_budget_cap() -> None:
    """The selector must never exceed the budget even if the LLM returns more names."""
    selector = ToolSelectorNode()
    # Provide 20 candidates, budget=5
    candidates = [{"name": f"tool_{i}", "description": f"Tool {i}", "parameters": {}} for i in range(20)]
    selected = selector.select(messages=[], candidates=candidates, budget=5)
    assert len(selected) <= 5


def test_namespace_classification_mcp_metadata() -> None:
    """Tools with server_name metadata must be classified to mcp:<server> namespace."""
    registry = ToolNamespaceRegistry()

    class MCPTool:
        name = "browse_page"
        metadata = {"server_name": "puppeteer"}

    ns = registry.classify_tool(MCPTool())
    assert ns == "mcp:puppeteer"


def test_namespace_classification_fallback_builtin() -> None:
    """Tools with no recognizable prefix or metadata must fall back to 'builtin'."""
    registry = ToolNamespaceRegistry()
    ns = registry.classify_tool({"name": "read_file"})
    assert ns == "builtin"

