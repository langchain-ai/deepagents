import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Overwrite

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.utils import (
    TRUNCATION_GUIDANCE,
    create_file_data,
    format_content_with_line_numbers,
    format_read_response,
    sanitize_tool_call_id,
    truncate_if_too_long,
    update_file_data,
)
from deepagents.middleware.filesystem import (
    FileData,
    FilesystemMiddleware,
    FilesystemState,
    _create_content_preview,
    _supports_execution,
)
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgentMiddleware


def build_composite_state_backend(runtime: ToolRuntime, *, routes):
    built_routes = {}
    for prefix, backend_or_factory in routes.items():
        if callable(backend_or_factory):
            built_routes[prefix] = backend_or_factory(runtime)
        else:
            built_routes[prefix] = backend_or_factory
    default_state = StateBackend(runtime)
    return CompositeBackend(default=default_state, routes=built_routes)


class TestAddMiddleware:
    def test_filesystem_middleware(self):
        middleware = [FilesystemMiddleware()]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools
        assert "edit_file" in agent_tools
        assert "glob" in agent_tools
        assert "grep" in agent_tools

    def test_subagent_middleware(self):
        middleware = [
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[{**GENERAL_PURPOSE_SUBAGENT, "model": "claude-sonnet-4-20250514", "tools": []}],
            )
        ]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "task" in agent.nodes["tools"].bound._tools_by_name

    def test_multiple_middleware(self):
        middleware = [
            FilesystemMiddleware(),
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[{**GENERAL_PURPOSE_SUBAGENT, "model": "claude-sonnet-4-20250514", "tools": []}],
            ),
        ]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools
        assert "edit_file" in agent_tools
        assert "glob" in agent_tools
        assert "grep" in agent_tools
        assert "task" in agent_tools


class TestFilesystemMiddleware:
    def test_init_default(self):
        middleware = FilesystemMiddleware()
        assert callable(middleware.backend)
        assert middleware._custom_system_prompt is None
        assert len(middleware.tools) == 7  # All tools including execute

    def test_init_with_composite_backend(self):
        def backend_factory(rt):
            return build_composite_state_backend(rt, routes={"/memories/": StoreBackend})

        middleware = FilesystemMiddleware(backend=backend_factory)
        assert callable(middleware.backend)
        assert middleware._custom_system_prompt is None
        assert len(middleware.tools) == 7  # All tools including execute

    def test_init_custom_system_prompt_default(self):
        middleware = FilesystemMiddleware(system_prompt="Custom system prompt")
        assert callable(middleware.backend)
        assert middleware._custom_system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 7  # All tools including execute

    def test_init_custom_system_prompt_with_composite(self):
        def backend_factory(rt):
            return build_composite_state_backend(rt, routes={"/memories/": StoreBackend})

        middleware = FilesystemMiddleware(backend=backend_factory, system_prompt="Custom system prompt")
        assert callable(middleware.backend)
        assert middleware._custom_system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 7  # All tools including execute

    def test_init_custom_tool_descriptions_default(self):
        middleware = FilesystemMiddleware(custom_tool_descriptions={"ls": "Custom ls tool description"})
        assert callable(middleware.backend)
        assert middleware._custom_system_prompt is None
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_init_custom_tool_descriptions_with_composite(self):
        def backend_factory(rt):
            return build_composite_state_backend(rt, routes={"/memories/": StoreBackend})

        middleware = FilesystemMiddleware(backend=backend_factory, custom_tool_descriptions={"ls": "Custom ls tool description"})
        assert callable(middleware.backend)
        assert middleware._custom_system_prompt is None
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_ls_shortterm(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/test2.txt": FileData(
                    content=["Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = ls_tool.invoke(
            {"runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}), "path": "/"}
        )
        assert result == str(["/test.txt", "/test2.txt"])

    def test_ls_shortterm_with_path(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/test2.txt": FileData(
                    content=["Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/charmander.txt": FileData(
                    content=["Ember"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/water/squirtle.txt": FileData(
                    content=["Water"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result_raw = ls_tool.invoke(
            {
                "path": "/pokemon/",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        result = result_raw
        # ls should only return files directly in /pokemon/, not in subdirectories
        assert "/pokemon/test2.txt" in result
        assert "/pokemon/charmander.txt" in result
        assert "/pokemon/water/squirtle.txt" not in result  # In subdirectory, should NOT be listed
        # ls should also list subdirectories with trailing /
        assert "/pokemon/water/" in result

    def test_ls_shortterm_lists_directories(self):
        """Test that ls lists directories with trailing / for traversal."""
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/charmander.txt": FileData(
                    content=["Ember"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/water/squirtle.txt": FileData(
                    content=["Water"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/docs/readme.md": FileData(
                    content=["Documentation"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result_raw = ls_tool.invoke(
            {
                "path": "/",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        result = result_raw
        # ls should list both files and directories at root level
        assert "/test.txt" in result
        assert "/pokemon/" in result
        assert "/docs/" in result
        # But NOT subdirectory files
        assert "/pokemon/charmander.txt" not in result
        assert "/pokemon/water/squirtle.txt" not in result

    def test_glob_search_shortterm_simple_pattern(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/test.py": FileData(
                    content=["print('hello')"],
                    modified_at="2021-01-02",
                    created_at="2021-01-01",
                ),
                "/pokemon/charmander.py": FileData(
                    content=["Ember"],
                    modified_at="2021-01-03",
                    created_at="2021-01-01",
                ),
                "/pokemon/squirtle.txt": FileData(
                    content=["Water"],
                    modified_at="2021-01-04",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        result = result_raw
        # Standard glob: *.py only matches files in root directory, not subdirectories
        assert result == str(["/test.py"])

    def test_glob_search_shortterm_wildcard_pattern(self):
        state = FilesystemState(
            messages=[],
            files={
                "/src/main.py": FileData(
                    content=["main code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/src/utils/helper.py": FileData(
                    content=["helper code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/tests/test_main.py": FileData(
                    content=["test code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "**/*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        result = result_raw
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" in result
        assert "/tests/test_main.py" in result

    def test_glob_search_shortterm_with_path(self):
        state = FilesystemState(
            messages=[],
            files={
                "/src/main.py": FileData(
                    content=["main code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/src/utils/helper.py": FileData(
                    content=["helper code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/tests/test_main.py": FileData(
                    content=["test code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "path": "/src",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        result = result_raw
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" not in result
        assert "/tests/test_main.py" not in result

    def test_glob_search_shortterm_brace_expansion(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["code"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/test.pyi": FileData(
                    content=["stubs"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/test.txt": FileData(
                    content=["text"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.{py,pyi}",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        result = result_raw
        assert "/test.py" in result
        assert "/test.pyi" in result
        assert "/test.txt" not in result

    def test_glob_search_shortterm_no_matches(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert result == str([])

    def test_glob_search_truncates_large_results(self):
        """Test that glob results are truncated when they exceed token limit."""
        # Create a large number of files that will exceed TOOL_RESULT_TOKEN_LIMIT
        # TOOL_RESULT_TOKEN_LIMIT = 20000, * 4 chars/token = 80000 chars
        # Create files with long paths to exceed this limit
        files = {}
        # Create 2000 files with 50-char paths = 100,000 chars total (exceeds 80k limit)
        for i in range(2000):
            path = f"/very_long_file_name_to_increase_size_{i:04d}.txt"
            files[path] = FileData(
                content=["content"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            )

        state = FilesystemState(messages=[], files=files)
        middleware = FilesystemMiddleware()
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.txt",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )

        # Result should be truncated
        result = result_raw
        assert isinstance(result, str)
        assert len(result.split(", ")) < 2000  # Should be truncated to fewer files
        # Last element should be the truncation message
        # Need to do the :-2 to account for the wrapping list characters
        assert result[:-2].endswith(TRUNCATION_GUIDANCE)

    def test_grep_search_shortterm_files_with_matches(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["import os", "import sys", "print('hello')"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/main.py": FileData(
                    content=["def main():", "    pass"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/helper.txt": FileData(
                    content=["import json"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py" in result
        assert "/helper.txt" in result
        assert "/main.py" not in result

    def test_grep_search_shortterm_content_mode(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["import os", "import sys", "print('hello')"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "output_mode": "content",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "1: import os" in result
        assert "2: import sys" in result
        assert "print" not in result

    def test_grep_search_shortterm_count_mode(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["import os", "import sys", "print('hello')"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/main.py": FileData(
                    content=["import json", "data = {}"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "output_mode": "count",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py:2" in result or "/test.py: 2" in result
        assert "/main.py:1" in result or "/main.py: 1" in result

    def test_grep_search_shortterm_with_include(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["import os"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/test.txt": FileData(
                    content=["import nothing"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "glob": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py" in result
        assert "/test.txt" not in result

    def test_grep_search_shortterm_with_path(self):
        state = FilesystemState(
            messages=[],
            files={
                "/src/main.py": FileData(
                    content=["import os"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/tests/test.py": FileData(
                    content=["import pytest"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "path": "/src",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/src/main.py" in result
        assert "/tests/test.py" not in result

    def test_grep_search_shortterm_regex_pattern(self):
        """Test grep with literal pattern (not regex)."""
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["def hello():", "def world():", "x = 5"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Search for literal "def " - literal search, not regex
        result = grep_search_tool.invoke(
            {
                "pattern": "def ",
                "output_mode": "content",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "1: def hello():" in result
        assert "2: def world():" in result
        assert "x = 5" not in result

    def test_grep_search_shortterm_no_matches(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["print('hello')"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert result == "No matches found"

    def test_grep_search_shortterm_invalid_regex(self):
        """Test grep with special characters (literal search, not regex)."""
        state = FilesystemState(
            messages=[],
            files={
                "/test.py": FileData(
                    content=["print('hello')"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Special characters are treated literally, so no matches expected
        result = grep_search_tool.invoke(
            {
                "pattern": "[invalid",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "No matches found" in result

    def test_search_store_paginated_empty(self):
        """Test pagination with no items."""
        store = InMemoryStore()
        result = StoreBackend._search_store_paginated(self, store, ("filesystem",))
        assert result == []

    def test_search_store_paginated_less_than_page_size(self):
        """Test pagination with fewer items than page size."""
        store = InMemoryStore()
        for i in range(5):
            store.put(
                ("filesystem",),
                f"/file{i}.txt",
                {
                    "content": [f"content {i}"],
                    "created_at": "2021-01-01",
                    "modified_at": "2021-01-01",
                },
            )

        result = StoreBackend._search_store_paginated(self, store, ("filesystem",), page_size=10)
        assert len(result) == 5
        # Check that all files are present (order may vary)
        keys = {item.key for item in result}
        assert keys == {f"/file{i}.txt" for i in range(5)}

    def test_search_store_paginated_exact_page_size(self):
        """Test pagination with exactly one page of items."""
        store = InMemoryStore()
        for i in range(10):
            store.put(
                ("filesystem",),
                f"/file{i}.txt",
                {
                    "content": [f"content {i}"],
                    "created_at": "2021-01-01",
                    "modified_at": "2021-01-01",
                },
            )

        result = StoreBackend._search_store_paginated(self, store, ("filesystem",), page_size=10)
        assert len(result) == 10
        keys = {item.key for item in result}
        assert keys == {f"/file{i}.txt" for i in range(10)}

    def test_search_store_paginated_multiple_pages(self):
        """Test pagination with multiple pages of items."""
        store = InMemoryStore()
        for i in range(250):
            store.put(
                ("filesystem",),
                f"/file{i}.txt",
                {
                    "content": [f"content {i}"],
                    "created_at": "2021-01-01",
                    "modified_at": "2021-01-01",
                },
            )

        result = StoreBackend._search_store_paginated(self, store, ("filesystem",), page_size=100)
        assert len(result) == 250
        keys = {item.key for item in result}
        assert keys == {f"/file{i}.txt" for i in range(250)}

    def test_search_store_paginated_with_filter(self):
        """Test pagination with filter parameter."""
        store = InMemoryStore()
        for i in range(20):
            store.put(
                ("filesystem",),
                f"/file{i}.txt",
                {
                    "content": [f"content {i}"],
                    "created_at": "2021-01-01",
                    "modified_at": "2021-01-01",
                    "type": "test" if i % 2 == 0 else "other",
                },
            )

        # Filter for type="test" (every other item, so 10 items)
        result = StoreBackend._search_store_paginated(self, store, ("filesystem",), filter={"type": "test"}, page_size=5)
        assert len(result) == 10
        # Verify all returned items have type="test"
        for item in result:
            assert item.value.get("type") == "test"

    def test_search_store_paginated_custom_page_size(self):
        """Test pagination with custom page size."""
        store = InMemoryStore()
        # Add 55 items
        for i in range(55):
            store.put(
                ("filesystem",),
                f"/file{i}.txt",
                {
                    "content": [f"content {i}"],
                    "created_at": "2021-01-01",
                    "modified_at": "2021-01-01",
                },
            )

        result = StoreBackend._search_store_paginated(self, store, ("filesystem",), page_size=20)
        # Should make 3 calls: 20, 20, 15
        assert len(result) == 55
        keys = {item.key for item in result}
        assert keys == {f"/file{i}.txt" for i in range(55)}

    def test_create_file_data_preserves_long_lines(self):
        """Test that create_file_data stores long lines as-is without splitting."""
        long_line = "a" * 3500
        short_line = "short line"
        content = f"{short_line}\n{long_line}"

        file_data = create_file_data(content)

        assert len(file_data["content"]) == 2
        assert file_data["content"][0] == short_line
        assert file_data["content"][1] == long_line
        assert len(file_data["content"][1]) == 3500

    def test_update_file_data_preserves_long_lines(self):
        """Test that update_file_data stores long lines as-is without splitting."""
        initial_file_data = create_file_data("initial content")

        long_line = "b" * 5000
        short_line = "another short line"
        new_content = f"{short_line}\n{long_line}"

        updated_file_data = update_file_data(initial_file_data, new_content)

        assert len(updated_file_data["content"]) == 2
        assert updated_file_data["content"][0] == short_line
        assert updated_file_data["content"][1] == long_line
        assert len(updated_file_data["content"][1]) == 5000

        assert updated_file_data["created_at"] == initial_file_data["created_at"]

    def test_format_content_with_line_numbers_short_lines(self):
        """Test that short lines (<=10000 chars) are displayed normally."""
        content = ["short line 1", "short line 2", "short line 3"]
        result = format_content_with_line_numbers(content, start_line=1)

        lines = result.split("\n")
        assert len(lines) == 3
        assert "     1\tshort line 1" in lines[0]
        assert "     2\tshort line 2" in lines[1]
        assert "     3\tshort line 3" in lines[2]

    def test_format_content_with_line_numbers_long_line_with_continuation(self):
        """Test that long lines (>5000 chars) are split with continuation markers."""
        long_line = "a" * 25000
        content = ["short line", long_line, "another short line"]
        result = format_content_with_line_numbers(content, start_line=1)

        lines = result.split("\n")
        assert len(lines) == 7  # 1 short + 5 continuation (2, 2.1, 2.2, 2.3, 2.4) + 1 short
        assert "     1\tshort line" in lines[0]
        assert "     2\t" in lines[1]
        assert lines[1].count("a") == 5000
        assert "   2.1\t" in lines[2]
        assert lines[2].count("a") == 5000
        assert "   2.2\t" in lines[3]
        assert lines[3].count("a") == 5000
        assert "   2.3\t" in lines[4]
        assert lines[4].count("a") == 5000
        assert "   2.4\t" in lines[5]
        assert lines[5].count("a") == 5000
        assert "     3\tanother short line" in lines[6]

    def test_format_content_with_line_numbers_multiple_long_lines(self):
        """Test multiple long lines in sequence with proper line numbering."""
        long_line_1 = "x" * 15000
        long_line_2 = "y" * 15000
        content = [long_line_1, "middle", long_line_2]
        result = format_content_with_line_numbers(content, start_line=5)
        lines = result.split("\n")
        assert len(lines) == 7  # 3 (line 5, 5.1, 5.2) + 1 middle + 3 (line 7, 7.1, 7.2)
        assert "     5\t" in lines[0]
        assert lines[0].count("x") == 5000
        assert "   5.1\t" in lines[1]
        assert lines[1].count("x") == 5000
        assert "   5.2\t" in lines[2]
        assert lines[2].count("x") == 5000
        assert "     6\tmiddle" in lines[3]
        assert "     7\t" in lines[4]
        assert lines[4].count("y") == 5000
        assert "   7.1\t" in lines[5]
        assert lines[5].count("y") == 5000
        assert "   7.2\t" in lines[6]
        assert lines[6].count("y") == 5000

    def test_format_content_with_line_numbers_exact_limit(self):
        """Test that a line exactly at the 5000 char limit is not split."""
        exact_line = "b" * 5000
        content = [exact_line]
        result = format_content_with_line_numbers(content, start_line=1)

        lines = result.split("\n")
        assert len(lines) == 1
        assert "     1\t" in lines[0]
        assert lines[0].count("b") == 5000

    def test_read_file_with_long_lines_shows_continuation_markers(self):
        """Test that read_file displays long lines with continuation markers."""
        long_line = "z" * 15000
        content = f"first line\n{long_line}\nthird line"
        file_data = create_file_data(content)
        result = format_read_response(file_data, offset=0, limit=100)
        lines = result.split("\n")
        assert len(lines) == 5  # 1 first + 3 continuation (2, 2.1, 2.2) + 1 third
        assert "     1\tfirst line" in lines[0]
        assert "     2\t" in lines[1]
        assert lines[1].count("z") == 5000
        assert "   2.1\t" in lines[2]
        assert lines[2].count("z") == 5000
        assert "   2.2\t" in lines[3]
        assert lines[3].count("z") == 5000
        assert "     3\tthird line" in lines[4]

    def test_read_file_with_offset_and_long_lines(self):
        """Test that read_file with offset handles long lines correctly."""
        long_line = "m" * 12000
        content = f"line1\nline2\n{long_line}\nline4"
        file_data = create_file_data(content)
        result = format_read_response(file_data, offset=2, limit=10)
        lines = result.split("\n")
        assert len(lines) == 4  # 3 continuation (3, 3.1, 3.2) + 1 line4
        assert "     3\t" in lines[0]
        assert lines[0].count("m") == 5000
        assert "   3.1\t" in lines[1]
        assert lines[1].count("m") == 5000
        assert "   3.2\t" in lines[2]
        assert lines[2].count("m") == 2000
        assert "     4\tline4" in lines[3]

    def test_intercept_short_toolmessage(self):
        """Test that small ToolMessages pass through unchanged."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        small_content = "x" * 1000
        tool_message = ToolMessage(content=small_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert result == tool_message

    def test_intercept_long_toolmessage(self):
        """Test that large ToolMessages are intercepted and saved to filesystem."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        assert "/large_tool_results/test_123" in result.update["files"]
        assert "Tool result too large" in result.update["messages"][0].content

    def test_intercept_long_toolmessage_preserves_name(self):
        """Test that ToolMessage name is preserved after eviction."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123", name="example_tool")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        assert result.update["messages"][0].name == "example_tool"

    def test_intercept_command_with_short_toolmessage(self):
        """Test that Commands with small messages pass through unchanged."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        small_content = "x" * 1000
        tool_message = ToolMessage(content=small_content, tool_call_id="test_123")
        command = Command(update={"messages": [tool_message], "files": {}})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == small_content

    def test_intercept_command_with_long_toolmessage(self):
        """Test that Commands with large messages are intercepted."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        large_content = "y" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        command = Command(update={"messages": [tool_message], "files": {}})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        assert "/large_tool_results/test_123" in result.update["files"]
        assert "Tool result too large" in result.update["messages"][0].content

    def test_intercept_command_with_files_and_long_toolmessage(self):
        """Test that file updates are properly merged with existing files and other keys preserved."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        large_content = "z" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        existing_file = FileData(content=["existing"], created_at="2021-01-01", modified_at="2021-01-01")
        command = Command(update={"messages": [tool_message], "files": {"/existing.txt": existing_file}, "custom_key": "custom_value"})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        assert "/existing.txt" in result.update["files"]
        assert "/large_tool_results/test_123" in result.update["files"]
        assert result.update["custom_key"] == "custom_value"

    def test_sanitize_tool_call_id(self):
        """Test that tool_call_id is sanitized to prevent path traversal."""
        assert sanitize_tool_call_id("call_123") == "call_123"
        assert sanitize_tool_call_id("call/123") == "call_123"
        assert sanitize_tool_call_id("test.id") == "test_id"

    def test_intercept_sanitizes_tool_call_id(self):
        """Test that tool_call_id with dangerous characters is sanitized in file path."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test/call.id")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        assert "/large_tool_results/test_call_id" in result.update["files"]

    def test_intercept_content_block_with_large_text(self):
        """Test that content blocks with large text get evicted and converted to string."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_cb", store=None, stream_writer=lambda _: None, config={})

        # Create list with content block with large text
        content_blocks = [{"type": "text", "text": "x" * 5000}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_cb")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        assert "/large_tool_results/test_cb" in result.update["files"]
        # After eviction, content is always converted to plain string
        returned_content = result.update["messages"][0].content
        assert isinstance(returned_content, str)
        assert "Tool result too large" in returned_content

    def test_intercept_content_block_with_small_text(self):
        """Test that content blocks with small text are not evicted."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_small_cb", store=None, stream_writer=lambda _: None, config={})

        # Create list with content block with small text
        content_blocks = [{"type": "text", "text": "small text"}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_small_cb")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        # Should return original message unchanged
        assert result == tool_message
        assert result.content == content_blocks

    def test_intercept_content_block_non_text_type(self):
        """Test that content blocks with non-text type get evicted if large when stringified."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_other", store=None, stream_writer=lambda _: None, config={})

        # Create list with content block with different type that's large when stringified
        content_blocks = [{"type": "image", "data": "x" * 5000}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_other")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        # All content types are evicted if large when converted to string
        assert isinstance(result, Command)
        assert "/large_tool_results/test_other" in result.update["files"]

    def test_intercept_list_content_gets_evicted_if_large(self):
        """Test that list content gets evicted if large when stringified."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_list", store=None, stream_writer=lambda _: None, config={})

        # Create list content that's large when stringified
        list_content = [{"key": "x" * 1000} for _ in range(50)]
        tool_message = ToolMessage(content=list_content, tool_call_id="test_list")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        # List content is evicted if large when converted to string
        assert isinstance(result, Command)
        assert "/large_tool_results/test_list" in result.update["files"]

    def test_single_text_block_extracts_text_directly(self):
        """Test that single text block extracts text content directly, not stringified structure."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_single", store=None, stream_writer=lambda _: None, config={})

        # Create single text block with large text
        content_blocks = [{"type": "text", "text": "Hello world! " * 1000}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_single")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        # Check that the file contains actual text, not stringified dict
        file_content = result.update["files"]["/large_tool_results/test_single"]["content"]
        file_text = "\n".join(file_content)
        # Should start with the actual text, not with "[{" which would indicate stringified dict
        assert file_text.startswith("Hello world!")
        assert not file_text.startswith("[{")

    def test_multiple_text_blocks_stringifies_structure(self):
        """Test that multiple text blocks stringify entire structure."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_multi", store=None, stream_writer=lambda _: None, config={})

        # Create multiple text blocks
        content_blocks = [
            {"type": "text", "text": "First block " * 500},
            {"type": "text", "text": "Second block " * 500},
        ]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_multi")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        # Check that the file contains stringified structure (starts with "[")
        file_content = result.update["files"]["/large_tool_results/test_multi"]["content"]
        file_text = "\n".join(file_content)
        # Should be stringified list of dicts
        assert file_text.startswith("[{")

    def test_mixed_content_blocks_stringifies_all(self):
        """Test that mixed content block types (text + image) stringify entire structure."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_mixed", store=None, stream_writer=lambda _: None, config={})

        # Create mixed content blocks
        content_blocks = [
            {"type": "text", "text": "Some text " * 200},
            {"type": "image", "url": "https://example.com/image.png"},
        ]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_mixed")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        # Check that the file contains stringified structure
        file_content = result.update["files"]["/large_tool_results/test_mixed"]["content"]
        file_text = "\n".join(file_content)
        assert file_text.startswith("[{")
        # Should contain both blocks in the stringified output
        assert "'type': 'text'" in file_text
        assert "'type': 'image'" in file_text

    def test_read_file_image_returns_standard_image_content_block(self):
        """Test image reads return standard image blocks with base64 + mime_type."""

        class ImageBackend(StateBackend):
            def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
                return [
                    FileDownloadResponse(
                        path=paths[0],
                        content=b"fake-image-bytes",
                        error=None,
                    )
                ]

        middleware = FilesystemMiddleware(backend=lambda rt: ImageBackend(rt))  # noqa: PLW0108
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="img-read-1",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = read_file_tool.invoke({"file_path": "/app/frame_001.jpg", "runtime": runtime})

        assert isinstance(result, ToolMessage)
        assert result.name == "read_file"
        assert result.tool_call_id == "img-read-1"
        assert result.additional_kwargs["read_file_path"] == "/app/frame_001.jpg"
        assert result.additional_kwargs["read_file_media_type"] == "image/jpeg"
        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "image"
        assert result.content[0]["mime_type"] == "image/jpeg"
        assert result.content[0]["base64"] == "ZmFrZS1pbWFnZS1ieXRlcw=="

    def test_read_file_image_returns_error_when_download_fails(self):
        """Image reads should return a clear backend error string."""

        class ImageBackend(StateBackend):
            def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
                return [FileDownloadResponse(path=paths[0], content=None, error="file_not_found")]

        middleware = FilesystemMiddleware(backend=lambda rt: ImageBackend(rt))  # noqa: PLW0108
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="img-read-err",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = read_file_tool.invoke({"file_path": "/app/missing.png", "runtime": runtime})

        assert isinstance(result, str)
        assert result == "Error reading image: file_not_found"

    def test_execute_tool_returns_error_when_backend_doesnt_support(self):
        """Test that execute tool returns friendly error instead of raising exception."""
        state = FilesystemState(messages=[], files={})
        middleware = FilesystemMiddleware()  # Default StateBackend doesn't support execution

        # Find the execute tool
        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")

        # Create runtime with StateBackend
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_exec",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        # Execute should return error message, not raise exception
        result = execute_tool.invoke({"command": "ls -la", "runtime": runtime})

        assert isinstance(result, str)
        assert "Error: Execution not available" in result
        assert "does not support command execution" in result

    def test_execute_tool_output_formatting(self):
        """Test execute tool formats output correctly."""

        # Mock sandbox backend that returns specific output
        class FormattingMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Hello world\nLine 2",
                    exit_code=0,
                    truncated=False,
                )

            @property
            def id(self):
                return "formatting-mock-sandbox-backend"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_fmt",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = FormattingMockSandboxBackend(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "echo test", "runtime": rt})

        assert "Hello world\nLine 2" in result
        assert "succeeded" in result
        assert "exit code 0" in result

    def test_execute_tool_output_formatting_with_failure(self):
        """Test execute tool formats failure output correctly."""

        # Mock sandbox backend that returns failure
        class FailureMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Error: command not found",
                    exit_code=127,
                    truncated=False,
                )

            @property
            def id(self):
                return "failure-mock-sandbox-backend"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_fail",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = FailureMockSandboxBackend(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "nonexistent", "runtime": rt})

        assert "Error: command not found" in result
        assert "failed" in result
        assert "exit code 127" in result

    def test_execute_tool_output_formatting_with_truncation(self):
        """Test execute tool formats truncated output correctly."""

        # Mock sandbox backend that returns truncated output
        class TruncatedMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Very long output...",
                    exit_code=0,
                    truncated=True,
                )

            @property
            def id(self):
                return "failure-mock-sandbox-backend"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_trunc",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TruncatedMockSandboxBackend(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "cat large_file", "runtime": rt})

        assert "Very long output..." in result
        assert "truncated" in result

    def test_supports_execution_helper_with_composite_backend(self):
        """Test _supports_execution correctly identifies CompositeBackend capabilities."""

        # Mock sandbox backend
        class TestSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="test", exit_code=0, truncated=False)

            @property
            def id(self) -> str:
                return "test-sandbox-backend"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        # StateBackend doesn't support execution
        state_backend = StateBackend(rt)
        assert not _supports_execution(state_backend)

        # TestSandboxBackend supports execution
        sandbox_backend = TestSandboxBackend(rt)
        assert _supports_execution(sandbox_backend)

        # CompositeBackend with sandbox default supports execution
        comp_with_sandbox = CompositeBackend(default=sandbox_backend, routes={})
        assert _supports_execution(comp_with_sandbox)

        # CompositeBackend with non-sandbox default doesn't support execution
        comp_without_sandbox = CompositeBackend(default=state_backend, routes={})
        assert not _supports_execution(comp_without_sandbox)

    def test_intercept_truncates_content_sample_lines(self):
        """Test that content sample shows head and tail with truncation notice and lines limited to 1000 chars."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_123", store=None, stream_writer=lambda _: None, config={})

        # Create content with 15 lines (more than head_lines + tail_lines = 10) to trigger truncation
        # Some lines are longer than 1000 chars to test line truncation
        lines_content = [
            "line 0",
            "a" * 2000,  # Long line in head
            "line 2",
            "line 3",
            "line 4",
            "line 5",  # This will be truncated
            "line 6",
            "line 7",
            "line 8",
            "line 9",
            "line 10",
            "line 11",
            "b" * 2000,  # Long line in tail
            "line 13",
            "line 14",
        ]
        large_content = "\n".join(lines_content)

        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, Command)
        processed_message = result.update["messages"][0]
        content_sample_section = processed_message.content

        # Verify the message contains the expected structure with head and tail
        assert "Tool result too large" in content_sample_section
        assert "head and tail" in content_sample_section

        # Verify truncation notice is present
        assert "lines truncated" in content_sample_section
        assert "[5 lines truncated]" in content_sample_section

        # Verify head lines are present (lines 0-4)
        assert "line 0" in content_sample_section
        assert "line 4" in content_sample_section

        # Verify tail lines are present (lines 10-14)
        assert "line 10" in content_sample_section
        assert "line 14" in content_sample_section

        # Verify middle lines are NOT present (lines 5-9)
        assert "line 5" not in content_sample_section
        assert "line 9" not in content_sample_section

        # Check each line in the content sample doesn't exceed 1000 chars
        lines = content_sample_section.split("\n")
        for line in lines:
            if line.strip() and "truncated" not in line:  # Skip empty lines and truncation notice
                assert len(line) <= 1010, f"Line exceeds 1000 chars: {len(line)} chars"

    @pytest.mark.parametrize(
        ("num_lines", "should_truncate"),
        [
            (0, False),  # Empty content
            (1, False),  # Single line
            (5, False),  # Fewer than head_lines + tail_lines
            (10, False),  # Exactly head_lines + tail_lines
            (11, True),  # Just over threshold
            (20, True),  # Well over threshold
        ],
    )
    def test_content_preview_edge_cases(self, num_lines, should_truncate):
        """Test _create_content_preview with various line counts."""
        # Create content with specified number of lines
        if num_lines == 0:
            content_str = ""
        else:
            lines = [f"line {i}" for i in range(num_lines)]
            content_str = "\n".join(lines)

        preview = _create_content_preview(content_str)

        if should_truncate:
            # Should have truncation notice
            assert "truncated" in preview
            # Should have head lines (0-4)
            assert "line 0" in preview
            assert "line 4" in preview
            # Should have tail lines
            assert f"line {num_lines - 5}" in preview
            assert f"line {num_lines - 1}" in preview
            # Should NOT have middle lines
            if num_lines > 11:
                assert "line 5" not in preview
                assert f"line {num_lines - 6}" not in preview
        else:
            # Should NOT have truncation notice
            assert "truncated" not in preview
            # Should have all lines
            for i in range(num_lines):
                assert f"line {i}" in preview


class TestPatchToolCallsMiddleware:
    def test_first_message(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert isinstance(state_update["messages"], Overwrite)
        patched_messages = state_update["messages"].value
        assert len(patched_messages) == 2
        assert patched_messages[0].type == "system"
        assert patched_messages[0].content == "You are a helpful assistant."
        assert patched_messages[1].type == "human"
        assert patched_messages[1].content == "Hello, how are you?"
        assert patched_messages[1].id == "2"

    def test_missing_tool_call(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="3",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="4"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert isinstance(state_update["messages"], Overwrite)
        patched_messages = state_update["messages"].value
        assert len(patched_messages) == 5
        assert patched_messages[0].type == "system"
        assert patched_messages[0].content == "You are a helpful assistant."
        assert patched_messages[1].type == "human"
        assert patched_messages[1].content == "Hello, how are you?"
        assert patched_messages[2].type == "ai"
        assert len(patched_messages[2].tool_calls) == 1
        assert patched_messages[2].tool_calls[0]["id"] == "123"
        assert patched_messages[2].tool_calls[0]["name"] == "get_events_for_days"
        assert patched_messages[2].tool_calls[0]["args"] == {"date_str": "2025-01-01"}
        assert patched_messages[3].type == "tool"
        assert patched_messages[3].name == "get_events_for_days"
        assert patched_messages[3].tool_call_id == "123"
        assert patched_messages[4].type == "human"
        assert patched_messages[4].content == "What is the weather in Tokyo?"

    def test_no_missing_tool_calls(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="3",
            ),
            ToolMessage(content="I have no events for that date.", tool_call_id="123", id="4"),
            HumanMessage(content="What is the weather in Tokyo?", id="5"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert isinstance(state_update["messages"], Overwrite)
        patched_messages = state_update["messages"].value
        assert len(patched_messages) == 5
        assert patched_messages[0].type == "system"
        assert patched_messages[0].content == "You are a helpful assistant."
        assert patched_messages[1].type == "human"
        assert patched_messages[1].content == "Hello, how are you?"
        assert patched_messages[2].type == "ai"
        assert len(patched_messages[2].tool_calls) == 1
        assert patched_messages[2].tool_calls[0]["id"] == "123"
        assert patched_messages[2].tool_calls[0]["name"] == "get_events_for_days"
        assert patched_messages[2].tool_calls[0]["args"] == {"date_str": "2025-01-01"}
        assert patched_messages[3].type == "tool"
        assert patched_messages[3].tool_call_id == "123"
        assert patched_messages[4].type == "human"
        assert patched_messages[4].content == "What is the weather in Tokyo?"

    def test_two_missing_tool_calls(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="3",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="4"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="456", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="5",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="6"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert isinstance(state_update["messages"], Overwrite)
        patched_messages = state_update["messages"].value
        assert len(patched_messages) == 8
        assert patched_messages[0].type == "system"
        assert patched_messages[0].content == "You are a helpful assistant."
        assert patched_messages[1].type == "human"
        assert patched_messages[1].content == "Hello, how are you?"
        assert patched_messages[2].type == "ai"
        assert len(patched_messages[2].tool_calls) == 1
        assert patched_messages[2].tool_calls[0]["id"] == "123"
        assert patched_messages[2].tool_calls[0]["name"] == "get_events_for_days"
        assert patched_messages[2].tool_calls[0]["args"] == {"date_str": "2025-01-01"}
        assert patched_messages[3].type == "tool"
        assert patched_messages[3].name == "get_events_for_days"
        assert patched_messages[3].tool_call_id == "123"
        assert patched_messages[4].type == "human"
        assert patched_messages[4].content == "What is the weather in Tokyo?"
        assert patched_messages[5].type == "ai"
        assert len(patched_messages[5].tool_calls) == 1
        assert patched_messages[5].tool_calls[0]["id"] == "456"
        assert patched_messages[5].tool_calls[0]["name"] == "get_events_for_days"
        assert patched_messages[5].tool_calls[0]["args"] == {"date_str": "2025-01-01"}
        assert patched_messages[6].type == "tool"
        assert patched_messages[6].name == "get_events_for_days"
        assert patched_messages[6].tool_call_id == "456"
        assert patched_messages[7].type == "human"
        assert patched_messages[7].content == "What is the weather in Tokyo?"


class TestTruncation:
    def test_truncate_list_result_no_truncation(self):
        items = ["/file1.py", "/file2.py", "/file3.py"]
        result = truncate_if_too_long(items)
        assert result == items

    def test_truncate_list_result_with_truncation(self):
        # Create a list that exceeds the token limit (20000 tokens * 4 chars = 80000 chars)
        large_items = [f"/very_long_file_path_{'x' * 100}_{i}.py" for i in range(1000)]
        result = truncate_if_too_long(large_items)

        # Should be truncated
        assert len(result) < len(large_items)
        # Last item should be the truncation message
        assert "results truncated" in result[-1]
        assert "try being more specific" in result[-1]

    def test_truncate_string_result_no_truncation(self):
        content = "short content"
        result = truncate_if_too_long(content)
        assert result == content

    def test_truncate_string_result_with_truncation(self):
        # Create string that exceeds the token limit (20000 tokens * 4 chars = 80000 chars)
        large_content = "x" * 100000
        result = truncate_if_too_long(large_content)

        # Should be truncated
        assert len(result) < len(large_content)
        # Should end with truncation message
        assert "results truncated" in result
        assert "try being more specific" in result


class TestBuiltinTruncationTools:
    def test_builtin_truncation_tool_not_evicted(self):
        """Test that tools excluded from eviction (grep, ls, glob, etc.) are NOT evicted to filesystem."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)  # Very low limit
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_grep_123", store=None, stream_writer=lambda _: None, config={})

        # Create a large tool result
        large_content = "x" * 5000
        expected_result = ToolMessage(content=large_content, tool_call_id="test_grep_123")

        # Mock handler that returns the large result
        def mock_handler(request):  # noqa: ARG001 - request required by handler interface
            return expected_result

        # Create a request for a tool in TOOLS_EXCLUDED_FROM_EVICTION
        request = ToolCallRequest(
            runtime=runtime,
            tool_call={"id": "test_grep_123", "name": "grep", "args": {"pattern": "test"}},
            state=state,
            tool=None,
        )

        # Call wrap_tool_call
        result = middleware.wrap_tool_call(request, mock_handler)

        # Result should NOT be intercepted - should be the original ToolMessage
        assert isinstance(result, ToolMessage)
        assert result == expected_result
        assert result.content == large_content

    def test_non_builtin_truncation_tool_evicted(self):
        """Test that tools NOT in TOOLS_EXCLUDED_FROM_EVICTION are evicted to filesystem."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)  # Very low limit
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(state=state, context=None, tool_call_id="test_custom_123", store=None, stream_writer=lambda _: None, config={})

        # Create a large tool result
        large_content = "y" * 5000
        large_result = ToolMessage(content=large_content, tool_call_id="test_custom_123")

        # Mock handler that returns the large result
        def mock_handler(request):  # noqa: ARG001 - request required by handler interface
            return large_result

        # Create a request for a tool NOT in TOOLS_EXCLUDED_FROM_EVICTION
        request = ToolCallRequest(
            runtime=runtime,
            tool_call={"id": "test_custom_123", "name": "custom_tool", "args": {"input": "test"}},
            state=state,
            tool=None,
        )

        # Call wrap_tool_call
        result = middleware.wrap_tool_call(request, mock_handler)

        # Result SHOULD be intercepted - should be a Command with files
        assert isinstance(result, Command)
        assert "/large_tool_results/test_custom_123" in result.update["files"]
        assert "Tool result too large" in result.update["messages"][0].content

    def test_execute_tool_large_output_evicted(self) -> None:
        """Test that execute tool with large output gets evicted to filesystem."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)  # Low threshold
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_exec_123",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        # Simulate large execute output (like a command that outputs many lines)
        large_execute_output = "x" * 10000
        large_execute_output += "\n[Command succeeded with exit code 0]"

        # Create a ToolMessage with the large execute output
        large_result = ToolMessage(content=large_execute_output, tool_call_id="test_exec_123", name="execute")

        # Mock handler that returns the large result
        def mock_handler(request):  # noqa: ARG001 - request required by handler interface
            return large_result

        # Create a request for the execute tool
        request = ToolCallRequest(
            runtime=runtime,
            tool_call={"id": "test_exec_123", "name": "execute", "args": {"command": "echo large output"}},
            state=state,
            tool=None,
        )

        # Call wrap_tool_call - this is where eviction happens
        result = middleware.wrap_tool_call(request, mock_handler)

        # Result SHOULD be intercepted - should be a Command with files
        assert isinstance(result, Command)
        assert "/large_tool_results/test_exec_123" in result.update["files"]
        assert "Tool result too large" in result.update["messages"][0].content

        # Verify the message has the tool name preserved
        assert result.update["messages"][0].name == "execute"

    def test_execute_tool_rejects_zero_timeout(self):
        """Middleware should return a friendly error for timeout=0, not crash."""

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_zero_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        # Should return a friendly error string, NOT raise an exception
        result = execute_tool.invoke({"command": "echo hello", "timeout": 0, "runtime": rt})

        assert isinstance(result, str)
        assert "error" in result.lower()

    def test_execute_tool_rejects_negative_timeout(self):
        """Middleware should return a friendly error for negative timeout."""

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_neg_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "echo hello", "timeout": -5, "runtime": rt})

        assert isinstance(result, str)
        assert "error" in result.lower()

    def test_execute_tool_forwards_valid_timeout_to_backend(self):
        """Middleware should forward a valid timeout to the backend."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_fwd_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        execute_tool.invoke({"command": "echo hello", "timeout": 300, "runtime": rt})

        assert captured_timeout["value"] == 300

    def test_execute_tool_rejects_timeout_exceeding_max(self):
        """Middleware should return a friendly error when timeout exceeds max_execute_timeout."""

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_max_execute_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend, max_execute_timeout=600)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "echo hello", "timeout": 601, "runtime": rt})

        assert isinstance(result, str)
        assert "error" in result.lower()
        assert "601" in result
        assert "600" in result

    def test_execute_tool_accepts_timeout_at_max(self):
        """Middleware should accept timeout exactly equal to max_execute_timeout."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_at_max_execute_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend, max_execute_timeout=300)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        execute_tool.invoke({"command": "echo hello", "timeout": 300, "runtime": rt})

        assert captured_timeout["value"] == 300

    def test_execute_tool_none_timeout_skips_max_check(self):
        """Middleware should not reject None timeout against max_execute_timeout."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_none_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend, max_execute_timeout=10)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        execute_tool.invoke({"command": "echo hello", "runtime": rt})

        # None should be forwarded without max_execute_timeout rejection
        assert captured_timeout["value"] is None

    def test_max_execute_timeout_init_validation(self):
        """FilesystemMiddleware should reject non-positive max_execute_timeout at init."""
        with pytest.raises(ValueError, match="max_execute_timeout must be positive"):
            FilesystemMiddleware(max_execute_timeout=0)

        with pytest.raises(ValueError, match="max_execute_timeout must be positive"):
            FilesystemMiddleware(max_execute_timeout=-1)
