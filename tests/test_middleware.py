import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore

from deepagents.middleware.filesystem import (
    FILESYSTEM_SYSTEM_PROMPT,
    FileData,
    FilesystemMiddleware,
    FilesystemState
)
from deepagents.backends import StoreBackend, CompositeBackend, StateBackend

from deepagents.backends.utils import create_file_data, update_file_data
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import DEFAULT_GENERAL_PURPOSE_DESCRIPTION, TASK_SYSTEM_PROMPT, TASK_TOOL_DESCRIPTION, SubAgentMiddleware
from deepagents.backends.utils import truncate_if_too_long

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
        middleware = [SubAgentMiddleware(default_tools=[], subagents=[], default_model="claude-sonnet-4-20250514")]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()

    def test_multiple_middleware(self):
        middleware = [FilesystemMiddleware(), SubAgentMiddleware(default_tools=[], subagents=[], default_model="claude-sonnet-4-20250514")]
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
        assert middleware.system_prompt == FILESYSTEM_SYSTEM_PROMPT
        assert len(middleware.tools) == 6

    def test_init_with_composite_backend(self):
        backend_factory = lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})
        middleware = FilesystemMiddleware(backend=backend_factory)
        assert callable(middleware.backend)
        assert middleware.system_prompt == FILESYSTEM_SYSTEM_PROMPT
        assert len(middleware.tools) == 6

    def test_init_custom_system_prompt_default(self):
        middleware = FilesystemMiddleware(system_prompt="Custom system prompt")
        assert callable(middleware.backend)
        assert middleware.system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 6

    def test_init_custom_system_prompt_with_composite(self):
        backend_factory = lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})
        middleware = FilesystemMiddleware(backend=backend_factory, system_prompt="Custom system prompt")
        assert callable(middleware.backend)
        assert middleware.system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 6

    def test_init_custom_tool_descriptions_default(self):
        middleware = FilesystemMiddleware(custom_tool_descriptions={"ls": "Custom ls tool description"})
        assert callable(middleware.backend)
        assert middleware.system_prompt == FILESYSTEM_SYSTEM_PROMPT
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_init_custom_tool_descriptions_with_composite(self):
        backend_factory = lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})
        middleware = FilesystemMiddleware(backend=backend_factory, custom_tool_descriptions={"ls": "Custom ls tool description"})
        assert callable(middleware.backend)
        assert middleware.system_prompt == FILESYSTEM_SYSTEM_PROMPT
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
            {
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
                "path": "/"
            }
        )
        assert result == ["/test.txt", "/test2.txt"]

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
        result = ls_tool.invoke(
            {
                "path": "/pokemon/",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
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
        result = ls_tool.invoke(
            {
                "path": "/",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
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
        print(glob_search_tool)
        result = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # Standard glob: *.py only matches files in root directory, not subdirectories
        assert "/test.py" in result
        assert "/test.txt" not in result
        assert "/pokemon/charmander.py" not in result
        assert len(result) == 1
        assert result[0] == "/test.py"

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
        result = glob_search_tool.invoke(
            {
                "pattern": "**/*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
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
        result = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "path": "/src",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
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
        result = glob_search_tool.invoke(
            {
                "pattern": "*.{py,pyi}",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
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
        print(glob_search_tool)
        assert result == []
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
        result = grep_search_tool.invoke(
            {
                "pattern": r"def \w+\(",
                "output_mode": "content",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        print(result)
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
                "pattern": "[invalid",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "Invalid regex pattern" in result

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

    def test_create_file_data_splits_long_lines(self):
        long_line = "a" * 3500
        short_line = "short line"
        content = f"{short_line}\n{long_line}"

        file_data = create_file_data(content)

        for line in file_data["content"]:
            assert len(line) <= 2000

        assert len(file_data["content"]) == 3
        assert file_data["content"][0] == short_line
        assert file_data["content"][1] == "a" * 2000
        assert file_data["content"][2] == "a" * 1500

    def test_update_file_data_splits_long_lines(self):
        initial_file_data = create_file_data("initial content")

        long_line = "b" * 5000
        short_line = "another short line"
        new_content = f"{short_line}\n{long_line}"

        updated_file_data = update_file_data(initial_file_data, new_content)

        for line in updated_file_data["content"]:
            assert len(line) <= 2000

        assert len(updated_file_data["content"]) == 4
        assert updated_file_data["content"][0] == short_line
        assert updated_file_data["content"][1] == "b" * 2000
        assert updated_file_data["content"][2] == "b" * 2000
        assert updated_file_data["content"][3] == "b" * 1000

        assert updated_file_data["created_at"] == initial_file_data["created_at"]


@pytest.mark.requires("langchain_openai")
class TestSubagentMiddleware:
    """Test the SubagentMiddleware class."""

    def test_subagent_middleware_init(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
        )
        assert middleware is not None
        assert middleware.system_prompt is TASK_SYSTEM_PROMPT
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        expected_desc = TASK_TOOL_DESCRIPTION.format(available_agents=f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")
        assert middleware.tools[0].description == expected_desc

    def test_default_subagent_with_tools(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
        )
        assert middleware is not None
        assert middleware.system_prompt == TASK_SYSTEM_PROMPT

    def test_default_subagent_custom_system_prompt(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        assert middleware.system_prompt == "Use the task tool to call a subagent."


class TestPatchToolCallsMiddleware:
    def test_first_message(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 3
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1].type == "system"
        assert state_update["messages"][1].content == "You are a helpful assistant."
        assert state_update["messages"][2].type == "human"
        assert state_update["messages"][2].content == "Hello, how are you?"
        assert state_update["messages"][2].id == "2"

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
        assert len(state_update["messages"]) == 6
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1] == input_messages[0]
        assert state_update["messages"][2] == input_messages[1]
        assert state_update["messages"][3] == input_messages[2]
        assert state_update["messages"][4].type == "tool"
        assert state_update["messages"][4].tool_call_id == "123"
        assert state_update["messages"][4].name == "get_events_for_days"
        assert state_update["messages"][5] == input_messages[3]
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 5
        assert updated_messages[0] == input_messages[0]
        assert updated_messages[1] == input_messages[1]
        assert updated_messages[2] == input_messages[2]
        assert updated_messages[3].type == "tool"
        assert updated_messages[3].tool_call_id == "123"
        assert updated_messages[3].name == "get_events_for_days"
        assert updated_messages[4] == input_messages[3]

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
        assert len(state_update["messages"]) == 6
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1:] == input_messages
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 5
        assert updated_messages == input_messages

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
        assert len(state_update["messages"]) == 9
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1] == input_messages[0]
        assert state_update["messages"][2] == input_messages[1]
        assert state_update["messages"][3] == input_messages[2]
        assert state_update["messages"][4].type == "tool"
        assert state_update["messages"][4].tool_call_id == "123"
        assert state_update["messages"][4].name == "get_events_for_days"
        assert state_update["messages"][5] == input_messages[3]
        assert state_update["messages"][6] == input_messages[4]
        assert state_update["messages"][7].type == "tool"
        assert state_update["messages"][7].tool_call_id == "456"
        assert state_update["messages"][7].name == "get_events_for_days"
        assert state_update["messages"][8] == input_messages[5]
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 8
        assert updated_messages[0] == input_messages[0]
        assert updated_messages[1] == input_messages[1]
        assert updated_messages[2] == input_messages[2]
        assert updated_messages[3].type == "tool"
        assert updated_messages[3].tool_call_id == "123"
        assert updated_messages[3].name == "get_events_for_days"
        assert updated_messages[4] == input_messages[3]
        assert updated_messages[5] == input_messages[4]
        assert updated_messages[6].type == "tool"
        assert updated_messages[6].tool_call_id == "456"
        assert updated_messages[6].name == "get_events_for_days"
        assert updated_messages[7] == input_messages[5]


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
