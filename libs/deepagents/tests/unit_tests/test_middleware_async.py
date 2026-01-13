"""Async tests for middleware filesystem tools."""

import pytest
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.middleware.filesystem import FileData, FilesystemMiddleware, FilesystemState


def build_composite_state_backend(runtime: ToolRuntime, *, routes):
    built_routes = {}
    for prefix, backend_or_factory in routes.items():
        if callable(backend_or_factory):
            built_routes[prefix] = backend_or_factory(runtime)
        else:
            built_routes[prefix] = backend_or_factory
    default_state = StateBackend(runtime)
    return CompositeBackend(default=default_state, routes=built_routes)


class TestFilesystemMiddlewareAsync:
    """Async tests for filesystem middleware tools."""

    @pytest.mark.asyncio
    async def test_als_shortterm(self):
        """Test async ls tool with state backend."""
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
        result = await ls_tool.ainvoke(
            {"runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}), "path": "/"}
        )
        assert result == str(["/test.txt", "/test2.txt"])

    @pytest.mark.asyncio
    async def test_als_shortterm_with_path(self):
        """Test async ls tool with specific path."""
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
        result = await ls_tool.ainvoke(
            {
                "path": "/pokemon/",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # ls should only return files directly in /pokemon/, not in subdirectories
        assert "/pokemon/test2.txt" in result
        assert "/pokemon/charmander.txt" in result
        assert "/pokemon/water/squirtle.txt" not in result  # In subdirectory
        assert "/pokemon/water/" in result

    @pytest.mark.asyncio
    async def test_als_shortterm_lists_directories(self):
        """Test async ls lists directories with trailing /."""
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
        result = await ls_tool.ainvoke(
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

    @pytest.mark.asyncio
    async def test_aglob_search_shortterm_simple_pattern(self):
        """Test async glob with simple pattern."""
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
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # Standard glob: *.py only matches files in root directory, not subdirectories
        assert result == str(["/test.py"])

    @pytest.mark.asyncio
    async def test_aglob_search_shortterm_wildcard_pattern(self):
        """Test async glob with wildcard pattern."""
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
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "**/*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" in result
        assert "/tests/test_main.py" in result

    @pytest.mark.asyncio
    async def test_aglob_search_shortterm_with_path(self):
        """Test async glob with specific path."""
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
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "path": "/src",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" not in result
        assert "/tests/test_main.py" not in result

    @pytest.mark.asyncio
    async def test_aglob_search_shortterm_brace_expansion(self):
        """Test async glob with brace expansion."""
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
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.{py,pyi}",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py" in result
        assert "/test.pyi" in result
        assert "/test.txt" not in result

    @pytest.mark.asyncio
    async def test_aglob_search_shortterm_no_matches(self):
        """Test async glob with no matches."""
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
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert result == str([])

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_files_with_matches(self):
        """Test async grep with files_with_matches mode."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py" in result
        assert "/helper.txt" in result
        assert "/main.py" not in result

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_content_mode(self):
        """Test async grep with content mode."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "output_mode": "content",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "1: import os" in result
        assert "2: import sys" in result
        assert "print" not in result

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_count_mode(self):
        """Test async grep with count mode."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "output_mode": "count",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py:2" in result or "/test.py: 2" in result
        assert "/main.py:1" in result or "/main.py: 1" in result

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_with_include(self):
        """Test async grep with glob filter."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "glob": "*.py",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/test.py" in result
        assert "/test.txt" not in result

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_with_path(self):
        """Test async grep with specific path."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "path": "/src",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/src/main.py" in result
        assert "/tests/test.py" not in result

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_regex_pattern(self):
        """Test async grep with regex pattern."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": r"def \w+\(",
                "output_mode": "content",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "1: def hello():" in result
        assert "2: def world():" in result
        assert "x = 5" not in result

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_no_matches(self):
        """Test async grep with no matches."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert result == "No matches found"

    @pytest.mark.asyncio
    async def test_agrep_search_shortterm_invalid_regex(self):
        """Test async grep with invalid regex."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "[invalid",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "Invalid regex pattern" in result

    @pytest.mark.asyncio
    async def test_aread_file(self):
        """Test async read_file tool."""
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world", "Line 2", "Line 3"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = await read_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "Hello world" in result
        assert "Line 2" in result
        assert "Line 3" in result

    @pytest.mark.asyncio
    async def test_aread_file_with_offset(self):
        """Test async read_file tool with offset."""
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Line 1", "Line 2", "Line 3", "Line 4"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = await read_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "offset": 1,
                "limit": 2,
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 1" not in result
        assert "Line 4" not in result

    @pytest.mark.asyncio
    async def test_awrite_file(self):
        """Test async write_file tool."""
        from langgraph.types import Command

        state = FilesystemState(messages=[], files={})
        middleware = FilesystemMiddleware()
        write_file_tool = next(tool for tool in middleware.tools if tool.name == "write_file")
        result = await write_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "content": "Hello world",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="tc1", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # StateBackend returns a Command with files_update
        assert isinstance(result, Command)
        assert "/test.txt" in result.update["files"]

    @pytest.mark.asyncio
    async def test_aedit_file(self):
        """Test async edit_file tool."""
        from langgraph.types import Command

        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world", "Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        edit_file_tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
        result = await edit_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "old_string": "Hello",
                "new_string": "Hi",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="tc2", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # StateBackend returns a Command with files_update
        assert isinstance(result, Command)
        assert "/test.txt" in result.update["files"]

    @pytest.mark.asyncio
    async def test_aedit_file_replace_all(self):
        """Test async edit_file tool with replace_all."""
        from langgraph.types import Command

        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world", "Hello again"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware()
        edit_file_tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
        result = await edit_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "old_string": "Hello",
                "new_string": "Hi",
                "replace_all": True,
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="tc3", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert isinstance(result, Command)
        assert "/test.txt" in result.update["files"]

    @pytest.mark.asyncio
    async def test_aexecute_tool_returns_error_when_backend_doesnt_support(self):
        """Test async execute tool returns friendly error instead of raising exception."""
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
        result = await execute_tool.ainvoke({"command": "ls -la", "runtime": runtime})

        assert isinstance(result, str)
        assert "Error: Execution not available" in result
        assert "does not support command execution" in result

    @pytest.mark.asyncio
    async def test_aexecute_tool_output_formatting(self):
        """Test async execute tool formats output correctly."""

        # Mock sandbox backend that returns specific output
        class FormattingMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Hello world\nLine 2",
                    exit_code=0,
                    truncated=False,
                )

            async def aexecute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Async Hello world\nAsync Line 2",
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
        result = await execute_tool.ainvoke({"command": "echo test", "runtime": rt})

        assert "Async Hello world\nAsync Line 2" in result
        assert "succeeded" in result
        assert "exit code 0" in result

    @pytest.mark.asyncio
    async def test_aexecute_tool_output_formatting_with_failure(self):
        """Test async execute tool formats failure output correctly."""

        # Mock sandbox backend that returns failure
        class FailureMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Error: command not found",
                    exit_code=127,
                    truncated=False,
                )

            async def aexecute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Async Error: command not found",
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
        result = await execute_tool.ainvoke({"command": "nonexistent", "runtime": rt})

        assert "Async Error: command not found" in result
        assert "failed" in result
        assert "exit code 127" in result

    @pytest.mark.asyncio
    async def test_aexecute_tool_output_formatting_with_truncation(self):
        """Test async execute tool formats truncated output correctly."""

        # Mock sandbox backend that returns truncated output
        class TruncatedMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Very long output...",
                    exit_code=0,
                    truncated=True,
                )

            async def aexecute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Async Very long output...",
                    exit_code=0,
                    truncated=True,
                )

            @property
            def id(self):
                return "truncated-mock-sandbox-backend"

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
        result = await execute_tool.ainvoke({"command": "cat large_file", "runtime": rt})

        assert "Async Very long output..." in result
        assert "truncated" in result
