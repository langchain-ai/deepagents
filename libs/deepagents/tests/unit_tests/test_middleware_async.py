"""Async tests for middleware filesystem tools."""

from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import create_file_data
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


class RecordingFilesystemBackend(StateBackend):
    """State-backed async test backend that records filesystem timeouts."""

    def __init__(self, runtime: ToolRuntime) -> None:
        super().__init__(runtime)
        self.timeouts: dict[str, int | None] = {}
        self.glob_result = GlobResult(matches=[])
        self.grep_result = GrepResult(matches=[])

    def ls_info(self, path: str, *, timeout: int | None = None) -> LsResult:
        self.timeouts["ls"] = timeout
        return LsResult(entries=[])

    def read(self, path: str, offset: int = 0, limit: int = 100, *, timeout: int | None = None) -> ReadResult:
        self.timeouts["read"] = timeout
        return ReadResult(file_data=create_file_data("content"))

    def write(self, path: str, content: str, *, timeout: int | None = None) -> WriteResult:
        self.timeouts["write"] = timeout
        return WriteResult(path=path, files_update=None)

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
        *,
        timeout: int | None = None,
    ) -> EditResult:
        self.timeouts["edit"] = timeout
        return EditResult(path=path, files_update=None, occurrences=1)

    def glob_info(self, pattern: str, path: str = "/", *, timeout: int | None = None) -> GlobResult:
        self.timeouts["glob"] = timeout
        return self.glob_result

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        timeout: int | None = None,
    ) -> GrepResult:
        self.timeouts["grep"] = timeout
        return self.grep_result


class TestFilesystemMiddlewareAsync:
    """Async tests for filesystem middleware tools."""

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

    async def test_glob_timeout_returns_error_message_async(self):
        rt = ToolRuntime(
            state={"messages": [], "files": {}},
            context=None,
            tool_call_id="aglob_timeout",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        backend = RecordingFilesystemBackend(rt)
        backend.glob_result = GlobResult(error="glob timed out after 1s. Try a more specific pattern or a narrower path.")
        middleware = FilesystemMiddleware(backend=backend, max_filesystem_timeout=5)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke({"pattern": "**/*", "timeout": 1, "runtime": rt})

        assert result == "Error: glob timed out after 1s. Try a more specific pattern or a narrower path."
        assert backend.timeouts["glob"] == 1

    async def test_async_filesystem_tools_apply_default_timeout_caps(self) -> None:
        rt = ToolRuntime(
            state={"messages": [], "files": {}},
            context=None,
            tool_call_id="async_filesystem_caps",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        backend = RecordingFilesystemBackend(rt)
        middleware = FilesystemMiddleware(backend=backend, max_filesystem_timeout=45)

        await next(tool for tool in middleware.tools if tool.name == "ls").ainvoke({"path": "/", "runtime": rt})
        await next(tool for tool in middleware.tools if tool.name == "read_file").ainvoke({"file_path": "/test.txt", "runtime": rt})
        await next(tool for tool in middleware.tools if tool.name == "write_file").ainvoke({"file_path": "/test.txt", "content": "x", "runtime": rt})
        await next(tool for tool in middleware.tools if tool.name == "edit_file").ainvoke(
            {"file_path": "/test.txt", "old_string": "x", "new_string": "y", "runtime": rt}
        )

        assert backend.timeouts["ls"] == 45
        assert backend.timeouts["read"] == 45
        assert backend.timeouts["write"] == 45
        assert backend.timeouts["edit"] == 45

    async def test_async_glob_and_grep_timeout_defaults_and_validation(self) -> None:
        rt = ToolRuntime(
            state={"messages": [], "files": {}},
            context=None,
            tool_call_id="async_glob_grep_timeout",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        backend = RecordingFilesystemBackend(rt)
        middleware = FilesystemMiddleware(backend=backend, max_filesystem_timeout=40)

        glob_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        grep_tool = next(tool for tool in middleware.tools if tool.name == "grep")

        await glob_tool.ainvoke({"pattern": "**/*", "runtime": rt})
        await grep_tool.ainvoke({"pattern": "needle", "runtime": rt})
        assert backend.timeouts["glob"] == 20
        assert backend.timeouts["grep"] == 30

        await glob_tool.ainvoke({"pattern": "**/*", "timeout": 15, "runtime": rt})
        await grep_tool.ainvoke({"pattern": "needle", "timeout": 16, "runtime": rt})
        assert backend.timeouts["glob"] == 15
        assert backend.timeouts["grep"] == 16

        result = await glob_tool.ainvoke({"pattern": "**/*", "timeout": 41, "runtime": rt})
        assert result == "Error: timeout 41s exceeds maximum allowed (40s)."

        result = await grep_tool.ainvoke({"pattern": "needle", "timeout": 0, "runtime": rt})
        assert result == "Error: timeout must be positive, got 0."

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

    async def test_agrep_search_shortterm_regex_pattern(self):
        """Test async grep with literal pattern (not regex)."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "def ",
                "output_mode": "content",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "1: def hello():" in result
        assert "2: def world():" in result
        assert "x = 5" not in result

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

    async def test_agrep_search_shortterm_invalid_regex(self):
        """Test async grep with special characters (literal search, not regex)."""
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
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "[invalid",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "No matches found" in result

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

    async def test_awrite_file(self):
        """Test async write_file tool."""
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

    async def test_aedit_file(self):
        """Test async edit_file tool."""
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

    async def test_aedit_file_replace_all(self):
        """Test async edit_file tool with replace_all."""
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

    async def test_aexecute_tool_forwards_zero_timeout_to_backend(self):
        """Async execute tool should forward timeout=0 for no-timeout backends."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="sync ok", exit_code=0, truncated=False)

            async def aexecute(
                self,
                command: str,
                *,
                timeout: int | None = None,  # noqa: ASYNC109
            ) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="async ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox-backend"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_zero_timeout_async",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox(rt)
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = await execute_tool.ainvoke({"command": "echo hello", "timeout": 0, "runtime": rt})

        assert "async ok" in result
        assert captured_timeout["value"] == 0

    async def test_aexecute_tool_output_formatting(self):
        """Test async execute tool formats output correctly."""

        # Mock sandbox backend that returns specific output
        class FormattingMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Hello world\nLine 2",
                    exit_code=0,
                    truncated=False,
                )

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
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

    async def test_aexecute_tool_output_formatting_with_failure(self):
        """Test async execute tool formats failure output correctly."""

        # Mock sandbox backend that returns failure
        class FailureMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Error: command not found",
                    exit_code=127,
                    truncated=False,
                )

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
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

    async def test_aexecute_tool_output_formatting_with_truncation(self):
        """Test async execute tool formats truncated output correctly."""

        # Mock sandbox backend that returns truncated output
        class TruncatedMockSandboxBackend(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(
                    output="Very long output...",
                    exit_code=0,
                    truncated=True,
                )

            async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
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
