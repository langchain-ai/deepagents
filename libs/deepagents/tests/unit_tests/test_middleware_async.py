"""Async tests for middleware filesystem tools."""

import asyncio
from unittest.mock import patch

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

import deepagents.middleware.filesystem as filesystem_middleware
from deepagents.backends import StateBackend, StoreBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.middleware.filesystem import FileData, FilesystemMiddleware, FilesystemState


def _make_backend(files=None):
    """Create a StoreBackend backed by InMemoryStore, optionally pre-populated with files."""
    mem_store = InMemoryStore()
    if files:
        for path, fdata in files.items():
            mem_store.put(
                ("filesystem",),
                path,
                {
                    "content": fdata["content"],
                    "encoding": fdata.get("encoding", "utf-8"),
                    "created_at": fdata.get("created_at", ""),
                    "modified_at": fdata.get("modified_at", ""),
                },
            )
    backend = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    return backend, mem_store


def _runtime():
    return ToolRuntime(state={}, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={})


class TestFilesystemMiddlewareAsync:
    """Async tests for filesystem middleware tools."""

    async def test_als_shortterm(self):
        """Test async ls tool with state backend."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = await ls_tool.ainvoke({"runtime": _runtime(), "path": "/"})
        assert result.content == str(["/test.txt", "/test2.txt"])

    async def test_als_shortterm_with_path(self):
        """Test async ls tool with specific path."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = await ls_tool.ainvoke(
            {
                "path": "/pokemon/",
                "runtime": _runtime(),
            }
        )
        # ls should only return files directly in /pokemon/, not in subdirectories
        assert "/pokemon/test2.txt" in result.content
        assert "/pokemon/charmander.txt" in result.content
        assert "/pokemon/water/squirtle.txt" not in result.content  # In subdirectory
        assert "/pokemon/water/" in result.content

    async def test_als_shortterm_lists_directories(self):
        """Test async ls lists directories with trailing /."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = await ls_tool.ainvoke(
            {
                "path": "/",
                "runtime": _runtime(),
            }
        )
        # ls should list both files and directories at root level
        assert "/test.txt" in result.content
        assert "/pokemon/" in result.content
        assert "/docs/" in result.content
        # But NOT subdirectory files
        assert "/pokemon/charmander.txt" not in result.content
        assert "/pokemon/water/squirtle.txt" not in result.content

    async def test_aglob_search_shortterm_simple_pattern(self):
        """Test async glob with simple pattern."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "runtime": _runtime(),
            }
        )
        # Standard glob: *.py only matches files in root directory, not subdirectories
        assert result.content == str(["/test.py"])

    async def test_aglob_search_shortterm_wildcard_pattern(self):
        """Test async glob with wildcard pattern."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "**/*.py",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/src/utils/helper.py" in result.content
        assert "/tests/test_main.py" in result.content

    async def test_aglob_search_shortterm_with_path(self):
        """Test async glob with specific path."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "path": "/src",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/src/utils/helper.py" not in result.content
        assert "/tests/test_main.py" not in result.content

    async def test_aglob_search_shortterm_brace_expansion(self):
        """Test async glob with brace expansion."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.{py,pyi}",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/test.pyi" in result.content
        assert "/test.txt" not in result.content

    async def test_aglob_search_shortterm_no_matches(self):
        """Test async glob with no matches."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "runtime": _runtime(),
            }
        )
        assert result.content == str([])

    async def test_glob_timeout_returns_error_message_async(self):
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        backend_obj = middleware._get_backend(_runtime())

        async def slow_aglob(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
            await asyncio.sleep(2)
            return []

        with (
            patch.object(filesystem_middleware, "GLOB_TIMEOUT", 0.5),
            patch.object(middleware, "_get_backend", return_value=backend_obj),
            patch.object(backend_obj, "aglob", side_effect=slow_aglob),
        ):
            result = await glob_search_tool.ainvoke(
                {
                    "pattern": "**/*",
                    "runtime": _runtime(),
                }
            )

        assert result.content == "Error: glob timed out after 0.5s. Try a more specific pattern or a narrower path."

    async def test_agrep_search_shortterm_files_with_matches(self):
        """Test async grep with files_with_matches mode."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/helper.txt" in result.content
        assert "/main.py" not in result.content

    async def test_agrep_search_shortterm_content_mode(self):
        """Test async grep with content mode."""
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "output_mode": "content",
                "runtime": _runtime(),
            }
        )
        assert "1: import os" in result.content
        assert "2: import sys" in result.content
        assert "print" not in result.content

    async def test_agrep_search_shortterm_count_mode(self):
        """Test async grep with count mode."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "output_mode": "count",
                "runtime": _runtime(),
            }
        )
        assert "/test.py:2" in result.content or "/test.py: 2" in result.content
        assert "/main.py:1" in result.content or "/main.py: 1" in result.content

    async def test_agrep_search_shortterm_with_include(self):
        """Test async grep with glob filter."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "glob": "*.py",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/test.txt" not in result.content

    async def test_agrep_search_shortterm_with_path(self):
        """Test async grep with specific path."""
        files = {
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
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "path": "/src",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/tests/test.py" not in result.content

    async def test_agrep_search_shortterm_regex_pattern(self):
        """Test async grep with literal pattern (not regex)."""
        files = {
            "/test.py": FileData(
                content=["def hello():", "def world():", "x = 5"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Search for literal "def " - literal search, not regex
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "def ",
                "output_mode": "content",
                "runtime": _runtime(),
            }
        )
        assert "1: def hello():" in result.content
        assert "2: def world():" in result.content
        assert "x = 5" not in result.content

    async def test_agrep_search_shortterm_no_matches(self):
        """Test async grep with no matches."""
        files = {
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "runtime": _runtime(),
            }
        )
        assert result.content == "No matches found"

    async def test_agrep_search_shortterm_invalid_regex(self):
        """Test async grep with special characters (literal search, not regex)."""
        files = {
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Special characters are treated literally, so no matches expected
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "[invalid",
                "runtime": _runtime(),
            }
        )
        content = result.content if isinstance(result, ToolMessage) else result
        assert "No matches found" in content

    async def test_aread_file(self):
        """Test async read_file tool."""
        files = {
            "/test.txt": FileData(
                content=["Hello world", "Line 2", "Line 3"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = await read_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "runtime": _runtime(),
            }
        )
        assert "Hello world" in result.content
        assert "Line 2" in result.content
        assert "Line 3" in result.content

    async def test_aread_file_with_offset(self):
        """Test async read_file tool with offset."""
        files = {
            "/test.txt": FileData(
                content=["Line 1", "Line 2", "Line 3", "Line 4"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = await read_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "offset": 1,
                "limit": 2,
                "runtime": _runtime(),
            }
        )
        assert "Line 2" in result.content
        assert "Line 3" in result.content
        assert "Line 1" not in result.content
        assert "Line 4" not in result.content

    async def test_awrite_file(self):
        """Test async write_file tool."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_file_tool = next(tool for tool in middleware.tools if tool.name == "write_file")
        result = await write_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "content": "Hello world",
                "runtime": ToolRuntime(state={}, context=None, tool_call_id="tc1", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert isinstance(result, ToolMessage)
        assert mem_store.get(("filesystem",), "/test.txt") is not None

    async def test_aedit_file(self):
        """Test async edit_file tool."""
        files = {
            "/test.txt": FileData(
                content=["Hello world", "Goodbye world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, mem_store = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        edit_file_tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
        result = await edit_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "old_string": "Hello",
                "new_string": "Hi",
                "runtime": ToolRuntime(state={}, context=None, tool_call_id="tc2", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert isinstance(result, ToolMessage)
        assert mem_store.get(("filesystem",), "/test.txt") is not None

    async def test_aedit_file_replace_all(self):
        """Test async edit_file tool with replace_all."""
        files = {
            "/test.txt": FileData(
                content=["Hello world", "Hello again"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, mem_store = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        edit_file_tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
        result = await edit_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "old_string": "Hello",
                "new_string": "Hi",
                "replace_all": True,
                "runtime": ToolRuntime(state={}, context=None, tool_call_id="tc3", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert isinstance(result, ToolMessage)
        assert mem_store.get(("filesystem",), "/test.txt") is not None

    async def test_aexecute_tool_returns_error_when_backend_doesnt_support(self):
        """Test async execute tool returns friendly error instead of raising exception."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)

        # Find the execute tool
        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")

        # Create runtime with StoreBackend
        runtime = ToolRuntime(
            state={},
            context=None,
            tool_call_id="test_exec",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        # Execute should return error message, not raise exception
        result = await execute_tool.ainvoke({"command": "ls -la", "runtime": runtime})

        assert isinstance(result, ToolMessage)
        assert "Error: Execution not available" in result.content
        assert "does not support command execution" in result.content

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

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = await execute_tool.ainvoke({"command": "echo hello", "timeout": 0, "runtime": rt})

        assert "async ok" in result.content
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

        backend = FormattingMockSandboxBackend()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = await execute_tool.ainvoke({"command": "echo test", "runtime": rt})

        assert "Async Hello world\nAsync Line 2" in result.content
        assert "succeeded" in result.content
        assert "exit code 0" in result.content

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

        backend = FailureMockSandboxBackend()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = await execute_tool.ainvoke({"command": "nonexistent", "runtime": rt})

        assert "Async Error: command not found" in result.content
        assert "failed" in result.content
        assert "exit code 127" in result.content

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

        backend = TruncatedMockSandboxBackend()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = await execute_tool.ainvoke({"command": "cat large_file", "runtime": rt})

        assert "Async Very long output..." in result.content
        assert "truncated" in result.content
