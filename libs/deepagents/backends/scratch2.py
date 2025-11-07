"""Simplified backend abstraction using explicit runtime parameters.

Backends provide LLM-optimized interfaces for filesystem and shell operations.
"""

import abc
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.protocol import EditResult


class FileData(TypedDict):
    """Structure for individual file data in state."""

    content: list[str]
    created_at: str
    modified_at: str


class FileState(TypedDict):
    """State schema for file storage."""

    files: dict[str, FileData]


def _create_edit_tool(backend: "Backend", replace_all: bool = False):
    """Helper to create an edit tool for a backend."""
    from langchain_core.tools import StructuredTool

    def _edit_tool(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
    ) -> EditResult:
        return backend.edit(
            file_path,
            old_string,
            new_string,
            runtime,
            replace_all=replace_all,
        )

    name = "edit_file_all" if replace_all else "edit_file"
    description = (
        "Edit a file by replacing all occurrences of a string." if replace_all else "Edit a file by replacing a single occurrence of a string."
    )

    return StructuredTool.from_function(
        func=_edit_tool,
        name=name,
        description=description,
    )


# This solution relies on inherting a default implementation of tools from the base Backend class.
class Backend(AgentMiddleware):
    """Base backend providing LLM-optimized filesystem and shell interfaces."""

    # Option A:
    # Not super nice since users need to remember to call super().__init__()
    # def __init__(self):
    #     self.tools = [
    #         _create_edit_tool(backend=self, replace_all=True),
    #     ]

    # Option B:
    # We'd need to figure out whether we can make this work nicely w/ typing
    # in langchain 1.0. This is a general disadvantage of using classvars in frameworks
    # they make some things convenient, but make dynamic behavior harder.
    @property  # <-- This doesn't place nicely with our classvar likely?
    def tools(self):
        """Create tools bound to this instance on first access (lazy)."""
        # you can add as many base tools as you want here
        return [
            _create_edit_tool(backend=self, replace_all=True),
            # ... create other tools
        ]

    @abc.abstractmethod
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit file by replacing strings."""
        ...


class ToolTruncationMiddledare(AgentMiddleware):
    def __init__(self, backend: Backend):
        self.backend = backend

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Insert logic for truncation"""
        # Logic for truncation goes here and can use the `self.backend` for
        # persisting tool result to the filesystem or state.


class StateBackend(Backend):
    """Store files in LangGraph agent state (ephemeral)."""

    state_schema = FileState

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
        replace_all: bool = False,
    ) -> EditResult:
        files = runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = "\n".join(file_data.get("content", []))

        result = _perform_string_replacement(content, old_string, new_string, replace_all)
        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        file_data["content"] = new_content.splitlines()
        file_data["modified_at"] = datetime.now(timezone.utc).isoformat()
        runtime.state["files"][file_path] = file_data

        return EditResult(
            path=file_path,
            files_update={file_path: file_data},
            occurrences=occurrences,
        )


class FilesystemBackend(Backend):
    """Read and write files directly from local filesystem."""

    def __init__(self, root_dir: str | None = None):
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()

    def _resolve_path(self, key: str):
        path = Path(key)
        return path if path.is_absolute() else (self.cwd / path).resolve()

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
        replace_all: bool = False,
    ) -> EditResult:
        resolved = self._resolve_path(file_path)

        if not resolved.exists() or not resolved.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return EditResult(error=f"Error reading file: {e}")

        result = _perform_string_replacement(content, old_string, new_string, replace_all)
        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result

        try:
            resolved.write_text(new_content, encoding="utf-8")
            return EditResult(path=file_path, files_update=None, occurrences=occurrences)
        except (OSError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error writing file: {e}")


class CompositeBackend(Backend):
    """Route operations to different backends based on path prefix."""

    def __init__(
        self,
        default: Backend,
        routes: dict[str, Backend],
    ):
        self.default = default
        self.routes = routes
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

        # Add children so framework can collect state schemas and lifecycle hooks
        # THERE IS A SEMANTIC ISSUE WITH CHILDREN DUE TO TOOLS.
        # Composite backends overrides all the tools, we don't want
        # to use the default tools of the children backends.
        # The problem is that the children life cycle hooks (e.g., wrap_model_call)
        # may assume that their tools are available.
        # It's unclear whether this would lead to issues or not, but definitely
        # looks like potential unexpected behavior.
        self.children = [default, *routes.values()]

    def _route(self, path: str) -> tuple[Backend, str]:
        """Find backend for path and strip prefix."""
        for prefix, backend in self.sorted_routes:
            if path.startswith(prefix):
                suffix = path[len(prefix) :]
                stripped = f"/{suffix}" if suffix else "/"
                return backend, stripped
        return self.default, path

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
        replace_all: bool = False,
    ) -> EditResult:
        backend, stripped = self._route(file_path)
        return backend.edit(stripped, old_string, new_string, runtime, replace_all)


# Helper function not part of abstraction
def _perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> str | tuple[str, int]:
    """Perform string replacement in content.

    Returns:
        Error string if replacement fails, or tuple of (new_content, occurrences) on success.
    """
    if replace_all:
        new_content = content.replace(old_string, new_string)
        occurrences = content.count(old_string)
    else:
        count = content.count(old_string)
        if count == 0:
            return f"Error: String not found: {old_string}"
        if count > 1:
            return f"Error: Multiple matches found ({count}). Use replace_all=True or provide more context."
        new_content = content.replace(old_string, new_string, 1)
        occurrences = 1

    if occurrences == 0:
        return f"Error: String not found: {old_string}"

    return new_content, occurrences
