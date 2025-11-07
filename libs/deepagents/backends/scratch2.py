"""Simplified filesystem middleware using explicit runtime parameters."""

import abc
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime

from deepagents.backends.protocol import EditResult


class FileData(TypedDict):
    """Structure for individual file data in state."""

    content: list[str]
    created_at: str
    modified_at: str


class FileState(TypedDict):
    """State schema for file storage."""

    files: dict[str, FileData]


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
            return (
                f"Error: Multiple matches found ({count}). "
                "Use replace_all=True or provide more context."
            )
        new_content = content.replace(old_string, new_string, 1)
        occurrences = 1

    if occurrences == 0:
        return f"Error: String not found: {old_string}"

    return new_content, occurrences


class FilesystemMiddleware(AgentMiddleware):
    """Base middleware for file operations."""

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


class StateFilesystemMiddleware(FilesystemMiddleware):
    """Store files in LangGraph agent state (ephemeral)."""

    state_schema = FileState

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
        *,
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


class LocalFilesystemMiddleware(FilesystemMiddleware):
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


class CompositeFilesystemMiddleware(FilesystemMiddleware):
    """Route operations to different middlewares based on path prefix."""

    def __init__(
        self,
        default: FilesystemMiddleware,
        routes: dict[str, FilesystemMiddleware],
    ):
        self.default = default
        self.routes = routes
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

        # Register children so framework can collect state schemas and lifecycle hooks
        self.children = [default, *routes.values()]

    def _route(self, path: str) -> tuple[FilesystemMiddleware, str]:
        """Find middleware for path and strip prefix."""
        for prefix, middleware in self.sorted_routes:
            if path.startswith(prefix):
                suffix = path[len(prefix) :]
                stripped = f"/{suffix}" if suffix else "/"
                return middleware, stripped
        return self.default, path

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime,
        replace_all: bool = False,
    ) -> EditResult:
        middleware, stripped = self._route(file_path)
        return middleware.edit(stripped, old_string, new_string, runtime, replace_all)
