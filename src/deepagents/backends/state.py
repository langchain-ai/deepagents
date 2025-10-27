"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

from typing import Literal

from deepagents.backends.protocol import Backend, EditResult, WriteResult
from deepagents.backends.utils import (
    _glob_search_files,
    _grep_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    truncate_if_too_long,
    update_file_data,
)


class StateBackend(Backend):
    """Backend that stores files in agent state (ephemeral).

    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.

    Storage Model: Checkpoint Storage
    ---------------------------------
    This backend stores files in LangGraph state, persisted via LangGraph's checkpoint
    system (Postgres, Redis, in-memory, etc.). Write and edit operations return
    WriteResult/EditResult with files_update populated, which the tool layer converts
    into Command objects for state updates.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """Initialize StateBackend with runtime.

        Args:
            runtime: Tool runtime with access to LangGraph state
        """
        self.runtime = runtime

    def ls(self, path: str) -> list[str]:
        """List files from state.

        Args:
            path: Absolute path to directory.

        Returns:
            List of file paths.
        """
        files = self.runtime.state.get("files", {})
        keys = list(files.keys())
        keys = [k for k in keys if k.startswith(path)]
        return truncate_if_too_long(keys)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to readReturns:
            Formatted file content with line numbers, or error message.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return f"Error: File '{file_path}' not found"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Absolute file path
            content: File content as a string

        Returns:
            WriteResult with files_update populated for framework state update.
        """
        files = self.runtime.state.get("files", {})

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)

        return WriteResult(path=file_path, files_update={file_path: new_file_data})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences

        Returns:
            EditResult with files_update populated for framework state update.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            # Error message from perform_string_replacement
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        return EditResult(path=file_path, files_update={file_path: new_file_data}, occurrences=occurrences)

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"Returns:
            Formatted search results based on output_mode.
        """
        files = self.runtime.state.get("files", {})

        return truncate_if_too_long(_grep_search_files(files, pattern, path, glob, output_mode))

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")Returns:
            List of absolute file paths matching the pattern.
        """
        files = self.runtime.state.get("files", {})

        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        return truncate_if_too_long(result.split("\n"))
