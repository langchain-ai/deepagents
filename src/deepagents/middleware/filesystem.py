"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

import os
from datetime import UTC, datetime
from typing import Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.memory.protocol import MemoryBackend
from deepagents.memory.backends import StateBackend, CompositeBackend

MEMORIES_PREFIX = "/memories/"
EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 2000


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """Merge file updates with support for deletions.

    This reducer enables file deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    """Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`) or does
            not start with an allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _format_content_with_line_numbers(
    content: str | list[str],
    *,
    format_style: Literal["pipe", "tab"] = "pipe",
    start_line: int = 1,
) -> str:
    r"""Format file content with line numbers for display.

    Converts file content to a numbered format similar to `cat -n` output,
    with support for two different formatting styles.

    Args:
        content: File content as a string or list of lines.
        format_style: Format style for line numbers:
            - `"pipe"`: Compact format like `"1|content"`
            - `"tab"`: Right-aligned format like `"     1\tcontent"` (lines truncated at 2000 chars)
        start_line: Starting line number (default: 1).

    Returns:
        Formatted content with line numbers prepended to each line.

    Example:
        ```python
        content = "Hello\nWorld"
        format_content_with_line_numbers(content, format_style="pipe")
        # Returns: "1|Hello\n2|World"

        format_content_with_line_numbers(content, format_style="tab", start_line=10)
        # Returns: "    10\tHello\n    11\tWorld"
        ```
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    if format_style == "pipe":
        return "\n".join(f"{i + start_line}|{line}" for i, line in enumerate(lines))

    return "\n".join(f"{i + start_line:{LINE_NUMBER_WIDTH}d}\t{line[:MAX_LINE_LENGTH]}" for i, line in enumerate(lines))


def _create_file_data(
    content: str | list[str],
    *,
    created_at: str | None = None,
) -> FileData:
    r"""Create a FileData object with automatic timestamp generation.

    Args:
        content: File content as a string or list of lines.
        created_at: Optional creation timestamp in ISO 8601 format.
            If `None`, uses the current UTC time.

    Returns:
        FileData object with content and timestamps.

    Example:
        ```python
        file_data = create_file_data("Hello\nWorld")
        # Returns: {"content": ["Hello", "World"], "created_at": "2024-...",
        #           "modified_at": "2024-..."}
        ```
    """
    lines = content.split("\n") if isinstance(content, str) else content
    # Split any lines exceeding MAX_LINE_LENGTH into chunks
    lines = [line[i:i+MAX_LINE_LENGTH] for line in lines for i in range(0, len(line) or 1, MAX_LINE_LENGTH)]
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def _update_file_data(
    file_data: FileData,
    content: str | list[str],
) -> FileData:
    """Update FileData with new content while preserving creation timestamp.

    Args:
        file_data: Existing FileData object to update.
        content: New file content as a string or list of lines.

    Returns:
        Updated FileData object with new content and updated `modified_at`
        timestamp. The `created_at` timestamp is preserved from the original.

    Example:
        ```python
        original = create_file_data("Hello")
        updated = update_file_data(original, "Hello World")
        # updated["created_at"] == original["created_at"]
        # updated["modified_at"] > original["modified_at"]
        ```
    """
    lines = content.split("\n") if isinstance(content, str) else content
    # Split any lines exceeding MAX_LINE_LENGTH into chunks
    lines = [line[i:i+MAX_LINE_LENGTH] for line in lines for i in range(0, len(line) or 1, MAX_LINE_LENGTH)]
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def _file_data_to_string(file_data: FileData) -> str:
    r"""Convert FileData to plain string content.

    Joins the lines stored in FileData with newline characters to produce
    a single string representation of the file content.

    Args:
        file_data: FileData object containing lines of content.

    Returns:
        File content as a single string with lines joined by newlines.

    Example:
        ```python
        file_data = {
            "content": ["Hello", "World"],
            "created_at": "...",
            "modified_at": "...",
        }
        file_data_to_string(file_data)  # Returns: "Hello\nWorld"
        ```
    """
    return "\n".join(file_data["content"])


def _check_empty_content(content: str) -> str | None:
    """Check if file content is empty and return a warning message.

    Args:
        content: File content to check.

    Returns:
        Warning message string if content is empty or contains only whitespace,
        `None` otherwise.

    Example:
        ```python
        check_empty_content("")  # Returns: "System reminder: File exists but has empty contents"
        check_empty_content("   ")  # Returns: "System reminder: File exists but has empty contents"
        check_empty_content("Hello")  # Returns: None
        ```
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None




class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the filesystem, optionally filtering by directory.

Usage:
- The list_files tool will return a list of all files in the filesystem.
- You can optionally provide a path parameter to list files in a specific directory.
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools."""
LIST_FILES_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = f"\n- Files from the longterm filesystem will be prefixed with the {MEMORIES_PREFIX} path."

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it."""
READ_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = f"\n- file_paths prefixed with the {MEMORIES_PREFIX} path will be read from the longterm filesystem."

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""
EDIT_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = (
    f"\n- You can edit files in the longterm filesystem by prefixing the filename with the {MEMORIES_PREFIX} path."
)

WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- The content parameter must be a string
- The write_file tool will create the a new file.
- Prefer to edit existing files over creating new ones when possible.
- file_paths prefixed with the /memories/ path will be written to the longterm filesystem."""
WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = (
    f"\n- file_paths prefixed with the {MEMORIES_PREFIX} path will be written to the longterm filesystem."
)

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list all files in the filesystem
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem"""
FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT = f"""

You also have access to a longterm filesystem in which you can store files that you want to keep around for longer than the current conversation.
In order to interact with the longterm filesystem, you can use those same tools, but filenames must be prefixed with the {MEMORIES_PREFIX} path.
Remember, to interact with the longterm filesystem, you must prefix the filename with the {MEMORIES_PREFIX} path."""


def _ls_tool_generator(
    backend: MemoryBackend,
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the ls (list files) tool.

    Args:
        backend: Backend to use for file storage.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured ls tool that lists files using the backend.
    """
    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def ls(runtime: ToolRuntime[None, FilesystemState], path: str | None = None) -> list[str]:
        prefix = _validate_path(path) if path is not None else None
        files = backend.ls(prefix)
        return files

    return ls


def _read_file_tool_generator(
    backend: MemoryBackend,
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the read_file tool.

    Args:
        backend: Backend to use for file storage.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured read_file tool that reads files using the backend.
    """
    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    def _read_file_data_content(file_data: FileData, offset: int, limit: int) -> str:
        """Read and format file content with line numbers.

        Args:
            file_data: The file data to read.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or an error message.
        """
        content = _file_data_to_string(file_data)
        empty_msg = _check_empty_content(content)
        if empty_msg:
            return empty_msg
        lines = content.splitlines()
        start_idx = offset
        end_idx = min(start_idx + limit, len(lines))
        if start_idx >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
        selected_lines = lines[start_idx:end_idx]
        return _format_content_with_line_numbers(selected_lines, format_style="tab", start_line=start_idx + 1)

    @tool(description=tool_description)
    def read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        file_path = _validate_path(file_path)
        file_data = backend.get(file_path)
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        return _read_file_data_content(file_data, offset, limit)

    return read_file


def _write_file_tool_generator(
    backend: MemoryBackend,
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the write_file tool.

    Args:
        backend: Backend to use for file storage.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured write_file tool that creates new files using the backend.
    """
    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        file_path = _validate_path(file_path)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for write_file invocation"
            raise ValueError(value_error_msg)
        
        existing = backend.get(file_path)
        if existing:
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        new_file_data = _create_file_data(content)
        result = backend.put(file_path, new_file_data)
        
        if isinstance(result, Command):
            return result
        elif isinstance(result, str):
            return result
        else:
            return f"Updated file {file_path}"

    return write_file


def _edit_file_tool_generator(
    backend: MemoryBackend,
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the edit_file tool.

    Args:
        backend: Backend to use for file storage.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured edit_file tool that performs string replacements in files using the backend.
    """
    tool_description = custom_description or EDIT_FILE_TOOL_DESCRIPTION

    def _perform_file_edit(
        file_data: FileData,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> tuple[FileData, str] | str:
        """Perform string replacement on file data.

        Args:
            file_data: The file data to edit.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences.

        Returns:
            Tuple of (updated_file_data, success_message) on success,
            or error string on failure.
        """
        content = _file_data_to_string(file_data)
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"
        if occurrences > 1 and not replace_all:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        new_content = content.replace(old_string, new_string)
        new_file_data = _update_file_data(file_data, new_content)
        result_msg = f"Successfully replaced {occurrences} instance(s) of the string"
        return new_file_data, result_msg

    @tool(description=tool_description)
    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        file_path = _validate_path(file_path)
        file_data = backend.get(file_path)
        if file_data is None:
            return f"Error: File '{file_path}' not found"

        result = _perform_file_edit(file_data, old_string, new_string, replace_all=replace_all)
        if isinstance(result, str):
            return result

        new_file_data, result_msg = result
        full_msg = f"{result_msg} in '{file_path}'"

        put_result = backend.put(file_path, new_file_data)
        
        if isinstance(put_result, Command):
            return Command(
                update={
                    **put_result.update,
                    "messages": [ToolMessage(full_msg, tool_call_id=runtime.tool_call_id)],
                }
            )
        else:
            return full_msg

    return edit_file


TOOL_GENERATORS = {
    "ls": _ls_tool_generator,
    "read_file": _read_file_tool_generator,
    "write_file": _write_file_tool_generator,
    "edit_file": _edit_file_tool_generator,
}


def _get_filesystem_tools(
    backend: MemoryBackend,
    custom_tool_descriptions: dict[str, str] | None = None,
) -> list[BaseTool]:
    """Get filesystem tools.

    Args:
        backend: Backend to use for file storage.
        custom_tool_descriptions: Optional custom descriptions for tools.

    Returns:
        List of configured filesystem tools (ls, read_file, write_file, edit_file).
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []
    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
        tools.append(tool)
    return tools


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
You can do this by specifying an offset and limit in the read_file tool call.
For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here are the first 10 lines of the result:
{content_sample}
"""


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem tools to an agent.

    This middleware adds four filesystem tools to the agent: ls, read_file, write_file,
    and edit_file. Files can be stored in two locations:
    - Short-term: In the agent's state (ephemeral, lasts only for the conversation)
    - Long-term: In a persistent store (persists across conversations when enabled)

    Args:
        backend: Optional backend for file storage. If not provided, defaults to StateBackend.
        long_term_backend: Optional backend for /memories/ files. If provided, creates CompositeBackend
            with StateBackend as default and long_term_backend for /memories/ prefix.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.

    Example:
        ```python
        from langchain.agents.middleware.filesystem import FilesystemMiddleware
        from langchain.agents import create_agent

        # Short-term memory only
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # With long-term memory
        from deepagents.memory.backends import StoreBackend
        agent = create_agent(middleware=[FilesystemMiddleware(long_term_backend=StoreBackend())])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: MemoryBackend | None = None,
        long_term_backend: MemoryBackend | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            backend: Optional backend for file storage. If provided, uses it directly.
            long_term_backend: Optional backend for /memories/ files. If provided, creates CompositeBackend.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.
        """
        self.tool_token_limit_before_evict = tool_token_limit_before_evict
        
        if backend is not None:
            self.backend = backend
        elif long_term_backend is not None:
            self.backend = CompositeBackend(
                default=StateBackend(),
                routes={MEMORIES_PREFIX: long_term_backend}
            )
        else:
            self.backend = StateBackend()
        
        self.system_prompt = FILESYSTEM_SYSTEM_PROMPT
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif isinstance(self.backend, CompositeBackend):
            self.system_prompt += FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT

        self.tools = _get_filesystem_tools(self.backend, custom_tool_descriptions)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt to include instructions on using the filesystem.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt to include instructions on using the filesystem.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return await handler(request)

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage) and isinstance(tool_result.content, str):
            content = tool_result.content
            if self.tool_token_limit_before_evict and len(content) > 4 * self.tool_token_limit_before_evict:
                file_path = f"/large_tool_results/{tool_result.tool_call_id}"
                file_data = _create_file_data(content)
                state_update = {
                    "messages": [
                        ToolMessage(
                            TOO_LARGE_TOOL_MSG.format(
                                tool_call_id=tool_result.tool_call_id,
                                file_path=file_path,
                                content_sample=_format_content_with_line_numbers(file_data["content"][:10], format_style="tab", start_line=1),
                            ),
                            tool_call_id=tool_result.tool_call_id,
                        )
                    ],
                    "files": {file_path: file_data},
                }
                return Command(update=state_update)
        elif isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            message_updates = update.get("messages", [])
            file_updates = update.get("files", {})

            edited_message_updates = []
            for message in message_updates:
                if self.tool_token_limit_before_evict and isinstance(message, ToolMessage) and isinstance(message.content, str):
                    content = message.content
                    if len(content) > 4 * self.tool_token_limit_before_evict:
                        file_path = f"/large_tool_results/{message.tool_call_id}"
                        file_data = _create_file_data(content)
                        edited_message_updates.append(
                            ToolMessage(
                                TOO_LARGE_TOOL_MSG.format(
                                    tool_call_id=message.tool_call_id,
                                    file_path=file_path,
                                    content_sample=_format_content_with_line_numbers(file_data["content"][:10], format_style="tab", start_line=1),
                                ),
                                tool_call_id=message.tool_call_id,
                            )
                        )
                        file_updates[file_path] = file_data
                        continue
                edited_message_updates.append(message)
            return Command(update={**update, "messages": edited_message_updates, "files": file_updates})
        return tool_result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return await handler(request)

        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result)
