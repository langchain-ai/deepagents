"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

import asyncio
import concurrent.futures
import contextvars
import mimetypes
import uuid
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, cast

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import (
    BACKEND_TYPES as BACKEND_TYPES,  # Re-export type here for backwards compatibility
    BackendProtocol,
    EditResult,
    FileData as FileData,  # Re-export for backwards compatibility
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    execute_accepts_timeout,
)
from deepagents.backends.utils import (
    _get_file_type,
    check_empty_content,
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
    validate_path,
)
from deepagents.middleware._utils import append_to_system_message

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
GLOB_TIMEOUT = 20.0  # seconds
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100
# Template for truncation message in read_file
# {file_path} will be filled in at runtime
READ_FILE_TRUNCATION_MSG = (
    "\n\n[Output was truncated due to size limits. "
    "The file content is very large. "
    "Consider reformatting the file to make it easier to navigate. "
    "For example, if this is JSON, use execute(command='jq . {file_path}') to pretty-print it with line breaks. "
    "For other formats, you can use appropriate formatting tools to split long lines.]"
)

# Approximate number of characters per token for truncation calculations.
# Using 4 chars per token as a conservative approximation (actual ratio varies by content)
# This errs on the high side to avoid premature eviction of content that might fit
NUM_CHARS_PER_TOKEN = 4


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


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


class LsSchema(BaseModel):
    """Input schema for the `ls` tool."""

    path: str = Field(description="Absolute path to the directory to list. Must be absolute, not relative.")


class ReadFileSchema(BaseModel):
    """Input schema for the `read_file` tool."""

    file_path: str = Field(description="Absolute path to the file to read. Must be absolute, not relative.")
    offset: int = Field(
        default=DEFAULT_READ_OFFSET,
        description="Line number to start reading from (0-indexed). Use for pagination of large files.",
    )
    limit: int = Field(
        default=DEFAULT_READ_LIMIT,
        description="Maximum number of lines to read. Use for pagination of large files.",
    )


class WriteFileSchema(BaseModel):
    """Input schema for the `write_file` tool."""

    file_path: str = Field(description="Absolute path where the file should be created. Must be absolute, not relative.")
    content: str = Field(description="The text content to write to the file. This parameter is required.")


class EditFileSchema(BaseModel):
    """Input schema for the `edit_file` tool."""

    file_path: str = Field(description="Absolute path to the file to edit. Must be absolute, not relative.")
    old_string: str = Field(description="The exact text to find and replace. Must be unique in the file unless replace_all is True.")
    new_string: str = Field(description="The text to replace old_string with. Must be different from old_string.")
    replace_all: bool = Field(
        default=False,
        description="If True, replace all occurrences of old_string. If False (default), old_string must be unique.",
    )


class GlobSchema(BaseModel):
    """Input schema for the `glob` tool."""

    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md').")
    path: str = Field(default="/", description="Base directory to search from. Defaults to root '/'.")


class GrepSchema(BaseModel):
    """Input schema for the `grep` tool."""

    pattern: str = Field(description="Text pattern to search for (literal string, not regex).")
    path: str | None = Field(default=None, description="Directory to search in. Defaults to current working directory.")
    glob: str | None = Field(default=None, description="Glob pattern to filter which files to search (e.g., '*.py').")
    output_mode: Literal["files_with_matches", "content", "count"] = Field(
        default="files_with_matches",
        description="Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
    )


class ExecuteSchema(BaseModel):
    """Input schema for the `execute` tool."""

    command: str = Field(description="Shell command to execute in the sandbox environment.")
    timeout: int | None = Field(
        default=None,
        description="Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
    )


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in a directory.

This is useful for exploring the filesystem and finding the right file to read or edit.
You should almost ALWAYS use this tool before using the read_file or edit_file tools."""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem.

Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- By default, it reads up to 100 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, etc.), audio and video files, and PDFs are returned as multimodal content blocks (see https://docs.langchain.com/oss/python/langchain/messages#multimodal).

For multimodal reads (image, audio, video, PDF, etc.):
- Use `read_file(file_path=...)`
- Do NOT use `offset`/`limit` for images (pagination is text-only)
- If file details were compacted from history, call `read_file` again on the same path

- You should ALWAYS make sure a file has been read before editing it."""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
- When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
- ALWAYS prefer editing existing files over creating new ones.
- Only use emojis if the user explicitly requests it."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The write_file tool will create the a new file.
- Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
"""

GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
Returns a list of absolute file paths that match the pattern.

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir"""

GREP_TOOL_DESCRIPTION = """Search for a text pattern across files.

Searches for literal text (not regex) and returns matching files or content based on output_mode.
Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`
- Search for code with special chars: `grep(pattern="def __init__(self):")`"""

EXECUTE_TOOL_DESCRIPTION = """Executes a shell command in an isolated sandbox environment.

Usage:
Executes a given command in the sandbox environment with proper handling and security measures.
Before executing the command, please follow these steps:
1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory
2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command
   - Capture the output of the command
Usage notes:
  - Commands run in an isolated sandbox environment
  - Returns combined stdout/stderr output with exit code
  - If the output is very large, it may be truncated
  - For long-running commands, use the optional timeout parameter to override the default timeout (e.g., execute(command="make build", timeout=300))
  - A timeout of 0 may disable timeouts on backends that support no-timeout execution
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")
    - execute(command="make build", timeout=300)

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE = """## Following Conventions

- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /. Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files

## Large Tool Results

When a tool result is too large, it may be offloaded into the filesystem instead of being returned inline. In those cases, use `read_file` to inspect the saved result in chunks, or use `grep` within `{large_tool_results_prefix}/` if you need to search across offloaded tool results and do not know the exact file path. Offloaded tool results are stored under `{large_tool_results_prefix}/<tool_call_id>`."""

FILESYSTEM_SYSTEM_PROMPT = _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(
    large_tool_results_prefix="/large_tool_results",
)

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


def supports_execution(backend: BackendProtocol) -> bool:
    """Check if a backend supports command execution.

    For CompositeBackend, checks if the default backend supports execution.
    For other backends, checks if they implement SandboxBackendProtocol.

    Args:
        backend: The backend to check.

    Returns:
        True if the backend supports execution, False otherwise.
    """
    # For CompositeBackend, check the default backend
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # For other backends, use isinstance check
    return isinstance(backend, SandboxBackendProtocol)


# Tools that should be excluded from the large result eviction logic.
#
# This tuple contains tools that should NOT have their results evicted to the filesystem
# when they exceed token limits. Tools are excluded for different reasons:
#
# 1. Tools with built-in truncation (ls, glob, grep):
#    These tools truncate their own output when it becomes too large. When these tools
#    produce truncated output due to many matches, it typically indicates the query
#    needs refinement rather than full result preservation. In such cases, the truncated
#    matches are potentially more like noise and the LLM should be prompted to narrow
#    its search criteria instead.
#
# 2. Tools with problematic truncation behavior (read_file):
#    read_file is tricky to handle as the failure mode here is single long lines
#    (e.g., imagine a jsonl file with very long payloads on each line). If we try to
#    truncate the result of read_file, the agent may then attempt to re-read the
#    truncated file using read_file again, which won't help.
#
# 3. Tools that never exceed limits (edit_file, write_file):
#    These tools return minimal confirmation messages and are never expected to produce
#    output large enough to exceed token limits, so checking them would be unnecessary.
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""

TOO_LARGE_HUMAN_MSG = """Message content too large and was saved to the filesystem at: {file_path}

You can read the full content using the read_file tool with pagination (offset and limit parameters).

Here is a preview showing the head and tail of the content:

{content_sample}
"""


def _build_evicted_human_content(
    message: HumanMessage,
    replacement_text: str,
) -> str | list[ContentBlock]:
    """Build replacement content for an evicted HumanMessage, preserving non-text blocks.

    For plain string content, returns the replacement text directly. For list content
    with mixed block types (e.g., text + image), replaces all text blocks with a single
    text block containing the replacement text while keeping non-text blocks intact.

    Args:
        message: The original HumanMessage being evicted.
        replacement_text: The truncation notice and preview text.

    Returns:
        Replacement content: a string or list of content blocks.
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]


def _build_truncated_human_message(message: HumanMessage, file_path: str) -> HumanMessage:
    """Build a truncated HumanMessage for the model request.

    Computes a preview from the full content still in state and returns a
    lightweight replacement the model will see. Pure string computation — no
    backend I/O.

    Args:
        message: The original HumanMessage (full content in state).
        file_path: The backend path where the content was evicted.

    Returns:
        A new HumanMessage with truncated content and the same `id`.
    """
    content_str = _extract_text_from_message(message)
    content_sample = _create_content_preview(content_str)
    replacement_text = TOO_LARGE_HUMAN_MSG.format(
        file_path=file_path,
        content_sample=content_sample,
    )
    evicted = _build_evicted_human_content(message, replacement_text)
    return message.model_copy(update={"content": evicted})


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> str:
    """Create a preview of content showing head and tail with truncation marker.

    Args:
        content_str: The full content string to preview.
        head_lines: Number of lines to show from the start.
        tail_lines: Number of lines to show from the end.

    Returns:
        Formatted preview string with line numbers.
    """
    lines = content_str.splitlines()

    if len(lines) <= head_lines + tail_lines:
        # If file is small enough, show all lines
        preview_lines = [line[:1000] for line in lines]
        return format_content_with_line_numbers(preview_lines, start_line=1)

    # Show head and tail with truncation marker
    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return head_sample + truncation_notice + tail_sample


def _extract_text_from_message(message: BaseMessage) -> str:
    """Extract text from a message using its `content_blocks` property.

    Joins all text content blocks and ignores non-text blocks (images, audio, etc.)
    so that binary payloads don't inflate the size measurement.

    Args:
        message: The BaseMessage to extract text from.

    Returns:
        Joined text from all text content blocks, or stringified content as fallback.
    """
    texts = [block["text"] for block in message.content_blocks if block["type"] == "text"]
    return "\n".join(texts)


def _build_evicted_content(message: ToolMessage, replacement_text: str) -> str | list[ContentBlock]:
    """Build replacement content for an evicted message, preserving non-text blocks.

    For plain string content, returns the replacement text directly. For list content
    with mixed block types (e.g., text + image), replaces all text blocks with a single
    text block containing the replacement text while keeping non-text blocks intact.

    Args:
        message: The original ToolMessage being evicted.
        replacement_text: The truncation notice and preview text.

    Returns:
        Replacement content: a string or list of content blocks.
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        # All content is text, so a plain string replacement is sufficient.
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]


class FilesystemMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
    """Middleware for providing filesystem and optional execution tools to an agent.

    This middleware adds filesystem tools to the agent: `ls`, `read_file`, `write_file`,
    `edit_file`, `glob`, and `grep`.

    Files can be stored using any backend that implements the `BackendProtocol`.

    If the backend implements `SandboxBackendProtocol`, an `execute` tool is also added
    for running shell commands.

    This middleware also automatically evicts large tool results to the file system when
    they exceed a token threshold, preventing context window saturation.

    Args:
        backend: Backend for file storage and optional execution.

            If not provided, defaults to `StateBackend` (ephemeral storage in agent state).

            For persistent storage or hybrid setups, use `CompositeBackend` with custom routes.

            For execution support, use a backend that implements `SandboxBackendProtocol`.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        tool_token_limit_before_evict: Token limit before evicting a tool result to the
            filesystem.

            When exceeded, writes the result using the configured backend and replaces it
            with a truncated preview and file reference.

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # Ephemeral storage only (default, no execution)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # With hybrid storage (ephemeral + persistent /memories/)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # With sandbox backend (supports execution)
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        human_message_token_limit_before_evict: int | None = 50000,
        max_execute_timeout: int = 3600,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            backend: Backend for file storage and optional execution, or a factory callable.
                Defaults to StateBackend if not provided.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.
            human_message_token_limit_before_evict: Optional token limit before
                evicting a HumanMessage to the filesystem.
            max_execute_timeout: Maximum allowed value in seconds for per-command timeout
                overrides on the execute tool.

                Defaults to 3600 seconds (1 hour). Any per-command timeout
                exceeding this value will be rejected with an error message.

        Raises:
            ValueError: If `max_execute_timeout` is not positive.
        """
        if max_execute_timeout <= 0:
            msg = f"max_execute_timeout must be positive, got {max_execute_timeout}"
            raise ValueError(msg)
        # Use provided backend or default to StateBackend instance
        self.backend = backend if backend is not None else StateBackend()

        artifacts_root = self.backend.artifacts_root if isinstance(self.backend, CompositeBackend) else "/"
        _root = artifacts_root.rstrip("/")
        self._large_tool_results_prefix = f"{_root}/large_tool_results"
        self._conversation_history_prefix = f"{_root}/conversation_history"

        # Store configuration (private - internal implementation details)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict
        self._human_message_token_limit_before_evict = human_message_token_limit_before_evict
        self._max_execute_timeout = max_execute_timeout

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            warnings.warn(
                "Passing a callable (factory) as `backend` is deprecated and "
                "will be removed in v0.7. Pass a `BackendProtocol` instance "
                "directly instead (e.g. `StateBackend()`).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.backend(runtime)  # ty: ignore[call-top-callable]
        return self.backend

    def _create_ls_tool(self) -> BaseTool:
        """Create the ls (list files) tool."""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> ToolMessage | str:
            """Synchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ls_result = resolved_backend.ls(validated_path)
            if ls_result.error:
                return f"Error: {ls_result.error}"
            infos = ls_result.entries or []
            paths = [fi.get("path", "") for fi in infos]
            return ToolMessage(
                content=str(truncate_if_too_long(paths)),
                artifact=ls_result,
                tool_call_id=runtime.tool_call_id,
                name="ls",
            )

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> ToolMessage | str:
            """Asynchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ls_result = await resolved_backend.als(validated_path)
            if ls_result.error:
                return f"Error: {ls_result.error}"
            infos = ls_result.entries or []
            paths = [fi.get("path", "") for fi in infos]
            return ToolMessage(
                content=str(truncate_if_too_long(paths)),
                artifact=ls_result,
                tool_call_id=runtime.tool_call_id,
                name="ls",
            )

        return StructuredTool.from_function(
            name="ls",
            description=tool_description,
            func=sync_ls,
            coroutine=async_ls,
            infer_schema=False,
            args_schema=LsSchema,
        )

    def _create_read_file_tool(self) -> BaseTool:  # noqa: C901
        """Create the read_file tool."""
        tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
        token_limit = self._tool_token_limit_before_evict

        def _truncate(content: str, file_path: str, limit: int) -> str:
            lines = content.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                content = "".join(lines)

            if token_limit and len(content) >= NUM_CHARS_PER_TOKEN * token_limit:
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=file_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                content = content[:max_content_length] + truncation_msg

            return content

        def _handle_read_result(
            read_result: ReadResult | str,
            validated_path: str,
            tool_call_id: str | None,
            offset: int,
            limit: int,
        ) -> ToolMessage | str:
            if isinstance(read_result, str):
                warnings.warn(
                    "Returning a plain `str` from `backend.read()` is deprecated. "
                    "Return a `ReadResult` instead. Returning `str` will not be "
                    "supported in v0.7.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Legacy backends already format with line numbers
                return _truncate(read_result, validated_path, limit)

            if read_result.error:
                return f"Error: {read_result.error}"

            if read_result.file_data is None:
                return f"Error: no data returned for '{validated_path}'"

            file_type = _get_file_type(validated_path)
            content = read_result.file_data["content"]

            if file_type != "text":
                mime_type = mimetypes.guess_type("file" + Path(validated_path).suffix)[0] or "application/octet-stream"
                return ToolMessage(
                    content_blocks=cast("list[ContentBlock]", [{"type": file_type, "base64": content, "mime_type": mime_type}]),
                    name="read_file",
                    tool_call_id=tool_call_id,
                    additional_kwargs={"read_file_path": validated_path, "read_file_media_type": mime_type},
                )

            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg

            content = format_content_with_line_numbers(content, start_line=offset + 1)
            # We apply truncation again after formatting content as continuation lines
            # can increase line count
            return _truncate(content, validated_path, limit)

        def sync_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | str:
            """Synchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            read_result = resolved_backend.read(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        async def async_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | str:
            """Asynchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            read_result = await resolved_backend.aread(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
            infer_schema=False,
            args_schema=ReadFileSchema,
        )

    def _create_write_file_tool(self) -> BaseTool:
        """Create the write_file tool."""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Synchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return res.error
            return f"Updated file {res.path}"

        async def async_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Asynchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            res: WriteResult = await resolved_backend.awrite(validated_path, content)
            if res.error:
                return res.error
            return f"Updated file {res.path}"

        return StructuredTool.from_function(
            name="write_file",
            description=tool_description,
            func=sync_write_file,
            coroutine=async_write_file,
            infer_schema=False,
            args_schema=WriteFileSchema,
        )

    def _create_edit_file_tool(self) -> BaseTool:
        """Create the edit_file tool."""
        tool_description = self._custom_tool_descriptions.get("edit_file") or EDIT_FILE_TOOL_DESCRIPTION

        def sync_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> str:
            """Synchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        async def async_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> str:
            """Asynchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            res: EditResult = await resolved_backend.aedit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
            infer_schema=False,
            args_schema=EditFileSchema,
        )

    def _create_glob_tool(self) -> BaseTool:
        """Create the glob tool."""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> ToolMessage | str:
            """Synchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: ctx.run(resolved_backend.glob, pattern, path=validated_path))
                try:
                    glob_result = future.result(timeout=GLOB_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
            if glob_result.error:
                return f"Error: {glob_result.error}"
            infos = glob_result.matches or []
            paths = [fi.get("path", "") for fi in infos]
            return ToolMessage(
                content=str(truncate_if_too_long(paths)),
                artifact=glob_result,
                tool_call_id=runtime.tool_call_id,
                name="glob",
            )

        async def async_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> ToolMessage | str:
            """Asynchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            try:
                glob_result = await asyncio.wait_for(
                    resolved_backend.aglob(pattern, path=validated_path),
                    timeout=GLOB_TIMEOUT,
                )
            except TimeoutError:
                return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
            if glob_result.error:
                return f"Error: {glob_result.error}"
            infos = glob_result.matches or []
            paths = [fi.get("path", "") for fi in infos]
            return ToolMessage(
                content=str(truncate_if_too_long(paths)),
                artifact=glob_result,
                tool_call_id=runtime.tool_call_id,
                name="glob",
            )

        return StructuredTool.from_function(
            name="glob",
            description=tool_description,
            func=sync_glob,
            coroutine=async_glob,
            infer_schema=False,
            args_schema=GlobSchema,
        )

    def _create_grep_tool(self) -> BaseTool:
        """Create the grep tool."""
        tool_description = self._custom_tool_descriptions.get("grep") or GREP_TOOL_DESCRIPTION

        def sync_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> ToolMessage | str:
            """Synchronous wrapper for grep tool."""
            if path is not None:
                try:
                    path = validate_path(path)
                except ValueError as e:
                    return f"Error: {e}"
            resolved_backend = self._get_backend(runtime)
            grep_result = resolved_backend.grep(pattern, path=path, glob=glob)
            if grep_result.error:
                return grep_result.error
            matches = grep_result.matches or []
            formatted = format_grep_matches(matches, output_mode)
            return ToolMessage(
                content=truncate_if_too_long(formatted),
                artifact={"result": grep_result, "output_mode": output_mode},
                tool_call_id=runtime.tool_call_id,
                name="grep",
            )

        async def async_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> ToolMessage | str:
            """Asynchronous wrapper for grep tool."""
            if path is not None:
                try:
                    path = validate_path(path)
                except ValueError as e:
                    return f"Error: {e}"
            resolved_backend = self._get_backend(runtime)
            grep_result = await resolved_backend.agrep(pattern, path=path, glob=glob)
            if grep_result.error:
                return grep_result.error
            matches = grep_result.matches or []
            formatted = format_grep_matches(matches, output_mode)
            return ToolMessage(
                content=truncate_if_too_long(formatted),
                artifact={"result": grep_result, "output_mode": output_mode},
                tool_call_id=runtime.tool_call_id,
                name="grep",
            )

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
            infer_schema=False,
            args_schema=GrepSchema,
        )

    def _create_execute_tool(self) -> BaseTool:  # noqa: C901
        """Create the execute tool for sandbox command execution."""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[
                int | None,
                "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
            ] = None,
        ) -> str:
            """Synchronous wrapper for execute tool."""
            if timeout is not None:
                if timeout < 0:
                    return f"Error: timeout must be non-negative, got {timeout}."
                if timeout > self._max_execute_timeout:
                    return f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s)."

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            # Safe cast: supports_execution validates that execute()/aexecute() exist
            # (either SandboxBackendProtocol or CompositeBackend with sandbox default)
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return (
                    "Error: This sandbox backend does not support per-command "
                    "timeout overrides. Update your sandbox package to the "
                    "latest version, or omit the timeout parameter."
                )
            try:
                result = executable.execute(command, timeout=timeout) if timeout is not None else executable.execute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        async def async_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
            # ASYNC109 - timeout is a semantic parameter forwarded to the
            # backend's implementation, not an asyncio.timeout() contract.
            timeout: Annotated[  # noqa: ASYNC109
                int | None,
                "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
            ] = None,
        ) -> str:
            """Asynchronous wrapper for execute tool."""
            if timeout is not None:
                if timeout < 0:
                    return f"Error: timeout must be non-negative, got {timeout}."
                if timeout > self._max_execute_timeout:
                    return f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s)."

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            # Safe cast: supports_execution validates that execute()/aexecute() exist
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return (
                    "Error: This sandbox backend does not support per-command "
                    "timeout overrides. Update your sandbox package to the "
                    "latest version, or omit the timeout parameter."
                )
            try:
                result = await executable.aexecute(command, timeout=timeout) if timeout is not None else await executable.aexecute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
            infer_schema=False,
            args_schema=ExecuteSchema,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """Update the system prompt, filter tools, and evict oversized HumanMessages.

        In addition to the system-prompt and tool-filtering logic, this method
        handles large HumanMessage eviction:

        1. Any message already tagged with `lc_evicted_to` in
           `additional_kwargs` is replaced with a truncated preview for the
           model request (content in state is unchanged).
        2. If the most recent message is an untagged HumanMessage exceeding the
           eviction threshold, its content is written to the backend and the
           message is tagged in state via `ExtendedModelResponse`.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response, or an `ExtendedModelResponse` with a state
            update tagging a newly evicted message.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [
                _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(
                    large_tool_results_prefix=self._large_tool_results_prefix,
                )
            ]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        eviction_result = self._evict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Also evicts oversized HumanMessages to the filesystem. See
        `wrap_model_call` for full documentation.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler, or an `ExtendedModelResponse`
            with a state update tagging newly evicted messages.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [
                _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(
                    large_tool_results_prefix=self._large_tool_results_prefix,
                )
            ]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        eviction_result = await self._aevict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = await handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, bool]:
        """Process a large ToolMessage by evicting its content to filesystem.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, evicted):
            - processed_message: New ToolMessage with truncated content and file reference
            - evicted: Whether the content was evicted to the filesystem

        Note:
            Text is extracted from all text content blocks, joined, and used for both the
            size check and eviction. Non-text blocks (images, audio, etc.) are preserved in
            the replacement message so multimodal context is not lost. The model can recover
            the full text by reading the offloaded file from the backend.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        # Check if content exceeds eviction threshold
        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"{self._large_tool_results_prefix}/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, False

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, True

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, bool]:
        """Async version of _process_large_message.

        Uses async backend methods to avoid sync calls in async context.
        See _process_large_message for full documentation.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        # Write content to filesystem using async method
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"{self._large_tool_results_prefix}/{sanitized_id}"
        result = await resolved_backend.awrite(file_path, content_str)
        if result.error:
            return message, False

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, True

    def _get_backend_from_runtime(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> BackendProtocol:
        """Resolve the backend from a bare `Runtime`.

        Constructs a `ToolRuntime` from the `Runtime` to satisfy the backend
        factory interface. Used by hooks like `before_agent` that receive
        `Runtime` rather than `ToolRuntime`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Resolved backend instance.
        """
        if not callable(self.backend):
            return self.backend
        config = cast("RunnableConfig", getattr(runtime, "config", {}))
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            stream_writer=runtime.stream_writer,
            store=runtime.store,
            config=config,
            tool_call_id=None,
        )
        return self.backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]

    def _check_eviction_needed(
        self,
        messages: list[AnyMessage],
    ) -> tuple[bool, bool]:
        """Check whether any message processing is needed.

        Args:
            messages: The message list to inspect.

        Returns:
            Tuple of (has_tagged, new_eviction_needed).
        """
        if not self._human_message_token_limit_before_evict:
            return False, False

        threshold = NUM_CHARS_PER_TOKEN * self._human_message_token_limit_before_evict
        has_tagged = any(isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_evicted_to") for msg in messages)
        new_eviction_needed = False
        if messages and isinstance(messages[-1], HumanMessage):
            last = messages[-1]
            if not last.additional_kwargs.get("lc_evicted_to") and len(_extract_text_from_message(last)) > threshold:
                new_eviction_needed = True
        return has_tagged, new_eviction_needed

    @staticmethod
    def _apply_eviction_and_truncate(
        messages: list[AnyMessage],
        write_result: WriteResult | None,
        file_path: str | None,
    ) -> tuple[list[AnyMessage], Command | None]:
        """Tag a newly evicted message and truncate all tagged messages.

        Args:
            messages: The message list (may be modified if write succeeded).
            write_result: Result of the backend write, or `None` if no new
                eviction was attempted.
            file_path: Path the content was written to.

        Returns:
            Tuple of (processed_messages, state_command).
        """
        state_command: Command | None = None

        if write_result is not None and file_path is not None and not write_result.error:
            last = messages[-1]
            tagged = last.model_copy(
                update={
                    "additional_kwargs": {
                        **last.additional_kwargs,
                        "lc_evicted_to": file_path,
                    }
                }
            )
            state_command = Command(update={"messages": [tagged]})
            messages = [*messages[:-1], tagged]

        processed: list[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_evicted_to"):
                processed.append(_build_truncated_human_message(msg, msg.additional_kwargs["lc_evicted_to"]))
            else:
                processed.append(msg)

        return processed, state_command

    def _evict_and_truncate_messages(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[list[AnyMessage], Command | None] | None:
        """Evict a new oversized HumanMessage and truncate all tagged messages.

        Returns `None` if no messages needed processing (fast path). Otherwise
        returns `(processed_messages, command)` where `command` is a state
        update tagging the newly evicted message, or `None` if only
        previously-tagged messages were truncated.

        Args:
            request: The model request being processed.

        Returns:
            Tuple of (messages, command) if any processing occurred, else `None`.
        """
        messages = list(request.messages)
        has_tagged, new_eviction_needed = self._check_eviction_needed(messages)
        if not has_tagged and not new_eviction_needed:
            return None

        write_result: WriteResult | None = None
        file_path: str | None = None
        if new_eviction_needed:
            backend = self._get_backend_from_runtime(request.state, request.runtime)
            file_path = f"{self._conversation_history_prefix}/{uuid.uuid4()}.md"
            write_result = backend.write(file_path, _extract_text_from_message(messages[-1]))

        return self._apply_eviction_and_truncate(messages, write_result, file_path)

    async def _aevict_and_truncate_messages(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[list[AnyMessage], Command | None] | None:
        """Async version of `_evict_and_truncate_messages`.

        Args:
            request: The model request being processed.

        Returns:
            Tuple of (messages, command) if any processing occurred, else `None`.
        """
        messages = list(request.messages)
        has_tagged, new_eviction_needed = self._check_eviction_needed(messages)
        if not has_tagged and not new_eviction_needed:
            return None

        write_result: WriteResult | None = None
        file_path: str | None = None
        if new_eviction_needed:
            backend = self._get_backend_from_runtime(request.state, request.runtime)
            file_path = f"{self._conversation_history_prefix}/{uuid.uuid4()}.md"
            write_result = await backend.awrite(file_path, _extract_text_from_message(messages[-1]))

        return self._apply_eviction_and_truncate(messages, write_result, file_path)

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a processed result with
            evicted content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, _evicted = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return processed_message

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, _evicted = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
            return Command(update={**update, "messages": processed_messages})
        msg = f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Async version of _intercept_large_tool_result.

        Uses async backend methods to avoid sync calls in async context.
        See _intercept_large_tool_result for full documentation.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, _evicted = await self._aprocess_large_message(
                tool_result,
                resolved_backend,
            )
            return processed_message

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, _evicted = await self._aprocess_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
            return Command(update={**update, "messages": processed_messages})
        msg = f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

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
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

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
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return await handler(request)

        tool_result = await handler(request)
        return await self._aintercept_large_tool_result(tool_result, request.runtime)
