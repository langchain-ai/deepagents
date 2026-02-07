"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

import base64
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite
from typing_extensions import TypedDict

from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    BACKEND_TYPES as BACKEND_TYPES,  # Re-export type here for backwards compatibility
    BackendProtocol,
    EditResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
)
from deepagents.middleware._utils import append_to_system_message

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100

# Image file extensions supported by LLM APIs (OpenAI and Anthropic)
# Note: BMP is NOT supported - only jpeg, png, gif, webp work with vision APIs
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
IMAGE_READING_TOOLS = frozenset({"read_file", "open_image"})
MAX_INLINE_IMAGE_HISTORY = 6


def _detect_model_provider(model_name: str | None) -> str:
    """Detect the model provider from the model name.

    Args:
        model_name: Model name string (e.g., "openai:gpt-4o", "anthropic:claude-sonnet-4-5-20250929")

    Returns:
        Provider string: "openai", "anthropic", etc.
    """
    if not model_name:
        return "anthropic"  # Default

    model_lower = model_name.lower()

    if model_lower.startswith("openai:") or "gpt" in model_lower:
        return "openai"
    if model_lower.startswith("anthropic:") or "claude" in model_lower:
        return "anthropic"
    if model_lower.startswith("google:") or "gemini" in model_lower:
        return "openai"  # Google uses similar format to OpenAI
    return "anthropic"  # Default to Anthropic format


def _create_image_content_block(image_b64: str, media_type: str, provider: str = "anthropic") -> dict:
    """Create a standard LangChain ImageContentBlock.

    LangChain chat models normalize this schema into provider-specific payloads.

    Args:
        image_b64: Base64-encoded image data.
        media_type: MIME type (e.g., ``image/png``).
        provider: Unused compatibility parameter retained for API stability.

    Returns:
        Standard image content block.
    """
    _ = provider  # Kept for backwards-compatible call signature.
    return {
        "type": "image",
        "base64": image_b64,
        "mime_type": media_type,
    }


def _is_inline_image_block(block: object) -> bool:
    """Check whether a content block contains inline base64 image bytes."""
    if not isinstance(block, dict):
        return False

    block_type = block.get("type")

    # OpenAI-style image block with data URL
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            return isinstance(url, str) and url.startswith("data:image/")
        return False

    # Anthropic-style image block
    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            return True
        # Also support LangChain standard content blocks: {"type":"image","base64":...}
        return isinstance(block.get("base64"), str)

    return False

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
    r"""Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    This function is designed for virtual filesystem paths and rejects
    Windows absolute paths (e.g., C:/..., F:/...) to maintain consistency
    and prevent path format ambiguity.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path (e.g., C:/...), or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Reject Windows absolute paths (e.g., C:\..., D:/...)
    # This maintains consistency in virtual filesystem paths
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files and directories in a given directory.

**When to Use:**
- Exploring the filesystem structure
- Finding files before reading or editing
- Verifying a directory exists before creating files in it

**When NOT to Use:**
- Finding files by pattern (use `glob` instead)
- Searching for files containing specific text (use `grep` instead)

**Usage:**
- Requires an absolute path starting with /
- Returns file and directory names in the specified path
- Use this before read_file, open_image, or edit_file to verify file locations"""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem.

**When to Use:**
- Understanding code before making changes
- Exploring unfamiliar codebases
- Checking configuration files
- Reading documentation
- ALWAYS read a file before attempting to edit it

**When NOT to Use:**
- Finding files by name pattern (use `glob` instead)
- Searching for text across multiple files (use `grep` instead)

**Unsupported Formats (use CLI tools instead):**
- Video files (.mp4, .mov, .avi, etc.) - use ffmpeg or similar
- Audio files (.mp3, .wav, etc.) - use ffprobe or similar
- PDF documents - use pdftotext or similar
- Office documents (.docx, .xlsx) - use appropriate converters
- Binary/executable files - use hexdump, strings, or file-specific tools

If you need to process these formats, use `execute()` with appropriate command-line tools.

**CRITICAL: You MUST read a file before editing it. The edit tool will fail if you haven't read the file first.**

**For image files (png, jpg, jpeg, gif, webp):**
- Use `open_image(file_path=...)` for visual analysis

**Pagination (IMPORTANT for large text files):**
- By default, reads up to 100 lines from the beginning
- Use `offset` and `limit` parameters for pagination:
  - First scan: `read_file(path, limit=100)` - See structure
  - Continue: `read_file(path, offset=100, limit=200)` - Next section
  - Full read: Only when necessary for immediate editing

**When to paginate:**
- Files >500 lines
- Exploring unfamiliar codebases (start with limit=100)
- Reading multiple files in sequence

**Output format:**
- Text files: cat -n format with line numbers starting at 1
- Lines >5,000 chars split into continuation lines (5.1, 5.2, etc.)
- Empty files return a system reminder warning

**Parallel reading:**
- You can call multiple read_file tools in a single response
- Read multiple potentially useful files in parallel to save time"""

OPEN_IMAGE_TOOL_DESCRIPTION = """Opens an image file for visual analysis.

**When to Use:**
- Reading screenshots
- Analyzing charts, diagrams, and plots
- Inspecting UI mockups or rendered pages
- Any task that requires understanding visual pixels

**Supported Formats:**
- PNG, JPG, JPEG, GIF, WebP

**Parameters:**
- `file_path`: Absolute path to the image file

**Important:**
- This tool is for images only
- For text/code/config files, use `read_file`
- No pagination parameters are needed"""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

**When to Use:**
- Modifying existing code
- Fixing bugs
- Adding new code to existing files
- Updating configuration

**When NOT to Use:**
- Creating new files (use `write_file` instead)
- Complete file rewrites (use `write_file` instead)

**CRITICAL: You MUST read the file first. This tool will ERROR if you haven't read the file in this conversation.**

**Parameters:**
- `file_path`: Absolute path to the file
- `old_string`: Exact text to find and replace (must be unique unless replace_all=True)
- `new_string`: Text to replace it with (must be different from old_string)
- `replace_all`: If True, replace all occurrences; if False (default), old_string must be unique

**Usage Guidelines:**
- Preserve exact indentation (tabs/spaces) from the read output
- Never include line number prefixes in old_string or new_string
- ALWAYS prefer editing existing files over creating new ones
- Make the smallest change necessary to accomplish the task
- Only use emojis if the user explicitly requests it

**If old_string is not unique:**
- Provide more surrounding context to make it unique, OR
- Use `replace_all=True` to replace all occurrences"""


WRITE_FILE_TOOL_DESCRIPTION = """Creates a new file or overwrites an existing file.

**When to Use:**
- Creating new files that don't exist
- Complete file rewrites where edit_file would be cumbersome
- Writing generated content (configs, scripts, etc.)

**When NOT to Use:**
- Modifying existing files (use `edit_file` instead - it's safer)
- Small changes to existing files (use `edit_file` instead)

**IMPORTANT:** ALWAYS prefer `edit_file` over `write_file` for existing files. Edit is safer because it preserves file content you didn't intend to change.

**Parameters:**
- `file_path`: Absolute path where the file should be created
- `content`: The text content to write to the file

**Usage Guidelines:**
- Will overwrite if file already exists - be careful!
- Parent directories must exist (use ls to verify first)
- Only use emojis if the user explicitly requests it
- NEVER proactively create documentation or README files"""

GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

**When to Use:**
- Finding files by name or extension pattern
- Discovering project structure
- Locating configuration files
- Finding all files of a certain type (*.py, *.js, etc.)

**When NOT to Use:**
- Searching for text content within files (use `grep` instead)
- Listing a single directory (use `ls` instead)

**Pattern Syntax:**
- `*` - Matches any characters within a path segment
- `**` - Matches any directories (recursive)
- `?` - Matches a single character

**Examples:**
- `**/*.py` - Find all Python files recursively
- `*.txt` - Find all text files in root only
- `/src/**/*.ts` - Find all TypeScript files under /src
- `**/test_*.py` - Find all test files
- `**/*config*` - Find all config files

**Parameters:**
- `pattern`: The glob pattern to match
- `path`: Base directory to search from (defaults to root /)

**Returns:** List of absolute file paths matching the pattern"""

GREP_TOOL_DESCRIPTION = """Search for a text pattern across files.

**When to Use:**
- Finding where a function/class/variable is defined or used
- Searching for error messages or log patterns
- Finding TODOs, FIXMEs, or other markers
- Locating imports or dependencies
- Understanding how code is connected

**When NOT to Use:**
- Finding files by name pattern (use `glob` instead)
- Reading a specific file (use `read_file` instead)

**Parameters:**
- `pattern`: Text to search for (literal string, not regex)
- `path`: Directory to search in (defaults to current working directory)
- `glob`: Filter which files to search (e.g., "*.py" for Python files only)
- `output_mode`:
  - `"files_with_matches"` (default): Just return file paths
  - `"content"`: Show matching lines with context
  - `"count"`: Show match counts per file

**Examples:**
- Find all TODOs: `grep(pattern="TODO")`
- Search Python files: `grep(pattern="import requests", glob="*.py")`
- Show matching lines: `grep(pattern="def process", output_mode="content")`
- Find in specific dir: `grep(pattern="error", path="/src/utils")`

**Note:** Searches literal text, not regex patterns."""

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
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file for text files and open_image for image files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file/open_image tools instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools

You have access to these filesystem tools. All file paths must be absolute (start with /).

**Tool Overview:**
- `ls`: List files in a directory
- `read_file`: Read text file contents (MUST read before editing)
- `open_image`: Open image files for visual analysis
- `write_file`: Create new files (prefer edit_file for existing files)
- `edit_file`: Modify existing files via string replacement
- `glob`: Find files by pattern (e.g., "**/*.py")
- `grep`: Search for text within files

**Key Rules:**
1. Use `open_image` for images, `read_file` for text files
2. Use `glob` to find files, not shell commands like `find`
3. Use `grep` tool to search content, not shell `grep`
4. Prefer `edit_file` over `write_file` for existing files
5. Call multiple read operations in parallel when possible"""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


def _supports_execution(backend: BackendProtocol) -> bool:
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
# 2. Tools with problematic truncation behavior (read_file, open_image):
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
    "open_image",
    "edit_file",
    "write_file",
)


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
You can do this by specifying an offset and limit in the read_file tool call.
For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form
... [N lines truncated] ...
indicate omitted lines in the middle of the content):

{content_sample}
"""


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


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem and optional execution tools to an agent.

    This middleware adds filesystem tools to the agent: `ls`, `read_file`,
    `open_image`, `write_file`, `edit_file`, `glob`, and `grep`.

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
        model_provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            backend: Backend for file storage and optional execution, or a factory callable.
                Defaults to StateBackend if not provided.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.
            model_provider: Model provider for image format ("anthropic" or "openai").
                If not specified, auto-detected from model_name.
            model_name: Model name used to auto-detect provider if model_provider not set.
        """
        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Store configuration (private - internal implementation details)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict
        # Auto-detect provider from model name if not explicitly set
        self._model_provider = model_provider or _detect_model_provider(model_name)

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_open_image_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    @staticmethod
    def _collect_recent_inline_image_indexes(messages: list[AnyMessage]) -> set[int]:
        """Return indexes for the most recent inline image tool messages to preserve."""
        keep_indexes: set[int] = set()
        kept = 0

        for idx in range(len(messages) - 1, -1, -1):
            if kept >= MAX_INLINE_IMAGE_HISTORY:
                break

            message = messages[idx]
            if not isinstance(message, ToolMessage):
                continue
            if message.name not in IMAGE_READING_TOOLS:
                continue
            if not isinstance(message.content, list):
                continue
            if not any(_is_inline_image_block(block) for block in message.content):
                continue

            keep_indexes.add(idx)
            kept += 1

        return keep_indexes

    @staticmethod
    def _compact_image_tool_message(message: ToolMessage) -> ToolMessage | None:
        """Replace inline image blocks with a compact text note."""
        if message.name not in IMAGE_READING_TOOLS or not isinstance(message.content, list):
            return None

        non_image_blocks: list[object] = []
        image_block_count = 0
        for block in message.content:
            if _is_inline_image_block(block):
                image_block_count += 1
            else:
                non_image_blocks.append(block)

        if image_block_count == 0:
            return None

        path_hint = ""
        file_path = (
            message.additional_kwargs.get("open_image_path")
            or message.additional_kwargs.get("read_file_path")
            or message.additional_kwargs.get("image_file_path")
        )
        if isinstance(file_path, str) and file_path:
            path_hint = f" Path: {file_path}."

        compact_note = {
            "type": "text",
            "text": (
                f"[{image_block_count} prior image block(s) omitted from history to preserve context."
                " Re-run open_image on the same file if you need pixels again."
                f"{path_hint}]"
            ),
        }
        compact_content = [*non_image_blocks, compact_note]
        return message.model_copy(update={"content": compact_content})

    def _compact_historical_image_messages(
        self,
        messages: list[AnyMessage],
    ) -> tuple[list[AnyMessage], bool]:
        """Compact image tool results while preserving only recent inline images."""
        if not messages:
            return messages, False

        keep_indexes = self._collect_recent_inline_image_indexes(messages)

        compacted_messages: list[AnyMessage] = []
        changed = False

        for idx, message in enumerate(messages):
            if isinstance(message, ToolMessage):
                if idx in keep_indexes:
                    compacted_messages.append(message)
                    continue
                compacted = self._compact_image_tool_message(message)
                if compacted is not None:
                    compacted_messages.append(compacted)
                    changed = changed or compacted != message
                    continue
            compacted_messages.append(message)

        return compacted_messages, changed

    def _create_ls_tool(self) -> BaseTool:
        """Create the ls (list files) tool."""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """Synchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(path)
            infos = resolved_backend.ls_info(validated_path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """Asynchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(path)
            infos = await resolved_backend.als_info(validated_path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="ls",
            description=tool_description,
            func=sync_ls,
            coroutine=async_ls,
        )

    def _create_read_file_tool(self) -> BaseTool:
        """Create the read_file tool."""
        tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
        token_limit = self._tool_token_limit_before_evict
        model_provider = self._model_provider

        def sync_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> Command | str:
            """Synchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)

            # Check if this is an image file
            ext = Path(validated_path).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                return self._load_image_sync(
                    resolved_backend=resolved_backend,
                    validated_path=validated_path,
                    tool_call_id=runtime.tool_call_id,
                    tool_name="read_file",
                    model_provider=model_provider,
                )

            # Text file handling (original logic)
            result = resolved_backend.read(validated_path, offset=offset, limit=limit)

            lines = result.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                result = "".join(lines)

            # Check if result exceeds token threshold and truncate if necessary
            if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
                # Calculate truncation message length to ensure final result stays under threshold
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                result = result[:max_content_length]
                result += truncation_msg

            return result

        async def async_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> Command | str:
            """Asynchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)

            # Check if this is an image file
            ext = Path(validated_path).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                return await self._load_image_async(
                    resolved_backend=resolved_backend,
                    validated_path=validated_path,
                    tool_call_id=runtime.tool_call_id,
                    tool_name="read_file",
                    model_provider=model_provider,
                )

            # Text file handling (original logic)
            result = await resolved_backend.aread(validated_path, offset=offset, limit=limit)

            lines = result.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                result = "".join(lines)

            # Check if result exceeds token threshold and truncate if necessary
            if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
                # Calculate truncation message length to ensure final result stays under threshold
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                result = result[:max_content_length]
                result += truncation_msg

            return result

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
        )

    def _create_open_image_tool(self) -> BaseTool:
        """Create the open_image tool."""
        tool_description = self._custom_tool_descriptions.get("open_image") or OPEN_IMAGE_TOOL_DESCRIPTION
        model_provider = self._model_provider

        def sync_open_image(
            file_path: Annotated[str, "Absolute path to the image file to open. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """Synchronous wrapper for open_image tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)
            ext = Path(validated_path).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                return (
                    "Error opening image: unsupported extension for open_image. "
                    "Use open_image only for .png/.jpg/.jpeg/.gif/.webp files."
                )
            return self._load_image_sync(
                resolved_backend=resolved_backend,
                validated_path=validated_path,
                tool_call_id=runtime.tool_call_id,
                tool_name="open_image",
                model_provider=model_provider,
            )

        async def async_open_image(
            file_path: Annotated[str, "Absolute path to the image file to open. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """Asynchronous wrapper for open_image tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)
            ext = Path(validated_path).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                return (
                    "Error opening image: unsupported extension for open_image. "
                    "Use open_image only for .png/.jpg/.jpeg/.gif/.webp files."
                )
            return await self._load_image_async(
                resolved_backend=resolved_backend,
                validated_path=validated_path,
                tool_call_id=runtime.tool_call_id,
                tool_name="open_image",
                model_provider=model_provider,
            )

        return StructuredTool.from_function(
            name="open_image",
            description=tool_description,
            func=sync_open_image,
            coroutine=async_open_image,
        )

    @staticmethod
    def _build_image_tool_message(
        *,
        image_bytes: bytes,
        validated_path: str,
        tool_call_id: str | None,
        tool_name: str,
        model_provider: str,
    ) -> Command:
        """Build a tool message command containing a normalized image block."""
        ext = Path(validated_path).suffix.lower()
        media_type = IMAGE_MEDIA_TYPES.get(ext, "image/png")
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        image_block = _create_image_content_block(image_b64, media_type, model_provider)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=[image_block],
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        additional_kwargs={
                            "image_file_path": validated_path,
                            "image_media_type": media_type,
                            "read_file_path": validated_path,
                            "open_image_path": validated_path,
                        },
                    )
                ],
            }
        )

    def _load_image_sync(
        self,
        *,
        resolved_backend: BackendProtocol,
        validated_path: str,
        tool_call_id: str | None,
        tool_name: str,
        model_provider: str,
    ) -> Command | str:
        """Load an image from backend and return a tool message command."""
        responses = resolved_backend.download_files([validated_path])
        if responses and responses[0].content is not None:
            return self._build_image_tool_message(
                image_bytes=responses[0].content,
                validated_path=validated_path,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                model_provider=model_provider,
            )
        if responses and responses[0].error:
            return f"Error reading image: {responses[0].error}"
        return "Error reading image: unknown error"

    async def _load_image_async(
        self,
        *,
        resolved_backend: BackendProtocol,
        validated_path: str,
        tool_call_id: str | None,
        tool_name: str,
        model_provider: str,
    ) -> Command | str:
        """Async variant for loading an image from backend."""
        responses = await resolved_backend.adownload_files([validated_path])
        if responses and responses[0].content is not None:
            return self._build_image_tool_message(
                image_bytes=responses[0].content,
                validated_path=validated_path,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                model_provider=model_provider,
            )
        if responses and responses[0].error:
            return f"Error reading image: {responses[0].error}"
        return "Error reading image: unknown error"

    def _create_write_file_tool(self) -> BaseTool:
        """Create the write_file tool."""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """Synchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)
            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return res.error
            # If backend returns state update, wrap into Command with ToolMessage
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        async def async_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """Asynchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)
            res: WriteResult = await resolved_backend.awrite(validated_path, content)
            if res.error:
                return res.error
            # If backend returns state update, wrap into Command with ToolMessage
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        return StructuredTool.from_function(
            name="write_file",
            description=tool_description,
            func=sync_write_file,
            coroutine=async_write_file,
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
        ) -> Command | str:
            """Synchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)
            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        async def async_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """Asynchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            validated_path = _validate_path(file_path)
            res: EditResult = await resolved_backend.aedit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
        )

    def _create_glob_tool(self) -> BaseTool:
        """Create the glob tool."""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """Synchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            infos = resolved_backend.glob_info(pattern, path=path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """Asynchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            infos = await resolved_backend.aglob_info(pattern, path=path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="glob",
            description=tool_description,
            func=sync_glob,
            coroutine=async_glob,
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
        ) -> str:
            """Synchronous wrapper for grep tool."""
            resolved_backend = self._get_backend(runtime)
            raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
            if isinstance(raw, str):
                return raw
            formatted = format_grep_matches(raw, output_mode)
            return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        async def async_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """Asynchronous wrapper for grep tool."""
            resolved_backend = self._get_backend(runtime)
            raw = await resolved_backend.agrep_raw(pattern, path=path, glob=glob)
            if isinstance(raw, str):
                return raw
            formatted = format_grep_matches(raw, output_mode)
            return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
        )

    def _create_execute_tool(self) -> BaseTool:
        """Create the execute tool for sandbox command execution."""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Synchronous wrapper for execute tool."""
            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            try:
                result = resolved_backend.execute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        async def async_execute(
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Asynchronous wrapper for execute tool."""
            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            try:
                result = await resolved_backend.aexecute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"

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
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

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
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return handler(request)

    def before_model(
        self,
        state: FilesystemState,
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Compact historical inline image blocks before model invocation."""
        messages = state.get("messages", [])
        compacted_messages, changed = self._compact_historical_image_messages(messages)
        if not changed:
            return None
        return {"messages": Overwrite(compacted_messages)}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

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
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return await handler(request)

    async def abefore_model(
        self,
        state: FilesystemState,
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of before_model for consistent compaction behavior."""
        messages = state.get("messages", [])
        compacted_messages, changed = self._compact_historical_image_messages(messages)
        if not changed:
            return None
        return {"messages": Overwrite(compacted_messages)}

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Process a large ToolMessage by evicting its content to filesystem.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, files_update):
            - processed_message: New ToolMessage with truncated content and file reference
            - files_update: Dict of file updates to apply to state, or None if eviction failed

        Note:
            The entire content is converted to string, written to /large_tool_results/{tool_call_id},
            and replaced with a truncated preview plus file reference. The replacement is always
            returned as a plain string for consistency, regardless of original content type.

            ToolMessage supports multimodal content blocks (images, audio, etc.), but these are
            uncommon in tool results. For simplicity, all content is stringified and evicted.
            The model can recover by reading the offloaded file from the backend.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        # Convert content to string once for both size check and eviction
        # Special case: single text block - extract text directly for readability
        if (
            isinstance(message.content, list)
            and len(message.content) == 1
            and isinstance(message.content[0], dict)
            and message.content[0].get("type") == "text"
            and "text" in message.content[0]
        ):
            content_str = str(message.content[0]["text"])
        elif isinstance(message.content, str):
            content_str = message.content
        else:
            # Multiple blocks or non-text content - stringify entire structure
            content_str = str(message.content)

        # Check if content exceeds eviction threshold
        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        # Always return as plain string after eviction
        processed_message = ToolMessage(
            content=replacement_text,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
        return processed_message, result.files_update

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Async version of _process_large_message.

        Uses async backend methods to avoid sync calls in async context.
        See _process_large_message for full documentation.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        # Convert content to string once for both size check and eviction
        # Special case: single text block - extract text directly for readability
        if (
            isinstance(message.content, list)
            and len(message.content) == 1
            and isinstance(message.content[0], dict)
            and message.content[0].get("type") == "text"
            and "text" in message.content[0]
        ):
            content_str = str(message.content[0]["text"])
        elif isinstance(message.content, str):
            content_str = message.content
        else:
            # Multiple blocks or non-text content - stringify entire structure
            content_str = str(message.content)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem using async method
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = await resolved_backend.awrite(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        # Always return as plain string after eviction
        processed_message = ToolMessage(
            content=replacement_text,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a Command with evicted
            content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}")

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Async version of _intercept_large_tool_result.

        Uses async backend methods to avoid sync calls in async context.
        See _intercept_large_tool_result for full documentation.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = await self._aprocess_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = await self._aprocess_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}")

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
