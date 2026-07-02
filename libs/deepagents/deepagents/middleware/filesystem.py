"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

import asyncio
import base64
import concurrent.futures
import contextlib
import contextvars
import mimetypes
import threading
import uuid
from binascii import Error as BinasciiError
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NotRequired, cast

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

import wcmatch.glob as wcglob
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
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.messages.content import ContentBlock
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.channels.delta import DeltaChannel
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepagents._api.deprecation import warn_deprecated
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend, StateBackend
from deepagents.backends.composite import _route_for_path
from deepagents.backends.protocol import (
    BACKEND_TYPES as BACKEND_TYPES,  # Re-export type here for backwards compatibility
    BackendProtocol,
    DeleteResult,
    EditResult,
    ExecuteOffloadResult,
    FileData as FileData,  # Re-export for backwards compatibility
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    _resolve_backend,
    _supports_delete,
    execute_accepts_timeout,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.utils import (
    _VIDEO_EXTRA_EXTENSIONS,
    MAX_VIDEO_INPUT_BYTES,
    FileType,
    _get_file_type,
    _glob_anchor,
    _paths_overlap,
    check_empty_content,
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id as sanitize_tool_call_id,
    truncate_if_too_long,
    validate_path,
)
from deepagents.middleware._message_eviction import (
    TOO_LARGE_TOOL_MSG as TOO_LARGE_TOOL_MSG,
    _aoffload_tool_message_content,
    _create_content_preview,
    _extract_text_from_message,
    _offload_tool_message_content,
)
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware._video import (
    VideoExtractionError,
    extract_video_frames,
    video_dependencies_available,
)

_FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR
"""wcmatch flags enabling brace expansion and `**` globstar recursion."""

_SYNC_GLOB_WORKERS = 4
"""Thread-pool size for synchronous glob operations."""

FilesystemOperation = Literal["read", "write"]
"""Classification of filesystem tools as read-only or mutating."""

_DEFAULT_FS_TOOL_OPS: dict[str, FilesystemOperation] = {
    "ls": "read",
    "read_file": "read",
    "glob": "read",
    "grep": "read",
    "write_file": "write",
    "edit_file": "write",
    "delete": "write",
}
"""Default mapping from filesystem tool name to its operation category."""

_READ_FILE_MEDIA_RESULT: Final = "read_file_media_result"
"""`additional_kwargs` key marking synthetic `HumanMessage` media from `read_file`."""

_VIDEO_SAMPLING_RATE: Final = 0.5
"""Seconds between sampled frames when extracting stills from a video."""


def _tool_error(name: str, tool_call_id: str | None, content: str) -> ToolMessage:
    """Build a `ToolMessage` carrying a plain text error."""
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error")


def _is_read_file_media_result(message: AnyMessage) -> bool:
    """Return whether `message` carries media emitted by a `read_file` tool result."""
    return isinstance(message, HumanMessage) and message.additional_kwargs.get(_READ_FILE_MEDIA_RESULT) is True


def _move_media_results_after_tool_results(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep synthetic media messages after the tool-result batch they describe.

    Tool-call providers require every `ToolMessage` for an assistant tool-call
    batch to arrive before any non-tool message. Video reads attach sampled
    frames as a synthetic `HumanMessage`; when multiple tools run in the same
    turn this helper keeps those attachments behind the full batch.
    """
    reordered: list[AnyMessage] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        reordered.append(message)
        i += 1
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue

        batch: list[AnyMessage] = []
        while i < len(messages):
            next_message = messages[i]
            if isinstance(next_message, ToolMessage) or _is_read_file_media_result(next_message):
                batch.append(next_message)
                i += 1
                continue
            break
        if batch:
            reordered.extend(message for message in batch if isinstance(message, ToolMessage))
            reordered.extend(message for message in batch if _is_read_file_media_result(message))
    return reordered


def _handle_video_read(
    content: str,
    validated_path: str,
    tool_call_id: str | None,
    offset: int,
    limit: int,
) -> ToolMessage | Command:
    """Slice a video byte payload into a sampled frame window for the model.

    `offset` is reinterpreted as seconds into the source to skip; `limit` as
    seconds of source to sample. The agent's supplied `limit` is authoritative
    (no per-call upper clamp), and supplying a non-positive value is rejected
    as a tool error. Output volume is bounded by the layered caps on the
    extractor (`MAX_VIDEO_DECODE_SECONDS`, `MAX_VIDEO_SAMPLED_FRAMES`,
    `MAX_VIDEO_EMITTED_BYTES`, `MAX_VIDEO_FRAME_PIXELS`, `MAX_VIDEO_FRAME_SIDE`).

    Errors are returned as `ToolMessage` errors so the turn still completes and
    the agent can recover (e.g. by retrying with a smaller window).
    """
    if limit <= 0:
        return _tool_error("read_file", tool_call_id, f"Error reading video {validated_path}: limit must be > 0, got {limit!r}")
    rate = _VIDEO_SAMPLING_RATE
    offset_seconds = max(0.0, float(offset))
    duration_seconds = float(limit)
    header = _video_window_header(validated_path, offset_seconds, duration_seconds, rate)

    def _err(msg: str) -> ToolMessage:
        return _tool_error("read_file", tool_call_id, f"Error reading video {validated_path}: {msg}\n{header}")

    try:
        raw_bytes = base64.b64decode(content, validate=True) if isinstance(content, str) else content
    except (ValueError, TypeError, BinasciiError) as exc:
        return _err(f"video bytes are not valid base64 ({exc})")
    if len(raw_bytes) > MAX_VIDEO_INPUT_BYTES:
        return _err(f"video payload exceeds maximum input size of {MAX_VIDEO_INPUT_BYTES} bytes")

    try:
        blocks = extract_video_frames(
            raw_bytes,
            offset_seconds=offset_seconds,
            duration_seconds=duration_seconds,
            sampling_rate=rate,
        )
    except VideoExtractionError as exc:
        return _err(str(exc))
    blocks.insert(0, {"type": "text", "text": header})
    frame_count = sum(1 for block in blocks if isinstance(block, dict) and block.get("type") == "image")
    frame_label = "frame" if frame_count == 1 else "frames"
    tool_message = ToolMessage(
        content=f"Read video {validated_path}: sampled {frame_count} {frame_label}. The sampled frames are attached in the following message.",
        name="read_file",
        tool_call_id=tool_call_id,
        additional_kwargs={"read_file_path": validated_path, "read_file_frame_count": frame_count},
        status="success",
    )
    media_message = HumanMessage(
        content_blocks=blocks,
        additional_kwargs={
            _READ_FILE_MEDIA_RESULT: True,
            "read_file_path": validated_path,
            "read_file_tool_call_id": tool_call_id,
        },
    )
    return Command(
        update={
            "messages": [
                tool_message,
                media_message,
            ],
        }
    )


def _video_window_header(path: str, offset_seconds: float, duration_seconds: float, rate: float) -> str:
    """Render the model-facing text header introducing a sampled frame window."""
    end = offset_seconds + duration_seconds
    if offset_seconds <= 0.0:
        return f"Reading first {int(duration_seconds)}s of {path} at {rate} fps."
    return f"Reading [{offset_seconds:.3f}s, {end:.3f}s) of {path} at {rate} fps."


def _get_read_file_type(path: str, *, video_enabled: bool) -> FileType:
    """Classify a file for `read_file`, gating optional video extensions."""
    file_type = _get_file_type(path)
    if video_enabled and PurePosixPath(path).suffix.lower() in _VIDEO_EXTRA_EXTENSIONS:
        return "video"
    return file_type


@dataclass
class FilesystemPermission:
    """A single access rule for filesystem operations."""

    operations: list[FilesystemOperation]
    paths: list[str]
    mode: Literal["allow", "deny", "interrupt"] = "allow"
    """Effect when a tool call matches this rule:

    - `"allow"` (default): the call proceeds.
    - `"deny"`: the tool returns a permission-denied error.
    - `"interrupt"`: the call is paused for human approval via
        [`HumanInTheLoopMiddleware`][langchain.agents.middleware.HumanInTheLoopMiddleware].

        Best paired with patterns that have a literal leading anchor (e.g.,
        `/secrets/**`, `/projects/*/secrets/**`). Bulk tools
        (`ls`/`glob`/`grep`) fire the interrupt based on whether their
        search subtree could overlap the rule's anchored prefix, so a fully
        unanchored pattern (`/**/secrets`) collapses to `/` and
        conservatively over-fires for any bulk call.
    """

    def __post_init__(self) -> None:
        """Validate permission path patterns."""
        for path in self.paths:
            if not path.startswith("/"):
                msg = f"Permission path must start with '/': {path!r}"
                raise ValueError(msg)
            parts = PurePosixPath(path.replace("\\", "/")).parts
            if ".." in parts:
                msg = f"Permission path must not contain '..': {path!r}"
                raise ValueError(msg)
            if "~" in parts:
                msg = f"Permission path must not contain '~': {path!r}"
                raise NotImplementedError(msg)


def _check_fs_permission(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path: str,
) -> Literal["allow", "deny", "interrupt"]:
    for rule in rules:
        if operation not in rule.operations:
            continue
        if any(wcglob.globmatch(path, pattern, flags=_FS_WCMATCH_FLAGS) for pattern in rule.paths):
            return rule.mode
    return "allow"


def _find_delete_deny_patterns(rules: list[FilesystemPermission], target: str) -> list[str]:
    """Return deny-write patterns that block deleting `target`.

    A recursive delete removes `target` and all descendants, so any overlapping
    deny-write pattern prevents the operation. The check is based only on
    permission rules and returns all matching patterns.

    Args:
        rules: Filesystem permission rules.
        target: Absolute, validated path being deleted.

    Returns:
        Matching deny-write patterns, or an empty list if the delete is allowed.
    """
    denying: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        if rule.mode != "deny" or "write" not in rule.operations:
            continue
        for pattern in rule.paths:
            if pattern not in seen and _paths_overlap(target, _glob_anchor(pattern)):
                seen.add(pattern)
                denying.append(pattern)
    return denying


def _filter_paths_by_permission(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    paths: list[str],
) -> list[str]:
    """Filter paths, removing only those denied by a rule.

    Interrupt-mode paths pass through here: the interrupt fires at the HITL
    stage *before* the tool runs (see `_build_interrupt_on_from_permissions`
    and its scope-aware predicate), so by the time result-filtering runs the
    user has already approved (or no rule matched). Filtering interrupt-mode
    results out here would silently empty the listing the user just approved.
    """
    if not rules:
        return paths
    return [p for p in paths if _check_fs_permission(rules, operation, p) != "deny"]


def _all_paths_scoped_to_routes(
    rules: list[FilesystemPermission],
    backend: BackendProtocol,
) -> bool:
    if not isinstance(backend, CompositeBackend):
        return False

    route_prefixes = list(backend.routes.keys())
    if not route_prefixes:
        return False

    for rule in rules:
        for path in rule.paths:
            if not any(path.startswith(prefix) for prefix in route_prefixes):
                return False
    return True


def _filter_file_infos_by_permission(
    rules: list[FilesystemPermission],
    infos: list[FileInfo],
    *,
    operation: FilesystemOperation,
) -> list[FileInfo]:
    """Filter file-info entries, removing only those denied by a rule.

    See `_filter_paths_by_permission` for why interrupt-mode entries
    pass through.
    """
    return [fi for fi in infos if _check_fs_permission(rules, operation, fi.get("path", "")) != "deny"]


def _filter_grep_matches_by_permission(
    rules: list[FilesystemPermission],
    matches: list[GrepMatch],
    *,
    operation: FilesystemOperation,
) -> list[GrepMatch]:
    """Filter grep matches, removing only those denied by a rule.

    See `_filter_paths_by_permission` for why interrupt-mode entries
    pass through.
    """
    return [m for m in matches if _check_fs_permission(rules, operation, m.get("path", "")) != "deny"]


def _format_grep_tool_result(
    result: GrepResult,
    output_mode: Literal["files_with_matches", "content", "count"],
) -> tuple[str, Literal["success", "error"]]:
    """Format a backend grep result for the tool boundary.

    Size-truncation is applied to the match body here, before any note is
    appended, so a trailing `SEARCH_TRUNCATION_NOTE` survives instead of being
    sliced off by an outer `truncate_if_too_long` at the call site. Callers
    should use the returned content as-is rather than re-truncating it.
    """
    matches = result.matches or []
    if result.error and not matches:
        return result.error, "error"

    formatted = truncate_if_too_long(format_grep_matches(matches, output_mode))
    if result.error:
        return f"{result.error}\n\nPartial matches:\n{formatted}", "error"
    if result.truncated:
        return f"{formatted}\n\n{SEARCH_TRUNCATION_NOTE}", "success"
    return formatted, "success"


def _apply_permissions_to_ls_results(
    rules: list[FilesystemPermission],
    entries: list[FileInfo],
) -> list[str]:
    """Filter ls entries by permission and return their paths."""
    filtered_entries = _filter_file_infos_by_permission(rules, entries, operation="read")
    return [fi.get("path", "") for fi in filtered_entries]


def _apply_permissions_to_glob_results(
    rules: list[FilesystemPermission],
    matches: list[FileInfo],
) -> list[str]:
    """Filter glob matches by permission and return their paths."""
    filtered_infos = _filter_file_infos_by_permission(rules, matches, operation="read")
    return [fi.get("path", "") for fi in filtered_infos]


def _format_file_paths(paths: list[str]) -> str:
    """Format filesystem path lists for tool output."""
    if not paths:
        return "No files found"
    return str(truncate_if_too_long(paths))


def _format_glob_tool_result(paths: list[str], *, truncated: bool) -> str:
    """Render glob paths for the tool boundary, appending the truncation note when partial."""
    content = _format_file_paths(paths)
    if truncated:
        return f"{content}\n\n{SEARCH_TRUNCATION_NOTE}"
    return content


EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
GLOB_TIMEOUT = 20.0  # seconds
LINE_NUMBER_WIDTH = 6
SEARCH_TRUNCATION_NOTE = (
    "Note: the search stopped early because it hit its time limit. The matches above are valid but incomplete. "
    "Narrow the search (a more specific pattern or a narrower path) to see the rest."
)


def _glob_timeout_message() -> str:
    """Build the glob-timeout error string.

    Reads `GLOB_TIMEOUT` at call time so tests and overrides keep the message
    in sync with the active deadline.
    """
    return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."


def _discard_task_result(task: asyncio.Future[Any]) -> None:
    """Consume a cancelled background task result to avoid event-loop warnings."""
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


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

    result: dict[str, FileData] = dict(left)
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _file_data_delta_reducer(
    left: dict[str, FileData] | None,
    values: list[dict[str, FileData | None]],
) -> dict[str, FileData]:
    """Batch reducer for use with DeltaChannel.

    `DeltaChannel` calls `reducer(base, list(values))` where values is a list of
    all writes in the current step.

    Single dict copy + one pass over all writes.
    """
    result: dict[str, FileData] = dict(left) if left else {}
    for writes in values:
        for key, value in writes.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = value
    return result


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], DeltaChannel(_file_data_delta_reducer, snapshot_frequency=50)]  # ty: ignore[invalid-argument-type]
    """Files in the filesystem. Uses DeltaChannel with snapshots every ~50 pregel steps to bound read depth."""


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


class ReadVideoFileSchema(ReadFileSchema):
    """Input schema for `read_file` when the optional video frame extraction is available.

    Identical to `ReadFileSchema`; only the `offset`/`limit` descriptions differ
    to document their video semantics (interpreted as seconds for video reads).
    """

    offset: int = Field(
        default=DEFAULT_READ_OFFSET,
        description="Line number to start reading from for text files (0-indexed). For videos, seconds into the source to start sampling.",
    )

    limit: int = Field(
        default=DEFAULT_READ_LIMIT,
        description="Maximum number of lines to read for text files. For videos, seconds of source to sample.",
    )


class WriteFileSchema(BaseModel):
    """Input schema for the `write_file` tool."""

    file_path: str = Field(description="Absolute path where the file should be written. Must be absolute, not relative.")

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


class DeleteSchema(BaseModel):
    """Input schema for the `delete` tool."""

    file_path: str = Field(description="Absolute path to the file to delete. Must be absolute, not relative.")


class GlobSchema(BaseModel):
    """Input schema for the `glob` tool."""

    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md').")

    path: str | None = Field(default=None, description="Base directory to search from. Defaults to the backend's default root.")


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

_READ_FILE_TOOL_DESCRIPTION_TEMPLATE = """Reads a file from the filesystem.

Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- {first_line}
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
    - First scan: read_file(file_path="...", limit=100) to see file structure
    - Read more sections: read_file(file_path="...", offset=100, limit=200) for next 200 lines
    - Omit `limit` to use the default window; increase it only when necessary for editing
- Specify offset and limit: read_file(file_path="...", offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). `limit` applies to source lines, so continuation rows do not consume the budget.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, etc.), audio and video files, and PDFs are returned as multimodal content blocks (see https://docs.langchain.com/oss/python/langchain/messages#multimodal).

For multimodal reads (image, audio, video, PDF, etc.):
- Use `read_file(file_path=...)`
{multimodal_bullets}
- If file details were compacted from history, call `read_file` again on the same path

- You should ALWAYS make sure a file has been read before editing it."""
"""Shared `read_file` description body for the text-only and video-aware variants.

The two variants differ only in the `{first_line}` and `{multimodal_bullets}`
fields, kept in a single template so the common guidance cannot drift between
them.
"""

_IMAGE_PDF_PAGINATION_BULLET = "- For images and PDFs, pagination via `offset`/`limit` is text-only - supply `file_path` only"
"""Multimodal bullet shared by both `read_file` descriptions (images/PDFs are not paginated)."""

READ_FILE_TOOL_DESCRIPTION = _READ_FILE_TOOL_DESCRIPTION_TEMPLATE.format(
    first_line="By default, it reads up to 100 lines starting from the beginning of the file",
    multimodal_bullets=_IMAGE_PDF_PAGINATION_BULLET,
)

READ_FILE_VIDEO_TOOL_DESCRIPTION = _READ_FILE_TOOL_DESCRIPTION_TEMPLATE.format(
    first_line="For text files, by default it reads up to 100 lines starting from the beginning of the file",
    multimodal_bullets=(
        f"{_IMAGE_PDF_PAGINATION_BULLET}\n"
        "- For videos, `offset`/`limit` are interpreted as seconds (default window 100 s; sampled at a fixed rate). Use smaller windows when you need more temporal detail."
    ),
)

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
- When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
- ALWAYS prefer editing existing files over creating new ones.
- Only use emojis if the user explicitly requests it."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes content to a file. Creates the file if it does not exist; replaces it entirely if it does.

Usage:
- Use this tool when you intend to create a new file or replace the whole file. You do not need to read the file first.
- Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
"""

DELETE_TOOL_DESCRIPTION = """Deletes a file or directory from the filesystem.

Usage:
- Permanently removes the file or directory at the given absolute path.
- Deleting a directory removes it and everything inside it, recursively. Prefer
  deleting a directory in one call over deleting each file individually.
- This cannot be undone, so only delete paths you are sure are no longer needed.
"""

GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
Returns a list of absolute file paths that match the pattern.

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in the backend's default root
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

## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /. Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- delete: delete a file or directory (recursively) from the filesystem
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


def _route_host_path_prompt(backend: BackendProtocol) -> str:
    """Build a prompt section mapping virtual route paths to host shell paths.

    `execute` runs on the default backend's shell, so virtual paths (e.g.
    `/common/`) may not exist there. Instead of rewriting shell commands, provide
    the model with prefix-substitution mappings so it can generate correct commands
    directly.

    A route exposes a usable host path only when its files live on the same
    filesystem the default's shell runs in, which requires the default to be a
    `LocalShellBackend` (its shell runs on the local host). For such a default, a
    `FilesystemBackend` route maps to a host path based on its mode:

    - virtual mode: the prefix maps to the backend's host root, `route.cwd`
        (e.g. `/common/` -> `/data/`, so `/common/x` is `/data/x` on the host).
    - non-virtual mode: the prefix is stripped and the remaining absolute path is
        used as-is (`root_dir` is ignored), i.e. the prefix maps to the filesystem
        root `/` (e.g. `/legacy/x` is `/x`).

    A remote/sandbox default runs its shell in a separate filesystem, so a local
    `FilesystemBackend` route is not reachable from it. Those routes, along with
    store-backed routes, have no host path mapping and must be accessed through the
    file tools instead.

    Returns an empty string if there are no routes to describe.
    """
    if not isinstance(backend, CompositeBackend):
        return ""

    # Host mappings are only valid when the default's shell shares the local
    # filesystem with the routes (LocalShellBackend). For a remote/sandbox
    # default, no local filesystem route is reachable from the shell.
    default_uses_local_shell = isinstance(backend.default, LocalShellBackend)

    # (virtual_prefix, host_prefix) pairs. A host_prefix of "/" means the virtual
    # prefix is stripped down to the filesystem root.
    host_mappings: list[tuple[str, str]] = []
    no_host_routes: list[str] = []
    for route_prefix, route_backend in backend.sorted_routes:
        if not (default_uses_local_shell and isinstance(route_backend, FilesystemBackend)):
            no_host_routes.append(route_prefix)
        elif route_backend.virtual_mode:
            # Virtual mode: prefix maps to the backend's host root directory.
            host_mappings.append((route_prefix, str(route_backend.cwd)))
        else:
            # Non-virtual mode: prefix is stripped, remaining absolute path used
            # as-is -> the prefix maps to the filesystem root.
            host_mappings.append((route_prefix, "/"))

    if not host_mappings and not no_host_routes:
        return ""

    def _norm(prefix: str) -> str:
        """Ensure a trailing slash so prefix substitution composes for subpaths."""
        return prefix if prefix.endswith("/") else f"{prefix}/"

    def _mapping_line(virtual_prefix: str, host_prefix: str) -> str:
        # Normalize both sides to end with "/" so replacing the virtual prefix with
        # the host prefix yields a correct host path for nested paths.
        virtual = _norm(virtual_prefix)
        host = _norm(host_prefix)
        example = f"`{virtual}dir/x.py` -> `{host}dir/x.py`"
        return f"- `{virtual}` -> `{host}` (e.g. {example})"

    lines = [
        "## Shell paths vs. virtual paths",
        "",
        "The `execute` tool runs commands in the host shell and can only access files that exist on the host filesystem.",
        "",
        "Some paths returned by the file tools are virtual mounts:",
        "",
        "- If a virtual mount has a host path mapping, replace its virtual prefix with the host prefix when running shell commands.",
        "- If a virtual mount does not have a host path mapping, it is not accessible "
        "from the shell. Use the file tools listed above to interact with those files.",
        "",
        "Do not assume that a path returned by a file tool can be used directly in a shell command.",
    ]

    if host_mappings:
        lines.append("")
        lines.append("Host path mappings:")
        lines.extend(_mapping_line(virtual_prefix, host_prefix) for virtual_prefix, host_prefix in host_mappings)

    if no_host_routes:
        lines.append("")
        lines.append("Virtual mounts without a host path mapping (not accessible from the shell):")
        lines.extend(f"- `{prefix}`" for prefix in no_host_routes)

    return "\n".join(lines)


def supports_execution(backend: BackendProtocol) -> bool:
    """Check if a backend supports command execution.

    For [`CompositeBackend`][deepagents.backends.composite.CompositeBackend],
    checks if the default backend supports execution.
    For other backends, checks if they implement
    [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol].

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
    "delete",
)


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


class FilesystemMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
    """Middleware for providing filesystem and optional execution tools to an agent.

    This middleware adds filesystem tools to the agent: `ls`, `read_file`, `write_file`,
    `edit_file`, `glob`, and `grep`.

    Files can be stored using any backend that implements the
    [`BackendProtocol`][deepagents.backends.protocol.BackendProtocol].

    If the backend implements
    [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol],
    an `execute` tool is also added for running shell commands.

    This middleware also automatically evicts large tool results to the file system when
    they exceed a token threshold, preventing context window saturation.

    Args:
        backend: Backend for file storage and optional execution.

            If not provided, defaults to
            [`StateBackend`][deepagents.backends.state.StateBackend]
            (ephemeral storage in agent state).

            For persistent storage or hybrid setups, use
            [`CompositeBackend`][deepagents.backends.composite.CompositeBackend]
            with custom routes.

            For execution support, use a backend that implements
            [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol].
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
        custom_tool_descriptions: Mapping[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        human_message_token_limit_before_evict: int | None = 50000,
        max_execute_timeout: int = 3600,
        _permissions: list[FilesystemPermission] | None = None,
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
            _permissions: Optional filesystem permission rules enforced directly
                by this middleware's tool implementations.

                Marked private for now because this is an internal
                implementation detail and may move to the backend layer in a
                future change.
        """
        if max_execute_timeout <= 0:
            msg = f"max_execute_timeout must be positive, got {max_execute_timeout}"
            raise ValueError(msg)
        # Use provided backend or default to StateBackend instance
        self.backend = backend if backend is not None else StateBackend()
        if (
            _permissions
            and isinstance(self.backend, BackendProtocol)
            and supports_execution(self.backend)
            and not _all_paths_scoped_to_routes(_permissions, self.backend)
        ):
            msg = (
                "FilesystemMiddleware does not yet support permissions with backends that "
                "provide command execution (SandboxBackendProtocol). Tool-level permissions "
                "for the execute tool are not implemented. Either remove permissions or use "
                "a backend without execution support."
            )
            raise NotImplementedError(msg)

        artifacts_root = self.backend.artifacts_root if isinstance(self.backend, CompositeBackend) else "/"
        _root = artifacts_root.rstrip("/")
        self._large_tool_results_prefix = f"{_root}/large_tool_results"
        self._conversation_history_prefix = f"{_root}/conversation_history"

        # Cache for dynamic system prompts keyed on the `include_execution`
        # flag. The text depends only on that flag and immutable config, so it
        # is computed at most twice per instance.
        self._dynamic_system_prompt_cache: dict[bool, str] = {}

        # Store configuration (private - internal implementation details)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict
        self._human_message_token_limit_before_evict = human_message_token_limit_before_evict
        self._max_execute_timeout = max_execute_timeout
        self._permissions = list(_permissions or [])

        # Shared executor for enforcing GLOB_TIMEOUT on the sync glob tool.
        # Timed-out worker threads keep running until the backend call returns,
        # so the semaphore rejects overload instead of queueing behind them.
        self._glob_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_SYNC_GLOB_WORKERS,
            thread_name_prefix="deepagents-glob",
        )
        self._glob_slots = threading.BoundedSemaphore(_SYNC_GLOB_WORKERS)

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_delete_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]

    def _build_dynamic_system_prompt(self, *, include_execution: bool) -> str:
        """Build (and memoize) the dynamic system prompt.

        The result depends only on `include_execution` and immutable config,
        so it is cached per instance to avoid rebuilding on every model call.
        The cache is intentionally lock-free even though sync and async model
        calls share it: writes are idempotent (a given flag always yields the
        same string), so a race at worst recomputes and re-stores that value.
        """
        cached = self._dynamic_system_prompt_cache.get(include_execution)
        if cached is not None:
            return cached
        prompt_parts = [_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(large_tool_results_prefix=self._large_tool_results_prefix)]
        if include_execution:
            prompt_parts.append(EXECUTION_SYSTEM_PROMPT)
        system_prompt = "\n\n".join(prompt_parts).strip()
        self._dynamic_system_prompt_cache[include_execution] = system_prompt
        return system_prompt

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            warn_deprecated(
                since="0.5.0",
                removal="0.7.0",
                message=(
                    "Passing a callable (factory) as `backend` is deprecated "
                    "and will be removed in deepagents==0.7.0. Pass a "
                    "`BackendProtocol` instance directly instead "
                    "(e.g. `StateBackend()`)."
                ),
                package="deepagents",
            )
            return _resolve_backend(self.backend, runtime)
        return self.backend

    def _create_ls_tool(self) -> BaseTool:
        """Create the ls (list files) tool."""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: str,
        ) -> ToolMessage:
            """Synchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="ls",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if _check_fs_permission(self._permissions, "read", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for read on {validated_path}",
                    name="ls",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            ls_result = resolved_backend.ls(validated_path)
            if ls_result.error:
                return ToolMessage(
                    content=f"Error: {ls_result.error}",
                    name="ls",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            infos = ls_result.entries or []
            paths = _apply_permissions_to_ls_results(self._permissions, infos)
            return ToolMessage(
                content=_format_file_paths(paths),
                tool_call_id=runtime.tool_call_id,
                name="ls",
                status="success",
            )

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: str,
        ) -> ToolMessage:
            """Asynchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="ls",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if _check_fs_permission(self._permissions, "read", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for read on {validated_path}",
                    name="ls",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            ls_result = await resolved_backend.als(validated_path)
            if ls_result.error:
                return ToolMessage(
                    content=f"Error: {ls_result.error}",
                    name="ls",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            infos = ls_result.entries or []
            paths = _apply_permissions_to_ls_results(self._permissions, infos)
            return ToolMessage(
                content=_format_file_paths(paths),
                tool_call_id=runtime.tool_call_id,
                name="ls",
                status="success",
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
        video_enabled = video_dependencies_available()
        default_description = READ_FILE_VIDEO_TOOL_DESCRIPTION if video_enabled else READ_FILE_TOOL_DESCRIPTION
        tool_description = self._custom_tool_descriptions.get("read_file") or default_description
        args_schema = ReadVideoFileSchema if video_enabled else ReadFileSchema
        token_limit = self._tool_token_limit_before_evict

        def _truncate(content: str, file_path: str, *, line_limit: int | None = None) -> str:
            if line_limit is not None:
                lines = content.splitlines(keepends=True)
                if len(lines) > line_limit:
                    content = "".join(lines[:line_limit])

            if token_limit and len(content) >= NUM_CHARS_PER_TOKEN * token_limit:
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=file_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                content = content[:max_content_length] + truncation_msg

            return content

        def _handle_read_result(  # noqa: PLR0911  # one branch per distinct read-result disposition
            read_result: ReadResult | str,
            validated_path: str,
            tool_call_id: str | None,
            offset: int,
            limit: int,
        ) -> ToolMessage | Command:
            if isinstance(read_result, str):
                warn_deprecated(
                    since="0.5.0",
                    removal="0.7.0",
                    message=(
                        "Returning a plain `str` from `backend.read()` is "
                        "deprecated and will be removed in deepagents==0.7.0. "
                        "Return a `ReadResult` instead."
                    ),
                    package="deepagents",
                )
                # Legacy backends already format with line numbers
                return ToolMessage(
                    content=_truncate(read_result, validated_path, line_limit=limit),
                    name="read_file",
                    tool_call_id=tool_call_id,
                    status="success",
                )

            if read_result.error:
                return ToolMessage(
                    content=f"Error: {read_result.error}",
                    name="read_file",
                    tool_call_id=tool_call_id,
                    status="error",
                )

            if read_result.file_data is None:
                return ToolMessage(
                    content=f"Error: no data returned for '{validated_path}'",
                    name="read_file",
                    tool_call_id=tool_call_id,
                    status="error",
                )

            file_type = _get_read_file_type(validated_path, video_enabled=video_enabled)
            encoding = read_result.file_data.get("encoding", "utf-8")
            content = read_result.file_data["content"]

            # Empty files get a uniform warning regardless of encoding/type, so
            # check before routing to avoid a degenerate empty content block for
            # binary reads.
            empty_msg = check_empty_content(content)
            if empty_msg:
                return ToolMessage(
                    content=empty_msg,
                    name="read_file",
                    tool_call_id=tool_call_id,
                    status="success",
                )

            # Video reads must be sliced into a sampled frame window before the
            # generic base64 branch runs; otherwise raw video bytes would reach
            # the model.
            if video_enabled and file_type == "video":
                return _handle_video_read(
                    content,
                    validated_path,
                    tool_call_id,
                    offset,
                    limit,
                )

            # Route on the backend-declared encoding first: `"base64"` means the
            # content is binary and must never be line-numbered as text, even
            # when the extension is absent from `_EXTENSION_TO_FILE_TYPE`.
            # The extension map is only consulted to pick the multimodal block
            # type; unknown binary extensions fall back to the generic `"file"`.
            if encoding == "base64" or file_type != "text":
                block_type = file_type if file_type != "text" else "file"
                mime_type = mimetypes.guess_type("file" + Path(validated_path).suffix)[0] or "application/octet-stream"
                return ToolMessage(
                    content_blocks=cast("list[ContentBlock]", [{"type": block_type, "base64": content, "mime_type": mime_type}]),
                    name="read_file",
                    tool_call_id=tool_call_id,
                    additional_kwargs={"read_file_path": validated_path, "read_file_media_type": mime_type},
                    status="success",
                )

            content = format_content_with_line_numbers(content, start_line=offset + 1)
            # `limit` already bounded raw source lines at the backend; do not
            # re-truncate by row count here, or wrapped continuation rows would
            # push real source lines off the end of the page (#2453).
            return ToolMessage(
                content=_truncate(content, validated_path),
                name="read_file",
                tool_call_id=tool_call_id,
                status="success",
            )

        def sync_read_file(
            file_path: str,
            runtime: ToolRuntime[None, FilesystemState],
            offset: int = DEFAULT_READ_OFFSET,
            limit: int = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | Command:
            """Synchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="read_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if _check_fs_permission(self._permissions, "read", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for read on {validated_path}",
                    name="read_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            read_result = resolved_backend.read(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        async def async_read_file(
            file_path: str,
            runtime: ToolRuntime[None, FilesystemState],
            offset: int = DEFAULT_READ_OFFSET,
            limit: int = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | Command:
            """Asynchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="read_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if _check_fs_permission(self._permissions, "read", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for read on {validated_path}",
                    name="read_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            read_result = await resolved_backend.aread(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
            infer_schema=False,
            args_schema=args_schema,
        )

    def _create_write_file_tool(self) -> BaseTool:
        """Create the write_file tool."""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: str,
            content: str,
            runtime: ToolRuntime[None, FilesystemState],
        ) -> ToolMessage:
            """Synchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="write_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            if _check_fs_permission(self._permissions, "write", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for write on {validated_path}",
                    name="write_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return ToolMessage(
                    content=res.error,
                    name="write_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=f"Updated file {res.path}",
                name="write_file",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        async def async_write_file(
            file_path: str,
            content: str,
            runtime: ToolRuntime[None, FilesystemState],
        ) -> ToolMessage:
            """Asynchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="write_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            if _check_fs_permission(self._permissions, "write", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for write on {validated_path}",
                    name="write_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            res: WriteResult = await resolved_backend.awrite(validated_path, content)
            if res.error:
                return ToolMessage(
                    content=res.error,
                    name="write_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=f"Updated file {res.path}",
                name="write_file",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

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
            file_path: str,
            old_string: str,
            new_string: str,
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: bool = False,
        ) -> ToolMessage:
            """Synchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="edit_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            if _check_fs_permission(self._permissions, "write", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for write on {validated_path}",
                    name="edit_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return ToolMessage(
                    content=res.error,
                    name="edit_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                name="edit_file",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        async def async_edit_file(
            file_path: str,
            old_string: str,
            new_string: str,
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: bool = False,
        ) -> ToolMessage:
            """Asynchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="edit_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            if _check_fs_permission(self._permissions, "write", validated_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for write on {validated_path}",
                    name="edit_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            res: EditResult = await resolved_backend.aedit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return ToolMessage(
                    content=res.error,
                    name="edit_file",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                name="edit_file",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
            infer_schema=False,
            args_schema=EditFileSchema,
        )

    def _create_delete_tool(self) -> BaseTool:  # Tool wiring + permission/support handling
        """Create the delete tool."""
        tool_description = self._custom_tool_descriptions.get("delete") or DELETE_TOOL_DESCRIPTION

        def sync_delete(
            file_path: str,
            runtime: ToolRuntime[None, FilesystemState],
        ) -> ToolMessage:
            """Synchronous wrapper for delete tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="delete",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            denying_patterns = _find_delete_deny_patterns(self._permissions, validated_path)
            if denying_patterns:
                return ToolMessage(
                    content=f"Error: permission denied for write on {validated_path} (matches deny rule(s): {', '.join(denying_patterns)})",
                    name="delete",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            res: DeleteResult = resolved_backend.delete(validated_path)
            if res.error:
                return ToolMessage(
                    content=res.error,
                    name="delete",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=f"Deleted {res.path}",
                name="delete",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        async def async_delete(
            file_path: str,
            runtime: ToolRuntime[None, FilesystemState],
        ) -> ToolMessage:
            """Asynchronous wrapper for delete tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="delete",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            denying_patterns = _find_delete_deny_patterns(self._permissions, validated_path)
            if denying_patterns:
                return ToolMessage(
                    content=f"Error: permission denied for write on {validated_path} (matches deny rule(s): {', '.join(denying_patterns)})",
                    name="delete",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            res: DeleteResult = await resolved_backend.adelete(validated_path)
            if res.error:
                return ToolMessage(
                    content=res.error,
                    name="delete",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=f"Deleted {res.path}",
                name="delete",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        return StructuredTool.from_function(
            name="delete",
            description=tool_description,
            func=sync_delete,
            coroutine=async_delete,
            infer_schema=False,
            args_schema=DeleteSchema,
        )

    def _create_glob_tool(self) -> BaseTool:  # noqa: C901, PLR0915  # Tool wiring + permission/result shaping + timeout handling
        """Create the glob tool."""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(  # noqa: PLR0911 - early returns for distinct error conditions
            pattern: str,
            runtime: ToolRuntime[None, FilesystemState],
            path: str | None = None,
        ) -> ToolMessage:
            """Synchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                permission_path = validate_path(path if path is not None else "/")
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if _check_fs_permission(self._permissions, "read", permission_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for read on {permission_path}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            backend_path = permission_path if path is not None else None
            ctx = contextvars.copy_context()
            # Submit to the shared executor rather than a per-call
            # ThreadPoolExecutor: a `with` block here would call
            # shutdown(wait=True) on timeout and block until the runaway glob
            # finished anyway, defeating the timeout.
            if not self._glob_slots.acquire(blocking=False):
                return ToolMessage(
                    content=("Error: too many glob calls are already running. Try again later with a more specific pattern or a narrower path."),
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            def run_glob() -> GlobResult:
                try:
                    return ctx.run(resolved_backend.glob, pattern, path=backend_path)
                finally:
                    self._glob_slots.release()

            try:
                future = self._glob_executor.submit(run_glob)
            except Exception:
                self._glob_slots.release()
                raise
            # Separate the wait deadline from result retrieval. On Python 3.11+
            # `concurrent.futures.TimeoutError is TimeoutError`, so catching the
            # future's wait-timeout would also swallow a builtin TimeoutError
            # raised *inside* the backend glob (e.g. a sandbox RPC timeout) and
            # misreport it as a glob-pattern timeout. `wait()` reports only
            # whether the deadline elapsed, leaving real backend exceptions to
            # surface through `future.result()` below.
            done, _ = concurrent.futures.wait([future], timeout=GLOB_TIMEOUT)
            if not done:
                # Deadline elapsed while the worker is still running; it cannot
                # be cancelled, so abandon it (run_glob's finally releases the
                # slot when it eventually returns). cancel() only succeeds if
                # the task never started, in which case release the slot here.
                if future.cancel():
                    self._glob_slots.release()
                return ToolMessage(
                    content=_glob_timeout_message(),
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            try:
                glob_result = future.result()
            except Exception as e:  # noqa: BLE001  # tool boundary: surface backend errors, never let them escape
                # run_glob's finally already released the slot before the
                # exception propagated, so do not release again here.
                return ToolMessage(
                    content=f"Error: glob failed: {e}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if glob_result.error:
                return ToolMessage(
                    content=f"Error: {glob_result.error}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            infos = glob_result.matches or []
            paths = _apply_permissions_to_glob_results(self._permissions, infos)
            return ToolMessage(
                content=_format_glob_tool_result(paths, truncated=glob_result.truncated),
                tool_call_id=runtime.tool_call_id,
                name="glob",
                status="success",
            )

        async def async_glob(
            pattern: str,
            runtime: ToolRuntime[None, FilesystemState],
            path: str | None = None,
        ) -> ToolMessage:
            """Asynchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                permission_path = validate_path(path if path is not None else "/")
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: {e}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if _check_fs_permission(self._permissions, "read", permission_path) == "deny":
                return ToolMessage(
                    content=f"Error: permission denied for read on {permission_path}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            backend_path = permission_path if path is not None else None
            # Run the backend glob as a task and wait on the deadline separately
            # so a `TimeoutError` raised *inside* the backend (rather than by the
            # deadline) is not misreported as a glob-pattern timeout, mirroring
            # the sync path. Other backend exceptions surface via `task.result()`.
            task = asyncio.ensure_future(resolved_backend.aglob(pattern, path=backend_path))
            done, _ = await asyncio.wait({task}, timeout=GLOB_TIMEOUT)
            if not done:
                task.add_done_callback(_discard_task_result)
                task.cancel()
                return ToolMessage(
                    content=_glob_timeout_message(),
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            try:
                glob_result = task.result()
            except Exception as e:  # noqa: BLE001  # tool boundary: surface backend errors, never let them escape
                return ToolMessage(
                    content=f"Error: glob failed: {e}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            if glob_result.error:
                return ToolMessage(
                    content=f"Error: {glob_result.error}",
                    name="glob",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            infos = glob_result.matches or []
            paths = _apply_permissions_to_glob_results(self._permissions, infos)
            return ToolMessage(
                content=_format_glob_tool_result(paths, truncated=glob_result.truncated),
                tool_call_id=runtime.tool_call_id,
                name="glob",
                status="success",
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
            pattern: str,
            runtime: ToolRuntime[None, FilesystemState],
            path: str | None = None,
            glob: str | None = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        ) -> ToolMessage:
            """Synchronous wrapper for grep tool."""
            if path is not None:
                try:
                    path = validate_path(path)
                except ValueError as e:
                    return ToolMessage(
                        content=f"Error: {e}",
                        name="grep",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
                if _check_fs_permission(self._permissions, "read", path) == "deny":
                    return ToolMessage(
                        content=f"Error: permission denied for read on {path}",
                        name="grep",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
            resolved_backend = self._get_backend(runtime)
            grep_result = resolved_backend.grep(pattern, path=path, glob=glob)
            matches = grep_result.matches or []
            filtered_matches = _filter_grep_matches_by_permission(self._permissions, matches, operation="read")
            formatted, status = _format_grep_tool_result(
                GrepResult(error=grep_result.error, matches=filtered_matches, truncated=grep_result.truncated),
                output_mode,
            )
            return ToolMessage(
                # `formatted` is already size-truncated inside
                # `_format_grep_tool_result` so the truncation note survives.
                content=formatted,
                tool_call_id=runtime.tool_call_id,
                name="grep",
                status=status,
            )

        async def async_grep(
            pattern: str,
            runtime: ToolRuntime[None, FilesystemState],
            path: str | None = None,
            glob: str | None = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        ) -> ToolMessage:
            """Asynchronous wrapper for grep tool."""
            if path is not None:
                try:
                    path = validate_path(path)
                except ValueError as e:
                    return ToolMessage(
                        content=f"Error: {e}",
                        name="grep",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
                if _check_fs_permission(self._permissions, "read", path) == "deny":
                    return ToolMessage(
                        content=f"Error: permission denied for read on {path}",
                        name="grep",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
            resolved_backend = self._get_backend(runtime)
            grep_result = await resolved_backend.agrep(pattern, path=path, glob=glob)
            matches = grep_result.matches or []
            filtered_matches = _filter_grep_matches_by_permission(self._permissions, matches, operation="read")
            formatted, status = _format_grep_tool_result(
                GrepResult(error=grep_result.error, matches=filtered_matches, truncated=grep_result.truncated),
                output_mode,
            )
            return ToolMessage(
                # `formatted` is already size-truncated inside
                # `_format_grep_tool_result` so the truncation note survives.
                content=formatted,
                tool_call_id=runtime.tool_call_id,
                name="grep",
                status=status,
            )

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
            infer_schema=False,
            args_schema=GrepSchema,
        )

    def _resolve_capture(self, resolved_backend: BackendProtocol, tool_call_id: str | None) -> tuple[BaseSandbox, str] | None:
        """Resolve the executing sandbox and offload path for capture-at-source.

        Capture-at-source writes output to a literal path via the sandbox shell
        and later reads it back through the backend, which requires `execute()`
        and `read_file` to resolve to the same filesystem at that path. Only
        `BaseSandbox` provides that guarantee, so it is gated on it; the offload
        path must also route to the executing backend rather than a different
        composite route.

        Whether capture is actually applied is left to the executor's
        `execute_with_offload` (which honors `enable_capture_offload`); this only
        decides whether the offload path is valid to attempt.

        Returns:
            `(executor, capture_path)` when capture-at-source can be attempted, or
            `None` to skip it (eviction disabled, no tool-call id, the backend is
            not a `BaseSandbox`, or the offload path routes elsewhere) — in which
            case the caller uses plain execute plus generic eviction.
        """
        if not self._tool_token_limit_before_evict or not tool_call_id:
            return None
        capture_path = f"{self._large_tool_results_prefix}/{sanitize_tool_call_id(tool_call_id)}"
        if isinstance(resolved_backend, CompositeBackend):
            default = resolved_backend.default
            if not isinstance(default, BaseSandbox):
                return None
            backend, _backend_path, route_prefix = _route_for_path(
                default=default,
                sorted_routes=resolved_backend.sorted_routes,
                path=capture_path,
            )
            # Safe only when the path falls through to the default backend
            # unchanged, since execute() also runs on the default.
            if route_prefix is None and backend is default:
                return default, capture_path
            return None
        if isinstance(resolved_backend, BaseSandbox):
            return resolved_backend, capture_path
        return None

    @staticmethod
    def _format_execute_output(output: str, exit_code: int | None, *, truncated: bool) -> str:
        """Format raw command output with status and truncation notes for the model."""
        parts = [output]
        if exit_code is not None:
            cmd_status = "succeeded" if exit_code == 0 else "failed"
            parts.append(f"\n[Command {cmd_status} with exit code {exit_code}]")
        if truncated:
            parts.append("\n[Output was truncated due to size limits]")
        return "".join(parts)

    def _interpret_capture_output(self, offload: ExecuteOffloadResult, capture_path: str, tool_call_id: str) -> str:
        """Build `ToolMessage` content from an `execute_with_offload` result."""
        response = offload.response
        if not offload.offloaded:
            return self._format_execute_output(response.output, response.exit_code, truncated=response.truncated)
        cmd_status = "succeeded" if response.exit_code == 0 else "failed"
        status_line = f"[Command {cmd_status} with exit code {response.exit_code}]"
        if response.truncated:
            status_line += "\n[Output exceeded the capture size limit and was truncated; the saved file is incomplete]"
        content_sample = f"{status_line}\n{response.output}"
        return TOO_LARGE_TOOL_MSG.format(
            tool_call_id=tool_call_id,
            file_path=capture_path,
            content_sample=content_sample,
        )

    def _create_execute_tool(self) -> BaseTool:  # noqa: C901
        """Create the execute tool for sandbox command execution."""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: str,
            runtime: ToolRuntime[None, FilesystemState],
            timeout: int | None = None,
        ) -> ToolMessage:
            """Synchronous wrapper for execute tool."""
            if timeout is not None:
                if timeout < 0:
                    return ToolMessage(
                        content=f"Error: timeout must be non-negative, got {timeout}.",
                        name="execute",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
                if timeout > self._max_execute_timeout:
                    return ToolMessage(
                        content=f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s).",
                        name="execute",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not supports_execution(resolved_backend):
                return ToolMessage(
                    content=(
                        "Error: Execution not available. This agent's backend "
                        "does not support command execution (SandboxBackendProtocol). "
                        "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                    ),
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            # Safe cast: supports_execution validates that execute()/aexecute() exist
            # (either SandboxBackendProtocol or CompositeBackend with sandbox default)
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return ToolMessage(
                    content=(
                        "Error: This sandbox backend does not support per-command "
                        "timeout overrides. Update your sandbox package to the "
                        "latest version, or omit the timeout parameter."
                    ),
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            capture = self._resolve_capture(resolved_backend, runtime.tool_call_id)
            try:
                if capture is not None:
                    executor, capture_path = capture
                    offload = executor.execute_with_offload(
                        command,
                        capture_path,
                        max_inline_bytes=NUM_CHARS_PER_TOKEN * cast("int", self._tool_token_limit_before_evict),
                        timeout=timeout,
                    )
                    content = self._interpret_capture_output(offload, capture_path, cast("str", runtime.tool_call_id))
                else:
                    result = executable.execute(command, timeout=timeout) if timeout is not None else executable.execute(command)
                    content = self._format_execute_output(result.output, result.exit_code, truncated=result.truncated)
            except NotImplementedError as e:
                return ToolMessage(
                    content=f"Error: Execution not available. {e}",
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: Invalid parameter. {e}",
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            return ToolMessage(
                content=content,
                name="execute",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        async def async_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: str,
            runtime: ToolRuntime[None, FilesystemState],
            timeout: int | None = None,  # noqa: ASYNC109  # forwarded to backend, not an asyncio contract
        ) -> ToolMessage:
            """Asynchronous wrapper for execute tool."""
            if timeout is not None:
                if timeout < 0:
                    return ToolMessage(
                        content=f"Error: timeout must be non-negative, got {timeout}.",
                        name="execute",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
                if timeout > self._max_execute_timeout:
                    return ToolMessage(
                        content=f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s).",
                        name="execute",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not supports_execution(resolved_backend):
                return ToolMessage(
                    content=(
                        "Error: Execution not available. This agent's backend "
                        "does not support command execution (SandboxBackendProtocol). "
                        "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                    ),
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            # Safe cast: supports_execution validates that execute()/aexecute() exist
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return ToolMessage(
                    content=(
                        "Error: This sandbox backend does not support per-command "
                        "timeout overrides. Update your sandbox package to the "
                        "latest version, or omit the timeout parameter."
                    ),
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            capture = self._resolve_capture(resolved_backend, runtime.tool_call_id)
            try:
                if capture is not None:
                    executor, capture_path = capture
                    offload = await executor.aexecute_with_offload(
                        command,
                        capture_path,
                        max_inline_bytes=NUM_CHARS_PER_TOKEN * cast("int", self._tool_token_limit_before_evict),
                        timeout=timeout,
                    )
                    content = self._interpret_capture_output(offload, capture_path, cast("str", runtime.tool_call_id))
                else:
                    result = await executable.aexecute(command, timeout=timeout) if timeout is not None else await executable.aexecute(command)
                    content = self._format_execute_output(result.output, result.exit_code, truncated=result.truncated)
            except NotImplementedError as e:
                return ToolMessage(
                    content=f"Error: Execution not available. {e}",
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            except ValueError as e:
                return ToolMessage(
                    content=f"Error: Invalid parameter. {e}",
                    name="execute",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )

            return ToolMessage(
                content=content,
                name="execute",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
            infer_schema=False,
            args_schema=ExecuteSchema,
        )

    def _filter_unsupported_tools_and_apply_prompt(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Drop capability-gated tools the backend can't serve, then apply the system prompt.

        Shared by the sync and async `wrap_model_call` paths (the only part that
        differs between them is sync vs. async message eviction). The `execute`
        and `delete` tools are optional per backend, so when the resolved
        backend doesn't support a capability the corresponding tool is filtered
        out of the request rather than advertised to the model and left to fail
        at call time. Resolving the backend and probing support is synchronous,
        so both paths route through here.

        Returns the request with unsupported tools removed and the filesystem
        system prompt appended.
        """
        tool_names = {(tool.name if hasattr(tool, "name") else tool.get("name")) for tool in request.tools}
        has_execute_tool = "execute" in tool_names
        has_delete_tool = "delete" in tool_names

        # `execute` and `delete` are optional per backend; resolve the
        # backend once and filter out any tool the backend can't serve.
        if has_execute_tool or has_delete_tool:
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            unsupported: set[str | None] = set()
            if has_execute_tool and not supports_execution(backend):
                unsupported.add("execute")
                has_execute_tool = False
            if has_delete_tool and not _supports_delete(backend):
                unsupported.add("delete")
            if unsupported:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) not in unsupported]
                request = request.override(tools=filtered_tools)

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

            # Add execution instructions only if the execute tool survived filtering
            if has_execute_tool:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)
                route_prompt = _route_host_path_prompt(backend)
                if route_prompt:
                    prompt_parts.append(route_prompt)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return request

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
        request = self._filter_unsupported_tools_and_apply_prompt(request)

        request_messages = _move_media_results_after_tool_results(list(request.messages))
        if request_messages != list(request.messages):
            request = request.override(messages=request_messages)

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
        request = self._filter_unsupported_tools_and_apply_prompt(request)

        request_messages = _move_media_results_after_tool_results(list(request.messages))
        if request_messages != list(request.messages):
            request = request.override(messages=request_messages)

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
            A tuple of `(processed_message, evicted)`:

                - processed_message: New `ToolMessage` with truncated content
                    and file reference
                - evicted: Whether the content was evicted to the filesystem

        !!! note

            Text is extracted from all text content blocks, joined, and used for
            both the size check and eviction. Non-text blocks
            (images, audio, etc.) are preserved in the replacement message
            so multimodal context is not lost. The model can recover
            the full text by reading the offloaded file from the backend.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        # Check if content exceeds eviction threshold
        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        processed_message = _offload_tool_message_content(
            message,
            content_str,
            resolved_backend,
            self._large_tool_results_prefix,
        )
        if processed_message is None:
            return message, False
        return processed_message, True

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, bool]:
        """Async version of _process_large_message.

        Uses async backend methods to avoid sync calls in async context.

        See `_process_large_message` for full documentation.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        processed_message = await _aoffload_tool_message_content(
            message,
            content_str,
            resolved_backend,
            self._large_tool_results_prefix,
        )
        if processed_message is None:
            return message, False
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
        return _resolve_backend(self.backend, tool_runtime)

    def _check_eviction_needed(
        self,
        messages: list[AnyMessage],
    ) -> tuple[bool, bool]:
        """Check whether any message processing is needed.

        Args:
            messages: The message list to inspect.

        Returns:
            Tuple of `(has_tagged, new_eviction_needed)`.
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

        When a new eviction fires, emits a `Command` whose messages update
        contains only the tagged `HumanMessage`. Because `ensure_message_ids`
        stamps a stable UUID onto the original write before it is checkpointed,
        the tagged copy (which reuses that ID) is deduped in-place by the
        `DeltaChannel` reducer — no `REMOVE_ALL_MESSAGES` sentinel is needed.
        Using a sentinel would also clobber the `AIMessage` that the model node
        writes in the same super-step.

        Args:
            messages: The message list (may be modified if write succeeded).
            write_result: Result of the backend write, or `None` if no new
                eviction was attempted.
            file_path: Path the content was written to.

        Returns:
            Tuple of `(processed_messages, state_command)`.
        """
        state_command: Command | None = None

        if write_result is not None and file_path is not None and not write_result.error:
            last = messages[-1]
            tagged = last.model_copy(
                update={
                    "id": last.id if last.id is not None else str(uuid.uuid4()),
                    "additional_kwargs": {
                        **last.additional_kwargs,
                        "lc_evicted_to": file_path,
                    },
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
        """Evict a new oversized `HumanMessage` and truncate all tagged messages.

        Returns `None` if no messages needed processing (fast path). Otherwise
        returns `(processed_messages, command)` where `command` is a state
        update tagging the newly evicted message, or `None` if only
        previously-tagged messages were truncated.

        Args:
            request: The model request being processed.

        Returns:
            Tuple of `(messages, command)` if any processing occurred, else `None`.
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
            Tuple of `(messages, command)` if any processing occurred, else `None`.
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

    @staticmethod
    def _unwrap_command_messages(update: Mapping[str, Any]) -> tuple[Any, bool]:
        """Return the message list from a Command update and whether it was prefixed with a `REMOVE_ALL_MESSAGES` sentinel.

        Tools that want to atomically replace the messages channel emit
        `[RemoveMessage(REMOVE_ALL_MESSAGES), *messages]`. Detect that
        sentinel so we can preserve it after processing.
        """
        command_messages = update.get("messages", [])
        if (
            isinstance(command_messages, list)
            and command_messages
            and isinstance(command_messages[0], RemoveMessage)
            and command_messages[0].id == REMOVE_ALL_MESSAGES
        ):
            return command_messages[1:], True
        return command_messages, False

    @staticmethod
    def _rewrap_command_messages(messages: list[AnyMessage], *, wrapped: bool) -> list[AnyMessage | RemoveMessage]:
        """Restore the `REMOVE_ALL_MESSAGES` sentinel when the original update used one."""
        if wrapped:
            return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]
        return list(messages)

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (`ToolMessage` or `Command`).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a processed result with
                evicted content written to filesystem and truncated message.

        !!! note

            Handles both single `ToolMessage` results and `Command` objects
            containing multiple messages. Large content is automatically
            offloaded to filesystem to prevent context window overflow.
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
            command_messages, wrapped = self._unwrap_command_messages(update)
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
            new_messages = self._rewrap_command_messages(processed_messages, wrapped=wrapped)
            return Command(
                goto=tool_result.goto,
                graph=tool_result.graph,
                update={**update, "messages": new_messages},
            )
        msg = f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Async version of _intercept_large_tool_result.

        Uses async backend methods to avoid sync calls in async context.

        See `_intercept_large_tool_result` for full documentation.
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
            command_messages, wrapped = self._unwrap_command_messages(update)
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
            new_messages = self._rewrap_command_messages(processed_messages, wrapped=wrapped)
            return Command(
                goto=tool_result.goto,
                graph=tool_result.graph,
                update={**update, "messages": new_messages},
            )
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
            The raw `ToolMessage`, or a pseudo tool message with the `ToolResult` in state.

        !!! note

            Tool-execution exceptions (including `ToolException`) propagate
            through this wrapper unhandled by design.
        """
        tool_result = handler(request)

        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return tool_result

        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw `ToolMessage`, or a pseudo tool message with the `ToolResult` in state.

        Note:
            Tool-execution exceptions (including `ToolException`) propagate
                through this wrapper unhandled by design.
        """
        tool_result = await handler(request)

        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return tool_result

        return await self._aintercept_large_tool_result(tool_result, request.runtime)
