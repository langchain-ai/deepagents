"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.

It also defines the SandboxClient abstract base class for third-party SDK
implementations to manage sandbox lifecycle (get, create, delete).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from abc import ABC, abstractmethod
from typing import Any, Generic, NotRequired, TypeVar

from typing_extensions import TypedDict

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)


class SandboxError(Exception):
    """Base exception for sandbox provider operations.

    Provider implementations may raise provider-specific exceptions, but when
    normalizing errors for callers, prefer raising `SandboxError` (or a subclass)
    and chain the original exception.

    The recommended pattern is:

    ```python
    try:
        ...
    except Exception as e:
        raise SandboxError("...") from e
    ```
    """

    @property
    def original_exc(self) -> BaseException | None:
        """Original exception this error was raised from.

        This is populated when raised using exception chaining:

        ```python
        raise SandboxError("...") from e
        ```
        """
        return self.__cause__


class SandboxNotFoundError(SandboxError):
    """Raised when a sandbox_id is provided but the sandbox does not exist."""


# Type variable for provider-specific metadata
MetadataT = TypeVar("MetadataT", covariant=True)
"""Type variable for sandbox metadata.

Providers can define their own TypedDict to specify the structure of sandbox metadata,
enabling type-safe access to metadata fields.

Example:
    ```python
    class ProviderMetadata(TypedDict, total=False):
        status: Literal["running", "stopped"]
        created_at: str
        template: str

    class MyProvider(SandboxProvider[ProviderMetadata]):
        def list(
            self, *, cursor=None, **kwargs: Any
        ) -> SandboxListResponse[ProviderMetadata]:
            # Extract kwargs as needed
            status = kwargs.get("status")
            ...
    ```
"""


class SandboxInfo(TypedDict, Generic[MetadataT]):
    """Metadata for a single sandbox instance.

    This lightweight structure is returned from list operations and provides
    basic information about a sandbox without requiring a full connection.

    Type Parameters:
        MetadataT: Type of the metadata field. Providers should define a TypedDict
            for type-safe metadata access.

    Attributes:
        sandbox_id: Unique identifier for the sandbox instance.
        metadata: Optional provider-specific metadata (e.g., creation time, status,
            resource limits, template information). Structure is provider-defined.

    Example:
        ```python
        # Using default dict[str, Any]
        info: SandboxInfo = {
            "sandbox_id": "sb_abc123",
            "metadata": {"status": "running", "created_at": "2024-01-15T10:30:00Z", "template": "python-3.11"},
        }


        # Using typed metadata
        class MyMetadata(TypedDict, total=False):
            status: Literal["running", "stopped"]
            created_at: str


        typed_info: SandboxInfo[MyMetadata] = {
            "sandbox_id": "sb_abc123",
            "metadata": {"status": "running", "created_at": "2024-01-15T10:30:00Z"},
        }
        ```
    """

    sandbox_id: str
    metadata: NotRequired[MetadataT]


class SandboxListResponse(TypedDict, Generic[MetadataT]):
    """Paginated response from a sandbox list operation.

    This structure supports cursor-based pagination for efficiently browsing
    large collections of sandboxes.

    Type Parameters:
        MetadataT: Type of the metadata field in SandboxInfo items.

    Attributes:
        items: List of sandbox metadata objects for the current page.
        cursor: Opaque continuation token for retrieving the next page.
            None indicates no more pages available. Clients should treat this
            as an opaque string and pass it to subsequent list() calls.

    Example:
        ```python
        response: SandboxListResponse[MyMetadata] = {
            "items": [{"sandbox_id": "sb_001", "metadata": {"status": "running"}}, {"sandbox_id": "sb_002", "metadata": {"status": "stopped"}}],
            "cursor": "eyJvZmZzZXQiOjEwMH0=",
        }

        # Fetch next page
        next_response = provider.list(cursor=response["cursor"])
        ```
    """

    items: list[SandboxInfo[MetadataT]]
    cursor: str | None


class SandboxProvider(ABC, Generic[MetadataT]):
    """Deprecated: use SandboxClient instead.

    This abstraction is deprecated in favor of the simpler `SandboxClient` interface
    and will be removed in 0.4.0.

    Defines the lifecycle management interface for sandbox providers. Implementations
    should integrate with their respective SDKs to provide standardized sandbox
    lifecycle operations (list, get_or_create, delete).

    Implementations can add provider-specific parameters as keyword-only arguments
    with defaults, maintaining compatibility while providing type-safe APIs.

    Sync/Async Convention: Following LangChain convention, providers should offer both
    sync and async methods in the same namespace if possible (doesn't hurt performance)
    (e.g., both `list()` and `alist()` in one class). The default async implementations
    delegate to sync methods via a thread pool. Providers can override async methods to
    provide optimized async implementations if needed.

    Alternatively, if necessary for performance optimization, providers may split into
    separate implementations (e.g., `MySyncProvider` and `MyAsyncProvider`). In this
    case, unimplemented methods should raise NotImplementedError with clear guidance
    (e.g., "This provider only supports async operations. Use 'await provider.alist()'
    or switch to MySyncProvider for synchronous code").

    Example Implementation:
        ```python
        class CustomMetadata(TypedDict, total=False):
            status: Literal["running", "stopped"]
            template: str
            created_at: str


        class CustomSandboxProvider(SandboxProvider[CustomMetadata]):
            def list(
                self, *, cursor=None, status: Literal["running", "stopped"] | None = None, template_id: str | None = None, **kwargs: Any
            ) -> SandboxListResponse[CustomMetadata]:
                # Type-safe parameters with IDE autocomplete
                # ... query provider API
                return {"items": [...], "cursor": None}

            def get_or_create(
                self, *, sandbox_id=None, template_id: str = "default", timeout_minutes: int | None = None, **kwargs: Any
            ) -> SandboxBackendProtocol:
                # Type-safe parameters with IDE autocomplete
                return CustomSandbox(sandbox_id or self._create_new(), template_id)

            def delete(self, *, sandbox_id: str, force: bool = False, **kwargs: Any) -> None:
                # Implementation
                self._client.delete(sandbox_id, force=force)
        ```
    """

    @abstractmethod
    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT]:
        """List available sandboxes with optional filtering and pagination.

        Args:
            cursor: Optional continuation token from a previous list() call.
                Pass None to start from the beginning. The cursor is opaque
                and provider-specific; clients should not parse or modify it.
            **kwargs: Provider-specific filter parameters. Implementations should
                expose these as named keyword-only parameters with defaults for
                type safety. Common examples include status filters, creation time
                ranges, template filters, or owner filters.

        Returns:
            SandboxListResponse containing:
                - items: List of sandbox metadata for the current page
                - cursor: Token for next page, or None if this is the last page

        Example:
            ```python
            # First page
            response = provider.list()
            for sandbox in response["items"]:
                print(sandbox["sandbox_id"])

            # Next page if available
            if response["cursor"]:
                next_response = provider.list(cursor=response["cursor"])

            # With filters (if provider supports them)
            running = provider.list(status="running")
            ```
        """

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox or create a new one.

        This method retrieves a connection to an existing sandbox if sandbox_id
        is provided, or creates a new sandbox instance if sandbox_id is None.
        The returned object implements SandboxBackendProtocol and can be used
        for all sandbox operations (execute, read, write, etc.).

        Important: If a sandbox_id is provided but does not exist, this method
        should raise an error rather than creating a new sandbox. Only when
        sandbox_id is explicitly None should a new sandbox be created.

        Args:
            sandbox_id: Unique identifier of an existing sandbox to retrieve.
                If None, creates a new sandbox instance. The new sandbox's ID
                can be accessed via the returned object's .id property.
                If a non-None value is provided but the sandbox doesn't exist,
                an error will be raised.
            **kwargs: Provider-specific creation/connection parameters. Implementations
                should expose these as named keyword-only parameters with defaults
                for type safety. Common examples include template_id, resource limits,
                environment variables, or timeout settings.

        Returns:
            An object implementing SandboxBackendProtocol that can execute
            commands, read/write files, and perform other sandbox operations.

        Raises:
            Implementation-specific exceptions for errors such as:
                - Sandbox not found (if sandbox_id provided but doesn't exist)
                - Insufficient permissions
                - Resource limits exceeded
                - Invalid template or configuration

        Example:
            ```python
            # Create a new sandbox
            sandbox = provider.get_or_create(sandbox_id=None, template_id="python-3.11", timeout_minutes=60)
            print(sandbox.id)  # "sb_new123"

            # Reconnect to existing sandbox
            existing = provider.get_or_create(sandbox_id="sb_new123")

            # Use the sandbox
            result = sandbox.execute("python --version")
            print(result.output)
            ```
        """

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox instance.

        This permanently destroys the sandbox and all its associated data.
        The operation is typically irreversible.

        Idempotency: This method should be idempotent - calling delete on a
        non-existent sandbox should succeed without raising an error. This makes
        cleanup code simpler and safe to retry.

        Args:
            sandbox_id: Unique identifier of the sandbox to delete.
            **kwargs: Provider-specific deletion options. Implementations should
                expose these as named keyword-only parameters with defaults for
                type safety. Common examples include force flags, grace periods,
                or cleanup options.

        Raises:
            Implementation-specific exceptions for errors such as:
                - Insufficient permissions
                - Sandbox is locked or in use
                - Network or API errors

        Example:
            ```python
            # Simple deletion
            provider.delete(sandbox_id="sb_123")

            # Safe to call multiple times (idempotent)
            provider.delete(sandbox_id="sb_123")  # No error even if already deleted

            # With options (if provider supports them)
            provider.delete(sandbox_id="sb_456", force=True)
            ```
        """

    async def alist(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT]:
        """Async version of list().

        By default, runs the synchronous list() method in a thread pool.
        Providers can override this for native async implementations.

        Args:
            cursor: Optional continuation token from a previous list() call.
            **kwargs: Provider-specific filter parameters.

        Returns:
            SandboxListResponse containing items and cursor for pagination.
        """
        return await asyncio.to_thread(self.list, cursor=cursor, **kwargs)

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Async version of get_or_create().

        By default, runs the synchronous get_or_create() method in a thread pool.
        Providers can override this for native async implementations.

        Important: If a sandbox_id is provided but does not exist, this method
        should raise an error rather than creating a new sandbox. Only when
        sandbox_id is explicitly None should a new sandbox be created.

        Args:
            sandbox_id: Unique identifier of an existing sandbox to retrieve.
                If None, creates a new sandbox instance. If a non-None value
                is provided but the sandbox doesn't exist, an error will be raised.
            **kwargs: Provider-specific creation/connection parameters.

        Returns:
            An object implementing SandboxBackendProtocol.
        """
        return await asyncio.to_thread(self.get_or_create, sandbox_id=sandbox_id, **kwargs)

    async def adelete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Async version of delete().

        By default, runs the synchronous delete() method in a thread pool.
        Providers can override this for native async implementations.

        Args:
            sandbox_id: Unique identifier of the sandbox to delete.
            **kwargs: Provider-specific deletion options.
        """
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)


class SandboxClient(ABC):
    """Standardized client for managing sandboxes.

    Defines a standardized interface for sandbox SDK integrations. Implementations
    should integrate with their respective SDKs to provide standardized sandbox
    lifecycle operations (get, create, delete).

    Implementations can add provider-specific parameters as keyword-only arguments
    with defaults, maintaining compatibility while providing type-safe APIs.

    Sync/Async Convention: Following LangChain convention, providers should offer both
    sync and async methods in the same namespace if possible (doesn't hurt performance).
    The default async implementations delegate to sync methods via a thread pool.
    Providers can override async methods to provide optimized async implementations if needed.

    Alternatively, if necessary for performance optimization, providers may split into
    separate implementations (e.g., `MySyncProvider` and `MyAsyncProvider`). In this
    case, unimplemented methods should raise NotImplementedError with clear guidance.

    Example Implementation:
        ```python
        class CustomMetadata(TypedDict, total=False):
            status: Literal["running", "stopped"]
            template: str
            created_at: str


        class CustomSandboxClient(SandboxClient):
            def get(self, *, sandbox_id: str, **kwargs: Any) -> SandboxBackendProtocol:
                return CustomSandbox(sandbox_id)

            def create(self, *, template_id: str = "default", timeout_minutes: int | None = None, **kwargs: Any) -> SandboxBackendProtocol:
                return CustomSandbox(self._create_new(), template_id)

            def delete(self, *, sandbox_id: str, force: bool = False, **kwargs: Any) -> None:
                # Implementation
                self._client.delete(sandbox_id, force=force)
        ```
    """

    @abstractmethod
    def get(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox.

        If the sandbox does not exist, implementations should raise an error.
        """

    @abstractmethod
    def create(
        self,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Create a new sandbox and return a connected backend."""

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox instance.

        This permanently destroys the sandbox and all its associated data.
        The operation is typically irreversible.

        Idempotency: This method should be idempotent - calling delete on a
        non-existent sandbox should succeed without raising an error. This makes
        cleanup code simpler and safe to retry.

        Args:
            sandbox_id: Unique identifier of the sandbox to delete.
            **kwargs: Provider-specific deletion options. Implementations should
                expose these as named keyword-only parameters with defaults for
                type safety. Common examples include force flags, grace periods,
                or cleanup options.

        Raises:
            Implementation-specific exceptions for errors such as:
                - Insufficient permissions
                - Sandbox is locked or in use
                - Network or API errors

        Example:
            ```python
            # Simple deletion
            provider.delete(sandbox_id="sb_123")

            # Safe to call multiple times (idempotent)
            provider.delete(sandbox_id="sb_123")  # No error even if already deleted

            # With options (if provider supports them)
            provider.delete(sandbox_id="sb_456", force=True)
            ```
        """

    async def aget(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Async version of get().

        By default, runs the synchronous get() method in a thread pool.
        Providers can override this for native async implementations.
        """
        return await asyncio.to_thread(self.get, sandbox_id=sandbox_id, **kwargs)

    async def acreate(
        self,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Async version of create().

        By default, runs the synchronous create() method in a thread pool.
        Providers can override this for native async implementations.
        """
        return await asyncio.to_thread(self.create, **kwargs)

    async def adelete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Async version of delete().

        By default, runs the synchronous delete() method in a thread pool.
        Providers can override this for native async implementations.

        Args:
            sandbox_id: Unique identifier of the sandbox to delete.
            **kwargs: Provider-specific deletion options.
        """
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)


_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json
import base64

# Decode base64-encoded parameters
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>/dev/null"""

# Use heredoc to pass content via stdin to avoid ARG_MAX limits on large files.
# ARG_MAX limits the total size of command-line arguments.
# Previously, base64-encoded content was interpolated directly into the command
# string, which would fail for files larger than ~100KB after base64 expansion.
# Heredocs bypass this by passing data through stdin rather than as arguments.
# Stdin format: first line is base64-encoded file path, second line is base64-encoded content.
_WRITE_COMMAND_TEMPLATE = """python3 -c "
import os
import sys
import base64
import json

# Read JSON payload from stdin containing file_path and content (both base64-encoded)
payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received for write operation', file=sys.stderr)
    sys.exit(1)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    content = base64.b64decode(data['content']).decode('utf-8')
except Exception as e:
    print(f'Error: Failed to decode write payload: {{e}}', file=sys.stderr)
    sys.exit(1)

# Check if file already exists (atomic with write)
if os.path.exists(file_path):
    print(f'Error: File \\'{{file_path}}\\' already exists', file=sys.stderr)
    sys.exit(1)

# Create parent directory if needed
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

with open(file_path, 'w') as f:
    f.write(content)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""

# Use heredoc to pass edit parameters via stdin to avoid ARG_MAX limits.
# Stdin format: base64-encoded JSON with {"path": str, "old": str, "new": str}.
# JSON bundles all parameters; base64 ensures safe transport of arbitrary content
# (special chars, newlines, etc.) through the heredoc without escaping issues.
_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys
import base64
import json
import os

# Read and decode JSON payload from stdin
payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received for edit operation', file=sys.stderr)
    sys.exit(4)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    old = data['old']
    new = data['new']
except Exception as e:
    print(f'Error: Failed to decode edit payload: {{e}}', file=sys.stderr)
    sys.exit(4)

# Check if file exists
if not os.path.isfile(file_path):
    sys.exit(3)  # File not found

# Read file content
with open(file_path, 'r') as f:
    text = f.read()

# Count occurrences
count = text.count(old)

# Exit with error codes if issues found
if count == 0:
    sys.exit(1)  # String not found
elif count > 1 and not {replace_all}:
    sys.exit(2)  # Multiple occurrences without replace_all

# Perform replacement
if {replace_all}:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

# Write back to file
with open(file_path, 'w') as f:
    f.write(result)

print(count)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""

_READ_COMMAND_TEMPLATE = """python3 -c "
import os
import sys

file_path = '{file_path}'
offset = {offset}
limit = {limit}

# Check if file exists
if not os.path.isfile(file_path):
    print('Error: File not found')
    sys.exit(1)

# Check if file is empty
if os.path.getsize(file_path) == 0:
    print('System reminder: File exists but has empty contents')
    sys.exit(0)

# Read file with offset and limit
with open(file_path, 'r') as f:
    lines = f.readlines()

# Apply offset and limit
start_idx = offset
end_idx = offset + limit
selected_lines = lines[start_idx:end_idx]

# Format with line numbers (1-indexed, starting from offset + 1)
for i, line in enumerate(selected_lines):
    line_num = offset + i + 1
    # Remove trailing newline for formatting, then add it back
    line_content = line.rstrip('\\n')
    print(f'{{line_num:6d}}\\t{{line_content}}')
" 2>&1"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with execute() as abstract method.

    This class provides default implementations for all protocol methods
    using shell commands. Subclasses only need to implement execute().
    """

    @abstractmethod
    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        ...

    def ls_info(self, path: str) -> list[FileInfo]:
        """Structured listing with file metadata using os.scandir."""
        cmd = f"""python3 -c "
import os
import json

path = '{path}'

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null"""

        result = self.execute(cmd)

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
            except json.JSONDecodeError:
                continue

        return file_infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using a single shell command."""
        # Use template for reading file with offset and limit
        cmd = _READ_COMMAND_TEMPLATE.format(file_path=file_path, offset=offset, limit=limit)
        result = self.execute(cmd)

        output = result.output.rstrip()
        exit_code = result.exit_code

        if exit_code != 0 or "Error: File not found" in output:
            return f"Error: File '{file_path}' not found"

        return output

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file. Returns WriteResult; error populated on failure."""
        # Create JSON payload with file path and base64-encoded content
        # This avoids shell injection via file_path and ARG_MAX limits on content
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload = json.dumps({"path": file_path, "content": content_b64})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        # Single atomic check + write command
        cmd = _WRITE_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
        result = self.execute(cmd)

        # Check for errors (exit code or error message in output)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        # External storage - no files_update needed
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences. Returns EditResult."""
        # Create JSON payload with file path, old string, and new string
        # This avoids shell injection via file_path and ARG_MAX limits on strings
        payload = json.dumps({"path": file_path, "old": old_string, "new": new_string})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        # Use template for string replacement
        cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64, replace_all=replace_all)
        result = self.execute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        # Map exit codes to error messages
        error_messages = {
            1: f"Error: String not found in file: '{old_string}'",
            2: f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.",
            3: f"Error: File '{file_path}' not found",
            4: f"Error: Failed to decode edit payload: {output}",
        }
        if exit_code in error_messages:
            return EditResult(error=error_messages[exit_code])
        if exit_code != 0:
            return EditResult(error=f"Error editing file (exit code {exit_code}): {output or 'Unknown error'}")

        count = int(output)
        # External storage - no files_update needed
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Structured search results or error string for invalid input."""
        search_path = shlex.quote(path or ".")

        # Build grep command to get structured output
        grep_opts = "-rHnF"  # recursive, with filename, with line number, fixed-strings (literal)

        # Add glob pattern if specified
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        # Escape pattern for shell
        pattern_escaped = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = self.execute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        # Parse grep output into GrepMatch objects
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # Format is: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append(
                    {
                        "path": parts[0],
                        "line": int(parts[1]),
                        "text": parts[2],
                    }
                )

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Structured glob matching returning FileInfo dicts."""
        # Encode pattern and path as base64 to avoid escaping issues
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        # Parse JSON output into FileInfo dicts
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append(
                    {
                        "path": data["path"],
                        "is_dir": data["is_dir"],
                    }
                )
            except json.JSONDecodeError:
                continue

        return file_infos

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in FileUploadResponse objects rather than raising.
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in FileDownloadResponse objects rather than raising.
        """
