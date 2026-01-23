"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.

It also defines the SandboxProvider protocol for third-party SDK implementations
to manage sandbox lifecycle (list, create, delete).
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

# Type variable for provider-specific metadata
MetadataT = TypeVar("MetadataT", covariant=True, default=dict[str, Any])
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
        MetadataT: Type of the metadata field. Defaults to dict[str, Any].
            Providers can specify their own TypedDict for type-safe metadata access.

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
        MetadataT: Type of the metadata field in SandboxInfo items. Defaults to dict[str, Any].

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
    """Abstract base class for third-party sandbox provider implementations.

    This protocol defines the lifecycle management interface for sandbox providers.
    Implementations should integrate with their respective SDKs to provide
    standardized sandbox lifecycle operations.

    Type Parameters:
        MetadataT: TypedDict defining the structure of sandbox metadata.
            Default is dict[str, Any]. Enables type-safe access to metadata fields.

    Example Implementation:
        ```python
        class CustomMetadata(TypedDict, total=False):
            status: Literal["running", "stopped"]
            template: str
            created_at: str


        class CustomSandboxProvider(SandboxProvider[CustomMetadata]):
            def list(self, *, cursor=None, **kwargs: Any) -> SandboxListResponse[CustomMetadata]:
                # Extract and use kwargs as needed
                status = kwargs.get("status")
                template_id = kwargs.get("template_id")
                # ... query provider API
                return {"items": [...], "cursor": None}

            def get_or_create(self, *, sandbox_id=None, **kwargs: Any) -> SandboxBackendProtocol:
                # Extract and use kwargs as needed
                template = kwargs.get("template_id")
                timeout = kwargs.get("timeout_minutes")
                return CustomSandbox(sandbox_id or self._create_new(), **kwargs)

            def delete(self, sandbox_id: str, **kwargs: Any) -> None:
                # Implementation
                self._client.delete(sandbox_id)
        ```

    Usage:
        ```python
        provider: SandboxProvider = CustomSandboxProvider(api_key="...")

        # List sandboxes with type-safe filters
        result = provider.list(status="running", template_id="python-3.11")
        for info in result["items"]:
            print(f"Sandbox {info['sandbox_id']}")

        # Get or create a sandbox
        sandbox = provider.get_or_create(
            sandbox_id="sb_123",  # Or None to create new
            template_id="python-3.11",
        )

        # Use the sandbox via SandboxBackendProtocol
        sandbox.execute("pip install numpy")

        # Delete when done
        provider.delete(sandbox_id="sb_123")
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
            **kwargs: Provider-specific filter parameters. Common examples include
                status filters, creation time ranges, template filters, or owner
                filters. Check provider documentation for available filters.

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

        Args:
            sandbox_id: Unique identifier of an existing sandbox to retrieve.
                If None, creates a new sandbox instance. The new sandbox's ID
                can be accessed via the returned object's .id property.
            **kwargs: Provider-specific creation/connection parameters. Common
                examples include template_id, resource limits, environment variables,
                or timeout settings. These are typically only used when creating
                new sandboxes.

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
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox instance.

        This permanently destroys the sandbox and all its associated data.
        The operation is typically irreversible.

        Args:
            sandbox_id: Unique identifier of the sandbox to delete.
            **kwargs: Provider-specific deletion options. Common examples include
                force flags, grace periods, or cleanup options. Check provider
                documentation for available options.

        Raises:
            Implementation-specific exceptions for errors such as:
                - Sandbox not found
                - Insufficient permissions
                - Sandbox is locked or in use

        Example:
            ```python
            # Simple deletion
            provider.delete(sandbox_id="sb_123")

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

        Args:
            sandbox_id: Unique identifier of an existing sandbox to retrieve.
                If None, creates a new sandbox instance.
            **kwargs: Provider-specific creation/connection parameters.

        Returns:
            An object implementing SandboxBackendProtocol.
        """
        return await asyncio.to_thread(self.get_or_create, sandbox_id=sandbox_id, **kwargs)

    async def adelete(
        self,
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
        await asyncio.to_thread(self.delete, sandbox_id, **kwargs)


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

_WRITE_COMMAND_TEMPLATE = """python3 -c "
import os
import sys
import base64

file_path = '{file_path}'

# Check if file already exists (atomic with write)
if os.path.exists(file_path):
    print(f'Error: File \\'{file_path}\\' already exists', file=sys.stderr)
    sys.exit(1)

# Create parent directory if needed
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

# Decode and write content
content = base64.b64decode('{content_b64}').decode('utf-8')
with open(file_path, 'w') as f:
    f.write(content)
" 2>&1"""

_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys
import base64

# Read file content
with open('{file_path}', 'r') as f:
    text = f.read()

# Decode base64-encoded strings
old = base64.b64decode('{old_b64}').decode('utf-8')
new = base64.b64decode('{new_b64}').decode('utf-8')

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
with open('{file_path}', 'w') as f:
    f.write(result)

print(count)
" 2>&1"""

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
        # Encode content as base64 to avoid any escaping issues
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # Single atomic check + write command
        cmd = _WRITE_COMMAND_TEMPLATE.format(file_path=file_path, content_b64=content_b64)
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
        # Encode strings as base64 to avoid any escaping issues
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")

        # Use template for string replacement
        cmd = _EDIT_COMMAND_TEMPLATE.format(file_path=file_path, old_b64=old_b64, new_b64=new_b64, replace_all=replace_all)
        result = self.execute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        if exit_code == 1:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if exit_code == 2:
            return EditResult(error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.")
        if exit_code != 0:
            return EditResult(error=f"Error: File '{file_path}' not found")

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
