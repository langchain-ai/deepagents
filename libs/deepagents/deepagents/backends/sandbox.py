"""Base sandbox implementation (`BaseSandbox`) implementing `SandboxBackendProtocol`.

File listing, grep, glob, and read use shell commands via `execute()`. Write
delegates content transfer to `upload_files()`. Edit uses server-side `execute()`
for small payloads and falls back to uploading old/new strings as temp files
with a server-side replace script for large ones.
"""

from __future__ import annotations

import base64
import json
import shlex
from abc import ABC, abstractmethod

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import _get_file_type, create_file_data

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
" 2>&1"""
"""Find files matching a pattern with metadata.

Uses base64-encoded parameters to avoid shell escaping issues.
"""

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
__DEEPAGENTS_EOF__\n"""
"""Write a file to the sandbox via heredoc stdin.

Uses base64-encoded JSON payload containing the file path and content,
passed via heredoc to avoid ARG_MAX limits on large files.
"""

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
__DEEPAGENTS_EOF__\n"""
"""Server-side file edit via `execute()`.

Reads the file, performs string replacement, and writes back — all on the
sandbox.  The payload (path, old/new strings) is passed as base64-encoded
JSON via heredoc stdin to avoid shell escaping issues.

Output: exit code 0 with occurrence count on success, or non-zero exit
codes for specific error conditions.
"""

_READ_COMMAND_TEMPLATE = """python3 -c "
import os, sys, base64, json

path = base64.b64decode('{path_b64}').decode('utf-8')

if not os.path.isfile(path):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)

if os.path.getsize(path) == 0:
    print(json.dumps({{'encoding': 'utf-8', 'content': 'System reminder: File exists but has empty contents'}}))
    sys.exit(0)

with open(path, 'rb') as f:
    raw = f.read()

try:
    text = raw.decode('utf-8')
except UnicodeDecodeError:
    print(json.dumps({{'encoding': 'base64', 'content': base64.b64encode(raw).decode('ascii')}}))
    sys.exit(0)

file_type = '{file_type}'
if file_type == 'text':
    lines = text.splitlines()
    offset = {offset}
    limit = {limit}
    if offset >= len(lines):
        print(json.dumps({{'error': 'Line offset ' + str(offset) + ' exceeds file length (' + str(len(lines)) + ' lines)'}}))
        sys.exit(0)
    text = chr(10).join(lines[offset:offset + limit])

print(json.dumps({{'encoding': 'utf-8', 'content': text}}))
" 2>&1"""
"""Read file content with server-side pagination.

Runs on the sandbox via `execute()`. Only the requested page is returned,
avoiding full-file transfer for paginated text reads. The path is
base64-encoded; `file_type`, `offset`, and `limit` are interpolated directly
(safe because they come from internal code, not user input).

Output: single-line JSON with either `{{"encoding": ..., "content": ...}}` on
success or `{{"error": ...}}` on failure.
"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with `execute()` as the core abstract method.

    This class provides default implementations for all protocol methods.
    File listing, grep, and glob use shell commands via `execute()`. Read uses
    a server-side Python script via `execute()` for paginated access. Write
    delegates content transfer to `upload_files()`. Edit uses a server-side
    script for small payloads and uploads old/new strings as temp files with
    a server-side replace for large ones.

    Subclasses must implement `execute()`, `upload_files()`, `download_files()`,
    and the `id` property.
    """

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """

    def ls(self, path: str) -> LsResult:
        """Structured listing with file metadata using os.scandir."""
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
        cmd = f"""python3 -c "
import os
import json
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')

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

        return LsResult(entries=file_infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content with server-side line-based pagination.

        Runs a Python script on the sandbox via `execute()` that reads the
        file, detects encoding, and applies offset/limit pagination for text
        files.  Only the requested page is returned over the wire.

        Binary files (non-UTF-8) are returned base64-encoded without
        pagination.

        Args:
            file_path: Absolute path to the file to read.
            offset: Starting line number (0-indexed).

                Only applied to text files.
            limit: Maximum number of lines to return.

                Only applied to text files.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        file_type = _get_file_type(file_path)
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")

        # Coerce to int to prevent injection if callers pass unvalidated strings.
        cmd = _READ_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            file_type=file_type,
            offset=int(offset),
            limit=int(limit),
        )
        result = self.execute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return ReadResult(error=f"File '{file_path}': not found")

        if not isinstance(data, dict):
            return ReadResult(error=f"File '{file_path}': not found")

        if "error" in data:
            return ReadResult(error=f"File '{file_path}': {data['error']}")

        return ReadResult(
            file_data=create_file_data(
                data["content"],
                encoding=data.get("encoding", "utf-8"),
            )
        )

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file, failing if it already exists.

        Runs a small preflight command to check existence and create parent
        directories, then transfers content via `upload_files()`.

        Args:
            file_path: Absolute path for the new file.
            content: UTF-8 text content to write.

        Returns:
            `WriteResult` with `path` on success or `error` on failure.
        """
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
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing exact string occurrences.

        For small payloads (combined old/new under `_EDIT_INLINE_MAX_BYTES`),
        runs a server-side Python script via `execute()` — single round-trip,
        no file transfer.  For larger payloads, uploads old/new strings as
        temp files and runs a server-side replace script — the source file
        never leaves the sandbox.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: The exact substring to find.
            new_string: The replacement string.
            replace_all: If `True`, replace every occurrence.

                If `False` (default), error when more than one
                occurrence exists.

        Returns:
            `EditResult` with `path` and `occurrences` on success, or `error`
                on failure.
        """
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

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents for a literal string using `grep -F`.

        Args:
            pattern: Literal string to search for (not a regex).
            path: Directory or file to search in.

                Defaults to `"."`.
            glob: Optional file-name glob to restrict the search
                (e.g. `'*.py'`).

        Returns:
            `GrepResult` with a list of `GrepMatch` dicts, or `error` on failure.
        """
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
            return GrepResult(matches=[])

        # Parse grep output into GrepMatch objects
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # Format is: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= 3:  # noqa: PLR2004  # Grep output field count
                matches.append(
                    {
                        "path": parts[0],
                        "line": int(parts[1]),
                        "text": parts[2],
                    }
                )

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Structured glob matching returning `GlobResult`."""
        # Encode pattern and path as base64 to avoid escaping issues
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

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

        return GlobResult(matches=file_infos)

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in `FileUploadResponse` objects rather than raising.
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in `FileDownloadResponse` objects rather than raising.
        """
