"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.

It also defines the BaseSandbox implementation used by the CLI sandboxes.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import shlex
from abc import ABC, abstractmethod

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

log = logging.getLogger("deepagents")

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
        replace_all: bool = False,  # noqa: FBT001, FBT002
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
            if len(parts) >= 3:  # noqa: PLR2004  # Grep output field count
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

    # -- File transfer via execute() -----------------------------------------
    #
    # Default implementations that use base64-encoded execute() calls.
    # This works with any sandbox backend (Docker tmpfs, microsandbox, etc.)
    # because execute() enters the sandbox's mount namespace.
    #
    # Subclasses may override these if the backend provides a more efficient
    # native file transfer mechanism (e.g. SSH's SFTP, Daytona's REST API).

    # Maximum base64 payload size for a single heredoc command.
    # Linux ARG_MAX is ~2MB, but the full command includes the Python
    # one-liner overhead. Stay well under that with 64KB chunks of raw
    # bytes (~87KB after base64 encoding).
    _UPLOAD_CHUNK_BYTES = 65_536

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files via base64-encoded execute() calls.

        Small files (under ``_UPLOAD_CHUNK_BYTES``) are written in a single
        command. Larger files are split into base64 chunks that each fit
        within the kernel's ARG_MAX limit, assembled into a temp file
        inside the sandbox, then decoded to the final path.

        Subclasses may override this if the backend provides a native file
        transfer mechanism.

        Args:
            files: List of (absolute_path, content_bytes) tuples.

        Returns:
            List of FileUploadResponse objects (one per input file).
        """
        responses: list[FileUploadResponse] = []

        for path, content in files:
            b64 = base64.b64encode(content).decode("ascii")

            result = self._upload_single(path, b64) if len(b64) <= self._UPLOAD_CHUNK_BYTES else self._upload_chunked(path, b64)

            if result.exit_code == 0:
                responses.append(FileUploadResponse(path=path))
            else:
                log.warning("Failed to upload %s: %s", path, result.output)
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )

        return responses

    def _upload_single(self, file_path: str, b64: str) -> ExecuteResponse:
        """Upload a small file in one execute() call.

        Args:
            file_path: Absolute path inside the sandbox.
            b64: Base64-encoded file content (must fit within ARG_MAX).

        Returns:
            ExecuteResponse from the sandbox.
        """
        cmd = (
            'python3 -c "\n'
            "import base64, pathlib, sys\n"
            "b64 = sys.stdin.read().strip()\n"
            "p = pathlib.Path('" + file_path + "')\n"
            "p.parent.mkdir(parents=True, exist_ok=True)\n"
            "p.write_bytes(base64.b64decode(b64))\n"
            "\" <<'__DEEPAGENTS_EOF__'\n"
            + b64 + "\n"
            "__DEEPAGENTS_EOF__"
        )
        return self.execute(cmd)

    def _upload_chunked(self, file_path: str, b64: str) -> ExecuteResponse:
        """Upload a large file by writing base64 chunks then decoding.

        Splits the base64 string into chunks that each fit within ARG_MAX,
        appends each chunk to a temp file inside the sandbox, then decodes
        the assembled base64 to the final destination.

        Args:
            file_path: Absolute path inside the sandbox.
            b64: Base64-encoded file content.

        Returns:
            ExecuteResponse from the final decode step.
        """
        tmp_b64 = file_path + ".__b64_tmp"

        # Ensure parent directory exists.
        result = self.execute(
            'python3 -c "'
            "import pathlib; "
            "pathlib.Path('" + file_path + "').parent.mkdir(parents=True, exist_ok=True)"
            '"'
        )
        if result.exit_code != 0:
            return result

        # Write base64 data in chunks that fit within ARG_MAX.
        chunk_size = self._UPLOAD_CHUNK_BYTES
        for i in range(0, len(b64), chunk_size):
            chunk = b64[i : i + chunk_size]
            append_cmd = (
                'python3 -c "\n'
                "import sys\n"
                "chunk = sys.stdin.read().strip()\n"
                "with open('" + tmp_b64 + "', 'a') as f:\n"
                "    f.write(chunk)\n"
                "\" <<'__DEEPAGENTS_EOF__'\n"
                + chunk + "\n"
                "__DEEPAGENTS_EOF__"
            )
            result = self.execute(append_cmd)
            if result.exit_code != 0:
                self.execute("rm -f '" + tmp_b64 + "'")
                return result

        # Decode the assembled base64 file to the final path.
        decode_cmd = (
            'python3 -c "'
            "import base64, pathlib; "
            "b64 = pathlib.Path('" + tmp_b64 + "').read_text(); "
            "pathlib.Path('" + file_path + "').write_bytes(base64.b64decode(b64)); "
            "pathlib.Path('" + tmp_b64 + "').unlink()"
            '"'
        )
        return self.execute(decode_cmd)

    # Maximum raw bytes to read per chunk during download.
    # Each chunk is base64-encoded in the sandbox and printed to stdout.
    # 64KB raw -> ~87KB base64, well within typical output truncation limits.
    _DOWNLOAD_CHUNK_BYTES = 65_536

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files via base64-encoded execute() calls.

        Checks each file's size first. Small files are downloaded in a
        single command. Larger files are read in chunks inside the sandbox,
        each chunk base64-encoded and printed to stdout, then reassembled
        on the host. This avoids output truncation by execute().

        Subclasses may override this if the backend provides a native file
        transfer mechanism.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            List of FileDownloadResponse objects (one per input path).
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            # Get file size to decide between single and chunked download.
            size_cmd = (
                'python3 -c "'
                "import os; "
                "print(os.path.getsize('" + path + "'))"
                '"'
            )
            size_result = self.execute(size_cmd)
            if size_result.exit_code != 0:
                responses.append(
                    FileDownloadResponse(path=path, error="file_not_found")
                )
                continue

            try:
                file_size = int(size_result.output.strip())
            except ValueError:
                responses.append(
                    FileDownloadResponse(path=path, error="file_not_found")
                )
                continue

            # Small files can be downloaded in one shot.
            response = self._download_single(path) if file_size <= self._DOWNLOAD_CHUNK_BYTES else self._download_chunked(path, file_size)
            responses.append(response)

        return responses

    def _download_single(self, file_path: str) -> FileDownloadResponse:
        """Download a small file in one execute() call.

        Args:
            file_path: Absolute path inside the sandbox.

        Returns:
            FileDownloadResponse with content or error.
        """
        cmd = (
            'python3 -c "'
            "import base64; "
            "print(base64.b64encode(open('" + file_path + "', 'rb').read()).decode())"
            '"'
        )
        result = self.execute(cmd)
        if result.exit_code == 0 and result.output.strip():
            try:
                content = base64.b64decode(result.output.strip())
                return FileDownloadResponse(path=file_path, content=content)
            except (ValueError, binascii.Error):
                return FileDownloadResponse(path=file_path, error="file_not_found")
        return FileDownloadResponse(path=file_path, error="file_not_found")

    def _download_chunked(
        self, file_path: str, file_size: int
    ) -> FileDownloadResponse:
        """Download a large file in chunks to avoid output truncation.

        Reads the file in fixed-size binary chunks inside the sandbox,
        base64-encodes each chunk, and reassembles them on the host.

        Args:
            file_path: Absolute path inside the sandbox.
            file_size: Total file size in bytes.

        Returns:
            FileDownloadResponse with content or error.
        """
        chunks: list[bytes] = []
        offset = 0

        while offset < file_size:
            chunk_cmd = (
                'python3 -c "'
                "import base64; "
                "f = open('" + file_path + "', 'rb'); "
                "f.seek(" + str(offset) + "); "
                "print(base64.b64encode(f.read(" + str(self._DOWNLOAD_CHUNK_BYTES) + ")).decode())"
                '"'
            )
            result = self.execute(chunk_cmd)
            if result.exit_code != 0 or not result.output.strip():
                return FileDownloadResponse(path=file_path, error="file_not_found")

            try:
                chunk = base64.b64decode(result.output.strip())
            except (ValueError, binascii.Error):
                return FileDownloadResponse(path=file_path, error="file_not_found")

            chunks.append(chunk)
            offset += len(chunk)

            # Safety: if we got zero bytes, the file ended.
            if not chunk:
                break

        return FileDownloadResponse(path=file_path, content=b"".join(chunks))
