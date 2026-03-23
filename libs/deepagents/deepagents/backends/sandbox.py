"""Base sandbox implementation.

Provides `BaseSandbox`, a base class that implements
`SandboxBackendProtocol`. File listing, grep, and glob use shell commands via
`execute()`. Read, write, and edit delegate data transfer to
`download_files()` / `upload_files()`.
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
" 2>/dev/null"""
"""Find files matching a pattern with metadata.

Uses base64-encoded parameters to avoid shell escaping issues.
"""

_WRITE_CHECK_TEMPLATE = """python3 -c "
import os, sys, base64

path = base64.b64decode('{path_b64}').decode('utf-8')
if os.path.exists(path):
    print('Error: File already exists: ' + repr(path), file=sys.stderr)
    sys.exit(1)
os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
" 2>&1"""
"""Preflight check for write operations: verify the target file does not already
exist and create parent directories.

Only the (small) base64-encoded *path* is interpolated — file content is
transferred separately via `upload_files()`.
"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with `execute()` as abstract method.

    This class provides default implementations for all protocol methods.

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
        """Read file content via `download_files()` with line-based pagination.

        Downloads file, then applies offset/limit slicing locally.

        Binary files (non-UTF-8) are returned base64-encoded without slicing.

        Args:
            file_path: Absolute path to the file to read.
            offset: Starting line number (0-indexed).

                Only applied to text files.
            limit: Maximum number of lines to return.

                Only applied to text files.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        responses = self.download_files([file_path])
        resp = responses[0] if responses else None
        if not resp or resp.error or resp.content is None:
            detail = resp.error if resp and resp.error else "not found"
            return ReadResult(error=f"File '{file_path}': {detail}")

        raw = resp.content

        if len(raw) == 0:
            return ReadResult(
                file_data=create_file_data(
                    "System reminder: File exists but has empty contents",
                    encoding="utf-8",
                )
            )

        try:
            content = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(raw).decode("ascii")
            encoding = "base64"

        file_type = _get_file_type(file_path)
        if encoding == "utf-8" and file_type == "text":
            lines = content.splitlines()
            if offset >= len(lines):
                return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")
            content = "\n".join(lines[offset : offset + limit])

        return ReadResult(file_data=create_file_data(content, encoding=encoding))

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
        # Step 1: existence check + mkdir
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
        check_cmd = _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)
        result = self.execute(check_cmd)

        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        # Step 2: transfer content via upload_files()
        responses = self.upload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            return WriteResult(error=f"Failed to write file '{file_path}': upload returned no response")
        if responses[0].error:
            return WriteResult(error=f"Failed to write file '{file_path}': {responses[0].error}")

        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing exact string occurrences.

        Downloads the file via `download_files()`, performs the replacement
        locally, and re-uploads via `upload_files()`.

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
        # Download → local edit → upload to bypass ARG_MAX on large payloads.
        responses = self.download_files([file_path])
        resp = responses[0] if responses else None
        if not resp or resp.error or resp.content is None:
            detail = resp.error if resp and resp.error else "not found"
            return EditResult(error=f"Error: Failed to read '{file_path}': {detail}")

        try:
            text = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            return EditResult(error=f"Error: File '{file_path}' is not a text file")

        count = text.count(old_string)
        if count == 0:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if count > 1 and not replace_all:
            return EditResult(error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.")

        result_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)

        upload_resp = self.upload_files([(file_path, result_text.encode("utf-8"))])
        upload_err = upload_resp[0].error if upload_resp else "upload returned no response"
        if upload_err:
            return EditResult(error=f"Error editing file '{file_path}': {upload_err}")

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
