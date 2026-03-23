"""Base sandbox implementation.

Provides `BaseSandbox`, a base class that implements
`SandboxBackendProtocol`. File listing, grep, glob, and read use shell
commands via `execute()`. Write delegates content transfer to
`upload_files()`. Edit uses server-side `execute()` for small payloads
and falls back to `download_files()` / `upload_files()` for large ones.
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

_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys, os, base64, json

payload = json.loads(base64.b64decode(sys.stdin.read().strip()).decode('utf-8'))
path, old, new = payload['path'], payload['old'], payload['new']
replace_all = payload.get('replace_all', False)

if not os.path.isfile(path):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)

with open(path, 'rb') as f:
    raw = f.read()

try:
    text = raw.decode('utf-8')
except UnicodeDecodeError:
    print(json.dumps({{'error': 'not_a_text_file'}}))
    sys.exit(0)

count = text.count(old)
if count == 0:
    print(json.dumps({{'error': 'string_not_found'}}))
    sys.exit(0)
if count > 1 and not replace_all:
    print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
    sys.exit(0)

result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
with open(path, 'wb') as f:
    f.write(result.encode('utf-8'))

print(json.dumps({{'count': count}}))
" 2>/dev/null <<'__DEEPAGENTS_EDIT_EOF__'
{payload_b64}
__DEEPAGENTS_EDIT_EOF__"""
"""Server-side file edit via `execute()`.

Reads the file, performs string replacement, and writes back — all on the
sandbox.  The payload (path, old/new strings, replace_all flag) is passed as
base64-encoded JSON via heredoc stdin to avoid shell escaping issues.

Output: single-line JSON with ``{{"count": N}}`` on success or
``{{"error": ...}}`` on failure.

Used for payloads under `_EDIT_INLINE_MAX_BYTES`; larger payloads fall back
to `download_files()` + local replace + `upload_files()`.
"""

# Maximum combined byte size of old_string + new_string for inline server-side
# edit.  Payloads above this fall back to download + local replace + upload.
_EDIT_INLINE_MAX_BYTES = 50_000

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
" 2>/dev/null"""
"""Read file content with server-side pagination.

Runs on the sandbox via `execute()`. Only the requested page is returned,
avoiding full-file transfer for paginated text reads. Uses base64-encoded
path parameter to avoid shell escaping issues.

Output: single-line JSON with either ``{{"encoding": ..., "content": ...}}``
on success or ``{{"error": ...}}`` on failure.
"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with `execute()` as the core abstract method.

    This class provides default implementations for all protocol methods.
    File listing, grep, and glob use shell commands via `execute()`. Read uses
    a server-side Python script via `execute()` for paginated access. Write
    delegates content transfer to `upload_files()`. Edit uses a server-side
    script for small payloads and falls back to file transfer for large ones.

    Subclasses must implement `execute()` and the `id` property. Default
    `upload_files()` / `download_files()` implementations use `execute()` with
    base64 encoding; subclasses with native transfer APIs should override them
    for efficiency.
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

        cmd = _READ_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            file_type=file_type,
            offset=offset,
            limit=limit,
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

        For small payloads (combined old/new under `_EDIT_INLINE_MAX_BYTES`),
        runs a server-side Python script via `execute()` — single round-trip,
        no file transfer.  For larger payloads, falls back to
        `download_files()` + local replace + `upload_files()`.

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
        payload_size = len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))

        if payload_size <= _EDIT_INLINE_MAX_BYTES:
            return self._edit_inline(file_path, old_string, new_string, replace_all)

        return self._edit_via_transfer(file_path, old_string, new_string, replace_all)

    def _edit_inline(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """Server-side replace via `execute()` — single round-trip."""
        payload = json.dumps(
            {
                "path": file_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
            }
        )
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
        result = self.execute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return EditResult(error=f"Error: File '{file_path}' not found")

        if not isinstance(data, dict):
            return EditResult(error=f"Error: File '{file_path}' not found")

        if "error" in data:
            return self._map_edit_error(data["error"], file_path, old_string)

        return EditResult(
            path=file_path,
            files_update=None,
            occurrences=data.get("count", 1),
        )

    def _edit_via_transfer(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """Download + local replace + upload for large payloads."""
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
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.",
            )

        result_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)

        upload_resp = self.upload_files([(file_path, result_text.encode("utf-8"))])
        upload_err = upload_resp[0].error if upload_resp else "upload returned no response"
        if upload_err:
            return EditResult(error=f"Error editing file '{file_path}': {upload_err}")

        return EditResult(path=file_path, files_update=None, occurrences=count)

    @staticmethod
    def _map_edit_error(error: str, file_path: str, old_string: str) -> EditResult:
        """Map server-side error codes to `EditResult` objects."""
        if error == "file_not_found":
            return EditResult(
                error=f"Error: Failed to read '{file_path}': file_not_found",
            )
        if error == "not_a_text_file":
            return EditResult(
                error=f"Error: File '{file_path}' is not a text file",
            )
        if error == "string_not_found":
            return EditResult(
                error=f"Error: String not found in file: '{old_string}'",
            )
        if error == "multiple_occurrences":
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.",
            )
        return EditResult(error=f"Error editing file '{file_path}': {error}")

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

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files via `execute()` with base64 stdin.

        Default implementation transfers each file by running a Python script
        on the sandbox that reads base64-encoded content from a heredoc and
        writes it to disk.

        Subclasses with native file transfer APIs (e.g. Modal, Daytona) should
        override this for efficiency.  The heredoc approach embeds the full
        base64 payload in the command string, which works well for local and
        moderate-size transfers but may hit limits on sandboxes that relay the
        command over HTTP.

        Args:
            files: List of `(path, content)` tuples to upload.

        Returns:
            List of `FileUploadResponse` objects, one per input file.
        """
        responses: list[FileUploadResponse] = []
        for path, data in files:
            path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
            content_b64 = base64.b64encode(data).decode("ascii")
            cmd = (
                f'python3 -c "\n'
                f"import os, sys, base64\n"
                f"path = base64.b64decode('{path_b64}').decode('utf-8')\n"
                f"data = base64.b64decode(sys.stdin.read().strip())\n"
                f"os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
                f"with open(path, 'wb') as f:\n"
                f"    f.write(data)\n"
                f"\" <<'__DEEPAGENTS_UPLOAD_EOF__'\n"
                f"{content_b64}\n"
                f"__DEEPAGENTS_UPLOAD_EOF__"
            )
            result = self.execute(cmd)
            if result.exit_code != 0:
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=result.output.strip() or "upload failed",
                    )
                )
            else:
                responses.append(FileUploadResponse(path=path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files via `execute()` with base64 stdout.

        Default implementation reads each file by running a Python script on
        the sandbox that base64-encodes the content and writes it to stdout.

        Subclasses with native file transfer APIs should override this for
        efficiency.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
            cmd = (
                f'python3 -c "\n'
                f"import sys, base64\n"
                f"path = base64.b64decode('{path_b64}').decode('utf-8')\n"
                f"with open(path, 'rb') as f:\n"
                f"    sys.stdout.write(base64.b64encode(f.read()).decode('ascii'))\n"
                f'" 2>/dev/null'
            )
            result = self.execute(cmd)
            if result.exit_code != 0:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            else:
                try:
                    content = base64.b64decode(result.output)
                    responses.append(FileDownloadResponse(path=path, content=content))
                except Exception:  # noqa: BLE001
                    responses.append(
                        FileDownloadResponse(
                            path=path,
                            error="failed to decode file content",
                        )
                    )
        return responses
