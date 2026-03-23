"""Harbor sandbox backend for executing commands in Harbor environments."""

import asyncio
import logging
import shlex
import tempfile
from pathlib import Path

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileOperationError,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import check_empty_content, create_file_data
from harbor.environments.base import BaseEnvironment

_SYNC_NOT_SUPPORTED = "This backend only supports async execution. Use the async variant instead."

logger = logging.getLogger(__name__)

DEFAULT_COMMAND_TIMEOUT_SEC = 300
"""Default per-command timeout (5 minutes) to prevent stuck command hangs."""

_PIPE_FIELD_COUNT = 2
"""Expected number of pipe-separated fields in `ls`/`glob` output."""

_GREP_FIELD_COUNT = 3
"""Minimum number of colon-separated fields in `grep` output."""

_COMMAND_PREVIEW_CHAR_LIMIT = 200
"""Maximum chars included in timeout error command previews."""


def _map_file_operation_error(exc: Exception) -> FileOperationError | None:
    """Map known download exceptions to standardized backend error codes."""
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, ValueError):
        return "invalid_path"

    msg = str(exc).lower()
    if "is a directory" in msg:
        return "is_directory"
    if "permission denied" in msg or "access denied" in msg:
        return "permission_denied"
    if "not found" in msg or "no such file" in msg or "does not exist" in msg:
        return "file_not_found"
    if "invalid path" in msg or "invalid argument" in msg:
        return "invalid_path"
    return None


class HarborSandbox(SandboxBackendProtocol):
    """A sandbox implementation using Harbor environments.

    Write and edit use Harbor's native file transfer (upload/download) for
    data content. Read, ls, grep, and glob execute shell commands in the
    environment.
    """

    def __init__(self, environment: BaseEnvironment) -> None:
        """Initialize HarborSandbox with the given environment."""
        self.environment = environment

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # Timeout parameter is forwarded to environment exec, not used as asyncio timeout
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the environment's default timeout.
        """
        timeout_sec = timeout if timeout is not None else DEFAULT_COMMAND_TIMEOUT_SEC
        try:
            if timeout_sec > 0:
                result = await asyncio.wait_for(
                    self.environment.exec(command),
                    timeout=timeout_sec,
                )
            else:
                result = await self.environment.exec(command)
        except TimeoutError:
            return ExecuteResponse(
                output=f"ERROR: Command timed out after {timeout_sec} seconds.\n"
                f"Command: {command[:_COMMAND_PREVIEW_CHAR_LIMIT]}"
                f"{'...' if len(command) > _COMMAND_PREVIEW_CHAR_LIMIT else ''}\n\n"
                f"SUGGESTION: This command is taking too long. Consider:\n"
                f"- Breaking it into smaller steps\n"
                f"- Using a shorter timeout with the timeout_sec parameter\n"
                f"- For package installs: use --no-install-recommends ...\n"
                f"- For long builds: run in background with nohup ...",
                exit_code=124,
                truncated=False,
            )

        # These errors appear in harbor environments when running bash commands
        # in non-interactive/non-TTY contexts. They're harmless artifacts.
        # Filter them from both stdout and stderr, then collect them to show in stderr.
        error_messages = [
            "bash: cannot set terminal process group (-1): Inappropriate ioctl for device",
            "bash: cannot set terminal process group (1): Inappropriate ioctl for device",
            "bash: no job control in this shell",
            "bash: initialize_job_control: no job control in background: Bad file descriptor",
        ]

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Collect the bash messages if they appear (to move to stderr)
        bash_messages = []
        for error_msg in error_messages:
            if error_msg in stdout:
                bash_messages.append(error_msg)
                stdout = stdout.replace(error_msg, "")
            if error_msg in stderr:
                stderr = stderr.replace(error_msg, "")

        stdout = stdout.strip()
        stderr = stderr.strip()

        # Add bash messages to stderr
        if bash_messages:
            bash_msg_text = "\n".join(bash_messages)
            stderr = f"{bash_msg_text}\n{stderr}".strip() if stderr else bash_msg_text

        # Only append stderr label if there's actual stderr content
        if stderr:
            output = stdout + "\n\n stderr: " + stderr if stdout else "\n stderr: " + stderr
        else:
            output = stdout
        return ExecuteResponse(
            output=output,
            exit_code=result.return_code,
        )

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self.environment.session_id

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read raw file content for the requested line range.

        Line-number formatting is applied by the middleware, so this method
        returns unformatted text matching the contract of `ReadResult`.
        """
        safe_path = shlex.quote(file_path)

        cmd = f"""
if [ ! -f {safe_path} ]; then
    echo "Error: File not found"
    exit 1
fi
if [ ! -s {safe_path} ]; then
    exit 0
fi
awk -v offset={offset} -v limit={limit} '
    NR > offset && NR <= offset + limit {{ print }}
    NR > offset + limit {{ exit }}
' {safe_path}
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0 or "Error: File not found" in result.output:
            return ReadResult(error=f"File '{file_path}' not found")

        content = result.output.rstrip()

        empty_msg = check_empty_content(content)
        if empty_msg:
            return ReadResult(file_data=create_file_data(empty_msg))

        return ReadResult(file_data=create_file_data(content))

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content with line numbers using shell commands."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file using Harbor's native file transfer.

        Uses `environment.upload_file()` instead of embedding content in
        the command string, which avoids OS ARG_MAX limits on large payloads.
        """
        safe_path = shlex.quote(file_path)

        # Step 1: existence check + mkdir (small command, no ARG_MAX risk).
        check_cmd = f"""
if [ -e {safe_path} ]; then
    echo 'Error: File already exists: '{safe_path} >&2
    exit 1
fi
mkdir -p "$(dirname {safe_path})" 2>/dev/null
"""
        result = await self.aexecute(check_cmd)

        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        # Step 2: transfer content via Harbor's native file upload
        # (bypasses ARG_MAX entirely — data never touches the command line).
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".tmp", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
        except OSError as exc:
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")
        try:
            await self.environment.upload_file(tmp_path, file_path)
        except Exception as exc:  # noqa: BLE001  # Harbor SDK; any failure → error
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temp file %s", tmp_path, exc_info=True)

        return WriteResult(path=file_path, files_update=None)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file (sync).

        Not supported; use `awrite`.
        """
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Downloads the file via Harbor's native file transfer, performs the
        replacement locally, and re-uploads. This keeps arbitrarily large
        payloads off the command line, avoiding OS `ARG_MAX` limits.
        """
        # Download → local edit → upload (no ARG_MAX risk).
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / "file"
            try:
                await self.environment.download_file(file_path, local)
            except Exception as exc:  # noqa: BLE001  # Harbor SDK; any failure type
                logger.warning("Failed to download %s for editing: %s", file_path, exc)
                return EditResult(error=f"Error: Failed to download '{file_path}': {exc}")

            try:
                text = local.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                return EditResult(error=f"Error: File '{file_path}' is not a text file")

            count = text.count(old_string)
            if count == 0:
                return EditResult(error=f"Error: String not found in file: '{old_string}'")
            if count > 1 and not replace_all:
                return EditResult(
                    error=f"Error: String '{old_string}' appears multiple times. "
                    "Use replace_all=True to replace all occurrences."
                )

            result_text = (
                text.replace(old_string, new_string)
                if replace_all
                else text.replace(old_string, new_string, 1)
            )

            local.write_bytes(result_text.encode("utf-8"))

            try:
                await self.environment.upload_file(local, file_path)
            except Exception as exc:  # noqa: BLE001  # Harbor SDK; any failure → error
                return EditResult(error=f"Error editing file '{file_path}': {exc}")

        return EditResult(path=file_path, files_update=None, occurrences=count)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences (sync).

        Not supported; use `aedit`.
        """
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def als(self, path: str) -> LsResult:
        """List directory contents with metadata using shell commands."""
        safe_path = shlex.quote(path)

        cmd = f"""
if [ ! -d {safe_path} ]; then
    exit 1
fi
for entry in {safe_path}/*; do
    if [ -e "$entry" ]; then
        name=$(basename "$entry")
        if [ -d "$entry" ]; then
            printf '%s|true\\n' "$name"
        else
            printf '%s|false\\n' "$name"
        fi
    fi
done
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0:
            detail = result.output.strip() if result.output else ""
            return LsResult(
                error=f"Directory not found or not accessible: {path}"
                + (f" ({detail})" if detail else "")
            )

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == _PIPE_FIELD_COUNT:
                file_infos.append({"path": parts[0], "is_dir": parts[1] == "true"})
            else:
                logger.debug("Skipping malformed ls output line: %r", line)

        return LsResult(entries=file_infos)

    def ls(self, path: str) -> LsResult:
        """List directory contents with metadata using shell commands."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for pattern in files using grep."""
        search_path = shlex.quote(path or ".")

        # Build grep command
        grep_opts = "-rHn"  # recursive, with filename, with line number

        # Add glob pattern if specified
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include={shlex.quote(glob)}"

        # Escape pattern for grep
        safe_pattern = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {safe_pattern} {search_path} 2>/dev/null"
        result = await self.aexecute(cmd)

        # grep exit codes: 0=matches found, 1=no matches, 2+=error
        if result.exit_code is not None and result.exit_code > 1:
            detail = result.output.strip() if result.output else ""
            return GrepResult(
                error=f"Grep failed (exit {result.exit_code})" + (f": {detail}" if detail else "")
            )

        output = result.output.rstrip()
        if not output:
            return GrepResult(matches=[])

        # Parse grep output into GrepMatch objects
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # Format is: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= _GREP_FIELD_COUNT:
                try:
                    matches.append(
                        {
                            "path": parts[0],
                            "line": int(parts[1]),
                            "text": parts[2],
                        }
                    )
                except ValueError:
                    logger.debug("Skipping malformed grep output line: %r", line)
                    continue

        return GrepResult(matches=matches)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for pattern in files using grep."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching glob pattern using shell commands.

        Please note that this implementation does not currently support all glob
        patterns.
        """
        safe_path = shlex.quote(path)
        safe_pattern = shlex.quote(pattern)

        cmd = f"""
cd {safe_path} 2>/dev/null || exit 1
# Use find with shell globbing
for file in {safe_pattern}; do
    if [ -e "$file" ]; then
        if [ -d "$file" ]; then
            printf '%s|true\\n' "$file"
        else
            printf '%s|false\\n' "$file"
        fi
    fi
done
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0:
            detail = result.output.strip() if result.output else ""
            return GlobResult(
                error=f"Path not found or not accessible: {path}"
                + (f" ({detail})" if detail else "")
            )

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

        # Parse output into FileInfo dicts
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == _PIPE_FIELD_COUNT:
                file_infos.append(
                    {
                        "path": parts[0],
                        "is_dir": parts[1] == "true",
                    }
                )
            else:
                logger.debug("Skipping malformed glob output line: %r", line)

        return GlobResult(matches=file_infos)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching glob pattern using shell commands."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    # -- file transfer via Harbor's native upload/download -------------------

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files using Harbor's native file transfer."""
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".tmp", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
            except OSError as exc:
                logger.warning("Failed to create temp file for upload %s: %s", path, exc)
                results.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                await self.environment.upload_file(tmp_path, path)
                results.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:
                error = _map_file_operation_error(exc)
                if error is None:
                    raise
                logger.warning("Failed to upload %s: %s", path, exc)
                results.append(FileUploadResponse(path=path, error=error))
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.warning("Failed to clean up temp file %s", tmp_path, exc_info=True)
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files (sync). Not supported; use `aupload_files`."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files using Harbor's native file transfer."""
        results: list[FileDownloadResponse] = []
        for path in paths:
            with tempfile.TemporaryDirectory() as tmpdir:
                local = Path(tmpdir) / (Path(path).name or "file")
                try:
                    await self.environment.download_file(path, local)
                    content = local.read_bytes()
                    results.append(FileDownloadResponse(path=path, content=content, error=None))
                except Exception as exc:
                    error = _map_file_operation_error(exc)
                    if error is None:
                        raise
                    logger.warning("Failed to download %s: %s", path, exc)
                    results.append(FileDownloadResponse(path=path, content=None, error=error))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files (sync). Not supported; use `adownload_files`."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)
