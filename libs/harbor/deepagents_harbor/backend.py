"""Implement harbor backend."""

import asyncio
import base64
import hashlib
import json
import os
import re
import shlex
import time

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import truncate_execute_output
from harbor.environments.base import BaseEnvironment

# Default per-command timeout (5 minutes) - prevents hanging on stuck commands
DEFAULT_COMMAND_TIMEOUT_SEC = 300
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})

# Patterns for commands that produce build artifacts needing cleanup
# These patterns match when the command STARTS with or has the build tool as a primary command
# (not when it's just a package name in apt-get/pip)
# Note: For g++, we use (?:\s|$) instead of \b since word boundary doesn't work after +
BUILD_ARTIFACT_PATTERNS = [
    r"^gcc\b",                 # gcc at start of command
    r"^g\+\+(?:\s|$)",         # g++ at start of command followed by space or end
    r"^make\b",                # make at start of command
    r"^cmake\b",               # cmake at start of command
    r"^cargo\s+build\b",       # cargo build at start of command
    r"^rustc\b",               # rustc at start of command
    r"&&\s*gcc\b",             # gcc after && (chained command)
    r"&&\s*g\+\+(?:\s|$)",     # g++ after && (chained command)
    r"&&\s*make\b",            # make after && (chained command)
    r"&&\s*cmake\b",           # cmake after && (chained command)
    r";\s*gcc\b",              # gcc after ; (chained command)
    r";\s*g\+\+(?:\s|$)",      # g++ after ; (chained command)
    r";\s*make\b",             # make after ; (chained command)
    r";\s*cmake\b",            # cmake after ; (chained command)
]


class HarborSandbox(SandboxBackendProtocol):
    """A sandbox implementation using shell commands.

    Note: The edit operation requires python3 for JSON parsing. Other operations
    (read, write, ls, grep, glob) use only standard shell utilities.
    """

    def __init__(self, environment: BaseEnvironment) -> None:
        """Initialize HarborSandbox with the given environment."""
        self.environment = environment

    async def aexecute(
        self,
        command: str,
        timeout_sec: int = DEFAULT_COMMAND_TIMEOUT_SEC,
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment.

        Args:
            command: The bash command to execute
            timeout_sec: Maximum time in seconds to wait for the command (default: 300s)
                        Set to 0 to disable timeout (not recommended)
        """
        # Apply per-command timeout to prevent hanging on stuck commands
        # This is critical for commands like apt-get that can hang indefinitely
        try:
            if timeout_sec > 0:
                result = await asyncio.wait_for(
                    self.environment.exec(command),
                    timeout=timeout_sec
                )
            else:
                result = await self.environment.exec(command)
        except asyncio.TimeoutError:
            return ExecuteResponse(
                output=f"ERROR: Command timed out after {timeout_sec} seconds.\n"
                       f"Command: {command[:200]}{'...' if len(command) > 200 else ''}\n\n"
                       f"SUGGESTION: This command is taking too long. Consider:\n"
                       f"- Breaking it into smaller steps\n"
                       f"- Using a shorter timeout with the timeout_sec parameter\n"
                       f"- For package installs: use --no-install-recommends or check if already installed\n"
                       f"- For long builds: run in background with nohup or use pre-built binaries",
                exit_code=124,  # Standard timeout exit code
                truncated=False,
            )

        # Check if this was a build command that may leave artifacts
        is_build_command = any(re.search(pattern, command) for pattern in BUILD_ARTIFACT_PATTERNS)

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

        # Truncate stdout if too long (prevents context overflow from verbose commands)
        # Save full output to a file so agent can access it if needed
        truncated = False
        if stdout and stdout.count('\n') > 200:
            # Save full output to a unique file before truncating
            hash_suffix = hashlib.md5(f"{time.time()}{command}".encode()).hexdigest()[:8]
            saved_path = f"/tmp/full_output_{hash_suffix}.txt"
            try:
                await self.environment.exec(
                    f"cat > {saved_path} << 'DEEPAGENTS_EOF'\n{stdout}\nDEEPAGENTS_EOF"
                )
            except Exception:
                saved_path = None  # Don't reference if save failed

            stdout, truncated = truncate_execute_output(stdout, saved_path=saved_path)

        # Only append stderr label if there's actual stderr content
        if stderr:
            output = stdout + "\n\n stderr: " + stderr if stdout else "\n stderr: " + stderr
        else:
            output = stdout

        # Add cleanup reminder after successful build commands
        # This helps agents remember to remove intermediate files before verification
        if is_build_command and result.return_code == 0:
            cleanup_reminder = (
                "\n\n[SYSTEM REMINDER: Build completed. Before finishing, ensure you:\n"
                "  1. Remove intermediate files (.o, .a, build/, __pycache__/) if they interfere with verification\n"
                "  2. Keep only the final executable/output file\n"
                "  3. Run 'make clean' or equivalent if provided by the build system]"
            )
            output = output + cleanup_reminder

        return ExecuteResponse(
            output=output,
            exit_code=result.return_code,
            truncated=truncated,
        )

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment."""
        raise NotImplementedError("This backend only supports async execution")

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self.environment.session_id

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using shell commands."""
        # Escape file path for shell
        safe_path = shlex.quote(file_path)

        # Check if file exists and handle empty files
        cmd = f"""
if [ ! -f {safe_path} ]; then
    echo "Error: File not found"
    exit 1
fi
if [ ! -s {safe_path} ]; then
    echo "System reminder: File exists but has empty contents"
    exit 0
fi
# Use awk to add line numbers and handle offset/limit
awk -v offset={offset} -v limit={limit} '
    NR > offset && NR <= offset + limit {{
        printf "%6d\\t%s\\n", NR, $0
    }}
    NR > offset + limit {{ exit }}
' {safe_path}
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0 or "Error: File not found" in result.output:
            return f"Error: File '{file_path}' not found"

        return result.output.rstrip()

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using shell commands."""
        raise NotImplementedError("Use aread instead")

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file using shell commands."""
        # Encode content as base64 to avoid escaping issues
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        safe_path = shlex.quote(file_path)

        # Use heredoc to pass content via stdin to avoid ARG_MAX limits on large files.
        # ARG_MAX limits the total size of command-line arguments.
        # Heredocs bypass this by passing data through stdin rather than as arguments.
        cmd = f"""
if [ -e {safe_path} ]; then
    echo "Error: File '{file_path}' already exists" >&2
    exit 1
fi
parent_dir=$(dirname {safe_path})
mkdir -p "$parent_dir" 2>/dev/null
if ! base64 -d > {safe_path} <<'__DEEPAGENTS_EOF__'
{content_b64}
__DEEPAGENTS_EOF__
then
    echo "Error: Failed to decode content for file '{file_path}'" >&2
    exit 1
fi
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        return WriteResult(path=file_path, files_update=None)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file using shell commands."""
        raise NotImplementedError("Use awrite instead")

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing exact string occurrences using Python.

        Uses a small Python helper inside the sandbox to avoid shell/grep/perl
        edge cases with multiline strings and special characters.
        """
        # Create JSON payload with old/new strings and replacement mode.
        payload = json.dumps(
            {"old": old_string, "new": new_string, "replace_all": replace_all}
        )
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        safe_path = shlex.quote(file_path)

        # Use Python for counting and replacement so multiline old/new strings
        # behave exactly like in-memory string replacement.
        cmd = f"""
if [ ! -f {safe_path} ]; then
    exit 3
fi

python3 - {safe_path} '{payload_b64}' <<'__DEEPAGENTS_EOF__'
import base64
import json
import pathlib
import sys

file_path = pathlib.Path(sys.argv[1])
payload_b64 = sys.argv[2]

try:
    payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
except Exception as exc:
    print(f"Error: Failed to decode edit payload: {{exc}}")
    raise SystemExit(4) from exc

try:
    content = file_path.read_text(encoding="utf-8")
except Exception as exc:
    print(f"Error: Failed to read file for edit: {{exc}}")
    raise SystemExit(5) from exc

old = payload["old"]
new = payload["new"]
replace_all = bool(payload.get("replace_all", False))

count = content.count(old)
if count == 0:
    raise SystemExit(1)
if count > 1 and not replace_all:
    raise SystemExit(2)

if replace_all:
    updated_content = content.replace(old, new)
else:
    updated_content = content.replace(old, new, 1)

try:
    file_path.write_text(updated_content, encoding="utf-8")
except Exception as exc:
    print(f"Error: Failed to write file for edit: {{exc}}")
    raise SystemExit(5) from exc

print(count)
__DEEPAGENTS_EOF__
"""
        result = await self.aexecute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        if exit_code == 1:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if exit_code == 2:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences."
            )
        if exit_code == 3:
            return EditResult(error=f"Error: File '{file_path}' not found")
        if exit_code == 4:
            return EditResult(error=f"Error: Failed to decode edit payload: {output}")
        if exit_code == 5:
            return EditResult(error=f"Error: Failed during edit file I/O: {output}")
        if exit_code != 0:
            return EditResult(
                error=f"Error editing file (exit code {exit_code}): {output or 'Unknown error'}"
            )

        try:
            count = int(output.split("\n")[0])
        except (ValueError, IndexError):
            count = 1

        return EditResult(path=file_path, files_update=None, occurrences=count)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences using shell commands."""
        raise NotImplementedError("Use aedit instead")

    async def als_info(self, path: str) -> list[FileInfo]:
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
            return []

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                file_infos.append({"path": parts[0], "is_dir": parts[1] == "true"})

        return file_infos

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents with metadata using shell commands."""
        raise NotImplementedError("Use als_info instead")

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
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

        cmd = f"grep {grep_opts} {glob_pattern} -e {safe_pattern} {search_path} 2>/dev/null || true"
        result = await self.aexecute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        # Parse grep output into GrepMatch objects
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # Format is: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    matches.append(
                        {
                            "path": parts[0],
                            "line": int(parts[1]),
                            "text": parts[2],
                        }
                    )
                except ValueError:
                    continue

        return matches

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files using grep."""
        raise NotImplementedError("Use agrep_raw instead")

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
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
            return []

        output = result.output.strip()
        if not output:
            return []

        # Parse output into FileInfo dicts
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                file_infos.append(
                    {
                        "path": parts[0],
                        "is_dir": parts[1] == "true",
                    }
                )

        return file_infos

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching glob pattern using shell commands."""
        raise NotImplementedError("Use aglob_info instead")

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files as raw bytes using base64 encoding over shell.

        This is primarily used for binary files like images that need to be
        returned as multimodal content.
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            safe_path = shlex.quote(path)

            # Check if file exists and read as base64
            # Use environment.exec directly to bypass truncation (base64 must not be truncated)
            cmd = f"""
if [ ! -f {safe_path} ]; then
    echo "FILE_NOT_FOUND"
    exit 1
fi
base64 {safe_path}
"""
            result = await self.environment.exec(cmd)

            if result.return_code != 0 or "FILE_NOT_FOUND" in (result.stdout or ""):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="file_not_found")
                )
            else:
                try:
                    # Normalize whitespace so wrapped base64 output can be decoded.
                    b64_data = "".join((result.stdout or "").split())
                    if not b64_data:
                        responses.append(
                            FileDownloadResponse(path=path, content=None, error="empty_output")
                        )
                        continue

                    # Decode the base64 output to get raw bytes
                    # validate=True catches noisy/corrupted output rather than silently
                    # dropping non-base64 characters.
                    content = base64.b64decode(b64_data, validate=True)

                    # Only apply image integrity checks for known image extensions.
                    # adownload_files() is used by other middleware too (e.g. text
                    # offloading for summarization), so non-image files must pass through.
                    ext = os.path.splitext(path)[1].lower()
                    if ext in IMAGE_EXTENSIONS and not _is_valid_image(content):
                        responses.append(
                            FileDownloadResponse(path=path, content=None, error="invalid_image_format")
                        )
                        continue

                    responses.append(
                        FileDownloadResponse(path=path, content=content, error=None)
                    )
                except Exception as e:
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error=f"decode_error: {e}")
                    )

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files as raw bytes."""
        raise NotImplementedError("Use adownload_files instead")


def _is_valid_image(data: bytes) -> bool:
    """Check if data is a valid image by examining signatures and footers.

    This is intentionally lightweight but stricter than header-only checks.
    It catches common truncation cases that otherwise produce model API errors.
    """
    if len(data) < 12:
        return False

    # PNG magic bytes
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return data.endswith(b'IEND\xaeB`\x82')

    # JPEG magic bytes (FFD8FF)
    if data[:3] == b'\xff\xd8\xff':
        return data[-2:] == b'\xff\xd9'

    # GIF magic bytes
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return data[-1:] == b';'

    # WebP magic bytes (RIFF....WEBP)
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        if len(data) < 16:
            return False
        riff_size = int.from_bytes(data[4:8], byteorder="little", signed=False)
        return riff_size + 8 == len(data)

    return False
