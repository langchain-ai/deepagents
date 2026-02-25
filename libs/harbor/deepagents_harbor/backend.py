"""Harbor sandbox backend for executing commands in Harbor environments."""

import base64
import json
import shlex

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from harbor.environments.base import BaseEnvironment

_SYNC_NOT_SUPPORTED = "This backend only supports async execution. Use the async variant instead."

# Shell exit codes used by the aedit command script
_EXIT_NOT_FOUND = 1
_EXIT_MULTIPLE_MATCHES = 2
_EXIT_FILE_MISSING = 3
_EXIT_DECODE_FAILED = 4

# Parsing constants
_PIPE_FIELD_COUNT = 2  # expected number of pipe-separated fields in ls/glob output
_GREP_FIELD_COUNT = 3  # minimum number of colon-separated fields in grep output


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
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # Timeout parameter is forwarded to environment exec, not used as asyncio timeout
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the environment's default timeout.
        """
        result = await self.environment.exec(command, timeout_sec=timeout)

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
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

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
    echo "Error: File '"{safe_path}"' already exists" >&2
    exit 1
fi
parent_dir=$(dirname {safe_path})
mkdir -p "$parent_dir" 2>/dev/null
if ! base64 -d > {safe_path} <<'__DEEPAGENTS_EOF__'
{content_b64}
__DEEPAGENTS_EOF__
then
    echo "Error: Failed to decode content for file '"{safe_path}"' " >&2
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
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences using shell commands."""
        # Create JSON payload with old and new strings, then base64 encode
        payload = json.dumps({"old": old_string, "new": new_string})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        safe_path = shlex.quote(file_path)
        replace_all_str = "true" if replace_all else "false"

        # Use heredoc to pass old/new strings via stdin to avoid ARG_MAX limits.
        # ARG_MAX limits the total size of command-line arguments.
        # Format: base64-encoded JSON with {{"old": str, "new": str}}.
        # The heredoc feeds into the brace group { ... } which reads and processes stdin.
        cmd = f"""
if [ ! -f {safe_path} ]; then
    exit 3
fi

{{
    # Read entire heredoc content using cat (read only gets first line)
    payload_b64=$(cat)
    if [ -z "$payload_b64" ]; then
        echo "Error: No payload received for edit operation" >&2
        exit 4
    fi

    # Decode base64 payload
    payload=$(echo "$payload_b64" | base64 -d) || {{
        echo "Error: Failed to decode payload" >&2
        exit 4
    }}

    # Extract old and new strings from JSON using python3
    old=$(echo "$payload" | python3 -c "import sys, json; print(json.load(sys.stdin)['old'], end='')") || {{
        echo "Error: Failed to parse JSON payload" >&2
        exit 4
    }}
    new=$(echo "$payload" | python3 -c "import sys, json; print(json.load(sys.stdin)['new'], end='')") || {{
        echo "Error: Failed to parse JSON payload" >&2
        exit 4
    }}

    # Count occurrences using grep -F (fixed strings)
    count=$(grep -o -F "$old" {safe_path} | wc -l)

    if [ "$count" -eq 0 ]; then
        exit 1
    elif [ "$count" -gt 1 ] && [ "{replace_all_str}" = "false" ]; then
        exit 2
    fi

    # Use perl for reliable string replacement (handles special chars).
    # Note: \\Q...\\E escapes the search pattern. The replacement string is not
    # escaped, so Perl special sequences (\\U, $1, etc.) in new will be interpreted.
    if [ "{replace_all_str}" = "true" ]; then
        perl -i -pe 's/\\Q'"$old"'\\E/'"$new"'/g' {safe_path}
    else
        perl -i -pe 's/\\Q'"$old"'\\E/'"$new"'/' {safe_path}
    fi

    echo "$count"
}} <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__
"""
        result = await self.aexecute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        if exit_code == _EXIT_NOT_FOUND:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if exit_code == _EXIT_MULTIPLE_MATCHES:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences."
            )
        if exit_code == _EXIT_FILE_MISSING:
            return EditResult(error=f"Error: File '{file_path}' not found")
        if exit_code == _EXIT_DECODE_FAILED:
            return EditResult(error=f"Error: Failed to decode edit payload: {output}")
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
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

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
            if len(parts) == _PIPE_FIELD_COUNT:
                file_infos.append({"path": parts[0], "is_dir": parts[1] == "true"})

        return file_infos

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents with metadata using shell commands."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

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
                    continue

        return matches

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files using grep."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)

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
            if len(parts) == _PIPE_FIELD_COUNT:
                file_infos.append(
                    {
                        "path": parts[0],
                        "is_dir": parts[1] == "true",
                    }
                )

        return file_infos

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching glob pattern using shell commands."""
        raise NotImplementedError(_SYNC_NOT_SUPPORTED)
