"""Implement harbor backend."""

import base64
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


class HarborSandbox(SandboxBackendProtocol):
    """A sandbox implementation without assuming that python3 is available."""

    def __init__(self, environment: BaseEnvironment) -> None:
        """Initialize HarborSandbox with the given environment."""
        self.environment = environment

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment."""
        result = await self.environment.exec(command)

        # These errors appear in harbor environments when running bash commands
        # in non-interactive/non-TTY contexts. They're harmless artifacts.
        # Filter them from both stdout and stderr, then collect them to show in stderr.
        error_messages = [
            "bash: cannot set terminal process group (-1): Inappropriate ioctl for device",
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

        cmd = f"""
if [ -e {safe_path} ]; then
    echo "Error: File '{file_path}' already exists" >&2
    exit 1
fi
parent_dir=$(dirname {safe_path})
mkdir -p "$parent_dir" 2>/dev/null
echo '{content_b64}' | base64 -d > {safe_path}
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
        """Edit a file by replacing string occurrences using shell commands."""
        # Encode strings as base64 to avoid escaping issues
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")
        safe_path = shlex.quote(file_path)
        replace_all_str = "true" if replace_all else "false"

        # Use a shell script with perl for reliable string replacement
        cmd = f"""
if [ ! -f {safe_path} ]; then
    exit 3
fi

old=$(echo '{old_b64}' | base64 -d)
new=$(echo '{new_b64}' | base64 -d)

# Count occurrences using grep -F (fixed strings)
count=$(grep -o -F "$old" {safe_path} | wc -l)

if [ "$count" -eq 0 ]; then
    exit 1
elif [ "$count" -gt 1 ] && [ "{replace_all_str}" = "false" ]; then
    exit 2
fi

# Use perl for reliable string replacement (handles special chars)
if [ "{replace_all_str}" = "true" ]; then
    perl -i -pe 's/\\Q'"$old"'\\E/'"$new"'/g' {safe_path}
else
    perl -i -pe 's/\\Q'"$old"'\\E/'"$new"'/' {safe_path}
fi

echo "$count"
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
        if exit_code != 0:
            return EditResult(error=f"Error editing file: {output}")

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
