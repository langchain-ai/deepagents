"""Daytona sandbox backend implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig

from deepagents.backends.pagination import PageResults, PaginationCursor
from deepagents.backends.process import ExecuteResponse, Process, ProcessCapabilities
from deepagents.backends.protocol import EditResult, FileInfo, GrepMatch, WriteResult
from deepagents.backends.sandbox import Sandbox, SandboxCapabilities, SandboxMetadata, SandboxProvider

if TYPE_CHECKING:
    from daytona import Sandbox as DaytonaSandboxClient


class DaytonaFileSystem:
    """Daytona filesystem implementation conforming to BackendProtocol."""

    def __init__(self, sandbox: DaytonaSandboxClient) -> None:
        """Initialize the DaytonaFileSystem with a Daytona sandbox client."""
        self._sandbox = sandbox

    def ls_info(self, path: str) -> list[FileInfo]:
        """Structured listing with file metadata."""
        files = self._sandbox.fs.list_files(path)

        result: list[FileInfo] = []
        for file in files:
            # Convert Daytona format to our FileInfo format
            file_info: FileInfo = {"path": file.name}

            # Add optional fields if present
            if hasattr(file, "is_dir"):
                file_info["is_dir"] = file.is_dir
            if hasattr(file, "size"):
                file_info["size"] = int(file.size)
            if hasattr(file, "mod_time"):
                file_info["modified_at"] = file.mod_time

            result.append(file_info)

        return result

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using a single shell command."""
        # Single command that checks file, reads lines, and formats with line numbers
        # tail -n +N starts from line N, head limits output, nl adds line numbers
        start_line = offset + 1
        cmd = (
            f"if [ ! -f '{file_path}' ]; then "
            f"echo 'Error: File not found'; exit 1; "
            f"elif [ ! -s '{file_path}' ]; then "
            f"echo 'System reminder: File exists but has empty contents'; "
            f"else "
            f"tail -n +{start_line} '{file_path}' | head -n {limit} | nl -ba -nrn -w6 -s$'\\t' -v{start_line}; "
            f"fi"
        )
        result = self._sandbox.process.exec(cmd)

        output = result.result.rstrip()
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
        # Escape content for shell safety
        content_escaped = content.replace("'", "'\\''")

        # Check if file already exists
        check_cmd = f"test -e '{file_path}' && echo 'exists' || echo 'not_exists'"
        check_result = self._sandbox.process.exec(check_cmd)

        if check_result.result.strip() == "exists":
            return WriteResult(error=f"Error: File '{file_path}' already exists")

        # Write the file using Python
        python_code = (
            f"import os; os.makedirs(os.path.dirname('{file_path}') or '.', exist_ok=True); open('{file_path}', 'w').write('''{content_escaped}''')"
        )

        cmd = f'python3 -c "{python_code}" 2>&1'
        result = self._sandbox.process.exec(cmd)

        if result.exit_code != 0:
            return WriteResult(error=f"Error: Failed to write file '{file_path}': {result.result.strip()}")

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
        # Escape single quotes in the strings for shell safety
        old_escaped = old_string.replace("'", "'\\''")
        new_escaped = new_string.replace("'", "'\\''")

        # Use Python one-liner for complex string replacement logic
        python_code = (
            f"import sys; "
            f"text = open('{file_path}', 'r').read(); "
            f"old = '''{old_escaped}'''; "
            f"new = '''{new_escaped}'''; "
            f"count = text.count(old); "
            f"sys.exit(1) if count == 0 else (sys.exit(2) if count > 1 and not {replace_all} else None); "
            f"result = text.replace(old, new) if {replace_all} else text.replace(old, new, 1); "
            f"open('{file_path}', 'w').write(result); "
            f"print(count)"
        )

        cmd = f'python3 -c "{python_code}" 2>&1'
        result = self._sandbox.process.exec(cmd)

        exit_code = result.exit_code
        output = result.result.strip()

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
        search_path = path or "/"

        # Build grep command to get structured output
        grep_opts = "-rHn"  # recursive, with filename, with line number

        # Add glob pattern if specified
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        # Escape pattern for shell
        pattern_escaped = pattern.replace("'", "'\\''")

        cmd = f"grep {grep_opts} {glob_pattern} -e '{pattern_escaped}' '{search_path}' 2>/dev/null || true"
        result = self._sandbox.process.exec(cmd)

        output = result.result.rstrip()
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
        # Escape pattern for shell
        pattern_escaped = pattern.replace("'", "'\\''")

        # Use Python's glob module for proper glob pattern matching
        python_code = (
            f"import glob; "
            f"import os; "
            f"os.chdir('{path}'); "
            f"results = sorted(glob.glob('{pattern_escaped}', recursive=True)); "
            f"print('\\n'.join(results))"
        )

        cmd = f'python3 -c "{python_code}" 2>/dev/null'
        result = self._sandbox.process.exec(cmd)

        output = result.result.strip()
        if not output:
            return []

        # Convert paths to FileInfo dicts
        file_infos: list[FileInfo] = []
        for file_path in output.split("\n"):
            file_infos.append({"path": file_path})

        return file_infos


class DaytonaProcess(Process):
    """Daytona process implementation."""

    def __init__(self, sandbox: DaytonaSandboxClient) -> None:
        """Initialize the DaytonaProcess with a Daytona sandbox client."""
        self._sandbox = sandbox

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the process.

        Args:
            command: Command to execute as a string.
            cwd: Working directory to execute the command in.
            timeout: Maximum execution time in seconds (default: 30 minutes).
        """
        response = self._sandbox.process.exec(command, cwd=cwd, timeout=timeout)
        return ExecuteResponse(
            result=response.result,
            exit_code=response.exit_code,
        )

    def get_capabilities(self) -> ProcessCapabilities:
        """Get the process capabilities."""
        return {
            "supports_exec": True,
        }


class DaytonaSandbox(Sandbox):
    """Daytona sandbox implementation."""

    def __init__(self, sandbox: DaytonaSandboxClient) -> None:
        """Initialize the DaytonaSandbox with a Daytona sandbox client."""
        self._sandbox = sandbox
        self._fs = DaytonaFileSystem(sandbox)
        self._process = DaytonaProcess(sandbox)

    @property
    def fs(self) -> DaytonaFileSystem:
        """Filesystem backend."""
        return self._fs

    @property
    def process(self) -> Process:
        """Process backend."""
        return self._process

    @property
    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox backend."""
        return {
            "fs": {
                "can_list_files": True,
                "can_read": True,
                "can_write": True,
                "can_edit": True,
                "can_grep": True,
                "can_glob": True,
            },
            "process": self._process.get_capabilities(),
        }

    @property
    def id(self) -> str:
        """Get the sandbox ID."""
        return self._sandbox.id


class DaytonaSandboxProvider(SandboxProvider):
    """Daytona sandbox provider implementation."""

    def __init__(
        self,
        *,
        client: Daytona | None = None,
        api_key: str | None = None,
        auto_stop_minutes: int | None = None,
        auto_delete_minutes: int | None = None,
    ) -> None:
        """Initialize the DaytonaSandboxProvider with a Daytona client.

        Args:
            client: An existing Daytona client instance
            api_key: API key for creating a new Daytona client
            auto_stop_minutes: Minutes of inactivity before sandbox auto-stops. Defaults to 15.
            auto_delete_minutes: Minutes after stopping before sandbox is deleted. Defaults to 0
                                (delete immediately on stop).
        """
        if client and api_key:
            raise ValueError("Provide either daytona_client or api_key, not both.")

        if client is None:
            api_key = api_key or os.environ.get("DAYTONA_API_KEY")
            if api_key is None:
                raise ValueError("Either daytona_client or api_key must be provided.")
            config = DaytonaConfig(api_key=api_key)
            client = Daytona(config)

        self._client = client
        self.auto_stop_interval = auto_stop_minutes
        self.auto_delete_interval = auto_delete_minutes

    def get_or_create(self, id: str | None = None, **kwargs) -> Sandbox:
        """Get or create a sandbox instance by ID.

        If id is None, creates a new sandbox.
        If id is provided, retrieves the existing sandbox.
        """
        if id is None:
            # Create a new sandbox with TTL parameters
            sandbox_client = self._client.create(
                params=CreateSandboxFromSnapshotParams(
                    auto_stop_interval=self.auto_stop_interval,
                    auto_delete_interval=self.auto_delete_interval,
                )
            )
            return DaytonaSandbox(sandbox_client)
        # Get existing sandbox
        sandbox_client = self._client.get(id)
        return DaytonaSandbox(sandbox_client)

    def delete(self, id: str) -> None:
        """Delete a sandbox instance by ID.

        Do not raise an error if the sandbox does not exist.
        """
        try:
            sandbox = self._client.get(id)
            self._client.delete(sandbox)
        except Exception:
            # Silently ignore if sandbox doesn't exist
            pass

    def list(self, *, cursor: PaginationCursor | None = None, **kwargs) -> PageResults[SandboxMetadata]:
        """List all sandbox IDs.

        Note: Daytona's list() method returns a simple list of IDs,
        so we don't support pagination at the API level.
        """
        # Daytona's list returns list[str] of sandbox IDs
        paginated_sandboxes = self._client.list()
        items: SandboxMetadata = [{"id": item.id} for item in paginated_sandboxes.items]
        # Convert to SandboxMetadata format
        items: list[SandboxMetadata] = items

        # Since Daytona doesn't support pagination, we return all items
        return PageResults(
            items=items,
            cursor=PaginationCursor(
                next_cursor=None,
                has_more=False,
            ),
        )

    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox provider."""
        return {
            "fs": {
                "can_list_files": True,
                "can_read": True,
                "can_write": True,
                "can_edit": True,
                "can_grep": True,
                "can_glob": True,
            },
            "process": {
                "supports_exec": True,
            },
        }
