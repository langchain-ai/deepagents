"""BackendProtocol implementation for Runloop."""

import datetime
import os
from typing import Optional

from runloop_api_client import Runloop

from deepagents.backends.protocol import (
    BackendProtocol,
    FileInfo,
    GrepMatch,
    WriteResult,
    EditResult,
)
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class RunloopBackend:
    """Backend that operates on files in a Runloop devbox.

    This implementation uses the Runloop API client to execute commands
    and manipulate files within a remote devbox environment.
    """

    # NOTE: As an example, this currently uses a pre-allocated devbox.
    # For the real version we would want to create a devbox as needed,
    # run one or more commands, and then clean up when finished.

    def __init__(
        self,
        devbox_id: str,
        client: Optional[Runloop] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        """Initialize Runloop backend.

        Args:
            devbox_id: ID of the Runloop devbox to operate on.
            client: Optional existing Runloop client instance
            bearer_token: Optional API key for creating a new client
                         (defaults to RUNLOOP_API_KEY environment variable)
        """
        if client and bearer_token:
            raise ValueError("Provide either client or bearer_token, not both.")

        if client is None:
            bearer_token = bearer_token or os.environ.get("RUNLOOP_API_KEY", None)
            if bearer_token is None:
                raise ValueError("Either client or bearer_token must be provided.")
            client = Runloop(bearer_token=bearer_token)

        self._client = client
        self._devbox_id = devbox_id

    def exec(self, command: str) -> tuple[str, int]:
        """Execute a command in the devbox and return (stdout, exit_status)."""
        result = self._client.devboxes.execute_and_await_completion(
            devbox_id=self._devbox_id,
            command=command,
        )
        # NOTE: could check exit status for error (non-zero) and
        # return stderr here instead / in addition to stdout.
        return (result.stdout or "", result.exit_status)


class RunloopProtocol(BackendProtocol):
    def __init__(self, backend):
        self._backend = backend

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Directory path to list files from.

        Returns:
            List of FileInfo dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        # Use find to list only direct children
        cmd = f"find '{path}' -maxdepth 1 -mindepth 1 -printf '%p %s %T@ %y %Y\\n' 2>/dev/null"
        stdout, exit_code = self._backend.exec(cmd)

        if exit_code != 0 or not stdout.strip():
            # NOTE: this silently ignores errors; not sure what error
            # handling semantics are needed here, but presumably not
            # this.  :)
            return []

        results: list[FileInfo] = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            # Parse out the listing info.
            (path, size, modified_secs, filetype, realtype) = line.split()
            modtime = datetime.datetime.fromtimestamp(float(modified_secs)).isoformat()

            file_info: FileInfo = {
                "path": path + "/" if filetype == "d" else path,
                "is_dir": filetype == "d",
                "size": int(size) if filetype == "f" else 0,
                "modified_at": modtime,
            }
            results.append(file_info)

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: File path to read
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            Formatted file content with line numbers, or error message.
        """
        # Check if file exists and get content
        start_line = offset + 1
        cmd = f"if [ ! -f '{file_path}' ]; then echo 'Error: File not found'; exit 1; else tail -n +{start_line} '{file_path}' | head -n {limit}; fi"
        stdout, exit_code = self._backend.exec(cmd)

        if exit_code != 0 or "Error: File not found" in stdout:
            return f"Error: File '{file_path}' not found"

        empty_msg = check_empty_content(stdout)
        if empty_msg:
            return empty_msg

        return format_content_with_line_numbers(stdout, start_line=start_line)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Path where to write the file
            content: Content to write

        Returns:
            WriteResult with path on success or error message on failure.
        """
        # QUESTIONS:
        # * is the intent here to only support text formats, as with read() and edit()?
        # * for text, any assumptions/requirements about the character set?

        # Check if file already exists
        check_cmd = f"test -e '{file_path}' && echo 'exists' || echo 'ok'"
        stdout, _ = self._backend.exec(check_cmd)

        if "exists" in stdout:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # Use the upload_file() method from the Runloop API client.
        try:
            self._backend._client.devboxes.upload_file(
                id=self._backend._devbox_id,
                path=file_path,
                file=content.encode("utf-8"),  # NOTE: might want a different type?
            )
        except Exception as e:
            # TODO: catch specific exception
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

        # External storage - no files_update needed
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Path to the file to edit
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences

        Returns:
            EditResult with path and occurrences on success or error on failure.
        """
        # QUESTIONS:
        # * this downloads the whole file to replace things locally; are files guaranteed to be small?
        # * what semantics do you want for non-existant / empty files?

        try:
            # fetch the file
            response = self._backend._client.devboxes.download_file(id=self._backend._devbox_id, path=file_path)

            # do the replacements
            new_text, occurrences = perform_string_replacement(response.text(), old_string, new_string, replace_all)

            # write back
            self._backend._client.devboxes.upload_file(
                id=self._backend._devbox_id,
                path=file_path,
                file=new_text.encode("utf-8"),  # NOTE: might want a different type?
            )
            # External storage - no files_update needed
            return EditResult(path=file_path, files_update=None, occurrences=occurrences)

        except Exception as e:
            # TODO: catch specific exception
            return EditResult(error=f"Error writing file '{file_path}': {e}")

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Search for a pattern in files.

        Args:
            pattern: Regular expression pattern to search for
            path: Base path to search from (defaults to current directory)
            glob: Optional glob pattern to filter files (e.g., "*.py")

        Returns:
            List of GrepMatch dicts on success, or error string on invalid input.
        """
        # Use grep to search files.  NOTE: might need something
        # differeent if you have other regex semantics.
        search_path = path or "."

        # Build grep command
        grep_opts = "-rHn"  # recursive, with filename, with line number

        # Add glob pattern if specified
        if glob:
            grep_opts += f" --include='{glob}'"

        # Escape pattern for shell
        pattern_escaped = pattern.replace("'", "\\'")

        cmd = f"grep {grep_opts} -e '{pattern_escaped}' '{search_path}' 2>/dev/null || true"
        stdout, _ = self._backend.exec(cmd)

        if not stdout.strip():
            return []

        # Parse grep output: path:line_number:content
        matches: list[GrepMatch] = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            # Split on first two colons to handle content with colons
            parts = line.split(":", 2)
            try:
                file_path = parts[0]
                line_num = int(parts[1])
                line_text = parts[2]
                matches.append(
                    {
                        "path": file_path,
                        "line": line_num,
                        "text": line_text,
                    }
                )
            except ValueError:
                continue

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.ts")
            path: Base path to search from

        Returns:
            List of FileInfo dicts for matching files.
        """
        # Use Python's glob module via remote execution
        pattern_escaped = pattern.replace("'", "'\\''")
        path_escaped = path.replace("'", "'\\''")

        # Use a more complicated command, to grab stat output from the
        # matching files.  Could be simplified if this isn't needed.
        python_cmd = (
            f'python3 -c "'
            f"import glob, os, json; "
            f"os.chdir('{path_escaped}'); "
            f"matches = glob.glob('{pattern_escaped}', recursive=True); "
            f"for m in matches: "
            f"  if os.path.isfile(m): "
            f"    s = os.stat(m); "
            f"    print(json.dumps({{'path': m, 'size': s.st_size, 'mtime': s.st_mtime}})); "
            f'" 2>/dev/null'
        )

        stdout, exit_code = self._backend.exec(python_cmd)

        if exit_code != 0 or not stdout.strip():
            return []

        results: list[FileInfo] = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            try:
                import json

                data = json.loads(line)
                # Convert relative path to absolute based on search path
                file_path = data["path"]
                if not file_path.startswith("/"):
                    if path == "/":
                        file_path = "/" + file_path
                    else:
                        file_path = path.rstrip("/") + "/" + file_path

                results.append(
                    {
                        "path": file_path,
                        "is_dir": False,
                        "size": data["size"],
                        "modified_at": str(data["mtime"]),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def exec(self, command: str) -> tuple[str, int]:
        """Execute a command in the devbox and return (stdout, exit_status)."""
        return self._backend.exec(command)
