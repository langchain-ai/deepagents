"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with execute() as abstract method.

    This class provides default implementations for all protocol methods
    using shell commands. Subclasses only need to implement execute().
    """

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum execution time in seconds (default: 30 minutes).

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        ...

    def ls_info(self, path: str) -> list[FileInfo]:
        """Structured listing with file metadata."""
        files = self.execute(f"ls -la '{path}' 2>/dev/null || true")

        # Parse ls output - this is a simple implementation
        # You might want to use find for more structured output
        result: list[FileInfo] = []
        for line in files.output.strip().split("\n"):
            if not line or line.startswith("total"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                filename = " ".join(parts[8:])
                is_dir = parts[0].startswith("d")
                result.append(
                    {
                        "path": filename,
                        "is_dir": is_dir,
                    }
                )

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
            f"tail -n +{start_line} '{file_path}' | head -n {limit} | nl -ba -nrn -w6 -s$'\\\\t' -v{start_line}; "
            f"fi"
        )
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
        # Escape content for shell safety
        content_escaped = content.replace("'", "'\\\\''")

        # Check if file already exists
        check_cmd = f"test -e '{file_path}' && echo 'exists' || echo 'not_exists'"
        check_result = self.execute(check_cmd)

        if check_result.output.strip() == "exists":
            return WriteResult(error=f"Error: File '{file_path}' already exists")

        # Write the file using Python
        python_code = (
            f"import os; os.makedirs(os.path.dirname('{file_path}') or '.', exist_ok=True); open('{file_path}', 'w').write('''{content_escaped}''')"
        )

        cmd = f'python3 -c "{python_code}" 2>&1'
        result = self.execute(cmd)

        if result.exit_code != 0:
            return WriteResult(error=f"Error: Failed to write file '{file_path}': {result.output.strip()}")

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
        old_escaped = old_string.replace("'", "'\\\\''")
        new_escaped = new_string.replace("'", "'\\\\''")

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
        result = self.execute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

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
        search_path = path or "."

        # Build grep command to get structured output
        grep_opts = "-rHn"  # recursive, with filename, with line number

        # Add glob pattern if specified
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        # Escape pattern for shell
        pattern_escaped = pattern.replace("'", "'\\\\''")

        cmd = f"grep {grep_opts} {glob_pattern} -e '{pattern_escaped}' '{search_path}' 2>/dev/null || true"
        result = self.execute(cmd)

        output = result.output.rstrip()
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
        pattern_escaped = pattern.replace("'", "'\\\\''")

        # Use Python's glob module for proper glob pattern matching
        python_code = (
            f"import glob; "
            f"import os; "
            f"os.chdir('{path}'); "
            f"results = sorted(glob.glob('{pattern_escaped}', recursive=True)); "
            f"print('\\\\n'.join(results))"
        )

        cmd = f'python3 -c "{python_code}" 2>/dev/null'
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        # Convert paths to FileInfo dicts
        file_infos: list[FileInfo] = []
        for file_path in output.split("\n"):
            file_infos.append({"path": file_path})

        return file_infos
