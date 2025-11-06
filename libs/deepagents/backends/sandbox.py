"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.
"""

from __future__ import annotations
import json

from abc import ABC, abstractmethod

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)

_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json

os.chdir('{path}')
matches = sorted(glob.glob('{pattern}', recursive=True))
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

_WRITE_COMMAND_TEMPLATE = """python3 -c "
import os

# Create parent directory if needed
parent_dir = os.path.dirname('{file_path}') or '.'
os.makedirs(parent_dir, exist_ok=True)

# Write content to file
with open('{file_path}', 'w') as f:
    f.write('''{content}''')
" 2>&1"""

_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys

# Read file content
with open('{file_path}', 'r') as f:
    text = f.read()

old = '''{old_string}'''
new = '''{new_string}'''

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
with open('{file_path}', 'w') as f:
    f.write(result)

print(count)
" 2>&1"""

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
        # Use template for reading file with offset and limit
        cmd = _READ_COMMAND_TEMPLATE.format(
            file_path=file_path,
            offset=offset,
            limit=limit
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

        # Write the file using template
        cmd = _WRITE_COMMAND_TEMPLATE.format(
            file_path=file_path,
            content=content_escaped
        )
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

        # Use template for string replacement
        cmd = _EDIT_COMMAND_TEMPLATE.format(
            file_path=file_path,
            old_string=old_escaped,
            new_string=new_escaped,
            replace_all=replace_all
        )
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
        # Escape pattern and path for shell
        pattern_escaped = pattern.replace("'", "'\\\\''")
        path_escaped = path.replace("'", "'\\\\''")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path=path_escaped, pattern=pattern_escaped)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        # Parse JSON output into FileInfo dicts
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append({
                    "path": data["path"],
                    "is_dir": data["is_dir"],
                })
            except json.JSONDecodeError:
                continue

        return file_infos
