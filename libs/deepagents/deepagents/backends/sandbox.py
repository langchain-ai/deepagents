"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.
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
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)

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

_WRITE_COMMAND_TEMPLATE = """
if [ -e {file_path} ]; then
    echo "Error: File already exists" >&2
    exit 1
fi
parent_dir=$(dirname {file_path})
mkdir -p "$parent_dir" 2>/dev/null
echo '{content_b64}' | base64 -d > {file_path}
"""

_EDIT_COMMAND_TEMPLATE = """
if [ ! -f {file_path} ]; then
    exit 3
fi

old=$(echo '{old_b64}' | base64 -d)
new=$(echo '{new_b64}' | base64 -d)

# Use awk for literal string replacement that handles multiline correctly
awk -v old="$old" -v new="$new" -v replace_all="{replace_all_str}" '
BEGIN {{
    RS = "^$"  # Read entire file as one record
    ORS = ""   # No extra newline on output
}}
{{
    content = $0
    count = 0

    # Count occurrences
    temp = content
    while ((pos = index(temp, old)) > 0) {{
        count++
        temp = substr(temp, pos + length(old))
    }}

    # Check error conditions
    if (count == 0) {{
        exit 1  # String not found
    }}
    if (count > 1 && replace_all == "false") {{
        exit 2  # Multiple occurrences without replace_all
    }}

    # Perform replacement
    if (replace_all == "true") {{
        # Replace all occurrences
        result = ""
        remaining = content
        while ((pos = index(remaining, old)) > 0) {{
            result = result substr(remaining, 1, pos - 1) new
            remaining = substr(remaining, pos + length(old))
        }}
        result = result remaining
    }} else {{
        # Replace first occurrence only
        pos = index(content, old)
        result = substr(content, 1, pos - 1) new substr(content, pos + length(old))
    }}

    # Write result and output count to stderr (so we can capture it)
    print result > {file_path}
    print count > "/dev/stderr"
}}
' {file_path} 2>&1 | tail -1
"""

_READ_COMMAND_TEMPLATE = """
if [ ! -f {file_path} ]; then
    echo "Error: File not found"
    exit 1
fi
if [ ! -s {file_path} ]; then
    echo "System reminder: File exists but has empty contents"
    exit 0
fi
# Use awk to add line numbers and handle offset/limit
awk -v offset={offset} -v limit={limit} '
    NR > offset && NR <= offset + limit {{
        printf "%6d\\\\t%s\\\\n", NR, $0
    }}
    NR > offset + limit {{ exit }}
' {file_path}
"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with execute() as abstract method.

    This class provides default implementations for all protocol methods
    using shell commands. Subclasses only need to implement execute().
    """

    @abstractmethod
    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        ...

    def ls_info(self, path: str) -> list[FileInfo]:
        """Structured listing with file metadata using shell commands."""
        # Escape path for safe shell execution
        safe_path = shlex.quote(path)
        # Use tab as delimiter (less likely to appear in filenames than pipe)
        cmd = f"""
if [ ! -d {safe_path} ]; then
    exit 1
fi
for entry in {safe_path}/*; do
    if [ -e "$entry" ]; then
        name=$(basename "$entry")
        if [ -d "$entry" ]; then
            printf '%s\\t1\\n' "$name"
        else
            printf '%s\\t0\\n' "$name"
        fi
    fi
done
"""

        result = self.execute(cmd)

        if result.exit_code != 0:
            return []

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                file_infos.append({"path": parts[0], "is_dir": parts[1] == "1"})

        return file_infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using a single shell command."""
        # Escape file path for safe shell execution
        safe_path = shlex.quote(file_path)
        # Use template for reading file with offset and limit
        cmd = _READ_COMMAND_TEMPLATE.format(file_path=safe_path, offset=offset, limit=limit)
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
        # Encode content as base64 to avoid any escaping issues
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        # Escape file path for safe shell execution
        safe_path = shlex.quote(file_path)

        # Single atomic check + write command
        cmd = _WRITE_COMMAND_TEMPLATE.format(file_path=safe_path, content_b64=content_b64)
        result = self.execute(cmd)

        # Check for errors (exit code or error message in output)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

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
        # Encode strings as base64 to avoid any escaping issues
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")
        replace_all_str = "true" if replace_all else "false"
        # Escape file path for safe shell execution
        safe_path = shlex.quote(file_path)

        # Use template for string replacement
        cmd = _EDIT_COMMAND_TEMPLATE.format(file_path=safe_path, old_b64=old_b64, new_b64=new_b64, replace_all_str=replace_all_str)
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
        # Use -E for extended regex to support patterns like test[0-9]+
        grep_opts = "-rHnE"  # recursive, with filename, with line number, extended regex

        # Add glob pattern if specified (escape for safe shell execution)
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include={shlex.quote(glob)}"

        # Escape pattern and path for safe shell execution
        safe_pattern = shlex.quote(pattern)
        safe_path = shlex.quote(search_path)

        cmd = f"grep {grep_opts} {glob_pattern} -e {safe_pattern} {safe_path} 2>/dev/null || true"
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
        # Encode pattern and path as base64 to avoid escaping issues
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

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

        return file_infos

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in FileUploadResponse objects rather than raising.
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in FileDownloadResponse objects rather than raising.
        """
