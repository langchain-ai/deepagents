"""Base sandbox implementation with execute() as the only abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.
"""

from __future__ import annotations

import json
import shlex
import base64
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
        # Use shlex.quote for path
        files = self.execute(f"ls -la {shlex.quote(path)} 2>/dev/null || true")

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
        # Create JSON payload with all parameters
        payload = json.dumps({"file_path": file_path, "offset": offset, "limit": limit})

        # Python code that reads from stdin
        python_code = """import sys, json, os
data = json.load(sys.stdin)
file_path = data['file_path']
offset = data['offset']
limit = data['limit']

if not os.path.isfile(file_path):
    print('Error: File not found')
    sys.exit(1)

if os.path.getsize(file_path) == 0:
    print('System reminder: File exists but has empty contents')
    sys.exit(0)

with open(file_path, 'r') as f:
    lines = f.readlines()

start_idx = offset
end_idx = offset + limit
selected_lines = lines[start_idx:end_idx]

for i, line in enumerate(selected_lines):
    line_num = offset + i + 1
    line_content = line.rstrip('\\n')
    print(f'{line_num:6d}\\t{line_content}')"""

        # Execute with JSON payload via stdin
        cmd = f"echo {shlex.quote(payload)} | python3 -c {shlex.quote(python_code)} 2>&1"
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
        # Use shlex.quote for file path in shell command
        file_path_quoted = shlex.quote(file_path)

        # Check if file already exists
        check_cmd = f"test -e {file_path_quoted} && echo 'exists' || echo 'not_exists'"
        check_result = self.execute(check_cmd)

        if check_result.output.strip() == "exists":
            return WriteResult(error=f"Error: File '{file_path}' already exists")

        # Create JSON payload with all parameters
        payload = json.dumps({"file_path": file_path, "content_b64": base64.b64encode(content.encode("utf-8")).decode("ascii")})

        # Python code that reads from stdin
        python_code = """import sys, json, base64, os
data = json.load(sys.stdin)
file_path = data['file_path']
content = base64.b64decode(data['content_b64']).decode('utf-8')
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)
with open(file_path, 'w') as f:
    f.write(content)"""

        # Execute with JSON payload via stdin
        cmd = f"echo {shlex.quote(payload)} | python3 -c {shlex.quote(python_code)} 2>&1"
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
        # Create JSON payload with all parameters
        payload = json.dumps(
            {
                "file_path": file_path,
                "old_b64": base64.b64encode(old_string.encode("utf-8")).decode("ascii"),
                "new_b64": base64.b64encode(new_string.encode("utf-8")).decode("ascii"),
                "replace_all": replace_all,
            }
        )

        # Python code that reads from stdin
        python_code = """import sys, json, base64
data = json.load(sys.stdin)
file_path = data['file_path']
old = base64.b64decode(data['old_b64']).decode('utf-8')
new = base64.b64decode(data['new_b64']).decode('utf-8')
replace_all = data['replace_all']

with open(file_path, 'r') as f:
    text = f.read()

count = text.count(old)

if count == 0:
    sys.exit(1)
elif count > 1 and not replace_all:
    sys.exit(2)

if replace_all:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

with open(file_path, 'w') as f:
    f.write(result)

print(count)"""

        # Execute with JSON payload via stdin
        cmd = f"echo {shlex.quote(payload)} | python3 -c {shlex.quote(python_code)} 2>&1"
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

        # Create JSON payload with all parameters
        payload = json.dumps({"pattern": pattern, "search_path": search_path, "glob": glob})

        # Python code that reads from stdin and performs grep
        python_code = """import sys, json, os, re
data = json.load(sys.stdin)
pattern = data['pattern']
search_path = data['search_path']
glob_pattern = data['glob']

def matches_glob(filepath, glob_pattern):
    if not glob_pattern:
        return True
    import fnmatch
    return fnmatch.fnmatch(os.path.basename(filepath), glob_pattern)

def grep_file(filepath, pattern):
    results = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if pattern in line:
                    results.append({
                        'path': filepath,
                        'line': line_num,
                        'text': line.rstrip('\\n')
                    })
    except:
        pass
    return results

all_matches = []
for root, dirs, files in os.walk(search_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        if matches_glob(filepath, glob_pattern):
            all_matches.extend(grep_file(filepath, pattern))

print(json.dumps(all_matches))"""

        # Execute with JSON payload via stdin
        cmd = f"echo {shlex.quote(payload)} | python3 -c {shlex.quote(python_code)} 2>&1"
        result = self.execute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        try:
            matches_data = json.loads(output)
            return matches_data
        except json.JSONDecodeError:
            return []

    def glob_info(self, pattern: str, path: str = ".") -> list[FileInfo]:
        """Structured glob matching returning FileInfo dicts."""
        # Create JSON payload with all parameters
        payload = json.dumps({"pattern": pattern, "path": path})

        # Python code that reads from stdin and performs glob
        python_code = """import sys, json, glob, os
data = json.load(sys.stdin)
pattern = data['pattern']
path = data['path']

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))

results = []
for m in matches:
    stat = os.stat(m)
    result = {
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }
    results.append(result)

print(json.dumps(results))"""

        # Execute with JSON payload via stdin
        cmd = f"echo {shlex.quote(payload)} | python3 -c {shlex.quote(python_code)} 2>&1"
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        try:
            data = json.loads(output)
            file_infos: list[FileInfo] = []
            for item in data:
                file_infos.append(
                    {
                        "path": item["path"],
                        "is_dir": item["is_dir"],
                    }
                )
            return file_infos
        except json.JSONDecodeError:
            return []
