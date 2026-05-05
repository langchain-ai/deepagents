"""`just-bash` backed sandbox for Deep Agents."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
from importlib.resources import files
from typing import Any, Final, Self
from uuid import uuid4

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
)

_DEFAULT_MAX_OUTPUT_BYTES: Final = 100_000


class JustBashError(RuntimeError):
    """Raised when the `just-bash` bridge cannot complete a request."""


class _JustBashClient:
    """Persistent Node bridge around one `just-bash` `Bash` instance."""

    def __init__(
        self,
        *,
        node_bin: str = "node",
        just_bash_package: str = "just-bash",
        javascript: bool = True,
        cwd: str | None = None,
    ) -> None:
        node = shutil.which(node_bin)
        if node is None:
            msg = f"Node.js executable not found: {node_bin}"
            raise JustBashError(msg)

        bridge = files("langchain_quickjs").joinpath("_just_bash_bridge.mjs")
        env = {
            **os.environ,
            "JUST_BASH_PACKAGE": just_bash_package,
            "JUST_BASH_JAVASCRIPT": "1" if javascript else "0",
        }
        self._process = subprocess.Popen(  # noqa: S603
            [node, str(bridge)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
        )
        self._lock = threading.Lock()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON request to the bridge."""
        with self._lock:
            if self._process.stdin is None or self._process.stdout is None:
                msg = "just-bash bridge is not connected"
                raise JustBashError(msg)
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()

        if not line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            msg = f"just-bash bridge exited unexpectedly: {stderr.strip()}"
            raise JustBashError(msg)

        response = json.loads(line)
        if not response.get("ok"):
            msg = response.get("error", "unknown just-bash bridge error")
            raise JustBashError(str(msg))
        return response

    def close(self) -> None:
        """Stop the bridge process."""
        if self._process.poll() is None:
            self._process.terminate()


class JustBashSandbox(SandboxBackendProtocol):
    """In-memory `just-bash` sandbox backend.

    Args:
        just_bash_package: Node import specifier for `just-bash`. Use the package
            name when installed, or a path to a built local package entrypoint.
        node_bin: Node.js executable to launch the bridge.
        javascript: Enable `just-bash`'s QuickJS-backed `js-exec` command.
        cwd: Directory used for npm package resolution and shell bridge startup.
        max_output_bytes: Maximum command output bytes returned by `execute`.
    """

    def __init__(
        self,
        *,
        just_bash_package: str = "just-bash",
        node_bin: str = "node",
        javascript: bool = True,
        cwd: str | None = None,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        _client: Any | None = None,
    ) -> None:
        """Initialize the sandbox and its bridge client."""
        self._client = _client or _JustBashClient(
            node_bin=node_bin,
            just_bash_package=just_bash_package,
            javascript=javascript,
            cwd=cwd,
        )
        self._sandbox_id = f"just-bash-{uuid4().hex[:8]}"
        self._max_output_bytes = max_output_bytes

    @property
    def id(self) -> str:
        """Unique identifier for this sandbox."""
        return self._sandbox_id

    def close(self) -> None:
        """Close the underlying bridge process."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def _files(self) -> dict[str, FileData]:
        response = self._client.request({"op": "files"})
        files_map: dict[str, FileData] = {}
        for item in response.get("files", []):
            content = base64.b64decode(item["content"])
            try:
                text = content.decode("utf-8")
                file_data = create_file_data(
                    text,
                    created_at=item.get("stat", {}).get("mtime"),
                    encoding="utf-8",
                )
            except UnicodeDecodeError:
                file_data = create_file_data(
                    item["content"],
                    created_at=item.get("stat", {}).get("mtime"),
                    encoding="base64",
                )
            if "stat" in item and "mtime" in item["stat"]:
                file_data["modified_at"] = item["stat"]["mtime"]
            files_map[item["path"]] = file_data
        return files_map

    def ls(self, path: str) -> LsResult:
        """List files and directories in `path`."""
        normalized = path.rstrip("/") or "/"
        prefix = "/" if normalized == "/" else f"{normalized}/"
        files_map = self._files()
        entries: list[FileInfo] = []
        dirs: set[str] = set()
        for file_path, file_data in files_map.items():
            if not file_path.startswith(prefix):
                continue
            relative = file_path[len(prefix) :]
            if not relative:
                continue
            if "/" in relative:
                dirs.add(prefix + relative.split("/", 1)[0] + "/")
                continue
            entries.append(
                {
                    "path": file_path,
                    "is_dir": False,
                    "size": len(file_data_to_string(file_data)),
                    "modified_at": file_data.get("modified_at", ""),
                }
            )
        entries.extend(
            FileInfo(path=d, is_dir=True, size=0, modified_at="")
            for d in sorted(dirs)
        )
        entries.sort(key=lambda item: item.get("path", ""))
        return LsResult(entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file from the virtual filesystem."""
        result = self.download_files([file_path])[0]
        if result.error is not None or result.content is None:
            return ReadResult(error=f"File '{file_path}' not found")
        try:
            text = result.content.decode("utf-8")
        except UnicodeDecodeError:
            encoded = base64.b64encode(result.content).decode("ascii")
            return ReadResult(file_data=create_file_data(encoded, encoding="base64"))
        file_data = create_file_data(text)
        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        return ReadResult(file_data={**file_data, "content": sliced})

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file in the virtual filesystem."""
        existing = self.download_files([file_path])[0]
        if existing.error is None:
            return WriteResult(
                error=(
                    f"Cannot write to {file_path} because it already exists. "
                    "Read and then make an edit, or write to a new path."
                )
            )
        response = self.upload_files([(file_path, content.encode("utf-8"))])[0]
        if response.error is not None:
            return WriteResult(error=response.error)
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a virtual file by exact string replacement."""
        result = self.download_files([file_path])[0]
        if result.error is not None or result.content is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        try:
            content = result.content.decode("utf-8")
        except UnicodeDecodeError:
            return EditResult(error="Error: Cannot edit binary file")
        replaced = perform_string_replacement(
            content,
            old_string,
            new_string,
            replace_all,
        )
        if isinstance(replaced, str):
            return EditResult(error=replaced)
        new_content, occurrences = replaced
        response = self.upload_files([(file_path, new_content.encode("utf-8"))])[0]
        if response.error is not None:
            return EditResult(error=response.error)
        return EditResult(path=file_path, occurrences=int(occurrences))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search virtual files for a literal text pattern."""
        base = path if path is not None else "/"
        return grep_matches_from_files(self._files(), pattern, base, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find virtual files matching a glob pattern."""
        files_map = self._files()
        result = _glob_search_files(files_map, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])
        matches = []
        for file_path in result.split("\n"):
            file_data = files_map[file_path]
            matches.append(
                {
                    "path": file_path,
                    "is_dir": False,
                    "size": len(file_data_to_string(file_data)),
                    "modified_at": file_data.get("modified_at", ""),
                }
            )
        return GlobResult(matches=matches)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the virtual filesystem."""
        response = self._client.request(
            {
                "op": "upload",
                "files": [
                    {"path": path, "content": base64.b64encode(content).decode("ascii")}
                    for path, content in files
                ],
            }
        )
        return [
            FileUploadResponse(path=item["path"], error=item.get("error"))
            for item in response.get("responses", [])
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the virtual filesystem."""
        response = self._client.request({"op": "download", "paths": paths})
        results = []
        for item in response.get("responses", []):
            content = item.get("content")
            results.append(
                FileDownloadResponse(
                    path=item["path"],
                    content=base64.b64decode(content) if content is not None else None,
                    error=item.get("error"),
                )
            )
        return results

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a command in the `just-bash` virtual shell."""
        response = self._client.request(
            {"op": "execute", "command": command, "timeout": timeout}
        )
        output = str(response.get("output", ""))
        encoded = output.encode("utf-8")
        truncated = len(encoded) > self._max_output_bytes
        if truncated:
            output = encoded[: self._max_output_bytes].decode("utf-8", errors="replace")
        return ExecuteResponse(
            output=output,
            exit_code=response.get("exitCode", 0),
            truncated=truncated,
        )

    def __enter__(self) -> Self:
        """Return this sandbox as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the bridge process on context exit."""
        self.close()
