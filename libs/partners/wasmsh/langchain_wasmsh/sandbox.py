"""Wasmsh sandbox implementation."""

from __future__ import annotations

import base64
import io
import json
import logging
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from wasmsh_pyodide_runtime import get_dist_dir, get_node_host_script

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_DIR = "/workspace"


def _encode_content(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def _decode_content(content: str) -> bytes:
    return base64.b64decode(content.encode("ascii"))


def _to_initial_files(
    files: dict[str, str | bytes] | None,
) -> list[dict[str, str]]:
    if not files:
        return []
    encoded: list[dict[str, str]] = []
    for path, content in files.items():
        payload = content.encode("utf-8") if isinstance(content, str) else content
        encoded.append({"path": path, "contentBase64": _encode_content(payload)})
    return encoded


def _extract_diagnostic(events: list[dict[str, Any]] | None) -> str | None:
    if not events:
        return None
    for event in events:
        diagnostic = event.get("Diagnostic")
        if isinstance(diagnostic, list) and len(diagnostic) >= 2:  # noqa: PLR2004
            return str(diagnostic[1])
    return None


def _map_error(message: str | None) -> FileOperationError:
    normalized = (message or "").lower()
    if "not found" in normalized:
        return "file_not_found"
    if "directory" in normalized:
        return "is_directory"
    if "permission" in normalized:
        return "permission_denied"
    return "invalid_path"


class WasmshSandbox(BaseSandbox):
    """Wasmsh sandbox using Deno (preferred) or Node.js as host runtime."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        runtime: str | None = None,
        dist_dir: str | Path | None = None,
        working_directory: str = DEFAULT_WORKSPACE_DIR,
        step_budget: int = 0,
        initial_files: dict[str, str | bytes] | None = None,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        """Create a wasmsh sandbox backed by a Deno or Node.js subprocess.

        Prefers Deno for its permission model (defense-in-depth: the subprocess
        is restricted to reading only the asset directory and accessing only the
        specified network hosts). Falls back to Node.js if Deno is not installed.

        Args:
            runtime: Explicit runtime path ("deno" or "node"). Auto-detected
                if not specified: prefers Deno, falls back to Node.js.
            dist_dir: Path to Pyodide distribution assets. Auto-resolved from
                the wasmsh-pyodide-runtime package if not specified.
            working_directory: Working directory for execute(). Defaults to
                "/workspace".
            step_budget: VM step budget per command. 0 means unlimited.
            initial_files: Files to seed at creation. Keys are absolute paths,
                values are str or bytes content.
            allowed_hosts: Hostnames allowed for network access. Under Deno
                this maps to --allow-net; under Node.js it is enforced at the
                wasmsh application level only.
        """
        resolved = self._resolve_runtime(runtime)
        self._runtime = resolved
        self._dist_dir = Path(dist_dir) if dist_dir is not None else get_dist_dir()
        self._working_directory = working_directory
        self._allowed_hosts = allowed_hosts or []
        self._id = f"wasmsh-python-{uuid4()}"
        self._lock = threading.Lock()

        cmd = self._build_cmd()
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._next_request_id = 0
        self._stderr_buffer = io.StringIO()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        try:
            self._request(
                "init",
                {
                    "stepBudget": step_budget,
                    "initialFiles": _to_initial_files(initial_files),
                    "allowedHosts": self._allowed_hosts,
                },
            )
        except Exception:
            self._kill_process()
            raise

    def _kill_process(self) -> None:
        """Forcibly terminate the host subprocess."""
        if self._process.stdin:
            self._process.stdin.close()
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.exception(
                    "wasmsh host process %d did not terminate after SIGKILL",
                    self._process.pid,
                )

    @staticmethod
    def _resolve_runtime(runtime: str | None) -> str:
        """Find Deno or Node.js on PATH, preferring Deno.

        Deno is preferred for its permission model: the subprocess is
        restricted to ``--allow-read=<assets>`` and ``--allow-net=<hosts>``.
        Falls back to Node.js if Deno is not installed.
        """
        if runtime is not None:
            path = shutil.which(runtime)
            if path is None:
                msg = f"Runtime not found: {runtime}"
                raise FileNotFoundError(msg)
            return path
        for name in ("deno", "node"):
            path = shutil.which(name)
            if path is not None:
                return path
        msg = "Neither deno nor node found on PATH"
        raise FileNotFoundError(msg)

    def _build_cmd(self) -> list[str]:
        host_script = str(get_node_host_script())
        asset_dir = str(self._dist_dir)
        if self._use_deno:
            cmd = [
                self._runtime,
                "run",
                f"--allow-read={asset_dir}",
                "--allow-env",
            ]
            if self._allowed_hosts:
                hosts = ",".join(self._allowed_hosts)
                cmd.append(f"--allow-net={hosts}")
            cmd.extend([host_script, "--asset-dir", asset_dir])
        else:
            cmd = [self._runtime, host_script, "--asset-dir", asset_dir]
            if self._allowed_hosts:
                logger.warning(
                    "allowed_hosts has no OS-level enforcement under "
                    "Node.js; install Deno for defense-in-depth",
                )
        return cmd

    @property
    def _use_deno(self) -> bool:
        return Path(self._runtime).name.startswith("deno")

    @property
    def id(self) -> str:
        """Return the sandbox identifier."""
        return self._id

    _MAX_STDERR_BYTES = 64 * 1024

    def _drain_stderr(self) -> None:
        """Continuously drain stderr to prevent pipe buffer deadlock."""
        if self._process.stderr is None:
            return
        for line in self._process.stderr:
            if self._stderr_buffer.tell() < self._MAX_STDERR_BYTES:
                self._stderr_buffer.write(line)

    _MAX_NON_JSON_LINES = 100

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._process.stdin or not self._process.stdout:
            msg = "wasmsh host is not available"
            raise RuntimeError(msg)

        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            payload = {"id": request_id, "method": method, "params": params}
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            except OSError as exc:
                stderr = self._stderr_buffer.getvalue().strip()
                msg = f"Failed to send '{method}' to wasmsh host: {exc}"
                if stderr:
                    msg += f"\nHost stderr: {stderr}"
                raise RuntimeError(msg) from exc

            response = None
            response_line = ""
            skipped = 0
            while True:
                response_line = self._process.stdout.readline()
                if not response_line:
                    break
                try:
                    response = json.loads(response_line)
                    break
                except json.JSONDecodeError:
                    skipped += 1
                    logger.debug(
                        "Skipping non-JSON host output: %s", response_line.rstrip()
                    )
                    if skipped >= self._MAX_NON_JSON_LINES:
                        msg = (
                            f"wasmsh host emitted {skipped} consecutive "
                            f"non-JSON lines without a response"
                        )
                        raise RuntimeError(msg) from None

        if not response_line or response is None:
            stderr = self._stderr_buffer.getvalue().strip()
            msg = stderr or "wasmsh host terminated unexpectedly"
            raise RuntimeError(msg)

        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "unknown wasmsh host error")))
        return response["result"]

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        if timeout is not None:
            logger.warning(
                "WasmshSandbox does not enforce execute() timeout; "
                "use step_budget instead",
            )
        result = self._request(
            "run",
            {
                "command": f"cd {shlex.quote(self._working_directory)} && {command}",
            },
        )
        return ExecuteResponse(
            output=str(result["output"]),
            exit_code=result.get("exitCode"),
            truncated=False,
        )

    def read(  # ty: ignore[invalid-method-override]  # returns str for langchain-tests v1 compat
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content via download_files.

        Overrides BaseSandbox which runs a Python script via execute() —
        that approach fails under wasmsh's Pyodide runtime with I/O errors.
        Returns str for compatibility with langchain-tests v1 standard suite.
        """
        responses = self.download_files([file_path])
        resp = responses[0]
        if resp.error or resp.content is None:
            detail = resp.error or "file not found"
            return f"Error: File '{file_path}': {detail}"
        content = resp.content.decode("utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        page = lines[offset : offset + limit]
        return "".join(f"{i + offset + 1:6d}\t{line}" for i, line in enumerate(page))

    def edit(  # noqa: C901, PLR0911
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file via download + string replace + upload.

        Overrides BaseSandbox which runs a Python script via execute() —
        that approach fails under wasmsh's Pyodide runtime with I/O errors.
        Uses download_files/upload_files directly instead.
        """
        responses = self.download_files([file_path])
        if responses[0].error or responses[0].content is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        text = responses[0].content.decode("utf-8", errors="replace")

        if not old_string:
            if text:
                return EditResult(
                    error="oldString must not be empty unless file is empty"
                )
            if not new_string:
                return EditResult(path=file_path, occurrences=0)
            data = new_string.encode("utf-8")
            upload = self.upload_files([(file_path, data)])
            if upload[0].error:
                return EditResult(
                    error=f"Failed to write '{file_path}': {upload[0].error}"
                )
            return EditResult(path=file_path, occurrences=1)

        idx = text.find(old_string)
        if idx == -1:
            return EditResult(error=f"String not found in file '{file_path}'")

        if old_string == new_string:
            return EditResult(path=file_path, occurrences=1)

        if replace_all:
            count = text.count(old_string)
            new_text = text.replace(old_string, new_string)
        else:
            second = text.find(old_string, idx + len(old_string))
            if second != -1:
                return EditResult(
                    error=f"Multiple occurrences found in '{file_path}'. "
                    "Use replace_all=True to replace all.",
                )
            count = 1
            new_text = text[:idx] + new_string + text[idx + len(old_string) :]

        data = new_text.encode("utf-8")
        upload = self.upload_files([(file_path, data)])
        if upload[0].error:
            return EditResult(error=f"Failed to write '{file_path}': {upload[0].error}")
        return EditResult(path=file_path, occurrences=count)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Checks for directories and unreadable files before attempting
        download, since Emscripten's VFS does not enforce permissions
        and reads directories as empty bytes.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue

            # Pre-check: detect directories since Emscripten's VFS reads
            # them as empty bytes instead of returning an error.
            try:
                check = self.execute(f"test -d {shlex.quote(path)} && echo DIR || true")
                if check.output.strip() == "DIR":
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="is_directory"
                        )
                    )
                    continue
            except RuntimeError:
                logger.debug(
                    "Directory pre-check failed for %s; proceeding with download",
                    path,
                    exc_info=True,
                )

            try:
                result = self._request("readFile", {"path": path})
            except RuntimeError as exc:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_map_error(str(exc)),
                    )
                )
                continue
            diagnostic = _extract_diagnostic(result.get("events"))
            if diagnostic:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_map_error(diagnostic),
                    )
                )
                continue
            responses.append(
                FileDownloadResponse(
                    path=path,
                    content=_decode_content(str(result["contentBase64"])),
                    error=None,
                )
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                result = self._request(
                    "writeFile",
                    {
                        "path": path,
                        "contentBase64": _encode_content(content),
                    },
                )
            except RuntimeError as exc:
                responses.append(
                    FileUploadResponse(path=path, error=_map_error(str(exc)))
                )
                continue
            diagnostic = _extract_diagnostic(result.get("events"))
            responses.append(
                FileUploadResponse(
                    path=path,
                    error=_map_error(diagnostic) if diagnostic else None,
                )
            )
        return responses

    def close(self) -> None:
        """Stop the host subprocess."""
        if self._process.poll() is not None:
            return
        try:
            self._request("close", {})
        except RuntimeError:
            logger.debug(
                "close request to node host failed (process will be terminated)",
                exc_info=True,
            )
        finally:
            self._kill_process()

    def stop(self) -> None:
        """Alias for `close()`."""
        self.close()
