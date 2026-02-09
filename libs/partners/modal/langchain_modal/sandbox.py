"""Modal sandbox implementation."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import modal
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox


class ModalSandbox(BaseSandbox):
    """Modal sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(self, sandbox: modal.Sandbox) -> None:
        """Create a backend wrapping an existing Modal sandbox."""
        self._sandbox = sandbox
        self._timeout = 30 * 60

    def _read_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error="invalid_path")

        error: str | None = None
        content_bytes: bytes | None = None

        try:
            f = self._sandbox.open(path, "rb")
            try:
                content = f.read()
            finally:
                with contextlib.suppress(Exception):
                    f.close()

            if isinstance(content, memoryview):
                content_bytes = content.tobytes()
            elif isinstance(content, str):
                content_bytes = content.encode()
            else:
                content_bytes = content
        except FileNotFoundError:
            error = "file_not_found"
        except modal.exception.FilesystemExecutionError as e:
            msg = str(e).lower()
            error = "is_directory" if "is a directory" in msg else "file_not_found"

        return FileDownloadResponse(path=path, content=content_bytes, error=error)

    def _write_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")

        try:
            f = self._sandbox.open(path, "wb")
            try:
                f.write(content)
            finally:
                with contextlib.suppress(Exception):
                    f.close()
            return FileUploadResponse(path=path, error=None)
        except PermissionError:
            return FileUploadResponse(path=path, error="permission_denied")
        except FileNotFoundError:
            return FileUploadResponse(path=path, error="file_not_found")

    @property
    def id(self) -> str:
        """Return the sandbox id."""
        return self._sandbox.object_id

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        process = self._sandbox.exec("bash", "-c", command, timeout=self._timeout)
        process.wait()

        stdout = process.stdout.read()
        stderr = process.stderr.read()

        output = stdout or ""
        if stderr:
            output += "\n" + stderr if output else stderr

        return ExecuteResponse(
            output=output,
            exit_code=process.returncode,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        return [self._read_file(path) for path in paths]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        return [self._write_file(path, content) for path, content in files]


class ModalSandboxClient:
    """Modal sandbox client implementation."""

    def __init__(
        self, *, app_name: str = "deepagents-sandbox", create_if_missing: bool = True
    ) -> None:
        """Create a sandbox client backed by the Modal SDK."""
        self._app_name = app_name
        self._app = modal.App.lookup(name=app_name, create_if_missing=create_if_missing)

    def get(self, *, sandbox_id: str, **kwargs: Any) -> SandboxBackendProtocol:
        """Get an existing Modal sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"ModalSandboxClient.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception as e:
            raise ValueError(sandbox_id) from e
        return ModalSandbox(sandbox)

    def create(
        self, *, workdir: str = "/workspace", timeout: int = 180, **kwargs: Any
    ) -> SandboxBackendProtocol:
        """Create a new Modal sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"ModalSandboxClient.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        try:
            sandbox = modal.Sandbox.create(
                app=self._app,
                workdir=workdir,
                image=modal.Image.debian_slim(python_version="3.13"),
            )
        except Exception as e:
            msg = "Modal sandbox create() failed"
            raise RuntimeError(msg) from e

        for _ in range(timeout // 2):
            if sandbox.poll() is not None:
                msg = "Modal sandbox terminated unexpectedly during startup"
                raise RuntimeError(msg)
            try:
                process = sandbox.exec("echo", "ready", timeout=5)
                process.wait()
                if process.returncode == 0:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            sandbox.terminate()
            msg = f"Modal sandbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return ModalSandbox(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a Modal sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"ModalSandboxClient.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception:  # noqa: BLE001
            return
        sandbox.terminate()
