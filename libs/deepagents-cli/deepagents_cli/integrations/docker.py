"""Docker sandbox backend implementation."""

from __future__ import annotations

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

import io
import tarfile


class DockerBackend(BaseSandbox):
    """Docker backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Docker SDK.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the DockerBackend with a Docker sandbox client.

        Args:
            sandbox: Docker sandbox instance
        """
        self._sandbox = sandbox
        self._timeout: int = 30 * 60  # 30 mins

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.id

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
        result = self._sandbox.exec_run(cmd=command, user="root", workdir="/root")

        output = result.output.decode('utf-8', errors='replace') if result.output else ""
        exit_code = result.exit_code

        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Docker sandbox.

        Leverages Docker's get_archive functionality.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """

        # Download files using Docker's get_archive
        responses = []
        try:
            for path in paths:
                strm, stat = self._sandbox.get_archive(path)
                file_like_object = io.BytesIO(b"".join(chunk for chunk in strm))
                print("Before tar")
                with tarfile.open(fileobj=file_like_object, mode='r') as tar:
                    print(f"{tar.getnames()}")
                    with tar.extractfile(stat['name']) as f:
                        content = f.read()
                        responses.append(FileDownloadResponse(path=path, content=content, error=None))
        except Exception as e:
            pass

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Docker sandbox.

        Leverages Docker's put_archiv functionality.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """

        for path, content in files:
            pw_tarstream = io.BytesIO()
            with tarfile.TarFile(fileobj=pw_tarstream, mode='w') as tar:
                data_size = len(content)
                data_io = io.BytesIO(content)
                info = tarfile.TarInfo(path)
                info.size = data_size
                tar.addfile(info, data_io)
            self._sandbox.put_archive(path, pw_tarstream)

        return [FileUploadResponse(path=path, error=None) for path, _ in files]
