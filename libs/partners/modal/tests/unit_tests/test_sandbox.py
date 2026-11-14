from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import modal
import pytest
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

from langchain_modal import ModalSandbox

if TYPE_CHECKING:
    from deepagents.backends.protocol import FileOperationError


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (modal.exception.SandboxFilesystemNotFoundError, "file_not_found"),
        (modal.exception.SandboxFilesystemNotADirectoryError, "file_not_found"),
        (modal.exception.SandboxFilesystemPermissionError, "permission_denied"),
        (modal.exception.SandboxFilesystemIsADirectoryError, "is_directory"),
    ],
)
def test_download_files_continues_after_error(
    exception: type[Exception], error: FileOperationError
) -> None:
    sandbox = Mock(spec=modal.Sandbox)
    content = b"\x00file contents\xff"
    sandbox.filesystem.read_bytes.side_effect = [
        exception("Cannot read path"),
        content,
    ]
    backend = ModalSandbox(sandbox=sandbox)

    responses = backend.download_files(["/blocked", "/valid.bin"])

    assert responses == [
        FileDownloadResponse(path="/blocked", content=None, error=error),
        FileDownloadResponse(path="/valid.bin", content=content, error=None),
    ]


def test_download_files_propagates_unmapped_filesystem_error() -> None:
    sandbox = Mock(spec=modal.Sandbox)
    sandbox.filesystem.read_bytes.side_effect = (
        modal.exception.SandboxFilesystemFileTooLargeError("File too large")
    )
    backend = ModalSandbox(sandbox=sandbox)

    with pytest.raises(modal.exception.SandboxFilesystemFileTooLargeError):
        backend.download_files(["/big.bin"])


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (modal.exception.SandboxFilesystemPermissionError, "permission_denied"),
        (modal.exception.SandboxFilesystemIsADirectoryError, "is_directory"),
        (modal.exception.SandboxFilesystemNotADirectoryError, "file_not_found"),
    ],
)
def test_upload_files_continues_after_error(
    exception: type[Exception], error: FileOperationError
) -> None:
    sandbox = Mock(spec=modal.Sandbox)
    content = b"\x00file contents\xff"
    sandbox.filesystem.write_bytes.side_effect = [exception("Cannot write path"), None]
    backend = ModalSandbox(sandbox=sandbox)

    responses = backend.upload_files(
        [("/blocked", b"blocked"), ("/valid.bin", content)]
    )

    assert responses == [
        FileUploadResponse(path="/blocked", error=error),
        FileUploadResponse(path="/valid.bin", error=None),
    ]
    sandbox.filesystem.write_bytes.assert_called_with(content, "/valid.bin")
