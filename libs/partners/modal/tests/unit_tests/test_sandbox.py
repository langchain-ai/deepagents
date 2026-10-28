from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import Mock

import modal
import pytest
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

from langchain_modal import ModalSandbox

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.backends.protocol import FileOperationError


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (FileNotFoundError, "file_not_found"),
        (PermissionError, "permission_denied"),
        (IsADirectoryError, "is_directory"),
    ],
)
def test_download_files_continues_after_error(
    exception: type[OSError], error: FileOperationError
) -> None:
    sandbox = Mock(spec=modal.Sandbox)
    content = b"\x00file contents\xff"
    with BytesIO(content) as file:
        sandbox.open.side_effect = [exception("Cannot read path"), file]
        backend = ModalSandbox(sandbox=sandbox)

        responses = backend.download_files(["/blocked", "/valid.bin"])

        assert responses == [
            FileDownloadResponse(path="/blocked", content=None, error=error),
            FileDownloadResponse(path="/valid.bin", content=content, error=None),
        ]
        assert file.closed


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (FileNotFoundError, "file_not_found"),
        (PermissionError, "permission_denied"),
        (IsADirectoryError, "is_directory"),
    ],
)
def test_upload_files_continues_after_error(
    tmp_path: Path, exception: type[OSError], error: FileOperationError
) -> None:
    sandbox = Mock(spec=modal.Sandbox)
    content = b"\x00file contents\xff"
    destination = tmp_path / "valid.bin"
    with destination.open("wb") as file:
        sandbox.open.side_effect = [exception("Cannot write path"), file]
        backend = ModalSandbox(sandbox=sandbox)

        responses = backend.upload_files(
            [("/blocked", b"blocked"), ("/valid.bin", content)]
        )

        assert responses == [
            FileUploadResponse(path="/blocked", error=error),
            FileUploadResponse(path="/valid.bin", error=None),
        ]
        assert file.closed
    assert destination.read_bytes() == content
