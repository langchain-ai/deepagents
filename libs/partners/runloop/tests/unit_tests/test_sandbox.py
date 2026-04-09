"""Tests for RunloopSandbox partial-success file operations."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_runloop.sandbox import RunloopSandbox


def _make_sandbox() -> tuple[RunloopSandbox, MagicMock]:
    devbox = MagicMock()
    devbox.id = "devbox-123"
    sandbox = RunloopSandbox(devbox=devbox)
    return sandbox, devbox


class TestDownloadFiles:
    def test_successful_download(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.download.return_value = b"hello"
        responses = sandbox.download_files(["/tmp/a.txt"])
        assert len(responses) == 1
        assert responses[0].content == b"hello"
        assert responses[0].error is None

    def test_file_not_found(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.download.side_effect = FileNotFoundError("missing")
        responses = sandbox.download_files(["/missing.txt"])
        assert len(responses) == 1
        assert responses[0].content is None
        assert responses[0].error == "file_not_found"

    def test_permission_denied(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.download.side_effect = PermissionError("denied")
        responses = sandbox.download_files(["/secret.txt"])
        assert len(responses) == 1
        assert responses[0].content is None
        assert responses[0].error == "permission_denied"

    def test_is_directory_error(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.download.side_effect = Exception("path is a directory")
        responses = sandbox.download_files(["/tmp"])
        assert len(responses) == 1
        assert responses[0].content is None
        assert responses[0].error == "is_directory"

    def test_partial_success(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.download.side_effect = [
            FileNotFoundError("missing"),
            b"content",
        ]
        responses = sandbox.download_files(["/missing.txt", "/exists.txt"])
        assert len(responses) == 2
        assert responses[0].error == "file_not_found"
        assert responses[0].content is None
        assert responses[1].error is None
        assert responses[1].content == b"content"


class TestUploadFiles:
    def test_successful_upload(self) -> None:
        sandbox, devbox = _make_sandbox()
        responses = sandbox.upload_files([("/tmp/a.txt", b"hello")])
        assert len(responses) == 1
        assert responses[0].error is None

    def test_file_not_found(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.upload.side_effect = FileNotFoundError("no parent dir")
        responses = sandbox.upload_files([("/no/parent/file.txt", b"data")])
        assert len(responses) == 1
        assert responses[0].error == "file_not_found"

    def test_permission_denied(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.upload.side_effect = PermissionError("denied")
        responses = sandbox.upload_files([("/proc/1/root", b"data")])
        assert len(responses) == 1
        assert responses[0].error == "permission_denied"

    def test_partial_success(self) -> None:
        sandbox, devbox = _make_sandbox()
        devbox.file.upload.side_effect = [
            PermissionError("denied"),
            None,
        ]
        responses = sandbox.upload_files([
            ("/proc/1/root", b"impossible"),
            ("/tmp/valid.txt", b"hello"),
        ])
        assert len(responses) == 2
        assert responses[0].error == "permission_denied"
        assert responses[1].error is None
