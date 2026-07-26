"""Unit tests for RunloopSandbox file transfer error handling.

These tests verify that ``download_files`` and ``upload_files`` correctly
implement the partial-success contract documented on
``SandboxBackendProtocol``: per-file errors are returned in the response
objects rather than raised.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_runloop.sandbox import RunloopSandbox


def _make_sandbox() -> RunloopSandbox:
    """Create a sandbox with a mocked devbox."""
    devbox = MagicMock()
    devbox.id = "test-devbox"
    return RunloopSandbox(devbox=devbox)


# ---------------------------------------------------------------------------
# download_files
# ---------------------------------------------------------------------------


class TestDownloadFilesPartialSuccess:
    """download_files must return per-file errors, never raise on one failure."""

    def test_single_failure_does_not_discard_batch(self) -> None:
        """A provider exception on one path must not crash the remaining downloads."""
        sandbox = _make_sandbox()
        sandbox._devbox.file.download.side_effect = [  # noqa: SLF001
            b"good content",
            RuntimeError("no such file"),
            b"also good",
        ]
        paths = ["/a.txt", "/missing.txt", "/b.txt"]
        results = sandbox.download_files(paths)
        assert len(results) == len(paths)
        assert results[0].error is None
        assert results[0].content == b"good content"
        assert results[1].error is not None
        assert results[1].content is None
        assert results[2].error is None
        assert results[2].content == b"also good"

    def test_relative_path_rejected_with_invalid_path(self) -> None:
        """Paths without a leading ``/`` must be rejected as ``invalid_path``."""
        sandbox = _make_sandbox()
        results = sandbox.download_files(["relative/path.txt"])
        assert len(results) == 1
        assert results[0].error == "invalid_path"
        assert results[0].content is None
        sandbox._devbox.file.download.assert_not_called()  # noqa: SLF001

    def test_happy_path_batch(self) -> None:
        """All valid absolute paths succeed without error."""
        sandbox = _make_sandbox()
        sandbox._devbox.file.download.side_effect = [b"one", b"two"]  # noqa: SLF001
        results = sandbox.download_files(["/one.txt", "/two.txt"])
        assert all(r.error is None for r in results)
        assert results[0].content == b"one"
        assert results[1].content == b"two"


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


class TestUploadFilesPartialSuccess:
    """upload_files must return per-file errors, never raise on one failure."""

    def test_single_failure_does_not_discard_batch(self) -> None:
        """A provider exception on one path must not crash the remaining uploads."""
        sandbox = _make_sandbox()
        sandbox._devbox.file.upload.side_effect = [  # noqa: SLF001
            None,
            RuntimeError("disk full"),
            None,
        ]
        files = [("/a.txt", b"a"), ("/b.txt", b"b"), ("/c.txt", b"c")]
        results = sandbox.upload_files(files)
        assert len(results) == len(files)
        assert results[0].error is None
        assert results[1].error is not None
        assert results[2].error is None

    def test_relative_path_rejected_with_invalid_path(self) -> None:
        """Paths without a leading ``/`` must be rejected as ``invalid_path``."""
        sandbox = _make_sandbox()
        results = sandbox.upload_files([("relative/path.txt", b"data")])
        assert len(results) == 1
        assert results[0].error == "invalid_path"
        sandbox._devbox.file.upload.assert_not_called()  # noqa: SLF001

    def test_happy_path_batch(self) -> None:
        """All valid absolute paths succeed without error."""
        sandbox = _make_sandbox()
        files = [("/a.txt", b"a"), ("/b.txt", b"b")]
        results = sandbox.upload_files(files)
        assert all(r.error is None for r in results)
        assert len(results) == len(files)
