"""Tests for the Harbor sandbox backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_harbor.backend import HarborSandbox


class _FakeHarborEnvironment:
    """Minimal async Harbor environment for backend tests."""

    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        download_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.session_id = "harbor-test-session"
        self.files = files or {}
        self.download_errors = download_errors or {}
        self.uploaded: dict[str, bytes] = {}

    async def exec(self, command: str):
        msg = "exec() should not be called in these tests"
        raise AssertionError(msg)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if source_path in self.download_errors:
            raise self.download_errors[source_path]
        Path(target_path).write_bytes(self.files[source_path])

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        content = Path(source_path).read_bytes()
        self.uploaded[target_path] = content
        self.files[target_path] = content


@pytest.mark.parametrize("line_ending", ["\r\n", "\r"])
async def test_aedit_preserves_existing_line_endings(line_ending: str) -> None:
    original = f"alpha{line_ending}old{line_ending}omega{line_ending}".encode()
    env = _FakeHarborEnvironment(files={"/app/test.txt": original})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/test.txt", "old", "new")

    assert result.error is None
    assert result.occurrences == 1
    assert env.uploaded["/app/test.txt"] == (
        f"alpha{line_ending}new{line_ending}omega{line_ending}".encode()
    )


@pytest.mark.parametrize(
    ("exc", "expected_error"),
    [
        (FileNotFoundError("missing file"), "file_not_found"),
        (PermissionError("permission denied"), "permission_denied"),
        (IsADirectoryError("is a directory"), "is_directory"),
        (ValueError("invalid path"), "invalid_path"),
        (RuntimeError("permission denied"), "permission_denied"),
        (RuntimeError("is a directory"), "is_directory"),
    ],
)
async def test_adownload_files_maps_known_errors(exc: Exception, expected_error: str) -> None:
    env = _FakeHarborEnvironment(download_errors={"/app/test.txt": exc})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.adownload_files(["/app/test.txt"])

    assert len(responses) == 1
    assert responses[0].path == "/app/test.txt"
    assert responses[0].content is None
    assert responses[0].error == expected_error


async def test_adownload_files_propagates_unknown_errors() -> None:
    env = _FakeHarborEnvironment(
        download_errors={"/app/test.txt": RuntimeError("transient download failure")}
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    with pytest.raises(RuntimeError, match="transient download failure"):
        await sandbox.adownload_files(["/app/test.txt"])
