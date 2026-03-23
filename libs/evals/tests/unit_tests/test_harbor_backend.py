"""Tests for the Harbor sandbox backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from deepagents_harbor.backend import HarborSandbox


@dataclass
class _FakeExecResult:
    """Minimal result type matching Harbor's exec() return contract."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class _FakeHarborEnvironment:
    """Minimal async Harbor environment for backend tests."""

    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        download_errors: dict[str, Exception] | None = None,
        upload_errors: dict[str, Exception] | None = None,
        exec_result: _FakeExecResult | None = None,
    ) -> None:
        self.session_id = "harbor-test-session"
        self.files = files or {}
        self.download_errors = download_errors or {}
        self.upload_errors = upload_errors or {}
        self.uploaded: dict[str, bytes] = {}
        self._exec_result = exec_result

    async def exec(self, command: str) -> _FakeExecResult:
        if self._exec_result is None:
            msg = "exec() should not be called in these tests"
            raise AssertionError(msg)
        return self._exec_result

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if source_path in self.download_errors:
            raise self.download_errors[source_path]
        if source_path not in self.files:
            msg = f"File not found: {source_path}"
            raise FileNotFoundError(msg)
        Path(target_path).write_bytes(self.files[source_path])

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if target_path in self.upload_errors:
            raise self.upload_errors[target_path]
        content = Path(source_path).read_bytes()
        self.uploaded[target_path] = content
        self.files[target_path] = content


# -- aedit tests ---------------------------------------------------------------


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


async def test_aedit_file_not_found() -> None:
    env = _FakeHarborEnvironment()
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/missing.txt", "old", "new")

    assert result.error is not None
    assert "file_not_found" in result.error


async def test_aedit_binary_file_returns_error() -> None:
    env = _FakeHarborEnvironment(files={"/app/binary.bin": b"\x80\x81\x82\xff"})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/binary.bin", "old", "new")

    assert result.error is not None
    assert "not a text file" in result.error


async def test_aedit_string_not_found() -> None:
    env = _FakeHarborEnvironment(files={"/app/test.txt": b"hello world"})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/test.txt", "missing", "new")

    assert result.error is not None
    assert "not found" in result.error.lower()


async def test_aedit_multiple_occurrences_without_replace_all() -> None:
    env = _FakeHarborEnvironment(files={"/app/test.txt": b"foo bar foo"})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/test.txt", "foo", "baz")

    assert result.error is not None
    assert "multiple times" in result.error.lower()


async def test_aedit_replace_all() -> None:
    env = _FakeHarborEnvironment(files={"/app/test.txt": b"foo bar foo"})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/test.txt", "foo", "baz", replace_all=True)

    assert result.error is None
    assert result.occurrences == 2
    assert env.uploaded["/app/test.txt"] == b"baz bar baz"


async def test_aedit_propagates_unknown_download_errors() -> None:
    env = _FakeHarborEnvironment(
        download_errors={"/app/test.txt": RuntimeError("transient failure")}
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    with pytest.raises(RuntimeError, match="transient failure"):
        await sandbox.aedit("/app/test.txt", "old", "new")


async def test_aedit_propagates_unknown_upload_errors() -> None:
    env = _FakeHarborEnvironment(
        files={"/app/test.txt": b"hello old world"},
        upload_errors={"/app/test.txt": RuntimeError("transient upload failure")},
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    with pytest.raises(RuntimeError, match="transient upload failure"):
        await sandbox.aedit("/app/test.txt", "old", "new")


async def test_aedit_maps_known_upload_errors() -> None:
    env = _FakeHarborEnvironment(
        files={"/app/test.txt": b"hello old world"},
        upload_errors={"/app/test.txt": PermissionError("denied")},
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.aedit("/app/test.txt", "old", "new")

    assert result.error is not None
    assert "permission_denied" in result.error


# -- awrite tests --------------------------------------------------------------


async def test_awrite_uploads_content() -> None:
    env = _FakeHarborEnvironment(exec_result=_FakeExecResult())
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.awrite("/app/new.txt", "hello world")

    assert result.error is None
    assert result.path == "/app/new.txt"
    assert env.uploaded["/app/new.txt"] == b"hello world"


async def test_awrite_propagates_unknown_upload_errors() -> None:
    env = _FakeHarborEnvironment(
        exec_result=_FakeExecResult(),
        upload_errors={"/app/new.txt": RuntimeError("transient failure")},
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    with pytest.raises(RuntimeError, match="transient failure"):
        await sandbox.awrite("/app/new.txt", "content")


async def test_awrite_maps_known_upload_errors() -> None:
    env = _FakeHarborEnvironment(
        exec_result=_FakeExecResult(),
        upload_errors={"/app/new.txt": PermissionError("denied")},
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.awrite("/app/new.txt", "content")

    assert result.error is not None
    assert "permission_denied" in result.error


async def test_awrite_returns_error_when_file_exists() -> None:
    env = _FakeHarborEnvironment(
        exec_result=_FakeExecResult(
            stderr="Error: File already exists: '/app/existing.txt'",
            return_code=1,
        ),
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    result = await sandbox.awrite("/app/existing.txt", "content")

    assert result.error is not None
    assert "Error:" in result.error


# -- adownload_files tests -----------------------------------------------------


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


async def test_adownload_files_returns_unknown_errors_in_response() -> None:
    env = _FakeHarborEnvironment(
        download_errors={"/app/test.txt": RuntimeError("transient download failure")}
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.adownload_files(["/app/test.txt"])

    assert len(responses) == 1
    assert responses[0].path == "/app/test.txt"
    assert responses[0].content is None
    assert responses[0].error == "transient download failure"


async def test_adownload_files_partial_success() -> None:
    env = _FakeHarborEnvironment(
        files={"/app/good.txt": b"content"},
        download_errors={"/app/bad.txt": PermissionError("denied")},
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.adownload_files(["/app/good.txt", "/app/bad.txt"])

    assert len(responses) == 2
    assert responses[0].error is None
    assert responses[0].content == b"content"
    assert responses[1].error == "permission_denied"
    assert responses[1].content is None


async def test_adownload_files_partial_success_on_unknown_errors() -> None:
    env = _FakeHarborEnvironment(
        files={"/app/good.txt": b"content"},
        download_errors={"/app/bad.txt": RuntimeError("transient download failure")},
    )
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.adownload_files(["/app/bad.txt", "/app/good.txt"])

    assert len(responses) == 2
    assert responses[0].error == "transient download failure"
    assert responses[0].content is None
    assert responses[1].error is None
    assert responses[1].content == b"content"


# -- aupload_files tests -------------------------------------------------------


async def test_aupload_files_happy_path() -> None:
    env = _FakeHarborEnvironment()
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.aupload_files([("/app/file.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].error is None
    assert env.uploaded["/app/file.txt"] == b"content"


async def test_aupload_files_maps_known_errors() -> None:
    env = _FakeHarborEnvironment(upload_errors={"/app/denied.txt": PermissionError("denied")})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.aupload_files([("/app/denied.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].error == "permission_denied"


async def test_aupload_files_returns_unknown_errors_in_response() -> None:
    env = _FakeHarborEnvironment(upload_errors={"/app/file.txt": RuntimeError("transient failure")})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.aupload_files([("/app/file.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].path == "/app/file.txt"
    assert responses[0].error == "transient failure"


async def test_aupload_files_partial_success_on_unknown_errors() -> None:
    env = _FakeHarborEnvironment(upload_errors={"/app/bad.txt": RuntimeError("transient failure")})
    sandbox = HarborSandbox(env)  # type: ignore[invalid-argument-type]

    responses = await sandbox.aupload_files(
        [("/app/bad.txt", b"first"), ("/app/good.txt", b"second")]
    )

    assert len(responses) == 2
    assert responses[0].error == "transient failure"
    assert responses[1].error is None
    assert env.uploaded["/app/good.txt"] == b"second"
