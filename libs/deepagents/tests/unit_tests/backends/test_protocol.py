"""Tests for BackendProtocol and SandboxBackendProtocol base class behavior.

Verifies that unimplemented protocol methods raise NotImplementedError
instead of silently returning None.
"""

import pytest

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)


class BareBackend(BackendProtocol):
    """Minimal subclass that delegates to super() so NotImplementedError propagates."""

    def ls_info(self, path: str) -> list[FileInfo]:
        return super().ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return super().read(file_path, offset, limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return super().grep_raw(pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return super().glob_info(pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:
        return super().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool | None = None,  # noqa: FBT001
    ) -> EditResult:
        return super().edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return super().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return super().download_files(paths)


class BareSandboxBackend(SandboxBackendProtocol):
    """Minimal subclass that delegates to super() so NotImplementedError propagates."""

    def ls_info(self, path: str) -> list[FileInfo]:
        return super().ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return super().read(file_path, offset, limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return super().grep_raw(pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return super().glob_info(pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:
        return super().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool | None = None,  # noqa: FBT001
    ) -> EditResult:
        return super().edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return super().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return super().download_files(paths)

    def execute(self, command: str) -> ExecuteResponse:
        return super().execute(command)


@pytest.fixture
def backend() -> BareBackend:
    return BareBackend()


@pytest.fixture
def sandbox_backend() -> BareSandboxBackend:
    return BareSandboxBackend()


class TestBackendProtocolRaisesNotImplemented:
    """All sync methods on BackendProtocol must raise NotImplementedError."""

    def test_ls_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.ls_info("/")

    def test_read(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.read("/file.txt")

    def test_grep_raw(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.grep_raw("pattern")

    def test_glob_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.glob_info("*.py")

    def test_write(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.write("/file.txt", "content")

    def test_edit(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.edit("/file.txt", "old", "new")

    def test_upload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.upload_files([("/file.txt", b"data")])

    def test_download_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.download_files(["/file.txt"])


class TestSandboxBackendProtocolRaisesNotImplemented:
    """SandboxBackendProtocol.execute must raise NotImplementedError."""

    def test_execute(self, sandbox_backend: BareSandboxBackend) -> None:
        with pytest.raises(NotImplementedError):
            sandbox_backend.execute("ls")


class TestAsyncMethodsPropagateNotImplemented:
    """Async wrappers delegate to sync methods, so NotImplementedError propagates."""

    @pytest.mark.asyncio
    async def test_als_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.als_info("/")

    @pytest.mark.asyncio
    async def test_aread(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aread("/file.txt")

    @pytest.mark.asyncio
    async def test_agrep_raw(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.agrep_raw("pattern")

    @pytest.mark.asyncio
    async def test_aglob_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aglob_info("*.py")

    @pytest.mark.asyncio
    async def test_awrite(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.awrite("/file.txt", "content")

    @pytest.mark.asyncio
    async def test_aedit(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aedit("/file.txt", "old", "new")

    @pytest.mark.asyncio
    async def test_aupload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aupload_files([("/file.txt", b"data")])

    @pytest.mark.asyncio
    async def test_adownload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.adownload_files(["/file.txt"])

    @pytest.mark.asyncio
    async def test_aexecute(self, sandbox_backend: BareSandboxBackend) -> None:
        with pytest.raises(NotImplementedError):
            await sandbox_backend.aexecute("ls")
