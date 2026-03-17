"""Tests for BackendProtocol and SandboxBackendProtocol base class behavior.

Verifies that unimplemented protocol methods raise NotImplementedError
instead of silently returning None.
"""

import pytest

from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol


class BareBackend(BackendProtocol):
    """Minimal subclass that implements nothing."""


class BareSandboxBackend(SandboxBackendProtocol):
    """Minimal subclass that implements nothing."""


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
            backend.ls("/")

    def test_read(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.read("/file.txt")

    def test_grep(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            backend.grep("pattern")

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

    async def test_als_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.als("/")

    async def test_aread(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aread("/file.txt")

    async def test_agrep(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.agrep("pattern")

    async def test_aglob_info(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aglob_info("*.py")

    async def test_awrite(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.awrite("/file.txt", "content")

    async def test_aedit(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aedit("/file.txt", "old", "new")

    async def test_aupload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.aupload_files([("/file.txt", b"data")])

    async def test_adownload_files(self, backend: BareBackend) -> None:
        with pytest.raises(NotImplementedError):
            await backend.adownload_files(["/file.txt"])

    async def test_aexecute(self, sandbox_backend: BareSandboxBackend) -> None:
        with pytest.raises(NotImplementedError):
            await sandbox_backend.aexecute("ls")
