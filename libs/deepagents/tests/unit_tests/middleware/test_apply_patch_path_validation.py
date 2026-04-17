"""Tests for path traversal validation in apply_patch helpers."""

from __future__ import annotations

import pytest

from deepagents.backends.protocol import BackendProtocol, EditResult, ReadResult, WriteResult
from deepagents.middleware.filesystem import _aapply_file_changes, _apply_file_changes


class _NeverCalledBackend(BackendProtocol):
    """Backend that fails if any operation is called."""

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        msg = f"unexpected read: {file_path}"
        raise AssertionError(msg)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        msg = f"unexpected aread: {file_path}"
        raise AssertionError(msg)

    def write(self, path: str, content: str) -> WriteResult:
        msg = f"unexpected write: {path}"
        raise AssertionError(msg)

    async def awrite(self, path: str, content: str) -> WriteResult:
        msg = f"unexpected awrite: {path}"
        raise AssertionError(msg)

    def edit(self, path: str, old: str, new: str) -> EditResult:
        msg = f"unexpected edit: {path}"
        raise AssertionError(msg)

    async def aedit(self, path: str, old: str, new: str) -> EditResult:
        msg = f"unexpected aedit: {path}"
        raise AssertionError(msg)


class _RecordingBackend(BackendProtocol):
    """Backend that records which paths are accessed."""

    def __init__(self) -> None:
        self.accessed: list[str] = []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.accessed.append(file_path)
        return ReadResult(error="not found")

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.accessed.append(file_path)
        return ReadResult(error="not found")

    def write(self, path: str, content: str) -> WriteResult:
        self.accessed.append(path)
        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        self.accessed.append(path)
        return WriteResult(path=path)


class TestApplyFileChangesPathValidation:
    """Path traversal attacks must be blocked before reaching the backend."""

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../etc/passwd",
            "foo/../../etc/shadow",
            "~/secret.txt",
            "C:\\Windows\\system32\\config",
        ],
    )
    def test_traversal_path_rejected_sync(self, malicious_path: str) -> None:
        backend = _NeverCalledBackend()
        result = _apply_file_changes(backend, {malicious_path: "pwned"})
        assert "Error" in result

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../etc/passwd",
            "foo/../../etc/shadow",
            "~/secret.txt",
            "C:\\Windows\\system32\\config",
        ],
    )
    async def test_traversal_path_rejected_async(self, malicious_path: str) -> None:
        backend = _NeverCalledBackend()
        result = await _aapply_file_changes(backend, {malicious_path: "pwned"})
        assert "Error" in result

    def test_valid_path_passes_through_sync(self) -> None:
        backend = _RecordingBackend()
        result = _apply_file_changes(backend, {"/app/safe.py": "content"})
        assert "Created '/app/safe.py'" in result
        assert "/app/safe.py" in backend.accessed

    async def test_valid_path_passes_through_async(self) -> None:
        backend = _RecordingBackend()
        result = await _aapply_file_changes(backend, {"/app/safe.py": "content"})
        assert "Created '/app/safe.py'" in result
        assert "/app/safe.py" in backend.accessed

    def test_mixed_valid_and_malicious_sync(self) -> None:
        """Valid paths succeed; traversal paths are rejected individually."""
        backend = _RecordingBackend()
        changes: dict[str, str | None] = {
            "/app/ok.py": "good",
            "../etc/passwd": "bad",
            "/app/also_ok.py": "fine",
        }
        result = _apply_file_changes(backend, changes)
        assert "Created '/app/ok.py'" in result
        assert "Created '/app/also_ok.py'" in result
        assert "Error" in result
        assert "/app/ok.py" in backend.accessed
        assert "/app/also_ok.py" in backend.accessed

    async def test_mixed_valid_and_malicious_async(self) -> None:
        """Valid paths succeed; traversal paths are rejected individually."""
        backend = _RecordingBackend()
        changes: dict[str, str | None] = {
            "/app/ok.py": "good",
            "../etc/passwd": "bad",
            "/app/also_ok.py": "fine",
        }
        result = await _aapply_file_changes(backend, changes)
        assert "Created '/app/ok.py'" in result
        assert "Created '/app/also_ok.py'" in result
        assert "Error" in result
        assert "/app/ok.py" in backend.accessed
        assert "/app/also_ok.py" in backend.accessed

    def test_delete_with_traversal_path_rejected_sync(self) -> None:
        """Deletion of a traversal path is also blocked."""
        backend = _NeverCalledBackend()
        result = _apply_file_changes(backend, {"../etc/passwd": None})
        assert "Error" in result

    async def test_delete_with_traversal_path_rejected_async(self) -> None:
        backend = _NeverCalledBackend()
        result = await _aapply_file_changes(backend, {"../etc/passwd": None})
        assert "Error" in result
