"""Standard integration contract tests for `BackendProtocol` implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol


class BackendIntegrationTests(ABC):
    """Reusable integration test contract for backend file operations."""

    @abstractmethod
    @pytest.fixture
    def backend(self) -> BackendProtocol:
        """Return a clean backend instance for a single test."""

    def test_lazy_create_on_first_write(self, backend: BackendProtocol) -> None:
        """Missing paths should read as not found, and writes should round-trip."""
        missing = backend.read("/notes.md")
        assert missing.error is not None

        write = backend.write("/notes.md", "# hi")
        assert write.error is None
        assert write.path == "/notes.md"

        read = backend.read("/notes.md")
        assert read.error is None
        assert read.file_data is not None
        assert read.file_data["content"] == "# hi"

    def test_round_trip_with_ls_grep_glob_edit(self, backend: BackendProtocol) -> None:
        """CRUD + search operations should behave consistently."""
        assert backend.write("/a.md", "hello\nworld").error is None
        assert backend.write("/b.md", "hello again").error is None
        assert backend.write("/notes/day1.md", "first note").error is None

        ls_root = backend.ls("/")
        assert ls_root.entries is not None
        root_paths = {entry["path"] for entry in ls_root.entries}
        assert "/a.md" in root_paths
        assert "/b.md" in root_paths
        assert any(path.rstrip("/") == "/notes" for path in root_paths)

        ls_nested = backend.ls("/notes")
        assert ls_nested.entries is not None
        assert {entry["path"] for entry in ls_nested.entries} == {"/notes/day1.md"}

        grep = backend.grep("hello")
        assert grep.matches is not None
        assert {match["path"] for match in grep.matches} == {"/a.md", "/b.md"}

        glob = backend.glob("*.md")
        assert glob.matches is not None
        glob_paths = {match["path"] for match in glob.matches}
        assert "/a.md" in glob_paths
        assert "/b.md" in glob_paths

        edit = backend.edit("/a.md", "world", "earth")
        assert edit.error is None
        assert edit.occurrences == 1

        updated = backend.read("/a.md")
        assert updated.error is None
        assert updated.file_data is not None
        assert "earth" in updated.file_data["content"]

    def test_download_files_round_trip(self, backend: BackendProtocol) -> None:
        """Download should return content for existing files and errors for missing paths."""
        assert backend.write("/blob.txt", "payload").error is None

        responses = backend.download_files(["/blob.txt", "/missing.txt"])
        assert len(responses) == 2
        assert responses[0].path == "/blob.txt"
        assert responses[0].content == b"payload"
        assert responses[0].error is None
        assert responses[1].path == "/missing.txt"
        assert responses[1].error is not None

    def test_upload_files_round_trip(self, backend: BackendProtocol) -> None:
        """Upload should support partial success and persist valid UTF-8 files."""
        responses = backend.upload_files(
            [
                ("/u1.md", b"one"),
                ("/u2.md", b"two"),
                ("/bad.bin", b"\x80\xff"),
            ]
        )
        assert responses[0].error is None
        assert responses[1].error is None
        assert responses[2].error == "invalid_path"

        first = backend.read("/u1.md")
        assert first.file_data is not None
        assert first.file_data["content"] == "one"

        second = backend.read("/u2.md")
        assert second.file_data is not None
        assert second.file_data["content"] == "two"
