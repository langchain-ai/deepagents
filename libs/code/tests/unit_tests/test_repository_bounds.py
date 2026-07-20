"""Unit tests for the shared repository-inspection bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import LsResult

from deepagents_code._repository_bounds import (
    REPOSITORY_DIRECTORY_ENTRY_LIMIT,
    REPOSITORY_GLOB_MATCH_LIMIT,
    REPOSITORY_GREP_MATCH_LIMIT,
    REPOSITORY_LISTING_ERROR,
    REPOSITORY_PATH_ERROR,
    REPOSITORY_READ_BYTE_LIMIT,
    REPOSITORY_READ_LINE_LIMIT,
    REPOSITORY_SIZE_ERROR,
    REPOSITORY_TOOL_RESULT_LIMIT,
    RepositoryBounds,
)

if TYPE_CHECKING:
    from pathlib import Path


def _backend(*, size: int = 10) -> MagicMock:
    backend = MagicMock()
    backend.ls.return_value = LsResult(
        entries=[{"path": "/src.py", "is_dir": False, "size": size}]
    )
    return backend


class TestRepositoryBoundsConstruction:
    """The root is validated and normalized at construction time."""

    @pytest.mark.parametrize("root", ["relative", "/a/../b", "~/x"])
    def test_rejects_unsafe_root(self, root: str) -> None:
        with pytest.raises(ValueError, match="absolute contained path"):
            RepositoryBounds(_backend(), root=root)

    def test_normalizes_root(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace/")
        assert bounds.root == "/workspace"


class TestSafePath:
    """Explicit paths must be absolute, non-traversing, and under the root."""

    @pytest.mark.parametrize(
        "path", ["../etc/passwd", "~/secrets", "relative/x", "/a/../b"]
    )
    def test_unsafe_paths_are_rejected(self, path: str) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert bounds.safe_path(path) is False

    def test_paths_outside_root_are_rejected(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert bounds.safe_path("/etc/passwd") is False

    def test_paths_under_root_are_allowed(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert bounds.safe_path("/workspace/pkg/app.py") is True

    def test_root_slash_allows_any_absolute_path(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        assert bounds.safe_path("/anything/here.py") is True


class TestClampArgs:
    """Read/search arguments are clamped to hard limits."""

    def test_read_limit_is_clamped(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        clamped = bounds.clamp_args("read_file", {"file_path": "/x", "limit": 10_000})
        assert clamped["limit"] == REPOSITORY_READ_LINE_LIMIT

    @pytest.mark.parametrize("limit", [True, 0, -3])
    def test_invalid_read_limit_falls_back(self, limit: object) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        clamped = bounds.clamp_args("read_file", {"file_path": "/x", "limit": limit})
        assert 1 <= clamped["limit"] <= REPOSITORY_READ_LINE_LIMIT

    def test_search_path_defaults_to_root(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert bounds.clamp_args("glob", {"pattern": "**/*.py"})["path"] == "/workspace"
        assert (
            bounds.clamp_args("grep", {"pattern": "x", "path": None})["path"]
            == "/workspace"
        )

    def test_grep_max_count_is_clamped(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        clamped = bounds.clamp_args("grep", {"pattern": "x", "max_count": 10_000})
        assert clamped["max_count"] == REPOSITORY_GREP_MATCH_LIMIT


class TestBoundText:
    """Result bodies are size and match bounded."""

    def test_long_content_is_truncated(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        bounded = bounds.bound_text(
            "read_file", "x" * (REPOSITORY_TOOL_RESULT_LIMIT + 500)
        )
        assert len(bounded) <= REPOSITORY_TOOL_RESULT_LIMIT

    def test_glob_matches_are_limited(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        paths = [f"/{index}.py" for index in range(REPOSITORY_GLOB_MATCH_LIMIT + 5)]
        bounded = bounds.bound_text("glob", str(paths))
        assert "Glob results limited" in bounded


class TestPreflight:
    """Preflight enforces path safety and backend metadata limits."""

    def test_rejects_unsafe_path(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert (
            bounds.preflight("read_file", {"file_path": "/etc/passwd"})
            == REPOSITORY_PATH_ERROR
        )

    def test_rejects_local_symlink_outside_root(self, tmp_path: Path) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        link = repository / "proof.txt"
        link.symlink_to(secret)
        backend = FilesystemBackend(root_dir=repository, virtual_mode=False)
        bounds = RepositoryBounds(backend, root=str(repository))

        assert (
            bounds.preflight("read_file", {"file_path": str(link)})
            == REPOSITORY_PATH_ERROR
        )

    async def test_async_rejects_local_symlink_outside_root(
        self, tmp_path: Path
    ) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        link = repository / "proof.txt"
        link.symlink_to(secret)
        backend = FilesystemBackend(root_dir=repository, virtual_mode=False)
        bounds = RepositoryBounds(backend, root=str(repository))

        assert (
            await bounds.apreflight("read_file", {"file_path": str(link)})
            == REPOSITORY_PATH_ERROR
        )

    def test_allows_local_symlink_within_root(self, tmp_path: Path) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        target = repository / "target.txt"
        target.write_text("safe")
        link = repository / "proof.txt"
        link.symlink_to(target)
        backend = FilesystemBackend(root_dir=repository, virtual_mode=False)
        bounds = RepositoryBounds(backend, root=str(repository))

        assert bounds.preflight("read_file", {"file_path": str(link)}) is None

    def test_allows_virtual_filesystem_path_within_root(self, tmp_path: Path) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        (repository / "proof.txt").write_text("safe")
        backend = FilesystemBackend(root_dir=repository, virtual_mode=True)
        bounds = RepositoryBounds(backend)

        assert bounds.preflight("read_file", {"file_path": "/proof.txt"}) is None

    def test_rejects_large_file(self) -> None:
        bounds = RepositoryBounds(
            _backend(size=REPOSITORY_READ_BYTE_LIMIT + 1), root="/"
        )
        assert (
            bounds.preflight("read_file", {"file_path": "/src.py"})
            == REPOSITORY_SIZE_ERROR
        )

    def test_rejects_large_directory(self) -> None:
        backend = MagicMock()
        backend.ls.return_value = LsResult(
            entries=[
                {"path": f"/{index}", "is_dir": False}
                for index in range(REPOSITORY_DIRECTORY_ENTRY_LIMIT + 1)
            ]
        )
        bounds = RepositoryBounds(backend, root="/")
        assert bounds.preflight("ls", {"path": "/"}) == REPOSITORY_LISTING_ERROR

    def test_backend_error_degrades_to_unavailable(self) -> None:
        backend = MagicMock()
        backend.ls.side_effect = RuntimeError("outage")
        bounds = RepositoryBounds(backend, root="/")
        assert (
            bounds.preflight("read_file", {"file_path": "/src.py"})
            == REPOSITORY_PATH_ERROR
        )

    def test_allows_valid_read(self) -> None:
        bounds = RepositoryBounds(_backend(), root="/")
        assert bounds.preflight("read_file", {"file_path": "/src.py"}) is None
