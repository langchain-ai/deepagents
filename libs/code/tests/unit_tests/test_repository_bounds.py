"""Unit tests for the shared repository-inspection bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import LsResult

from deepagents_code._repository_bounds import REPOSITORY_PATH_ERROR, RepositoryBounds

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


class TestSafePath:
    """Explicit paths must be absolute, non-traversing, and under the root."""

    @pytest.mark.parametrize(
        "path", ["../etc/passwd", "~/secrets", "relative/x", "/a/../b"]
    )
    def test_unsafe_paths_are_rejected(self, path: str) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert bounds.safe_path(path) is False


class TestClampArgs:
    """Read/search arguments are clamped to hard limits."""


class TestBoundText:
    """Result bodies are size and match bounded."""


class TestPreflight:
    """Preflight enforces path safety and backend metadata limits."""

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
