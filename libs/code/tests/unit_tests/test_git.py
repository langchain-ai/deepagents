"""Unit tests for the deepagents_code._git module."""

import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code._git import (
    RepositoryMetadata,
    _abbreviate_git_ref,
    _git_dir_cache,
    _normalize_lookup_path,
    _parse_git_dir_pointer,
    find_git_common_dir,
    find_git_dir,
    find_git_root,
    parse_repository_metadata,
    read_git_branch_from_filesystem,
    read_git_branch_via_subprocess,
    read_git_commit_sha_from_filesystem,
    read_git_commit_sha_via_subprocess,
    read_git_remote_url_from_filesystem,
    read_git_remote_url_via_subprocess,
    resolve_git_branch,
    resolve_git_commit_sha,
    resolve_git_remote_url,
)

_FULL_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
"""A valid 40-char SHA-1 used across the commit-resolution tests."""


def _run_git(root: Path, *args: str) -> None:
    """Run Git in a throwaway repository."""
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git_repo(root: Path) -> None:
    """Create a committed repository suitable for real worktree tests."""
    root.mkdir()
    _run_git(root, "init")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test")
    _run_git(root, "config", "commit.gpgsign", "false")
    (root / "tracked.txt").write_text("initial\n")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "-m", "initial")


def _add_git_worktree(root: Path, worktree: Path, branch: str) -> None:
    """Add a linked worktree backed by `root`'s common metadata."""
    _run_git(root, "worktree", "add", "-b", branch, str(worktree), "HEAD")


@pytest.fixture(autouse=True)
def clear_git_dir_cache() -> Iterator[None]:
    _git_dir_cache.clear()
    yield
    _git_dir_cache.clear()


class TestNormalizeLookupPath:
    @patch("pathlib.Path.resolve")
    def test_normalize_os_error_fallback(
        self, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        mock_resolve.side_effect = OSError("Permission denied")
        assert _normalize_lookup_path(tmp_path) == tmp_path


class TestFindGitCommonDir:
    def test_main_and_genuine_sibling_worktrees_share_identity(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        _init_git_repo(main)
        _add_git_worktree(main, first, "first")
        _add_git_worktree(main, second, "second")

        expected = (main / ".git").resolve()
        assert find_git_common_dir(main) == expected
        assert find_git_common_dir(first) == expected
        assert find_git_common_dir(second) == expected

    def test_direct_pointer_to_another_common_dir_is_rejected(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        forged.mkdir()
        (forged / ".git").write_text(f"gitdir: {main / '.git'}\n")

        assert find_git_common_dir(forged) is None

    def test_independent_repositories_have_distinct_identities(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        _init_git_repo(first)
        _init_git_repo(second)

        assert find_git_common_dir(first) == (first / ".git").resolve()
        assert find_git_common_dir(second) == (second / ".git").resolve()
        assert find_git_common_dir(first) != find_git_common_dir(second)

    def test_non_git_directory_has_no_common_identity(self, tmp_path: Path) -> None:
        assert find_git_common_dir(tmp_path) is None

    def test_nested_non_repo_directory_does_not_inherit_parent_identity(
        self, tmp_path: Path
    ) -> None:
        outer = tmp_path / "outer"
        _init_git_repo(outer)
        child = outer / "child"
        child.mkdir()

        assert find_git_common_dir(child) is None

    def test_missing_worktree_common_dir_is_rejected(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        _init_git_repo(main)
        _add_git_worktree(main, worktree, "worktree")
        git_dir = _parse_git_dir_pointer(worktree / ".git")
        assert git_dir is not None
        (git_dir / "commondir").unlink()

        assert find_git_common_dir(worktree) is None

    def test_malformed_worktree_common_dir_is_rejected(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        _init_git_repo(main)
        _add_git_worktree(main, worktree, "worktree")
        git_dir = _parse_git_dir_pointer(worktree / ".git")
        assert git_dir is not None
        (git_dir / "commondir").write_text("../..\nunexpected\n")

        assert find_git_common_dir(worktree) is None

    def test_worktree_missing_head_is_rejected(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        _init_git_repo(main)
        _add_git_worktree(main, worktree, "worktree")
        git_dir = _parse_git_dir_pointer(worktree / ".git")
        assert git_dir is not None
        (git_dir / "HEAD").unlink()

        assert find_git_common_dir(worktree) is None

    def test_worktree_missing_backlink_is_rejected(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        _init_git_repo(main)
        _add_git_worktree(main, worktree, "worktree")
        git_dir = _parse_git_dir_pointer(worktree / ".git")
        assert git_dir is not None
        (git_dir / "gitdir").unlink()

        assert find_git_common_dir(worktree) is None

    def test_forged_pointer_to_another_worktree_is_rejected(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        genuine = tmp_path / "genuine"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        _add_git_worktree(main, genuine, "genuine")
        git_dir = _parse_git_dir_pointer(genuine / ".git")
        assert git_dir is not None
        forged.mkdir()
        (forged / ".git").write_text(f"gitdir: {git_dir}\n")

        assert find_git_common_dir(genuine) == (main / ".git").resolve()
        assert find_git_common_dir(forged) is None

    def test_symlinked_stale_backlink_does_not_validate_forged_worktree(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        genuine = tmp_path / "genuine"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        _add_git_worktree(main, genuine, "genuine")
        git_dir = _parse_git_dir_pointer(genuine / ".git")
        assert git_dir is not None
        forged.mkdir()
        (forged / ".git").write_text(f"gitdir: {git_dir}\n")
        (genuine / ".git").unlink()
        (genuine / ".git").symlink_to(forged / ".git")

        assert find_git_common_dir(forged) is None

    def test_self_consistent_forgery_outside_worktrees_dir_is_rejected(
        self, tmp_path: Path
    ) -> None:
        # A fully self-consistent forgery: the attacker's admin dir has a valid
        # `commondir` pointing at the victim repo, a present `HEAD`, and a correct
        # self-referential `gitdir` backlink. Every acceptance check passes EXCEPT
        # the admin dir's location, so the `worktrees/`-parent guard is the only
        # defense that rejects it. Removing that guard makes this test fail.
        main = tmp_path / "main"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        forged.mkdir()
        common = (main / ".git").resolve()
        admin = tmp_path / "attacker" / "wt"
        admin.mkdir(parents=True)
        (admin / "commondir").write_text(f"{common}\n")
        (admin / "HEAD").write_text("ref: refs/heads/forged\n")
        (admin / "gitdir").write_text(f"{forged.resolve() / '.git'}\n")
        (forged / ".git").write_text(f"gitdir: {admin}\n")

        assert admin.parent != common / "worktrees"
        assert find_git_common_dir(forged) is None

    def test_symlinked_git_entry_is_rejected(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        forged.mkdir()
        (forged / ".git").symlink_to(main / ".git", target_is_directory=True)

        assert find_git_common_dir(forged) is None

    def test_directory_link_is_rejected_even_when_not_reported_as_symlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        main = tmp_path / "main"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        forged.mkdir()
        git_entry = forged / ".git"
        git_entry.symlink_to(main / ".git", target_is_directory=True)
        original_is_symlink = Path.is_symlink

        def _hide_git_entry_link(path: Path) -> bool:
            if path == git_entry:
                return False
            return original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", _hide_git_entry_link)

        assert find_git_common_dir(forged) is None


class TestReadGitBranchFromFilesystem:
    @patch("pathlib.Path.read_text")
    def test_read_os_error(self, mock_read: MagicMock, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").touch()

        mock_read.side_effect = OSError("Permission denied")
        assert read_git_branch_from_filesystem(tmp_path) is None


class TestReadGitRemoteUrlFromFilesystem:
    def _write_config(self, tmp_path: Path, body: str) -> None:
        git_dir = tmp_path / ".git"
        (git_dir / "objects").mkdir(parents=True, exist_ok=True)
        (git_dir / "refs").mkdir(exist_ok=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (git_dir / "config").write_text(body)

    def test_reads_origin_url_from_linked_worktree(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        _init_git_repo(main)
        _run_git(
            main,
            "remote",
            "add",
            "origin",
            "https://github.com/langchain-ai/deepagents.git",
        )
        _add_git_worktree(main, worktree, "worktree")
        nested = worktree / "src"
        nested.mkdir()

        assert (
            read_git_remote_url_from_filesystem(nested)
            == "https://github.com/langchain-ai/deepagents.git"
        )

    def test_forged_pointer_does_not_expose_remote(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        forged = tmp_path / "forged"
        _init_git_repo(main)
        _run_git(
            main,
            "remote",
            "add",
            "origin",
            "https://github.com/langchain-ai/deepagents.git",
        )
        forged.mkdir()
        (forged / ".git").write_text(f"gitdir: {main / '.git'}\n")

        assert read_git_remote_url_from_filesystem(forged) == ""

    def test_reads_origin_url_from_submodule(self, tmp_path: Path) -> None:
        child = tmp_path / "child"
        parent = tmp_path / "parent"
        _init_git_repo(child)
        _init_git_repo(parent)
        _run_git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(child),
            "child",
        )

        assert read_git_remote_url_from_filesystem(parent / "child") == str(child)

    def test_submodule_pointer_with_mismatched_worktree_is_rejected(
        self, tmp_path: Path
    ) -> None:
        child = tmp_path / "child"
        parent = tmp_path / "parent"
        forged = tmp_path / "forged"
        _init_git_repo(child)
        _init_git_repo(parent)
        _run_git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(child),
            "child",
        )
        forged.mkdir()
        git_dir = _parse_git_dir_pointer(parent / "child" / ".git")
        assert git_dir is not None
        (forged / ".git").write_text(f"gitdir: {git_dir}\n")

        assert read_git_remote_url_from_filesystem(forged) == ""


class TestParseRepositoryMetadata:
    def test_strips_embedded_credentials(self) -> None:
        result = parse_repository_metadata(
            "https://user:token@gitlab.com/group/project.git"
        )
        assert result is not None
        url, provider, name = result
        assert url == "https://gitlab.com/group/project"
        assert provider == "gitlab"
        assert name == "group/project"
