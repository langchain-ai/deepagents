"""Unit tests for the deepagents_cli._git module."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deepagents_cli._git import (
    _abbreviate_git_ref,
    _normalize_lookup_path,
    _parse_git_dir_pointer,
    find_git_dir,
    find_git_root,
    read_git_branch_from_filesystem,
    read_git_branch_via_subprocess,
    resolve_git_branch,
)


class TestAbbreviateGitRef:
    def test_abbreviate_heads(self) -> None:
        assert _abbreviate_git_ref("refs/heads/main") == "main"

    def test_abbreviate_remotes(self) -> None:
        assert _abbreviate_git_ref("refs/remotes/origin/main") == "origin/main"

    def test_abbreviate_tags(self) -> None:
        assert _abbreviate_git_ref("refs/tags/v1.0") == "v1.0"

    def test_abbreviate_other_refs(self) -> None:
        assert _abbreviate_git_ref("refs/stash") == "stash"

    def test_abbreviate_no_prefix(self) -> None:
        assert _abbreviate_git_ref("main") == "main"


class TestParseGitDirPointer:
    def test_parse_valid_pointer(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "actual_git_dir"
        target_dir.mkdir()
        git_file = tmp_path / ".git"
        git_file.write_text(f"gitdir: {target_dir}\n")
        assert _parse_git_dir_pointer(git_file) == target_dir

    def test_parse_relative_pointer(self, tmp_path: Path) -> None:
        git_file = tmp_path / ".git"
        git_file.write_text("gitdir: ../some/path\n")
        expected = (tmp_path / "../some/path").resolve(strict=False)
        assert _parse_git_dir_pointer(git_file) == expected

    def test_parse_invalid_prefix(self, tmp_path: Path) -> None:
        git_file = tmp_path / ".git"
        git_file.write_text("notagitdir: /some/path")
        assert _parse_git_dir_pointer(git_file) is None

    def test_parse_empty_pointer(self, tmp_path: Path) -> None:
        git_file = tmp_path / ".git"
        git_file.write_text("gitdir:    \n")
        assert _parse_git_dir_pointer(git_file) is None

    def test_parse_os_error(self, tmp_path: Path) -> None:
        git_file = tmp_path / ".git"
        # file does not exist
        assert _parse_git_dir_pointer(git_file) is None


class TestNormalizeLookupPath:
    def test_normalize_valid_path(self, tmp_path: Path) -> None:
        assert _normalize_lookup_path(tmp_path) == tmp_path.resolve()

    @patch("pathlib.Path.resolve")
    def test_normalize_os_error_fallback(self, mock_resolve, tmp_path: Path) -> None:
        mock_resolve.side_effect = OSError("Permission denied")
        assert _normalize_lookup_path(tmp_path) == tmp_path


@pytest.fixture(autouse=True)
def clear_git_dir_cache() -> None:
    from deepagents_cli._git import _git_dir_cache

    _git_dir_cache.clear()


class TestFindGitDirAndRoot:
    def test_find_standard_repo(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        git_dir = repo_root / ".git"
        git_dir.mkdir()

        subdir = repo_root / "src" / "subdir"
        subdir.mkdir(parents=True)

        assert find_git_dir(subdir) == git_dir
        assert find_git_root(subdir) == repo_root

        # Test caching
        assert find_git_dir(subdir) == git_dir

    def test_find_worktree_repo(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        actual_git_dir = tmp_path / "actual_git_dir"
        actual_git_dir.mkdir()

        git_file = repo_root / ".git"
        git_file.write_text(f"gitdir: {actual_git_dir}")

        subdir = repo_root / "src"
        subdir.mkdir()

        assert find_git_dir(subdir) == actual_git_dir
        assert find_git_root(subdir) == repo_root

    def test_find_invalid_worktree(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        git_file = repo_root / ".git"
        git_file.write_text("invalid_content")

        subdir = repo_root / "src"
        subdir.mkdir()
        
        # Test passing a file instead of a dir to trigger `not current.is_dir()`
        some_file = subdir / "file.txt"
        some_file.touch()

        assert find_git_dir(some_file) is None
        assert find_git_root(some_file) is None

    def test_find_no_repo(self, tmp_path: Path) -> None:
        assert find_git_dir(tmp_path) is None
        assert find_git_root(tmp_path) is None


class TestReadGitBranchFromFilesystem:
    def test_read_named_branch(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        head_file = git_dir / "HEAD"
        head_file.write_text("ref: refs/heads/feature-branch\n")

        assert read_git_branch_from_filesystem(tmp_path) == "feature-branch"

    def test_read_detached_head(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        head_file = git_dir / "HEAD"
        head_file.write_text("a1b2c3d4e5f6\n")

        assert read_git_branch_from_filesystem(tmp_path) == "HEAD"

    def test_read_empty_head(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        head_file = git_dir / "HEAD"
        head_file.write_text("")

        assert read_git_branch_from_filesystem(tmp_path) == ""

    def test_read_missing_head(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        assert read_git_branch_from_filesystem(tmp_path) is None

    @patch("pathlib.Path.read_text")
    def test_read_os_error(self, mock_read, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").touch()

        mock_read.side_effect = OSError("Permission denied")
        assert read_git_branch_from_filesystem(tmp_path) is None

    def test_read_not_in_repo(self, tmp_path: Path) -> None:
        assert read_git_branch_from_filesystem(tmp_path) == ""


class TestReadGitBranchViaSubprocess:
    @patch("subprocess.run")
    def test_read_success(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        assert read_git_branch_via_subprocess("/some/path") == "main"

    @patch("subprocess.run")
    def test_read_timeout(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=2)
        assert read_git_branch_via_subprocess("/some/path") == ""

    @patch("subprocess.run")
    def test_read_file_not_found(self, mock_run) -> None:
        mock_run.side_effect = FileNotFoundError()
        assert read_git_branch_via_subprocess("/some/path") == ""

    @patch("subprocess.run")
    def test_read_os_error(self, mock_run) -> None:
        mock_run.side_effect = OSError("error")
        assert read_git_branch_via_subprocess("/some/path") == ""

    @patch("subprocess.run")
    def test_read_failure_code(self, mock_run) -> None:
        mock_run.return_value.returncode = 128
        assert read_git_branch_via_subprocess("/some/path") == ""


class TestResolveGitBranch:
    @patch("deepagents_cli._git.read_git_branch_from_filesystem")
    @patch("deepagents_cli._git.read_git_branch_via_subprocess")
    def test_resolve_from_fs(self, mock_sub, mock_fs) -> None:
        mock_fs.return_value = "main"
        assert resolve_git_branch("/some/path") == "main"
        mock_sub.assert_not_called()

    @patch("deepagents_cli._git.read_git_branch_from_filesystem")
    @patch("deepagents_cli._git.read_git_branch_via_subprocess")
    def test_resolve_fallback(self, mock_sub, mock_fs) -> None:
        mock_fs.return_value = None
        mock_sub.return_value = "fallback-branch"
        assert resolve_git_branch("/some/path") == "fallback-branch"
        mock_sub.assert_called_once()
