"""Unit tests for the deepagents_code._git module."""

import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code._git import (
    _abbreviate_git_ref,
    _git_dir_cache,
    _normalize_lookup_path,
    _parse_git_dir_pointer,
    find_git_dir,
    find_git_root,
    parse_repository_metadata,
    read_git_branch_from_filesystem,
    read_git_branch_via_subprocess,
    read_git_commit_sha_from_filesystem,
    read_git_remote_url_from_filesystem,
    resolve_git_branch,
    resolve_git_commit_sha,
    resolve_git_remote_url,
)

_FULL_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
"""A valid 40-char SHA-1 used across the commit-resolution tests."""


@pytest.fixture(autouse=True)
def clear_git_dir_cache() -> Iterator[None]:
    _git_dir_cache.clear()
    yield
    _git_dir_cache.clear()


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

    def test_parse_prefix_without_pointer(self, tmp_path: Path) -> None:
        # File content whose stripped form is just the prefix without a space
        # fails the prefix check at line 54 (not the empty-pointer guard).
        git_file = tmp_path / ".git"
        git_file.write_text("gitdir:    \n")
        assert _parse_git_dir_pointer(git_file) is None

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        git_file = tmp_path / ".git"
        assert _parse_git_dir_pointer(git_file) is None


class TestNormalizeLookupPath:
    def test_normalize_valid_path(self, tmp_path: Path) -> None:
        assert _normalize_lookup_path(tmp_path) == tmp_path.resolve()

    @patch("pathlib.Path.resolve")
    def test_normalize_os_error_fallback(
        self, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        mock_resolve.side_effect = OSError("Permission denied")
        assert _normalize_lookup_path(tmp_path) == tmp_path


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

        # Second call exercises the positive cache hit
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

    def test_find_invalid_gitdir_pointer(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        git_file = repo_root / ".git"
        git_file.write_text("invalid_content")

        assert find_git_dir(repo_root) is None
        assert find_git_root(repo_root) is None

    def test_find_gitdir_pointer_target_missing(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        git_file = repo_root / ".git"
        git_file.write_text(f"gitdir: {tmp_path / 'does_not_exist'}\n")

        assert find_git_dir(repo_root) is None
        assert find_git_root(repo_root) is None

    def test_find_with_file_path_input(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        git_dir = repo_root / ".git"
        git_dir.mkdir()

        src = repo_root / "src"
        src.mkdir()
        module_file = src / "module.py"
        module_file.touch()

        assert find_git_dir(module_file) == git_dir
        assert find_git_root(module_file) == repo_root

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

    def test_read_tag_ref(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        head_file = git_dir / "HEAD"
        head_file.write_text("ref: refs/tags/v1.0\n")

        assert read_git_branch_from_filesystem(tmp_path) == "v1.0"

    def test_read_missing_head(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        assert read_git_branch_from_filesystem(tmp_path) is None

    @patch("pathlib.Path.read_text")
    def test_read_os_error(self, mock_read: MagicMock, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").touch()

        mock_read.side_effect = OSError("Permission denied")
        assert read_git_branch_from_filesystem(tmp_path) is None

    def test_read_not_in_repo(self, tmp_path: Path) -> None:
        assert read_git_branch_from_filesystem(tmp_path) == ""


class TestReadGitBranchViaSubprocess:
    @patch("subprocess.run")
    def test_read_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        assert read_git_branch_via_subprocess("/some/path") == "main"

    @patch("subprocess.run")
    def test_read_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=2)
        assert read_git_branch_via_subprocess("/some/path") == ""

    @patch("subprocess.run")
    def test_read_file_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        assert read_git_branch_via_subprocess("/some/path") == ""

    @patch("subprocess.run")
    def test_read_os_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("error")
        assert read_git_branch_via_subprocess("/some/path") == ""

    @patch("subprocess.run")
    def test_read_failure_code(self, mock_run: MagicMock) -> None:
        mock_run.return_value.returncode = 128
        assert read_git_branch_via_subprocess("/some/path") == ""


class TestResolveGitBranch:
    @patch("deepagents_code._git.read_git_branch_from_filesystem")
    @patch("deepagents_code._git.read_git_branch_via_subprocess")
    def test_resolve_from_fs(self, mock_sub: MagicMock, mock_fs: MagicMock) -> None:
        mock_fs.return_value = "main"
        assert resolve_git_branch("/some/path") == "main"
        mock_sub.assert_not_called()

    @patch("deepagents_code._git.read_git_branch_from_filesystem")
    @patch("deepagents_code._git.read_git_branch_via_subprocess")
    def test_resolve_empty_string_skips_subprocess(
        self, mock_sub: MagicMock, mock_fs: MagicMock
    ) -> None:
        # Empty string is the not-in-repo signal and must short-circuit the
        # subprocess fallback (branch is not None).
        mock_fs.return_value = ""
        assert resolve_git_branch("/some/path") == ""
        mock_sub.assert_not_called()

    @patch("deepagents_code._git.read_git_branch_from_filesystem")
    @patch("deepagents_code._git.read_git_branch_via_subprocess")
    def test_resolve_fallback(self, mock_sub: MagicMock, mock_fs: MagicMock) -> None:
        mock_fs.return_value = None
        mock_sub.return_value = "fallback-branch"
        assert resolve_git_branch("/some/path") == "fallback-branch"
        mock_sub.assert_called_once()


class TestReadGitCommitShaFromFilesystem:
    def test_loose_ref(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        (git_dir / "refs" / "heads").mkdir(parents=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (git_dir / "refs" / "heads" / "main").write_text(f"{_FULL_SHA}\n")
        assert read_git_commit_sha_from_filesystem(tmp_path) == _FULL_SHA

    def test_detached_head(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text(f"{_FULL_SHA}\n")
        assert read_git_commit_sha_from_filesystem(tmp_path) == _FULL_SHA

    def test_packed_ref(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (git_dir / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\n"
            f"{_FULL_SHA} refs/heads/main\n"
        )
        assert read_git_commit_sha_from_filesystem(tmp_path) == _FULL_SHA

    def test_not_in_repo_returns_empty(self, tmp_path: Path) -> None:
        # Empty string (not None) is the not-in-repo signal so resolve() can
        # short-circuit the subprocess fallback.
        assert read_git_commit_sha_from_filesystem(tmp_path) == ""

    def test_unresolvable_ref_returns_none(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/missing\n")
        assert read_git_commit_sha_from_filesystem(tmp_path) is None


class TestResolveGitCommitSha:
    @patch("deepagents_code._git.read_git_commit_sha_via_subprocess")
    @patch("deepagents_code._git.read_git_commit_sha_from_filesystem")
    def test_filesystem_wins(self, mock_fs: MagicMock, mock_sub: MagicMock) -> None:
        mock_fs.return_value = _FULL_SHA
        assert resolve_git_commit_sha("/some/path") == _FULL_SHA
        mock_sub.assert_not_called()

    @patch("deepagents_code._git.read_git_commit_sha_via_subprocess")
    @patch("deepagents_code._git.read_git_commit_sha_from_filesystem")
    def test_empty_string_skips_subprocess(
        self, mock_fs: MagicMock, mock_sub: MagicMock
    ) -> None:
        mock_fs.return_value = ""
        assert resolve_git_commit_sha("/some/path") == ""
        mock_sub.assert_not_called()

    @patch("deepagents_code._git.read_git_commit_sha_via_subprocess")
    @patch("deepagents_code._git.read_git_commit_sha_from_filesystem")
    def test_fallback_on_none(self, mock_fs: MagicMock, mock_sub: MagicMock) -> None:
        mock_fs.return_value = None
        mock_sub.return_value = _FULL_SHA
        assert resolve_git_commit_sha("/some/path") == _FULL_SHA
        mock_sub.assert_called_once()


class TestReadGitRemoteUrlFromFilesystem:
    def _write_config(self, tmp_path: Path, body: str) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir(exist_ok=True)
        (git_dir / "config").write_text(body)

    def test_reads_origin_url(self, tmp_path: Path) -> None:
        self._write_config(
            tmp_path,
            '[remote "origin"]\n'
            "\turl = https://github.com/langchain-ai/deepagents.git\n"
            "\tfetch = +refs/heads/*:refs/remotes/origin/*\n",
        )
        assert (
            read_git_remote_url_from_filesystem(tmp_path)
            == "https://github.com/langchain-ai/deepagents.git"
        )

    def test_ignores_other_remotes(self, tmp_path: Path) -> None:
        self._write_config(
            tmp_path,
            '[remote "upstream"]\n\turl = https://github.com/other/fork.git\n'
            '[remote "origin"]\n\turl = git@github.com:langchain-ai/deepagents.git\n',
        )
        assert (
            read_git_remote_url_from_filesystem(tmp_path)
            == "git@github.com:langchain-ai/deepagents.git"
        )

    def test_no_origin_returns_none(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, "[core]\n\tbare = false\n")
        assert read_git_remote_url_from_filesystem(tmp_path) is None

    def test_not_in_repo_returns_empty(self, tmp_path: Path) -> None:
        assert read_git_remote_url_from_filesystem(tmp_path) == ""


class TestResolveGitRemoteUrl:
    @patch("deepagents_code._git.read_git_remote_url_via_subprocess")
    @patch("deepagents_code._git.read_git_remote_url_from_filesystem")
    def test_empty_string_skips_subprocess(
        self, mock_fs: MagicMock, mock_sub: MagicMock
    ) -> None:
        mock_fs.return_value = ""
        assert resolve_git_remote_url("/some/path") == ""
        mock_sub.assert_not_called()

    @patch("deepagents_code._git.read_git_remote_url_via_subprocess")
    @patch("deepagents_code._git.read_git_remote_url_from_filesystem")
    def test_fallback_on_none(self, mock_fs: MagicMock, mock_sub: MagicMock) -> None:
        mock_fs.return_value = None
        mock_sub.return_value = "https://github.com/org/repo.git"
        assert resolve_git_remote_url("/p") == "https://github.com/org/repo.git"
        mock_sub.assert_called_once()


class TestParseRepositoryMetadata:
    def test_https_github(self) -> None:
        assert parse_repository_metadata(
            "https://github.com/langchain-ai/deepagents.git"
        ) == (
            "https://github.com/langchain-ai/deepagents",
            "github",
            "langchain-ai/deepagents",
        )

    def test_scp_style_ssh(self) -> None:
        assert parse_repository_metadata(
            "git@github.com:langchain-ai/deepagents.git"
        ) == (
            "https://github.com/langchain-ai/deepagents",
            "github",
            "langchain-ai/deepagents",
        )

    def test_strips_embedded_credentials(self) -> None:
        result = parse_repository_metadata(
            "https://user:token@gitlab.com/group/project.git"
        )
        assert result is not None
        url, provider, name = result
        assert url == "https://gitlab.com/group/project"
        assert provider == "gitlab"
        assert name == "group/project"

    def test_bitbucket_provider(self) -> None:
        result = parse_repository_metadata("https://bitbucket.org/team/repo.git")
        assert result is not None
        assert result[1] == "bitbucket"

    def test_unknown_host_is_other(self) -> None:
        result = parse_repository_metadata("git@git.example.com:internal/tool.git")
        assert result is not None
        _, provider, name = result
        assert provider == "other"
        assert name == "internal/tool"

    def test_nested_group_path(self) -> None:
        result = parse_repository_metadata(
            "https://gitlab.com/group/subgroup/project.git"
        )
        assert result is not None
        assert result[2] == "group/subgroup/project"

    def test_empty_returns_none(self) -> None:
        assert parse_repository_metadata("") is None
        assert parse_repository_metadata("   ") is None

    def test_malformed_returns_none(self) -> None:
        assert parse_repository_metadata("not-a-url") is None
