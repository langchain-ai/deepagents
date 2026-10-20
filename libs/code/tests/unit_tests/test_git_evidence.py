"""Real-checkout coverage for backend-local Git work evidence."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from deepagents_code.git_evidence import aprobe_git, changed_snapshots, probe_git


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def repository(root: Path, name: str = "example/repo") -> Path:
    root.mkdir(parents=True)
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.com")
    git(root, "remote", "add", "origin", f"https://secret:token@github.com/{name}.git")
    (root / "file.txt").write_text("initial\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


@pytest.mark.parametrize("sandbox", [False, True])
def test_edits_and_commits_not_preexisting_dirt(tmp_path: Path, sandbox: bool) -> None:
    root = repository(tmp_path / "repo")
    backend = (
        LocalShellBackend(root_dir=root)
        if sandbox
        else FilesystemBackend(root_dir=root, virtual_mode=False)
    )
    (root / "file.txt").write_text("old dirt\n")
    before = probe_git(backend, str(root))
    assert before[0].repository is not None
    assert before[0].repository.url == "https://github.com/example/repo"
    git(root, "status", "--short")
    assert changed_snapshots(before, probe_git(backend, str(root))) == []
    (root / "file.txt").write_text("new dirt\n")
    edited = probe_git(backend, str(root))
    assert changed_snapshots(before, edited) == edited
    git(root, "add", ".")
    git(root, "commit", "-m", "change")
    committed = probe_git(backend, str(root))
    assert changed_snapshots(edited, committed) == committed
    assert committed[0].commit != edited[0].commit


def test_branch_switch_and_worktree(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    before = probe_git(backend, str(root))
    git(root, "switch", "-c", "feature")
    after = probe_git(backend, str(root))
    assert changed_snapshots(before, after)[0].branch == "feature"
    worktree = tmp_path / "worktree"
    git(root, "worktree", "add", "-b", "linked", str(worktree))
    linked = probe_git(backend, str(worktree))
    assert linked[0].branch == "linked"
    assert linked[0].commit == after[0].commit
    assert linked[0].repository == after[0].repository
    assert linked[0].root != after[0].root


@pytest.mark.parametrize("key", ["file_path", "path", "cwd", "command"])
def test_explicit_candidates_outside_default_workspace(
    tmp_path: Path, key: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = repository(tmp_path / "elsewhere" / "repo with spaces")
    backend = FilesystemBackend(root_dir=workspace, virtual_mode=False)
    value = str(root / "new.txt") if key != "command" else f"git -C '{root}' status"
    snapshots = probe_git(backend, str(workspace), {key: value})
    assert [snapshot.root for snapshot in snapshots] == [str(root)]


def test_workspace_multi_repo_and_new_checkout(tmp_path: Path) -> None:
    first = repository(tmp_path / "first", "org/first")
    second = repository(tmp_path / "nested" / "second", "org/second")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    before = probe_git(backend, str(tmp_path))
    assert {snapshot.root for snapshot in before} == {str(first), str(second)}
    (second / "file.txt").write_text("changed\n")
    changed = changed_snapshots(before, probe_git(backend, str(tmp_path)))
    assert [snapshot.root for snapshot in changed] == [str(second)]
    third = repository(tmp_path / "third", "org/third")
    changed = changed_snapshots(before, probe_git(backend, str(tmp_path)))
    assert {snapshot.root for snapshot in changed} == {str(second), str(third)}


async def test_async_backend_environment_and_quoted_candidates(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = repository(tmp_path / "repo'quoted")
    backend = LocalShellBackend(root_dir=workspace)
    snapshots = await aprobe_git(
        backend, str(workspace), {"file_path": str(root / "file.txt")}
    )
    assert [snapshot.root for snapshot in snapshots] == [str(root)]
    assert not (workspace / "file.txt").exists()


def test_virtual_filesystem_mapping_and_routed_paths(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    snapshots = probe_git(backend, "/repo", {"file_path": "/repo/file.txt"})
    assert [snapshot.root for snapshot in snapshots] == [str(root)]
    assert (
        probe_git(backend, "/memories", {"path": "/conversation_history/session"}) == []
    )


@pytest.mark.parametrize(
    "response",
    [
        ExecuteResponse(output="[]", exit_code=1),
        ExecuteResponse(output="[]", exit_code=0, truncated=True),
        ExecuteResponse(output="not json", exit_code=0),
        ExecuteResponse(output="{}", exit_code=0),
        ExecuteResponse(output='[["incomplete"]]', exit_code=0),
    ],
)
def test_unknown_probe_is_not_a_valid_empty_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: ExecuteResponse
) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    monkeypatch.setattr(backend, "execute", Mock(return_value=response))
    with pytest.raises(ValueError, match=r"Git evidence|Expecting value"):
        probe_git(backend, str(tmp_path))
