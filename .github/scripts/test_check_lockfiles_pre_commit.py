"""Tests for check_lockfiles_pre_commit changed path selection."""

from pathlib import Path
from types import SimpleNamespace

import check_lockfiles_pre_commit
from check_lockfiles_pre_commit import (
    LIBS_ROOT,
    REPO_ROOT,
    _lock_command,
    _lockfile_error,
    _packages_for_paths,
    main,
)


def _paths(packages: list[Path]) -> list[str]:
    return [package.relative_to(REPO_ROOT).as_posix() for package in packages]


def test_unrelated_paths_skip_talon() -> None:
    """Unrelated multi-package changes select only those packages, never Talon."""
    packages = _paths(
        _packages_for_paths(["libs/deepagents/deepagents/graph.py", "libs/cli/uv.lock"])
    )
    assert packages == ["libs/cli", "libs/deepagents"]
    assert "libs/talon" not in packages


def test_talon_source_includes_talon() -> None:
    """A Talon source/config edit selects Talon for validation."""
    assert _paths(_packages_for_paths(["libs/talon/deepagents_talon/__init__.py"])) == [
        "libs/talon"
    ]
    assert _paths(_packages_for_paths(["libs/talon/pyproject.toml"])) == ["libs/talon"]


def test_talon_lockfile_includes_talon() -> None:
    """A direct edit to libs/talon/uv.lock selects Talon."""
    assert _paths(_packages_for_paths(["libs/talon/uv.lock"])) == ["libs/talon"]


def test_empty_paths_check_all_packages() -> None:
    """No paths preserves full-check behavior for manual runs."""
    packages = _paths(_packages_for_paths([]))
    assert "libs/deepagents" in packages
    assert "libs/talon" in packages
    assert "examples/async-subagent-server" in packages


def test_changed_paths_check_only_touched_packages() -> None:
    """Changed paths do not force unrelated lockfile updates."""
    packages = _paths(
        _packages_for_paths(
            [
                "libs/deepagents/deepagents/graph.py",
                "libs/deepagents/uv.lock",
            ]
        )
    )
    assert packages == ["libs/deepagents"]


def test_changed_partner_path_checks_only_that_partner() -> None:
    """Nested partner paths match the owning partner package only."""
    packages = _paths(_packages_for_paths(["libs/partners/daytona/pyproject.toml"]))
    assert packages == ["libs/partners/daytona"]


def test_unowned_paths_skip_lock_check() -> None:
    """Non-package edits should not run repo-wide lock checks in PR mode."""
    assert _packages_for_paths([".github/workflows/check_lockfiles.yml"]) == []


def test_lock_command_uses_package_specific_python_version() -> None:
    """The suggested and checked commands preserve package Python requirements."""
    assert _lock_command(LIBS_ROOT / "evals", check=True) == [
        "uv",
        "lock",
        "--check",
        "--directory",
        "libs/evals",
        "--python",
        "3.12",
    ]
    assert _lock_command(LIBS_ROOT / "acp", check=False) == [
        "uv",
        "lock",
        "--directory",
        "libs/acp",
        "--python",
        "3.14",
    ]


def test_lockfile_error_names_package_and_fix_command() -> None:
    """Failure output points at the stale lockfile and the exact relock command."""
    assert _lockfile_error(LIBS_ROOT / "evals") == (
        "::error file=libs/evals/uv.lock,title=Out-of-date uv.lock::"
        "libs/evals/uv.lock is out of sync with libs/evals/pyproject.toml. "
        "From the repository root, run `uv lock --directory libs/evals "
        "--python 3.12` and commit the updated lockfile."
    )


def test_main_prints_actionable_error_on_lock_failure(monkeypatch, capsys) -> None:
    """A stale lockfile failure includes the package path and exact fix command."""
    package = LIBS_ROOT / "evals"
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool, cwd: Path) -> SimpleNamespace:
        commands.append(command)
        assert check is False
        assert cwd == REPO_ROOT
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(check_lockfiles_pre_commit, "_package_dirs", lambda: [package])
    monkeypatch.setattr(check_lockfiles_pre_commit.subprocess, "run", fake_run)

    assert main(["libs/evals/pyproject.toml"]) == 1
    captured = capsys.readouterr()

    assert commands == [
        [
            "uv",
            "lock",
            "--check",
            "--directory",
            "libs/evals",
            "--python",
            "3.12",
        ]
    ]
    assert "🔍 Checking evals" in captured.out
    assert "libs/evals/uv.lock is out of sync" in captured.err
    assert "uv lock --directory libs/evals --python 3.12" in captured.err
