"""Tests for check_lockfiles_pre_commit changed path selection."""

from pathlib import Path

from check_lockfiles_pre_commit import REPO_ROOT, _packages_for_paths


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
