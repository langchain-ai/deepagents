"""Tests for check_lockfiles_pre_commit (pre-commit Talon path selection)."""

from check_lockfiles_pre_commit import _include_talon


def test_unrelated_paths_skip_talon() -> None:
    """Changes to other packages never force Talon validation."""
    paths = ["libs/deepagents/deepagents/graph.py", "libs/cli/uv.lock"]
    assert _include_talon(paths) is False


def test_talon_source_includes_talon() -> None:
    """A Talon source/config edit forces Talon validation."""
    assert _include_talon(["libs/talon/deepagents_talon/__init__.py"]) is True
    assert _include_talon(["libs/talon/pyproject.toml"]) is True


def test_talon_lockfile_includes_talon() -> None:
    """A direct edit to libs/talon/uv.lock forces Talon validation."""
    assert _include_talon(["libs/talon/uv.lock"]) is True


def test_empty_paths_include_talon() -> None:
    """No paths preserves the full-check behavior (Talon included)."""
    assert _include_talon([]) is True
