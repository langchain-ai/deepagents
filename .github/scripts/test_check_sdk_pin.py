"""Tests for check_sdk_pin pin/version comparison."""

from pathlib import Path

import pytest
from check_sdk_pin import main


def _write_repo(tmp_path: Path, sdk_version: str, pin: str) -> Path:
    return _write_repo_raw(
        tmp_path,
        sdk_section=f'[project]\nname = "deepagents"\nversion = "{sdk_version}"\n',
        code_deps=f'"deepagents=={pin}", "deepagents-acp>=0.0.8", "rich>=15"',
    )


def _write_repo_raw(tmp_path: Path, sdk_section: str, code_deps: str) -> Path:
    (tmp_path / "libs" / "deepagents").mkdir(parents=True)
    (tmp_path / "libs" / "code").mkdir(parents=True)
    (tmp_path / "libs" / "deepagents" / "pyproject.toml").write_text(sdk_section)
    (tmp_path / "libs" / "code" / "pyproject.toml").write_text(
        "[project]\n"
        'name = "deepagents-code"\n'
        'version = "0.1.0"\n'
        f"dependencies = [{code_deps}]\n"
    )
    return tmp_path


def test_pin_matches(tmp_path) -> None:
    """Matching pin and SDK version returns 0."""
    assert main(_write_repo(tmp_path, "0.6.10", "0.6.10")) == 0


def test_pin_drift(tmp_path) -> None:
    """A pin that lags the SDK version returns 1."""
    assert main(_write_repo(tmp_path, "0.6.11", "0.6.10")) == 1


def test_acp_dependency_not_mistaken_for_sdk(tmp_path) -> None:
    """`deepagents-acp` must not be parsed as the `deepagents` pin.

    The SDK and the (real) `deepagents` pin agree at 0.6.10 while
    `deepagents-acp` sits at a different version; a matcher that grabbed the
    acp line would read 0.0.8 and report drift.
    """
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents-acp>=0.0.8", "deepagents==0.6.10", "rich>=15"',
    )
    assert main(repo) == 0


def test_missing_pin_raises(tmp_path) -> None:
    """A non-`==` deepagents dependency yields a clear ValueError."""
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents>=0.6.0", "rich>=15"',
    )
    with pytest.raises(ValueError, match="No `deepagents==X.Y.Z` pin"):
        main(repo)


def test_missing_sdk_version_raises(tmp_path) -> None:
    """A SDK pyproject without project.version yields a ValueError."""
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\n',
        code_deps='"deepagents==0.6.10"',
    )
    with pytest.raises(ValueError, match="project.version"):
        main(repo)


def test_non_semver_pin_not_recognized(tmp_path) -> None:
    """A pin without three numeric segments is not treated as a valid pin."""
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents==0.6"',
    )
    with pytest.raises(ValueError, match="No `deepagents==X.Y.Z` pin"):
        main(repo)
