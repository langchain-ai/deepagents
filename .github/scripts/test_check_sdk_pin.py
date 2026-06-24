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
    with pytest.raises(ValueError, match="No `deepagents==<version>` pin"):
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


def test_prerelease_pin_in_sync(tmp_path) -> None:
    """A PEP 440 prerelease pin matching the SDK returns 0.

    The extractor must capture the full `0.7.0a2` token; a strict `X.Y.Z`
    pattern would truncate it to `0.7.0` and report false drift against an
    in-sync prerelease SDK. Guards parity with the release workflow's gate.
    """
    assert main(_write_repo(tmp_path, "0.7.0a2", "0.7.0a2")) == 0


def test_prerelease_pin_drift(tmp_path) -> None:
    """Two distinct prereleases must be compared verbatim, not truncated."""
    assert main(_write_repo(tmp_path, "0.7.0a3", "0.7.0a2")) == 1


def test_two_segment_pin_recognized(tmp_path) -> None:
    """Any `==` token is accepted and compared verbatim (parity with sed).

    The extractor no longer requires three numeric segments — it mirrors the
    `deepagents==([^", <>=;]+)` sed pattern in the workflows, which captures
    whatever follows `==`. A two-segment `0.6` pin is therefore recognized and
    reported as drift against the SDK's `0.6.10` (not silently rejected).
    """
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents==0.6"',
    )
    assert main(repo) == 1
