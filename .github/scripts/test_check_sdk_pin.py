"""Tests for check_sdk_pin pin/version comparison."""

from pathlib import Path

from check_sdk_pin import main


def _write_repo(tmp_path: Path, sdk_version: str, pin: str) -> Path:
    (tmp_path / "libs" / "deepagents").mkdir(parents=True)
    (tmp_path / "libs" / "code").mkdir(parents=True)
    (tmp_path / "libs" / "deepagents" / "pyproject.toml").write_text(
        f'[project]\nname = "deepagents"\nversion = "{sdk_version}"\n'
    )
    (tmp_path / "libs" / "code" / "pyproject.toml").write_text(
        "[project]\n"
        'name = "deepagents-code"\n'
        'version = "0.1.0"\n'
        f'dependencies = ["deepagents=={pin}", "deepagents-acp>=0.0.8", "rich>=15"]\n'
    )
    return tmp_path


def test_pin_matches(tmp_path) -> None:
    """Matching pin and SDK version returns 0."""
    assert main(_write_repo(tmp_path, "0.6.10", "0.6.10")) == 0


def test_pin_drift(tmp_path) -> None:
    """A pin that lags the SDK version returns 1."""
    assert main(_write_repo(tmp_path, "0.6.11", "0.6.10")) == 1


def test_acp_dependency_not_mistaken_for_sdk(tmp_path) -> None:
    """`deepagents-acp` must not be parsed as the `deepagents` pin."""
    assert main(_write_repo(tmp_path, "0.6.10", "0.6.10")) == 0
