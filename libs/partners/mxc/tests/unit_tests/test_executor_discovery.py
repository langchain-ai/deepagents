"""Executor discovery order and the not-found error."""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_mxc._executor import (
    ENV_BIN_DIR,
    ENV_EXECUTOR_PATH,
    SDK_PACKAGE,
    MXCExecutorNotFoundError,
    MXCUnsupportedPlatformError,
    candidate_executor_paths,
    executor_name,
    find_executor,
    install_instructions,
    sdk_arch,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_executor_name_per_platform() -> None:
    assert executor_name("win32") == "wxc-exec.exe"
    assert executor_name("darwin") == "mxc-exec-mac"
    assert executor_name("linux") == "lxc-exec"
    with pytest.raises(MXCUnsupportedPlatformError):
        executor_name("freebsd14")


def test_sdk_arch() -> None:
    assert sdk_arch("arm64") == "arm64"
    assert sdk_arch("aarch64") == "arm64"
    assert sdk_arch("x86_64") == "x64"
    assert sdk_arch("AMD64") == "x64"


def test_explicit_path_wins(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "custom" / "mxc-exec-mac")
    assert find_executor(exe, platform="darwin", machine="arm64", environ={}, cwd=tmp_path, home=tmp_path, use_npm=False) == exe


def test_explicit_path_missing_raises_with_instructions(tmp_path: Path) -> None:
    with pytest.raises(MXCExecutorNotFoundError) as excinfo:
        find_executor(tmp_path / "nope", platform="darwin", machine="arm64", environ={}, cwd=tmp_path, home=tmp_path, use_npm=False)
    assert "npm install @microsoft/mxc-sdk" in str(excinfo.value)
    assert "nope" in str(excinfo.value)


def test_env_var_is_checked_before_node_modules(tmp_path: Path) -> None:
    env_exe = _touch(tmp_path / "env" / "mxc-exec-mac")
    _touch(tmp_path / "node_modules" / SDK_PACKAGE / "bin" / "arm64" / "mxc-exec-mac")
    found = find_executor(
        None, platform="darwin", machine="arm64", environ={ENV_EXECUTOR_PATH: str(env_exe)}, cwd=tmp_path, home=tmp_path, use_npm=False
    )
    assert found == env_exe


def test_bin_dir_env_var(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "bin" / "x64" / "lxc-exec")
    found = find_executor(
        None, platform="linux", machine="x86_64", environ={ENV_BIN_DIR: str(tmp_path / "bin")}, cwd=tmp_path, home=tmp_path, use_npm=False
    )
    assert found == exe


def test_local_node_modules_in_cwd_and_parents(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "node_modules" / SDK_PACKAGE / "bin" / "arm64" / "mxc-exec-mac")
    nested = tmp_path / "apps" / "agent"
    nested.mkdir(parents=True)
    found = find_executor(None, platform="darwin", machine="arm64", environ={}, cwd=nested, home=tmp_path / "home", use_npm=False)
    assert found == exe


def test_global_npm_prefix_under_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    exe = _touch(home / ".npm-global" / "lib" / "node_modules" / SDK_PACKAGE / "bin" / "x64" / "lxc-exec")
    found = find_executor(None, platform="linux", machine="x86_64", environ={}, cwd=tmp_path / "proj", home=home, use_npm=False)
    assert found == exe


def test_nvm_versions_are_searched(tmp_path: Path) -> None:
    home = tmp_path / "home"
    exe = _touch(home / ".nvm" / "versions" / "node" / "v24.1.0" / "lib" / "node_modules" / SDK_PACKAGE / "bin" / "arm64" / "mxc-exec-mac")
    found = find_executor(None, platform="darwin", machine="arm64", environ={}, cwd=tmp_path / "proj", home=home, use_npm=False)
    assert found == exe


def test_windows_global_locations(tmp_path: Path) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    exe = _touch(appdata / "npm" / "node_modules" / SDK_PACKAGE / "bin" / "x64" / "wxc-exec.exe")
    found = find_executor(
        None, platform="win32", machine="AMD64", environ={"APPDATA": str(appdata)}, cwd=tmp_path / "proj", home=tmp_path, use_npm=False
    )
    assert found == exe


def test_search_order(tmp_path: Path) -> None:
    candidates = candidate_executor_paths(
        platform="darwin",
        machine="arm64",
        environ={ENV_EXECUTOR_PATH: "/x/exe", ENV_BIN_DIR: "/y/bin"},
        cwd=tmp_path / "a" / "b",
        home=tmp_path / "home",
    )
    assert candidates[0] == Path("/x/exe")
    assert candidates[1] == Path("/y/bin/arm64/mxc-exec-mac")
    assert candidates[2] == tmp_path / "a" / "b" / "node_modules" / SDK_PACKAGE / "bin" / "arm64" / "mxc-exec-mac"
    assert candidates[3] == tmp_path / "a" / "node_modules" / SDK_PACKAGE / "bin" / "arm64" / "mxc-exec-mac"
    assert Path("/opt/homebrew/lib/node_modules") / SDK_PACKAGE / "bin" / "arm64" / "mxc-exec-mac" in candidates


def test_not_found_lists_searched_locations(tmp_path: Path) -> None:
    with pytest.raises(MXCExecutorNotFoundError) as excinfo:
        find_executor(None, platform="darwin", machine="arm64", environ={}, cwd=tmp_path, home=tmp_path / "home", use_npm=False)
    message = str(excinfo.value)
    assert "Could not find mxc-exec-mac" in message
    assert "npm install @microsoft/mxc-sdk" in message
    assert "MXC_EXECUTOR_PATH" in message
    assert "executor_path=" in message
    assert "Searched:" in message
    assert str(tmp_path / "node_modules") in message


def test_install_instructions_mention_requirements() -> None:
    text = install_instructions("win32", "AMD64")
    assert "wxc-exec.exe" in text
    assert "Windows 11 24H2" in text
    assert "macOS 15+" in text
    assert "bubblewrap" in text
