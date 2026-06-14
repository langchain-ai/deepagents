"""Tests for the shell install script argument construction."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "install.sh"

PRERELEASE_STRATEGIES = (
    "disallow",
    "allow",
    "if-necessary",
    "explicit",
    "if-necessary-or-explicit",
)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_fake_tools(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    tools = tmp_path / "tools"
    bin_dir.mkdir()
    home.mkdir()
    tools.mkdir()

    # Raw f-string: the embedded bash must keep `\n` as the two literal
    # characters (an f-string would otherwise turn `\n` into a newline). `{{ }}`
    # still escape to literal braces; the `{...!r}` slots interpolate paths.
    uv = bin_dir / "uv"
    uv.write_text(
        rf"""#!/usr/bin/env bash
set -euo pipefail
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "dir" ]; then
  printf '%s\n' {str(tools)!r}
  exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "install" ]; then
  printf '%s\n' "$@" > {str(tmp_path / "uv-args.txt")!r}
  exit 0
fi
printf 'unexpected uv args: %s\n' "$*" >&2
exit 1
"""
    )
    _make_executable(uv)

    dcode = bin_dir / "dcode"
    dcode.write_text(
        r"""#!/usr/bin/env bash
if [ "${1:-}" = "-v" ]; then
  printf 'deepagents-code 0.0.1\n'
  exit 0
fi
exit 0
"""
    )
    _make_executable(dcode)
    return bin_dir, home, uv


def _invoke(
    tmp_path: Path, extra_env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `install.sh` with fake `uv`/`dcode` on `PATH`.

    Returns the completed process (never raising on non-zero exit) and the path
    where the fake `uv` records its `tool install` argv — which only exists if
    `uv tool install` was actually invoked.
    """
    bin_dir, home, uv = _write_fake_tools(tmp_path)
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
        **extra_env,
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc, tmp_path / "uv-args.txt"


def _run_install_script(tmp_path: Path, extra_env: dict[str, str]) -> list[str]:
    """Run the script expecting success and return the argv passed to uv."""
    proc, args_path = _invoke(tmp_path, extra_env)
    if proc.returncode != 0:
        msg = f"install.sh exited {proc.returncode}\nstderr:\n{proc.stderr}"
        raise AssertionError(msg)
    return args_path.read_text().splitlines()


def test_install_script_default_invocation_installs_plain_package(
    tmp_path: Path,
) -> None:
    """With no version/prerelease/extras, uv gets the bare package name.

    Guards the most common `curl ... | bash` path against accidentally
    appending a version pin, extras, or a `--prerelease` flag.
    """
    args = _run_install_script(tmp_path, {})

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code"
    assert "--prerelease" not in args


def test_install_script_supports_exact_version_with_extras(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_VERSION` pins the requirement, after the extras."""
    args = _run_install_script(
        tmp_path,
        {
            "DEEPAGENTS_CODE_VERSION": "0.1.0rc1",
            "DEEPAGENTS_CODE_EXTRAS": "nvidia,ollama",
        },
    )

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code[nvidia,ollama]==0.1.0rc1"
    assert "--prerelease" not in args


def test_install_script_supports_exact_version_without_extras(tmp_path: Path) -> None:
    """The version spec appends directly to the package name when no extras."""
    args = _run_install_script(tmp_path, {"DEEPAGENTS_CODE_VERSION": "0.1.0rc1"})

    assert args[-1] == "deepagents-code==0.1.0rc1"


@pytest.mark.parametrize("strategy", PRERELEASE_STRATEGIES)
def test_install_script_forwards_each_prerelease_strategy(
    tmp_path: Path, strategy: str
) -> None:
    """`DEEPAGENTS_CODE_PRERELEASE` forwards each valid strategy verbatim to uv."""
    args = _run_install_script(tmp_path, {"DEEPAGENTS_CODE_PRERELEASE": strategy})

    # The flag is forwarded immediately before the (unpinned) package name.
    assert args[-3:] == ["--prerelease", strategy, "deepagents-code"]


@pytest.mark.parametrize(
    "bad_version",
    [
        "0.1.0; rm -rf /",  # shell metacharacters
        "1.0 --force",  # whitespace + smuggled flag
        ">=1.0",  # range operator, not an exact pin
        "-U",  # leading dash reads as an option
    ],
)
def test_install_script_rejects_invalid_version(
    tmp_path: Path, bad_version: str
) -> None:
    """An invalid version fails before uv runs, so nothing is installed."""
    proc, args_path = _invoke(tmp_path, {"DEEPAGENTS_CODE_VERSION": bad_version})

    assert proc.returncode != 0
    assert not args_path.exists()  # uv tool install was never invoked
    assert "DEEPAGENTS_CODE_VERSION" in proc.stderr


def test_install_script_rejects_invalid_prerelease(tmp_path: Path) -> None:
    """An unknown pre-release strategy fails before uv runs."""
    proc, args_path = _invoke(tmp_path, {"DEEPAGENTS_CODE_PRERELEASE": "maybe"})

    assert proc.returncode != 0
    assert not args_path.exists()
    assert "DEEPAGENTS_CODE_PRERELEASE" in proc.stderr


def test_install_script_rejects_version_and_prerelease_together(
    tmp_path: Path,
) -> None:
    """Pinning a version and a pre-release strategy at once is rejected."""
    proc, args_path = _invoke(
        tmp_path,
        {
            "DEEPAGENTS_CODE_VERSION": "0.1.0rc1",
            "DEEPAGENTS_CODE_PRERELEASE": "allow",
        },
    )

    assert proc.returncode != 0
    assert not args_path.exists()
    assert "mutually exclusive" in proc.stderr
