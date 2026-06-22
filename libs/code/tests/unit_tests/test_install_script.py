"""Tests for the shell install script argument construction."""

from __future__ import annotations

import os
import pty
import re
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


def _write_fake_tools(
    tmp_path: Path,
    *,
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
    curl_fails: bool = False,
) -> tuple[Path, Path, Path]:
    """Stage fake `uv`, `curl`, and (optionally) `dcode` binaries on `PATH`.

    `installed_version` controls whether `dcode -v` reports an existing install
    (`None` simulates a fresh machine). `latest_version` is the version the
    fake `curl` reports from PyPI; `curl_fails` makes that probe error out so
    the script's offline fallback can be exercised.
    """
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
  if [ -n "${{FAKE_UV_INSTALL_STDERR:-}}" ]; then
    printf '%s\n' "$FAKE_UV_INSTALL_STDERR" >&2
  fi
  exit 0
fi
printf 'unexpected uv args: %s\n' "$*" >&2
exit 1
"""
    )
    _make_executable(uv)

    # Shadow the real `curl` so the latest-version probe never hits the network.
    curl = bin_dir / "curl"
    if curl_fails or latest_version is None:
        curl.write_text("#!/usr/bin/env bash\nexit 1\n")
    else:
        payload = f'{{"info":{{"version":"{latest_version}"}}}}'
        curl.write_text(f"#!/usr/bin/env bash\nprintf '%s' '{payload}'\n")
    _make_executable(curl)

    if installed_version is not None:
        dcode = bin_dir / "dcode"
        dcode.write_text(
            f"""#!/usr/bin/env bash
if [ "${{1:-}}" = "-v" ]; then
  printf 'deepagents-code {installed_version}\\n'
  exit 0
fi
exit 0
"""
        )
        _make_executable(dcode)
    return bin_dir, home, uv


def _env(
    tmp_path: Path,
    extra_env: dict[str, str],
    *,
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
    curl_fails: bool = False,
) -> dict[str, str]:
    bin_dir, home, uv = _write_fake_tools(
        tmp_path,
        installed_version=installed_version,
        latest_version=latest_version,
        curl_fails=curl_fails,
    )
    return {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
        **extra_env,
    }


def _invoke(
    tmp_path: Path,
    extra_env: dict[str, str],
    *,
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
    curl_fails: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `install.sh` non-interactively with the fake tools on `PATH`.

    `start_new_session` detaches the controlling terminal so `/dev/tty` is
    unopenable — the deterministic "no TTY to prompt" path. Returns the
    completed process (never raising) and the path where the fake `uv` records
    its `tool install` argv, which only exists if uv was actually invoked.
    """
    env = _env(
        tmp_path,
        extra_env,
        installed_version=installed_version,
        latest_version=latest_version,
        curl_fails=curl_fails,
    )
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc, tmp_path / "uv-args.txt"


def _invoke_interactive(
    tmp_path: Path,
    extra_env: dict[str, str],
    *,
    answer: str,
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
) -> tuple[int, str, Path]:
    """Run `install.sh` with a pty stdin and feed `answer` to its prompt.

    A pty makes `[ -t 0 ]` true, so the script treats the run as interactive and
    reads the y/n answer from stdin. Returns the exit code, combined output
    (ANSI stripped), and the uv-argv path.
    """
    env = _env(
        tmp_path,
        extra_env,
        installed_version=installed_version,
        latest_version=latest_version,
    )
    primary, secondary = pty.openpty()
    proc = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env,
        stdin=secondary,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.close(secondary)
    os.write(primary, f"{answer}\n".encode())
    output = proc.stdout.read() if proc.stdout else ""
    proc.wait(timeout=30)
    os.close(primary)
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return proc.returncode, clean, tmp_path / "uv-args.txt"


def _extract_shell_function(name: str) -> str:
    """Return the source text of a top-level `name() { ... }` block from the script.

    Pulls the real implementation out of `install.sh` so helper-function tests
    exercise the shipped code rather than a copy. Assumes the closing brace sits
    at column 0 (the script's style), matching the first such block.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{.*?^\}}", text, re.MULTILINE | re.DOTALL
    )
    if match is None:
        msg = f"could not locate shell function {name!r} in {SCRIPT}"
        raise AssertionError(msg)
    return match.group(0)


def _eval_can_prompt(
    tmp_path: Path, *, is_interactive: bool, stdin_is_tty: bool
) -> bool:
    """Run the real `can_prompt` from `install.sh` in isolation.

    Writes the extracted function to a temp script (macOS ships bash 3.2, where
    `source <(...)` does not define the function), then reports its exit status
    under a controlled `IS_INTERACTIVE` and stdin. With `stdin_is_tty=False` the
    child is detached from any controlling terminal (`start_new_session`, stdin
    from `/dev/null`), so the `/dev/tty` open fails — the case that distinguishes
    the real open-probe from merely trusting `IS_INTERACTIVE`.
    """
    script = tmp_path / "can_prompt_harness.sh"
    script.write_text(
        f"{_extract_shell_function('can_prompt')}\n"
        f"IS_INTERACTIVE={'true' if is_interactive else 'false'}\n"
        "can_prompt\n",
        encoding="utf-8",
    )
    if stdin_is_tty:
        primary, secondary = pty.openpty()
        proc = subprocess.run(
            ["bash", str(script)],
            stdin=secondary,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        os.close(secondary)
        os.close(primary)
        return proc.returncode == 0
    proc = subprocess.run(
        ["bash", str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        start_new_session=True,
    )
    return proc.returncode == 0


def _run_install_script(
    tmp_path: Path,
    extra_env: dict[str, str],
    *,
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
    curl_fails: bool = False,
) -> list[str]:
    """Run the script expecting success and return the argv passed to uv."""
    proc, args_path = _invoke(
        tmp_path,
        extra_env,
        installed_version=installed_version,
        latest_version=latest_version,
        curl_fails=curl_fails,
    )
    if proc.returncode != 0:
        msg = f"install.sh exited {proc.returncode}\nstderr:\n{proc.stderr}"
        raise AssertionError(msg)
    return args_path.read_text().splitlines()


def test_install_script_default_invocation_installs_plain_package(
    tmp_path: Path,
) -> None:
    """A fresh machine installs the bare package with no prompt.

    Guards the most common `curl ... | bash` path against accidentally
    appending a version pin, extras, or a `--prerelease` flag.
    """
    args = _run_install_script(tmp_path, {}, installed_version=None)

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


def test_install_script_already_up_to_date_skips_uv(tmp_path: Path) -> None:
    """When the installed version matches PyPI's latest, uv is not invoked."""
    proc, args_path = _invoke(
        tmp_path, {}, installed_version="0.1.0", latest_version="0.1.0"
    )

    assert proc.returncode == 0
    assert not args_path.exists()
    assert "already up to date" in proc.stdout


def test_install_script_latest_version_with_extras_installs_requested_extra(
    tmp_path: Path,
) -> None:
    """An extras request still runs uv when the base package is up to date."""
    args = _run_install_script(
        tmp_path,
        {"DEEPAGENTS_CODE_EXTRAS": "ollama"},
        installed_version="0.1.0",
        latest_version="0.1.0",
    )

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code[ollama]"


def test_install_script_latest_version_with_extras_skips_prompt(
    tmp_path: Path,
) -> None:
    """An up-to-date extras request is not gated behind the update prompt."""
    code, output, args_path = _invoke_interactive(
        tmp_path,
        {"DEEPAGENTS_CODE_EXTRAS": "ollama"},
        answer="n",
        installed_version="0.1.0",
        latest_version="0.1.0",
    )

    assert code == 0
    assert "0.1.0 → 0.1.0" not in output
    args = args_path.read_text().splitlines()
    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code[ollama]"


def test_install_script_out_of_date_with_extras_skips_prompt(
    tmp_path: Path,
) -> None:
    """An extras request is explicit intent to reinstall, even across updates."""
    code, output, args_path = _invoke_interactive(
        tmp_path,
        {"DEEPAGENTS_CODE_EXTRAS": "ollama"},
        answer="n",
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert code == 0
    assert "Keeping deepagents-code 0.1.0" not in output
    args = args_path.read_text().splitlines()
    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code[ollama]"


def test_install_script_latest_version_with_python_rebuilds_tool_env(
    tmp_path: Path,
) -> None:
    """An explicit Python request rebuilds even when the package is current."""
    args = _run_install_script(
        tmp_path,
        {"DEEPAGENTS_CODE_PYTHON": "3.12"},
        installed_version="0.1.0",
        latest_version="0.1.0",
    )

    assert args[:5] == ["tool", "install", "-U", "--python", "3.12"]
    assert args[-1] == "deepagents-code"


def test_install_script_out_of_date_auto_updates_without_tty(tmp_path: Path) -> None:
    """Out of date with no TTY to prompt: upgrade automatically (legacy path)."""
    args = _run_install_script(
        tmp_path, {}, installed_version="0.1.0", latest_version="0.2.0"
    )

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code"


def test_install_script_assume_yes_updates_without_prompt(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_YES=1` upgrades an out-of-date install without asking."""
    args = _run_install_script(
        tmp_path,
        {"DEEPAGENTS_CODE_YES": "1"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code"


def test_install_script_unreachable_pypi_falls_back_to_upgrade(tmp_path: Path) -> None:
    """If the latest version can't be fetched, uv still attempts an upgrade."""
    args = _run_install_script(tmp_path, {}, installed_version="0.1.0", curl_fails=True)

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code"


def test_install_script_interactive_decline_keeps_current(tmp_path: Path) -> None:
    """Answering 'n' to the update prompt keeps the current version (no uv)."""
    code, output, args_path = _invoke_interactive(
        tmp_path, {}, answer="n", installed_version="0.1.0", latest_version="0.2.0"
    )

    assert code == 0
    assert not args_path.exists()
    assert "0.1.0 → 0.2.0" in output
    assert "Keeping deepagents-code 0.1.0" in output


def test_install_script_interactive_accept_updates(tmp_path: Path) -> None:
    """Answering 'y' to the update prompt runs `uv tool install -U`."""
    code, output, args_path = _invoke_interactive(
        tmp_path, {}, answer="y", installed_version="0.1.0", latest_version="0.2.0"
    )

    assert code == 0
    # The accept-path uv argv is identical to the auto-update and assume-yes
    # paths, so assert the "Updating ..." line to prove the prompt was shown and
    # answered yes rather than bypassed.
    assert "Updating deepagents-code 0.1.0 → 0.2.0" in output
    args = args_path.read_text().splitlines()
    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code"


def test_install_script_pinned_version_skips_prompt_over_existing_install(
    tmp_path: Path,
) -> None:
    """A pinned `DEEPAGENTS_CODE_VERSION` installs directly, never prompting.

    Guards the dispatch gate (`[ -z "$VERSION" ]`) that routes an explicit pin
    past the update prompt: answering 'n' must not stop the install, and neither
    the prompt arrow nor the "Keeping" decline message should appear.
    """
    code, output, args_path = _invoke_interactive(
        tmp_path,
        {"DEEPAGENTS_CODE_VERSION": "0.2.0"},
        answer="n",
        installed_version="0.1.0",
        latest_version="0.3.0",
    )

    assert code == 0
    assert "→" not in output
    assert "Keeping deepagents-code" not in output
    args = args_path.read_text().splitlines()
    assert args[:3] == ["tool", "install", "-U"]
    assert args[-1] == "deepagents-code==0.2.0"


def test_can_prompt_false_when_not_interactive(tmp_path: Path) -> None:
    """`can_prompt` short-circuits to false when `IS_INTERACTIVE` is false."""
    assert _eval_can_prompt(tmp_path, is_interactive=False, stdin_is_tty=True) is False


def test_can_prompt_true_when_stdin_is_a_tty(tmp_path: Path) -> None:
    """A real tty on stdin satisfies the `[ -t 0 ]` fast path."""
    assert _eval_can_prompt(tmp_path, is_interactive=True, stdin_is_tty=True) is True


def test_can_prompt_false_without_usable_tty(tmp_path: Path) -> None:
    """No openable `/dev/tty` yields false even when `IS_INTERACTIVE` is true.

    Guards the load-bearing line: `can_prompt` must actually open `/dev/tty`
    rather than trusting `IS_INTERACTIVE` (which only access-checks the device).
    A regression that returned 0 right after the `IS_INTERACTIVE` check would
    wrongly report the unanswerable cron/systemd/CI case as promptable.
    """
    assert _eval_can_prompt(tmp_path, is_interactive=True, stdin_is_tty=False) is False


_FRESH_INSTALL_DIFF = (
    " + agent-client-protocol==0.10.1\n + deepagents-code==0.1.19\n + zstandard==0.25.0"
)

_UPGRADE_DIFF = (
    " - deepagents-code==0.1.18\n + deepagents-code==0.1.19\n + brand-new-dep==1.0.0"
)

_REMOVAL_DIFF = (
    " - deepagents-code==0.1.18\n + deepagents-code==0.1.19\n - dropped-dep==2.0.0"
)


def test_install_script_fresh_install_hides_packages(tmp_path: Path) -> None:
    """A fresh install hides every dependency touched by uv."""
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _FRESH_INSTALL_DIFF},
        installed_version=None,
    )

    assert proc.returncode == 0
    assert "Installed 3 packages" not in proc.stderr
    assert "Installed packages:" not in proc.stderr
    assert "agent-client-protocol" not in proc.stderr


def test_install_script_verbose_lists_every_package(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_VERBOSE=1` opts back in to the full dependency list."""
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _FRESH_INSTALL_DIFF, "DEEPAGENTS_CODE_VERBOSE": "1"},
        installed_version=None,
    )

    assert proc.returncode == 0
    assert "agent-client-protocol==0.10.1" in proc.stderr
    assert "zstandard==0.25.0" in proc.stderr
    assert "Installed 3 packages" not in proc.stderr


def test_install_script_upgrade_still_shows_diff(tmp_path: Path) -> None:
    """An upgrade keeps its compact changed-package diff."""
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _UPGRADE_DIFF},
        installed_version="0.1.18",
        latest_version="0.1.19",
    )

    assert proc.returncode == 0
    assert "Updated packages:" in proc.stderr
    assert "0.1.18 \u2192 0.1.19" in proc.stderr
    assert "brand-new-dep" in proc.stderr
    assert "(new)" in proc.stderr
    assert "Installed 3 packages" not in proc.stderr


def test_install_script_upgrade_marks_removed_packages(tmp_path: Path) -> None:
    """An upgrade that drops a transitive dependency labels it `(removed)`."""
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _REMOVAL_DIFF},
        installed_version="0.1.18",
        latest_version="0.1.19",
    )

    assert proc.returncode == 0
    assert "Updated packages:" in proc.stderr
    assert "0.1.18 → 0.1.19" in proc.stderr
    assert "dropped-dep" in proc.stderr
    assert "(removed)" in proc.stderr


def test_install_script_interactive_empty_answer_keeps_current(tmp_path: Path) -> None:
    """An empty answer at the prompt declines rather than defaulting to upgrade.

    Guards `prompt_yn`'s default: pressing Enter (or any reply that is not
    `^[Yy]$`) must not be mistaken for consent, so uv is never invoked.
    """
    code, output, args_path = _invoke_interactive(
        tmp_path, {}, answer="", installed_version="0.1.0", latest_version="0.2.0"
    )

    assert code == 0
    assert not args_path.exists()
    assert "Keeping deepagents-code 0.1.0" in output
