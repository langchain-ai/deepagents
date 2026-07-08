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
    dcode_verify_fails: bool = False,
) -> tuple[Path, Path, Path]:
    """Stage fake `uv`, `curl`, and (optionally) `dcode` binaries on `PATH`.

    `installed_version` controls whether `dcode -v` reports an existing install
    (`None` simulates a fresh machine). `latest_version` is the version the
    fake `curl` reports from PyPI; `curl_fails` makes that probe error out so
    the script's offline fallback can be exercised. `dcode_verify_fails` makes
    `dcode -v` exit non-zero (`VERIFY_OK=false`) so the eager managed-ripgrep
    guard can be exercised against a present-but-broken binary.
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
  exit "${{FAKE_UV_INSTALL_RC:-0}}"
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
        tools_log = tmp_path / "dcode-tools.txt"
        verify_rc = 1 if dcode_verify_fails else 0
        dcode.write_text(
            f"""#!/usr/bin/env bash
if [ "${{1:-}}" = "-v" ]; then
  printf 'deepagents-code {installed_version}\\n'
  exit {verify_rc}
fi
if [ "${{1:-}}" = "tools" ]; then
  printf '%s\\n' "$*" >> {str(tools_log)!r}
  printf 'Using ripgrep already on PATH at /tmp/fake-rg\\n'
  exit "${{FAKE_DCODE_TOOLS_RC:-0}}"
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
    dcode_verify_fails: bool = False,
) -> dict[str, str]:
    bin_dir, home, uv = _write_fake_tools(
        tmp_path,
        installed_version=installed_version,
        latest_version=latest_version,
        curl_fails=curl_fails,
        dcode_verify_fails=dcode_verify_fails,
    )
    return {
        **os.environ,
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
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
    dcode_verify_fails: bool = False,
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
        dcode_verify_fails=dcode_verify_fails,
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
    appending a version pin or extras, while allowing stable releases that pin
    pre-release dependencies to resolve.
    """
    args = _run_install_script(tmp_path, {}, installed_version=None)

    assert args[:3] == ["tool", "install", "-U"]
    assert args[-3:] == ["--prerelease", "allow", "deepagents-code"]


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

_DEPENDENCY_UPDATE_DIFF = " - boto3==1.43.33\n + boto3==1.43.34"

# A pure-addition diff: uv pulled in a brand-new transitive dep without any
# version change to an existing package.
_DEPENDENCY_ADDITION_DIFF = " + brand-new-dep==1.0.0"

# uv ran but moved nothing — only timing/summary noise, no `± pkg==ver` lines.
_NO_PACKAGE_CHANGE_STDERR = (
    "Resolved 5 packages in 12ms\n"
    "Resolved in 12ms\n"
    "Prepared 1 package for build in 20ms\n"
    "Checked in 1ms\n"
    "Audited 5 packages in 1ms"
)

_UV_PROGRESS_STDERR = (
    "Downloading uvloop (1.3MiB)\n"
    " Downloading pygments (1.2MiB)\n"
    "Downloaded uvloop\n"
    "Building forbiddenfruit==0.1.4\n"
    "Built forbiddenfruit==0.1.4"
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


def test_install_script_hides_uv_download_and_build_progress(tmp_path: Path) -> None:
    """Non-verbose installs hide uv's download and build progress lines."""
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _UV_PROGRESS_STDERR},
        installed_version=None,
    )

    assert proc.returncode == 0
    assert "Downloading uvloop" not in proc.stderr
    assert "Downloaded uvloop" not in proc.stderr
    assert "Building forbiddenfruit" not in proc.stderr
    assert "Built forbiddenfruit" not in proc.stderr


def test_install_script_verbose_shows_uv_download_and_build_progress(
    tmp_path: Path,
) -> None:
    """Verbose installs preserve uv's raw download and build progress lines."""
    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _UV_PROGRESS_STDERR,
            "DEEPAGENTS_CODE_VERBOSE": "1",
        },
        installed_version=None,
    )

    assert proc.returncode == 0
    assert "Downloading uvloop" in proc.stderr
    assert "Downloaded uvloop" in proc.stderr
    assert "Building forbiddenfruit" in proc.stderr
    assert "Built forbiddenfruit" in proc.stderr


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


def test_install_script_same_version_with_dependency_updates_says_dependencies_updated(
    tmp_path: Path,
) -> None:
    """Unchanged app version + a uv dependency diff reports the deps were updated.

    The fake `dcode -v` reports the same version before and after install, so
    `PRE_VERSION == NEW_VERSION` and the same-version branch fires; the `± pkg==`
    diff in stderr must steer it away from the flat "already up to date" message.
    Also verifies the raw uv diff is persisted to the cache install log and that
    the success line points the user at it via the `Details:` suffix.
    """
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF},
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated. "
        "Details: ~/.cache/deepagents-code/install.log"
    ) in proc.stdout
    assert "deepagents-code 0.1.8 already up to date" not in proc.stdout
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_DEPENDENCY_UPDATE_DIFF}\n"
    )
    assert "✔ Dependencies updated. Run: dcode" in proc.stdout
    assert "✔ Already installed. Run: dcode" not in proc.stdout


def test_install_script_same_version_no_dependency_changes_says_up_to_date(
    tmp_path: Path,
) -> None:
    """Unchanged app version + no uv package diff keeps the flat no-op message.

    The negative mirror of the dependency-update test: when uv runs but moves
    nothing (only timing/summary noise), the flag must stay false so the plain
    "already up to date" message is emitted. Guards against the flag defaulting
    on, the conditional inverting, or the grep matching uv's noise lines. The
    log is still written (the no-op stderr) but the `Details:` suffix is
    suppressed, since there's no dependency change worth pointing at.
    """
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _NO_PACKAGE_CHANGE_STDERR},
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert "deepagents-code 0.1.8 already up to date." in proc.stdout
    assert "dependencies were updated" not in proc.stdout
    assert "Details: ~/.cache/deepagents-code/install.log" not in proc.stdout
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_NO_PACKAGE_CHANGE_STDERR}\n"
    )
    assert "✔ Already installed. Run: dcode" in proc.stdout


def test_install_script_same_version_with_new_dependency_says_dependencies_updated(
    tmp_path: Path,
) -> None:
    """A pure-addition diff also counts as a dependency change.

    A new transitive dep (a `+ pkg==` line with no matching `-`) trips the flag
    just like an upgrade does, so the same-version branch reports the change
    rather than a flat no-op. Pins this `+`-only semantics deliberately, and
    verifies the addition-only diff is persisted to the install log.
    """
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _DEPENDENCY_ADDITION_DIFF},
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated. "
        "Details: ~/.cache/deepagents-code/install.log"
    ) in proc.stdout
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_DEPENDENCY_ADDITION_DIFF}\n"
    )


def test_install_script_dependency_update_without_writable_log_omits_details(
    tmp_path: Path,
) -> None:
    """When the log dir can't be created, the message drops the `Details:` suffix.

    Points `XDG_CACHE_HOME` under a regular file so `mkdir -p` fails, leaving
    `INSTALL_LOG` empty. The dependency-update message must still fire, just
    without a pointer to a log that was never written — guards against the
    suffix being appended unconditionally.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("")  # regular file; mkdir -p underneath must fail

    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
            "XDG_CACHE_HOME": str(blocker / "cache"),
        },
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated."
        in proc.stdout
    )
    assert "Details:" not in proc.stdout
    assert not (blocker / "cache").exists()


def test_install_script_dependency_update_with_failed_log_copy_omits_details(
    tmp_path: Path,
) -> None:
    """When log creation succeeds but copying fails, the message omits `Details:`."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can write through directory permissions")

    cache = tmp_path / "cache"
    install_log_dir = cache / "deepagents-code"
    install_log_dir.mkdir(parents=True)
    install_log_dir.chmod(0o500)

    try:
        proc, _ = _invoke(
            tmp_path,
            {
                "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
                "XDG_CACHE_HOME": str(cache),
            },
            installed_version="0.1.8",
            latest_version="0.1.20",
        )
    finally:
        install_log_dir.chmod(0o700)

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated."
        in proc.stdout
    )
    assert "Details:" not in proc.stdout
    assert not (install_log_dir / "install.log").exists()


def test_install_script_refuses_symlinked_log_dir(tmp_path: Path) -> None:
    """A pre-existing log-dir symlink disables the persistent install log."""
    cache = tmp_path / "cache"
    target = tmp_path / "target"
    install_log_dir = cache / "deepagents-code"
    cache.mkdir()
    target.mkdir()
    install_log_dir.symlink_to(target, target_is_directory=True)

    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
            "XDG_CACHE_HOME": str(cache),
        },
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated."
        in proc.stdout
    )
    assert "Details:" not in proc.stdout
    assert not (target / "install.log").exists()


def test_install_script_refuses_symlinked_log_file(tmp_path: Path) -> None:
    """A pre-existing log-file symlink disables the persistent install log."""
    cache = tmp_path / "cache"
    install_log_dir = cache / "deepagents-code"
    target = tmp_path / "target.log"
    install_log_dir.mkdir(parents=True)
    target.write_text("keep me\n")
    (install_log_dir / "install.log").symlink_to(target)

    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
            "XDG_CACHE_HOME": str(cache),
        },
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated."
        in proc.stdout
    )
    assert "Details:" not in proc.stdout
    assert target.read_text() == "keep me\n"


def test_install_script_unset_xdg_cache_home_falls_back_to_home_cache(
    tmp_path: Path,
) -> None:
    """An empty `XDG_CACHE_HOME` falls back to `~/.cache` for the log path.

    `_env` always sets `XDG_CACHE_HOME`, which would otherwise mask the
    fallback branch — the primary path on machines (e.g. macOS) that don't
    export it. Overriding it to empty exercises that branch directly.
    """
    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
            "XDG_CACHE_HOME": "",
        },
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated. "
        "Details: ~/.cache/deepagents-code/install.log"
    ) in proc.stdout
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_DEPENDENCY_UPDATE_DIFF}\n"
    )


def test_install_script_log_path_outside_home_stays_absolute(tmp_path: Path) -> None:
    """A log path outside `$HOME` is shown verbatim, not tilde-collapsed.

    The `~` collapse only fires for paths under `$HOME`; an `XDG_CACHE_HOME`
    elsewhere must surface the absolute path in the `Details:` suffix.
    """
    external = tmp_path / "external-cache"

    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
            "XDG_CACHE_HOME": str(external),
        },
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    expected_log = external / "deepagents-code" / "install.log"
    assert f"Details: {expected_log}" in proc.stdout
    assert "Details: ~/" not in proc.stdout
    assert expected_log.read_text() == f"{_DEPENDENCY_UPDATE_DIFF}\n"


def test_install_script_failed_install_points_to_log(tmp_path: Path) -> None:
    """A failed `uv tool install` still writes the log and points the user at it.

    The log is copied from uv's captured stderr before the failure exit, so the
    error path can hand the user the full output — the case where a persistent
    log matters most. Guards the `cp`-before-`exit` ordering.
    """
    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF,
            "FAKE_UV_INSTALL_RC": "1",
        },
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode != 0
    assert "Failed to install" in proc.stderr
    assert "Full install log: ~/.cache/deepagents-code/install.log" in proc.stderr
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_DEPENDENCY_UPDATE_DIFF}\n"
    )


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


def _path_without_dcode() -> str:
    """Return the host `PATH` with any directory that already provides dcode dropped.

    The test venv installs a real `dcode`/`deepagents-code` on `PATH`. Tests that
    need to exercise the `~/.local/bin` fallback must ensure neither resolves via
    `PATH`, while keeping the system directories the script's coreutils need.
    Filtering the real `PATH` is portable across hosts, unlike hardcoding
    `/usr/bin:/bin`.
    """
    kept = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
        and not any(
            (Path(entry) / name).exists() for name in ("dcode", "deepagents-code")
        )
    ]
    return os.pathsep.join(kept)


def _invoke_with_os(
    tmp_path: Path,
    *,
    uname_os: str,
    xcode_select_rc: int,
    installed_version: str | None = None,
    latest_version: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `install.sh` with faked `uname`/`xcode-select` os probes.

    Pins the detected OS and the Xcode Command Line Tools check deterministically,
    independent of the host running the suite, on top of the usual fake tool rig.
    Returns the completed process and the path where the fake `uv` records its
    `tool install` argv — absent if the script exited before invoking uv.
    """
    bin_dir, home, uv = _write_fake_tools(
        tmp_path,
        installed_version=installed_version,
        latest_version=latest_version,
    )
    uname = bin_dir / "uname"
    uname.write_text(f"#!/usr/bin/env bash\necho {uname_os}\n")
    _make_executable(uname)
    xcode_select = bin_dir / "xcode-select"
    xcode_select.write_text(f"#!/usr/bin/env bash\nexit {xcode_select_rc}\n")
    _make_executable(xcode_select)

    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
        **(extra_env or {}),
    }
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


def _run_install_uv(
    tmp_path: Path,
    *,
    verbose: bool,
    fails: bool = False,
    mktemp_fails: bool = False,
    no_shebang: bool = False,
    download_fails: bool = False,
    use_wget: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the real `install_uv` from `install.sh` against a fake uv installer.

    A fake downloader (``curl`` by default, or ``wget`` when ``use_wget`` is set)
    writes a trivial "installer" to the file named by its output flag (``-o`` for
    curl, ``-O`` for wget); the harness runs it via ``sh``, so the noise lands in
    the captured output. When ``fails`` is set, that installer also exits
    non-zero, exercising the surface-output-on-failure branch. When ``no_shebang``
    is set, the installer content starts with an HTML tag instead of a shell
    shebang, exercising the shebang-verification rejection. When ``download_fails``
    is set, the fake downloader writes an error to stderr and exits non-zero
    *without* creating the file, exercising the download-failure branch and
    proving the downloader's own error is surfaced. Returns the completed process
    so callers can assert on whether the noise reached the terminal and on the
    exit code.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    first_line = "'<html>error</html>'" if no_shebang else "'#!/bin/sh'"
    installer = first_line + " 'echo UV_INSTALLER_NOISE'"
    if fails:
        installer += " 'exit 3'"

    # The fake downloader must handle its output flag (curl ``-o`` / wget ``-O``)
    # and write the installer content there instead of stdout. With
    # ``download_fails`` it instead emits an error to stderr and exits non-zero
    # without creating the file, so install_uv sees a failed download.
    downloader_name = "wget" if use_wget else "curl"
    out_flag = "-O" if use_wget else "-o"
    if download_fails:
        write_body = (
            "printf 'DOWNLOADER_ERROR: could not resolve host\\n' >&2\nexit 7\n"
        )
    else:
        write_body = f"printf '%s\\n' {installer} >\"${{out:-/dev/stdout}}\"\n"
    downloader = bin_dir / downloader_name
    downloader.write_text(
        "#!/usr/bin/env bash\n"
        "out=''\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        f'    {out_flag}) out="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n" + write_body
    )
    _make_executable(downloader)
    if mktemp_fails:
        mktemp = bin_dir / "mktemp"
        mktemp.write_text("#!/usr/bin/env bash\nexit 1\n")
        _make_executable(mktemp)

    # install_uv branches on is_snap_curl. For the curl path, stub it to the
    # non-snap answer so the normal curl branch runs (and no stray "command not
    # found" hits stderr). For the wget path, report curl as a snap so install_uv
    # skips the curl branch and falls through to the wget branch — regardless of
    # a real curl on the host PATH.
    is_snap_curl_rc = "0" if use_wget else "1"
    script = tmp_path / "install_uv_harness.sh"
    script.write_text(
        "set -euo pipefail\n"
        "log_info() { :; }\n"
        'log_error() { printf "%s\\n" "$*" >&2; }\n'
        "register_temp() { :; }\n"
        f"is_snap_curl() {{ return {is_snap_curl_rc}; }}\n"
        f"VERBOSE={'1' if verbose else '0'}\n"
        f"{_extract_shell_function('install_uv')}\n"
        "install_uv\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def test_install_uv_hides_installer_output_by_default(tmp_path: Path) -> None:
    """The chatty upstream uv installer output is suppressed on a normal run."""
    proc = _run_install_uv(tmp_path, verbose=False)

    assert proc.returncode == 0
    assert "UV_INSTALLER_NOISE" not in proc.stdout
    assert "UV_INSTALLER_NOISE" not in proc.stderr


def test_install_uv_verbose_shows_installer_output(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_VERBOSE=1` opts back in to the uv installer's output."""
    proc = _run_install_uv(tmp_path, verbose=True)

    assert proc.returncode == 0
    assert "UV_INSTALLER_NOISE" in proc.stderr


def test_install_uv_surfaces_output_on_failure(tmp_path: Path) -> None:
    """A failed uv install replays the captured output even when not verbose.

    The surface-on-failure half of the gate (`uv_install_rc -ne 0`) is the only
    diagnostic the user gets when the upstream installer dies, so it must fire
    regardless of `DEEPAGENTS_CODE_VERBOSE` and the script must exit non-zero.
    """
    proc = _run_install_uv(tmp_path, verbose=False, fails=True)

    assert proc.returncode != 0
    assert "UV_INSTALLER_NOISE" in proc.stderr
    assert "uv installation failed" in proc.stderr


def test_install_uv_requires_secure_temp_file(tmp_path: Path) -> None:
    """`install_uv` fails closed if secure temporary file creation is unavailable."""
    proc = _run_install_uv(tmp_path, verbose=False, mktemp_fails=True)

    assert proc.returncode != 0
    assert "mktemp is required to create a secure temp file" in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stderr


def test_install_uv_rejects_non_shell_response(tmp_path: Path) -> None:
    """A download that doesn't start with a shell shebang is rejected before exec.

    Simulates a transparent proxy or captive portal returning 200 with HTML
    instead of the uv installer. The shebang check must catch it and exit with
    an actionable error, rather than piping the HTML into ``sh``.
    """
    proc = _run_install_uv(tmp_path, verbose=False, no_shebang=True)

    assert proc.returncode != 0
    assert "does not start with a shell shebang" in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stdout


def test_install_uv_surfaces_download_failure(tmp_path: Path) -> None:
    """A failed download exits non-zero and surfaces the downloader's own error.

    Exercises the download-failure branch (`uv_install_rc -ne 0`): the fake curl
    exits non-zero and writes its error to stderr without creating the installer
    file. `install_uv` must relay that captured error — not just a generic
    message — include the downloader's exit code, and never execute a payload.
    """
    proc = _run_install_uv(tmp_path, verbose=False, download_fails=True)

    assert proc.returncode != 0
    assert "Failed to download uv installer" in proc.stderr
    # The downloader's captured stderr is surfaced, not discarded to /dev/null.
    assert "DOWNLOADER_ERROR: could not resolve host" in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stdout


def test_install_uv_downloads_via_wget(tmp_path: Path) -> None:
    """The wget branch downloads to `-O <file>` and the script then runs it.

    curl is reported as a snap so `install_uv` falls through to the wget branch.
    Verbose mode surfaces the installer's output, proving wget wrote a valid
    shebang file that passed verification and executed.
    """
    proc = _run_install_uv(tmp_path, verbose=True, use_wget=True)

    assert proc.returncode == 0, proc.stderr
    assert "UV_INSTALLER_NOISE" in proc.stderr


def _run_signal_traps(tmp_path: Path, *, interrupt: bool) -> str:
    """Wire the real EXIT + INT/TERM traps from `install.sh` and trip one.

    Extracts the shipped `cleanup_on_signal`/`cleanup_on_interrupt` handlers and
    installs them exactly as the script does. With `interrupt=True` the process
    sends itself SIGINT (the Ctrl-C path); otherwise it exits non-zero without a
    signal (the ordinary-failure path). Returns combined stderr so callers can
    assert which trap message the user actually sees.
    """
    script = tmp_path / "signal_trap_harness.sh"
    body = "kill -INT $$\nsleep 5\n" if interrupt else "exit 2\n"
    script.write_text(
        "set -uo pipefail\n"
        'log_warn()  { printf "%s\\n" "$*" >&2; }\n'
        'log_error() { printf "%s\\n" "$*" >&2; }\n'
        "cleanup_temp_files() { :; }\n"
        f"{_extract_shell_function('cleanup_on_signal')}\n"
        f"{_extract_shell_function('cleanup_on_interrupt')}\n"
        "trap cleanup_on_signal EXIT\n"
        "trap cleanup_on_interrupt INT TERM\n"
        f"{body}",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        start_new_session=True,
    )
    return proc.stderr


def test_interrupt_shows_notice_without_failure_message(tmp_path: Path) -> None:
    """Ctrl-C prints only the interrupt notice, not the EXIT trap's failure line.

    `cleanup_on_interrupt` disarms the EXIT trap (`trap - EXIT`) before exiting,
    so the friendly "Installation interrupted." message isn't followed by a
    contradictory "Installation failed (exit code 1)". Guards against dropping
    that disarm, which would surface both messages on a single Ctrl-C.
    """
    stderr = _run_signal_traps(tmp_path, interrupt=True)

    assert "Installation interrupted." in stderr
    assert "Installation failed" not in stderr


def test_exit_trap_reports_failure_on_ordinary_error(tmp_path: Path) -> None:
    """A non-signal, non-zero exit still fires the EXIT trap's failure message.

    The interrupt handler's `trap - EXIT` must be scoped to the interrupt path
    only: an ordinary failure exit still needs `cleanup_on_signal` to tell the
    user the install failed and where to get help.
    """
    stderr = _run_signal_traps(tmp_path, interrupt=False)

    assert "Installation failed (exit code 2)." in stderr
    assert "Installation interrupted." not in stderr


def test_install_script_macos_without_clt_exits_early(tmp_path: Path) -> None:
    """On macOS, missing Xcode Command Line Tools fails fast before uv runs.

    Pins `uname`→Darwin and a failing `xcode-select -p` so the pre-flight check
    trips. The script must exit non-zero with an actionable message and must do
    so before invoking uv (the fake `uv` records no argv), rather than letting a
    downstream tool trigger the macOS "install developer tools" GUI popup.
    """
    proc, uv_args = _invoke_with_os(
        tmp_path, uname_os="Darwin", xcode_select_rc=2, installed_version="0.0.1"
    )

    assert proc.returncode != 0
    assert "Xcode Command Line Tools" in proc.stderr
    assert "xcode-select --install" in proc.stderr
    assert not uv_args.exists()


def test_install_script_macos_skip_xcode_check_proceeds_without_clt(
    tmp_path: Path,
) -> None:
    """The macOS CLT check can be bypassed for managed install environments."""
    proc, uv_args = _invoke_with_os(
        tmp_path,
        uname_os="Darwin",
        xcode_select_rc=2,
        installed_version="0.0.1",
        latest_version="0.2.0",
        extra_env={"DEEPAGENTS_CODE_SKIP_XCODE_CHECK": "1"},
    )

    assert proc.returncode == 0
    assert "Xcode Command Line Tools" not in proc.stderr
    assert uv_args.exists()


def test_install_script_macos_with_clt_proceeds_to_install(tmp_path: Path) -> None:
    """On macOS with Xcode CLT present, the pre-flight check passes through to uv.

    Pins `uname`→Darwin and a succeeding `xcode-select -p` so the gate's no-fire
    branch is asserted deterministically rather than relying on the host's own
    CLT state. The run must reach `uv tool install` without emitting the CLT
    error.
    """
    proc, uv_args = _invoke_with_os(
        tmp_path,
        uname_os="Darwin",
        xcode_select_rc=0,
        installed_version="0.0.1",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "Xcode Command Line Tools" not in proc.stderr
    assert uv_args.exists()


def test_install_script_linux_skips_clt_check(tmp_path: Path) -> None:
    """The CLT gate is macOS-only: a failing `xcode-select` is ignored on Linux.

    Pins `uname`→Linux with a failing `xcode-select -p`; the `$OS = macos` guard
    must short-circuit so the check never trips and the install proceeds.
    """
    proc, uv_args = _invoke_with_os(
        tmp_path,
        uname_os="Linux",
        xcode_select_rc=2,
        installed_version="0.0.1",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "Xcode Command Line Tools" not in proc.stderr
    assert uv_args.exists()


def _invoke_with_local_uv_not_on_path(
    tmp_path: Path, *, env_file_content: str | None = None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run with uv present only in ~/.local/bin, absent from PATH."""
    bin_dir, home, uv = _write_fake_tools(
        tmp_path, installed_version=None, latest_version="0.2.0"
    )

    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    local_uv = local_bin / "uv"
    local_uv.write_text(uv.read_text())
    _make_executable(local_uv)
    uv.unlink()
    if env_file_content is not None:
        (local_bin / "env").write_text(env_file_content)

    path_without_uv = os.pathsep.join(
        entry
        for entry in _path_without_dcode().split(os.pathsep)
        if entry and not (Path(entry) / "uv").exists()
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{path_without_uv}",
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
    }
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


def test_install_script_uses_local_uv_when_not_on_path(tmp_path: Path) -> None:
    """A minimal MDM PATH must not reinstall uv when ~/.local/bin/uv exists."""
    proc, uv_args = _invoke_with_local_uv_not_on_path(tmp_path)

    assert proc.returncode == 0
    assert uv_args.exists()
    assert "uv not found — installing" not in proc.stdout + proc.stderr
    assert uv_args.read_text().splitlines()[:3] == ["tool", "install", "-U"]


def test_install_script_sources_uv_env_file_defensively(tmp_path: Path) -> None:
    """A non-zero command in uv's env file must not abort the installer."""
    proc, uv_args = _invoke_with_local_uv_not_on_path(
        tmp_path,
        env_file_content='export PATH="$HOME/.local/bin:$PATH"\nfalse\n',
    )

    assert proc.returncode == 0
    assert uv_args.exists()
    assert "uv not found — installing" not in proc.stdout + proc.stderr
    assert uv_args.read_text().splitlines()[:3] == ["tool", "install", "-U"]


def test_install_script_rejects_invalid_uv_bin_without_installing(
    tmp_path: Path,
) -> None:
    """A bad `UV_BIN` should fail clearly instead of reinstalling uv."""
    cases = [
        (tmp_path / "missing", tmp_path / "missing" / "uv"),
        (tmp_path / "directory", tmp_path / "directory" / "uv"),
    ]
    cases[1][1].mkdir(parents=True)

    for root, uv_bin in cases:
        root.mkdir(exist_ok=True)
        proc, uv_args = _invoke(root, {"UV_BIN": str(uv_bin)})

        assert proc.returncode != 0
        assert not uv_args.exists()
        assert (
            f"UV_BIN is set but does not point to an executable uv: {uv_bin}"
            in proc.stderr
        )


def _invoke_with_local_dcode_not_on_path(
    tmp_path: Path, *, create_env_file: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run with a working `dcode` in ~/.local/bin but outside the original PATH."""
    bin_dir, home, uv = _write_fake_tools(tmp_path, installed_version=None)

    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    dcode = local_bin / "dcode"
    dcode.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-v" ]; then printf "deepagents-code 0.1.0\\n"; exit 0; fi\n'
        "exit 0\n"
    )
    _make_executable(dcode)
    if create_env_file:
        (local_bin / "env").write_text('export PATH="$HOME/.local/bin:$PATH"\n')

    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{_path_without_dcode()}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
        "SHELL": "/bin/zsh",
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_install_script_adds_local_bin_when_dcode_installed_but_not_on_path(
    tmp_path: Path,
) -> None:
    """A fresh install resolved only via ~/.local/bin adds it to PATH setup.

    Simulates `uv tool install` dropping the binary in ~/.local/bin without the
    current shell having picked it up: `command -v dcode` misses, the fallback
    path hits, and the script verifies it directly. The success path should not
    replace the installed executable with a self-referential symlink when the
    binary path and intended symlink path are the same.
    """
    proc = _invoke_with_local_dcode_not_on_path(tmp_path)

    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    dcode = tmp_path / "home/.local/bin/dcode"
    assert not dcode.is_symlink()
    assert "deepagents-code 0.1.0" in dcode.read_text()
    assert "Added ~/.local/bin to PATH" in combined
    assert "isn't on your PATH yet" not in combined
    profile_texts = [
        profile.read_text()
        for profile in (
            tmp_path / "home/.zshrc",
            tmp_path / "home/.bashrc",
            tmp_path / "home/.bash_profile",
        )
        if profile.exists()
    ]
    assert any('export PATH="$HOME/.local/bin:$PATH"' in text for text in profile_texts)
    assert "source ~/.local/bin/env" not in combined


def test_install_script_uses_uv_env_file_path_hint_when_available(
    tmp_path: Path,
) -> None:
    """When uv wrote ~/.local/bin/env, a source hint is shown for stale shells.

    uv's env file handles PATH setup for *new* shells, so no profile
    modification is needed. But the current shell still lacks ~/.local/bin on
    PATH (the binary resolved only via the installer's absolute-path fallback),
    so the script emits a `source ~/.local/bin/env` reload hint instead of
    silently returning success — a fresh `dcode` invocation would otherwise fail
    until the user restarts their shell.
    """
    proc = _invoke_with_local_dcode_not_on_path(tmp_path, create_env_file=True)

    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "isn't on your PATH yet" not in combined
    assert "source ~/.local/bin/env" in combined
    assert not (tmp_path / "home/.zshrc").exists()
    assert not (tmp_path / "home/.bashrc").exists()
    assert not (tmp_path / "home/.bash_profile").exists()


def test_install_script_stale_shell_with_profile_already_set_shows_reload_hint(
    tmp_path: Path,
) -> None:
    """~/.local/bin already in the profile still warns when the shell is stale.

    The profile already has the PATH export, so no file modification is needed.
    But the current shell's PATH lacks ~/.local/bin (the binary resolved only
    via the installer's absolute-path fallback), so the script must emit a
    reload/source hint rather than silently returning success — otherwise the
    user sees "Run: dcode" but dcode won't resolve until they restart.
    """
    bin_dir, home, uv = _write_fake_tools(tmp_path, installed_version=None)

    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    dcode = local_bin / "dcode"
    dcode.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-v" ]; then printf "deepagents-code 0.1.0\\n"; exit 0; fi\n'
        "exit 0\n"
    )
    _make_executable(dcode)

    # Pre-seed the shell profile so `local_bin_in_profile` returns true.
    zshrc = home / ".zshrc"
    zshrc.write_text('export PATH="$HOME/.local/bin:$PATH"\n')

    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{_path_without_dcode()}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
        "SHELL": "/bin/zsh",
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    # No duplicate PATH export was appended.
    assert combined.count('export PATH="$HOME/.local/bin:$PATH"') == 1
    # But the reload hint is shown because the current shell is stale.
    assert "Restart your shell, or run:" in combined
    assert 'export PATH="$HOME/.local/bin:$PATH"' in combined


def test_install_script_no_path_warning_when_dcode_on_path(tmp_path: Path) -> None:
    """When `dcode` resolves via PATH, the not-on-PATH hint is suppressed."""
    proc, _ = _invoke(tmp_path, {}, installed_version="0.1.0", latest_version="0.2.0")

    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "isn't on your PATH yet" not in combined


def test_install_script_managed_ripgrep_calls_tools_install(tmp_path: Path) -> None:
    """Default (`managed`) mode eagerly runs `dcode tools install`."""
    proc, _ = _invoke(
        tmp_path,
        {"DEEPAGENTS_CODE_SKIP_OPTIONAL": "0"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    tools_log = tmp_path / "dcode-tools.txt"
    assert tools_log.exists(), proc.stdout + proc.stderr
    assert "tools install" in tools_log.read_text()
    combined = proc.stdout + proc.stderr
    assert "Setting up ripgrep..." not in combined
    assert "Using ripgrep already on PATH" not in combined
    assert "opt out with DEEPAGENTS_CODE_RIPGREP_INSTALLER=system" not in combined


def test_install_script_managed_ripgrep_verbose_reports_tools_install(
    tmp_path: Path,
) -> None:
    """Verbose mode prints the otherwise quiet managed-ripgrep setup details."""
    proc, _ = _invoke(
        tmp_path,
        {"DEEPAGENTS_CODE_SKIP_OPTIONAL": "0", "DEEPAGENTS_CODE_VERBOSE": "1"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Setting up ripgrep..." in combined
    assert "Using ripgrep already on PATH" in combined


def test_install_script_system_ripgrep_skips_tools_install(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_RIPGREP_INSTALLER=system` keeps the package-manager path."""
    proc, _ = _invoke(
        tmp_path,
        {
            "DEEPAGENTS_CODE_SKIP_OPTIONAL": "0",
            "DEEPAGENTS_CODE_RIPGREP_INSTALLER": "system",
        },
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "dcode-tools.txt").exists()


def test_install_script_skip_optional_skips_tools_install(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_SKIP_OPTIONAL=1` skips the managed install entirely."""
    proc, _ = _invoke(
        tmp_path,
        {"DEEPAGENTS_CODE_SKIP_OPTIONAL": "1"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "dcode-tools.txt").exists()


def test_install_script_managed_ripgrep_failure_warns(tmp_path: Path) -> None:
    """A failed `dcode tools install` falls back with a slow-grep warning.

    The captured command output is surfaced on failure — the whole reason the
    quiet path writes to a temp file instead of discarding to `/dev/null`.
    """
    proc, _ = _invoke(
        tmp_path,
        {"DEEPAGENTS_CODE_SKIP_OPTIONAL": "0", "FAKE_DCODE_TOOLS_RC": "1"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "slower fallback" in combined
    assert "Using ripgrep already on PATH" in combined


def test_install_script_managed_ripgrep_verbose_failure_warns(
    tmp_path: Path,
) -> None:
    """Verbose mode still warns and shows setup output when the install fails."""
    proc, _ = _invoke(
        tmp_path,
        {
            "DEEPAGENTS_CODE_SKIP_OPTIONAL": "0",
            "DEEPAGENTS_CODE_VERBOSE": "1",
            "FAKE_DCODE_TOOLS_RC": "1",
        },
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Setting up ripgrep..." in combined
    assert "Using ripgrep already on PATH" in combined
    assert "slower fallback" in combined


def test_install_script_skips_managed_install_when_verify_failed(
    tmp_path: Path,
) -> None:
    """A present-but-broken `dcode` (`VERIFY_OK=false`) is not run for `tools`.

    The eager managed-ripgrep block is gated on `VERIFY_OK = true`, so a binary
    that fails its `-v` probe must not be invoked as `dcode tools install`.
    """
    proc, _ = _invoke(
        tmp_path,
        {"DEEPAGENTS_CODE_SKIP_OPTIONAL": "0"},
        installed_version="0.1.0",
        latest_version="0.2.0",
        dcode_verify_fails=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "dcode-tools.txt").exists(), proc.stdout + proc.stderr


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_install_script_help_flag_prints_usage_and_exits(
    tmp_path: Path, flag: str
) -> None:
    """`--help` / `-h` prints the env-var reference and exits 0 before any install.

    Guards the early-returns in the CLI-flag loop: the script must not reach uv
    or any network probe. The output must mention key environment variables so
    the user can discover their options without reading source.
    """
    env = _env(tmp_path, {}, installed_version=None, latest_version="0.2.0")
    proc = subprocess.run(
        ["bash", str(SCRIPT), flag],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert proc.returncode == 0
    assert "DEEPAGENTS_CODE_VERSION" in proc.stdout
    assert "DEEPAGENTS_CODE_EXTRAS" in proc.stdout
    assert "baseten" in proc.stdout
    assert "basesten" not in proc.stdout
    assert "DEEPAGENTS_CODE_PYTHON" in proc.stdout
    assert not (tmp_path / "uv-args.txt").exists()


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_install_script_version_flag_prints_version_and_exits(
    tmp_path: Path, flag: str
) -> None:
    """`--version` / `-v` prints the installer version and exits 0."""
    env = _env(tmp_path, {}, installed_version=None, latest_version="0.2.0")
    proc = subprocess.run(
        ["bash", str(SCRIPT), flag],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert proc.returncode == 0
    # Assert the exact version string, not just a substring: the help body also
    # contains "installer", so a weaker check wouldn't catch --version being
    # mis-wired to print_help. The absent "Usage:" marker pins that distinction
    # and doubles as a drift guard on INSTALLER_VERSION.
    assert "deepagents-code installer 1.0" in proc.stdout
    assert "Usage:" not in proc.stdout
    assert not (tmp_path / "uv-args.txt").exists()


def test_install_script_rejects_unknown_flag(tmp_path: Path) -> None:
    """An unrecognized argument exits non-zero before any install work.

    Guards the `*)` arm of the CLI-flag loop: a typo like `--verison` must
    surface an error and skip the install, rather than being silently ignored
    and proceeding to a full install.
    """
    env = _env(tmp_path, {}, installed_version=None, latest_version="0.2.0")
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--verison"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert proc.returncode == 2
    assert "Unrecognized argument" in proc.stderr
    assert not (tmp_path / "uv-args.txt").exists()
