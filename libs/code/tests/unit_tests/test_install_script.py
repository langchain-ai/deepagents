"""Tests for the shell install script argument construction."""

from __future__ import annotations

import os
import pty
import re
import stat
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SCRIPT = Path(__file__).parents[2] / "scripts" / "install.sh"

# Sentinel answer for `_invoke_interactive`: send a bare EOT (Ctrl-D) rather
# than a line. Identity-compared, so it cannot collide with a real answer.
_CTRL_D = "<ctrl-d>"

# Where `copy_install_log` stages the log before publishing it. Mirrors the
# `mktemp -d` template in install.sh; kept here so the cleanup assertion looks
# in the same place the script writes.
_STAGE_ROOT = Path("/tmp")
_STAGE_GLOB = "deepagents-code-install-log.*"

# Root bypasses the permission bits several tests rely on. `hasattr` because
# `os.geteuid` does not exist on Windows, where these tests are skipped whole.
_RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

PRERELEASE_STRATEGIES = (
    "disallow",
    "allow",
    "if-necessary",
    "explicit",
    "if-necessary-or-explicit",
)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _clean_environ() -> dict[str, str]:
    """Return `os.environ` without vars that redirect the installer's writes.

    A developer with `ZDOTDIR`, `XDG_CONFIG_HOME`, or `XDG_DATA_HOME` set in
    their own shell would otherwise redirect writes away from the fake HOME.
    The data directory also selects the fallback installer-lock root when
    `uv tool dir` is unavailable. Tests that exercise those variables set them
    explicitly.

    `SHELL` is dropped for the same reason: it selects the candidate profile
    set, so a contributor whose login shell is fish would otherwise exercise a
    different code path than CI. Callers that care set it explicitly.
    `DEEPAGENTS_CODE_*` is dropped so a developer's own installer settings
    (an assumed-yes, a pinned version, an opt-out) can't leak into a run.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "DEEPAGENTS_HOME",
            "SHELL",
            "UV_TOOL_DIR",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "ZDOTDIR",
        }
        and not key.startswith("DEEPAGENTS_CODE_")
    }


def test_clean_environ_drops_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host data-directory settings cannot redirect installer test locks."""
    monkeypatch.setenv("XDG_DATA_HOME", "/host-specific/data")

    assert "XDG_DATA_HOME" not in _clean_environ()


def _host_path_without_dcode() -> str:
    """The host `PATH` with any directory holding a real `dcode` dropped.

    The fake tools shadow `dcode` only when the fixture stages one; a run
    configured as a fresh machine otherwise finds whatever the developer has
    installed and reports it as a pre-existing install. That makes
    `PRE_VERSION` — and every branch keyed on it — depend on who is running
    the suite.
    """
    kept = []
    for entry in os.environ["PATH"].split(os.pathsep):
        if not entry:
            continue
        directory = Path(entry)
        if any((directory / name).exists() for name in ("dcode", "deepagents-code")):
            continue
        kept.append(entry)
    return os.pathsep.join(kept)


def _write_fake_tools(
    tmp_path: Path,
    *,
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
    curl_fails: bool = False,
    curl_failures_before_success: int = 0,
    dcode_verify_fails: bool = False,
    rg_version_fails: bool = False,
    mktemp_fails: bool = False,
    stage_uv_receipt: bool = True,
) -> tuple[Path, Path, Path]:
    """Stage fake `uv`, `curl`, and (optionally) `dcode` binaries on `PATH`.

    `installed_version` controls whether `dcode -v` reports an existing install
    (`None` simulates a fresh machine). `latest_version` is the version the
    fake `curl` reports from PyPI; `curl_fails` makes that probe error out so
    the script's offline fallback can be exercised. `dcode_verify_fails` makes
    `dcode -v` exit non-zero (`VERIFY_OK=false`) so the eager managed-ripgrep
    guard can be exercised against a present-but-broken binary.
    `rg_version_fails` stages an `rg` that fails its version probe.

    An existing install also gets a bare `uv-receipt.toml`, because that is
    what a real `uv tool install` leaves behind: the script treats a uv-managed
    install whose receipt it cannot find as "couldn't tell which extras this
    has" and warns, so a fixture without one would put every unrelated test on
    that warning path. A test that staged its own receipt keeps it, and
    `stage_uv_receipt=False` opts out to reach the missing-receipt branches
    deliberately.
    """
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    tools = tmp_path / "tools"
    bin_dir.mkdir()
    # exist_ok: a test may seed `home/.cache/deepagents-code/install.log` before
    # invoking, to assert what happens to a previous run's log.
    home.mkdir(exist_ok=True)
    # exist_ok: a test may stage a uv tool receipt under `tools/deepagents-code`
    # before invoking, which creates `tools` as a side effect.
    tools.mkdir(exist_ok=True)
    if (
        stage_uv_receipt
        and installed_version is not None
        and not (tools / "deepagents-code").exists()
    ):
        _write_uv_receipt(tools, None)

    # Raw f-string: the embedded bash must keep `\n` as the two literal
    # characters (an f-string would otherwise turn `\n` into a newline). `{{ }}`
    # still escape to literal braces; the `{...!r}` slots interpolate paths.
    default_tool_bin = bin_dir if installed_version is not None else home / ".local/bin"
    uv = bin_dir / "uv"
    uv.write_text(
        rf"""#!/usr/bin/env bash
set -euo pipefail
default_tool_bin={str(default_tool_bin)!r}
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "dir" ]; then
  if [ "${{3:-}}" = "--bin" ]; then
    if [ "${{FAKE_UV_TOOL_DIR_BIN_UNSUPPORTED:-}}" = "1" ]; then
      exit 2
    fi
    printf '%s\n' "${{FAKE_UV_TOOL_BIN_DIR:-$default_tool_bin}}"
  else
    if [ "${{FAKE_UV_TOOL_DIR_UNSUPPORTED:-}}" = "1" ]; then
      exit 2
    fi
    printf '%s\n' {str(tools)!r}
  fi
  exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "install" ]; then
  printf '%s\n' "$@" > {str(tmp_path / "uv-args.txt")!r}
  if [ "${{FAKE_UV_CREATE_LOCAL_DCODE:-}}" = "1" ]; then
    tool_bin="${{FAKE_UV_TOOL_BIN_DIR:-$default_tool_bin}}"
    mkdir -p "$tool_bin"
    cat > "$tool_bin/dcode" <<'DCODE'
#!/usr/bin/env bash
if [ "${{1:-}}" = "-v" ]; then
  printf 'deepagents-code %s\n' "${{FAKE_LOCAL_DCODE_VERSION:-0.2.0}}"
  exit 0
fi
exit 0
DCODE
    chmod +x "$tool_bin/dcode"
  fi
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
    elif curl_failures_before_success:
        payload = f'{{"info":{{"version":"{latest_version}"}}}}'
        attempts = tmp_path / "curl-attempts.txt"
        curl.write_text(
            f"""#!/usr/bin/env bash
count=0
if [ -f {str(attempts)!r} ]; then
  read -r count < {str(attempts)!r}
fi
count=$((count + 1))
printf '%s\n' "$count" > {str(attempts)!r}
if [ "$count" -le {curl_failures_before_success} ]; then
  exit 7
fi
printf '%s' '{payload}'
"""
        )
    else:
        payload = f'{{"info":{{"version":"{latest_version}"}}}}'
        curl.write_text(f"#!/usr/bin/env bash\nprintf '%s' '{payload}'\n")
    _make_executable(curl)

    sleep = bin_dir / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    _make_executable(sleep)

    if rg_version_fails:
        rg = bin_dir / "rg"
        rg.write_text("#!/usr/bin/env bash\nexit 1\n")
        _make_executable(rg)

    if mktemp_fails:
        mktemp = bin_dir / "mktemp"
        mktemp.write_text("#!/usr/bin/env bash\nexit 1\n")
        _make_executable(mktemp)

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
  if [ -n "${{FAKE_MANAGED_BIN_DIR:-}}" ]; then
    mkdir -p "$FAKE_MANAGED_BIN_DIR"
    : >"$FAKE_MANAGED_BIN_DIR/rg"
  fi
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
    curl_failures_before_success: int = 0,
    dcode_verify_fails: bool = False,
    rg_version_fails: bool = False,
    mktemp_fails: bool = False,
    stage_uv_receipt: bool = True,
) -> dict[str, str]:
    bin_dir, home, uv = _write_fake_tools(
        tmp_path,
        installed_version=installed_version,
        latest_version=latest_version,
        curl_fails=curl_fails,
        curl_failures_before_success=curl_failures_before_success,
        dcode_verify_fails=dcode_verify_fails,
        rg_version_fails=rg_version_fails,
        mktemp_fails=mktemp_fails,
        stage_uv_receipt=stage_uv_receipt,
    )
    return {
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{_host_path_without_dcode()}",
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
    curl_failures_before_success: int = 0,
    dcode_verify_fails: bool = False,
    rg_version_fails: bool = False,
    mktemp_fails: bool = False,
    stage_uv_receipt: bool = True,
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
        curl_failures_before_success=curl_failures_before_success,
        dcode_verify_fails=dcode_verify_fails,
        rg_version_fails=rg_version_fails,
        mktemp_fails=mktemp_fails,
        stage_uv_receipt=stage_uv_receipt,
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
    answer: str | list[str],
    installed_version: str | None = "0.0.1",
    latest_version: str | None = None,
) -> tuple[int, str, Path]:
    """Run `install.sh` with a pty stdin and feed `answer` to its prompt.

    A pty makes `[ -t 0 ]` true, so the script treats the run as interactive and
    reads the y/n answer from stdin. `answer` may be a single line or a list of
    lines (fed in order) when the script prompts more than once; the sentinel
    `_CTRL_D` sends a bare EOT instead, which the terminal delivers as a real
    end-of-file to whichever `read` is waiting. Returns the exit code, combined
    output (ANSI stripped), and the uv-argv path.
    """
    env = _env(
        tmp_path,
        extra_env,
        installed_version=installed_version,
        latest_version=latest_version,
    )
    answers = [answer] if isinstance(answer, str) else answer
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
    try:
        for line in answers:
            payload = b"\x04" if line is _CTRL_D else f"{line}\n".encode()
            os.write(primary, payload)
        assert proc.stdout is not None
        output = proc.stdout.read()
        proc.wait(timeout=30)
    finally:
        # A `TimeoutExpired` here would otherwise leak the pipe and the pty and
        # orphan the child — precisely when a cascade of unraisable warnings
        # would obscure the real failure.
        if proc.stdout is not None:
            proc.stdout.close()
        os.close(primary)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
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


def _write_prompt_yn_harness(
    tmp_path: Path, name: str = "prompt_yn_harness.sh"
) -> Path:
    """Write a script that runs the real `prompt_yn` and reports its status.

    Extracts the shipped function (bash 3.2 on macOS cannot `source <(...)`)
    alongside the `log_warn` it calls, then prints `rc=<status>` so a caller
    can tell the three outcomes apart: accepted (0), declined (1), and
    unaskable (2). The status is the whole contract — every caller in the
    script branches on exactly these three values.
    """
    script = tmp_path / name
    script.write_text(
        "log_warn() { printf 'WARN %s\\n' \"$1\" >&2; }\n"
        f"{_extract_shell_function('prompt_yn')}\n"
        "IS_INTERACTIVE=true\n"
        "rc=0\n"
        'prompt_yn "Continue?" || rc=$?\n'
        'printf "[rc=%s]\\n" "$rc" >&2\n'
        "exit 0\n",
        encoding="utf-8",
    )
    return script


def _run_prompt_yn_on_tty(
    tmp_path: Path, keystrokes: bytes, *, stdin_is_tty: bool
) -> str:
    r"""Drive the real `prompt_yn` against a controlling terminal.

    Forks with the pty as the child's controlling terminal, so `/dev/tty` opens
    in both configurations. `stdin_is_tty` selects the branch under test: True
    leaves the pty on stdin (`[ -t 0 ]`), False redirects stdin from
    `/dev/null` — the shape of the documented `curl … | bash` install, where
    the script's own stdin is a pipe but a terminal is still reachable.

    `keystrokes` goes to the pty verbatim, so `b"\\x04"` is a real Ctrl-D.
    Note a terminal needs *two* EOTs to end a line that has text on it: the
    first only flushes the pending characters to the reader, and the second —
    on a now-empty buffer — is what makes `read(2)` return 0. Returns the
    child's combined output.
    """
    script = _write_prompt_yn_harness(tmp_path)
    pid, primary = pty.fork()
    if pid == 0:  # pragma: no cover - child process never returns
        try:
            if not stdin_is_tty:
                devnull = os.open(os.devnull, os.O_RDONLY)
                os.dup2(devnull, 0)
            os.execvp("bash", ["bash", str(script)])
        finally:
            os._exit(127)
    os.write(primary, keystrokes)
    chunks = []
    try:
        while True:
            chunk = os.read(primary, 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError:
        # The pty reports EIO rather than EOF once the child side is gone.
        pass
    os.waitpid(pid, 0)
    os.close(primary)
    return b"".join(chunks).decode(errors="replace")


def _run_with_args(
    tmp_path: Path,
    args: tuple[str, ...],
    extra_env: dict[str, str] | None = None,
    *,
    installed_version: str | None = None,
    latest_version: str | None = "0.2.0",
) -> subprocess.CompletedProcess[str]:
    """Run `install.sh` with positional `args` and the fake tools on `PATH`."""
    env = _env(
        tmp_path,
        extra_env or {},
        installed_version=installed_version,
        latest_version=latest_version,
    )
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


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


@pytest.mark.parametrize(
    "bad_target",
    [
        "0.1.0; rm -rf /",  # shell metacharacters
        "1.0 --force",  # whitespace + smuggled flag
        ">=1.0",  # range operator, not an exact pin
    ],
)
def test_install_script_rejects_invalid_positional_version(
    tmp_path: Path, bad_target: str
) -> None:
    """An invalid positional target is rejected before uv runs (injection guard).

    The positional arg is a brand-new untrusted input that flows into uv's argv;
    this pins the `^[A-Za-z0-9][A-Za-z0-9_.!+-]*$` guard that blocks metacharacter
    and smuggled-flag payloads independently of the DEEPAGENTS_CODE_VERSION check.
    """
    proc = _run_with_args(tmp_path, (bad_target,))

    assert proc.returncode == 2
    assert "Invalid version target" in proc.stderr
    assert not (tmp_path / "uv-args.txt").exists()


def test_install_script_already_up_to_date_skips_uv(tmp_path: Path) -> None:
    """When installed matches PyPI's latest, uv is skipped and no lock is taken.

    The `~/.deepagents` assertion pins that the early up-to-date exit returns
    before `acquire_install_lock`, so the no-op path leaves no lock directory
    behind. `~/.cache` pins the same property for the install log:
    `prepare_install_log_dir` creates the cache root and the package
    subdirectory, so computing the log path above this exit would make a run
    that installs nothing still leave directories on the machine.
    """
    proc, args_path = _invoke(
        tmp_path, {}, installed_version="0.1.0", latest_version="0.1.0"
    )

    assert proc.returncode == 0
    assert not args_path.exists()
    assert "Already up to date!" in proc.stdout
    assert not (tmp_path / "home/.deepagents").exists()
    assert not (tmp_path / "home/.cache").exists()


def test_install_script_retries_transient_pypi_failure(tmp_path: Path) -> None:
    """Two transient metadata failures are retried before updating."""
    proc, args_path = _invoke(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
        curl_failures_before_success=2,
    )

    assert proc.returncode == 0
    assert (tmp_path / "curl-attempts.txt").read_text().strip() == "3"
    assert "Could not determine the latest version" not in proc.stderr
    assert args_path.read_text().splitlines()[:3] == ["tool", "install", "-U"]


def test_install_script_uv_output_uses_cache_log_not_predictable_tmp(
    tmp_path: Path,
) -> None:
    """Non-root installs stream uv output to the cache log, not a `/tmp` file.

    The live-log path means an unprivileged run never calls `mktemp` for uv's
    output, so a broken `mktemp` no longer aborts it. The property the original
    test protected — never fall back to a predictable `/tmp` name — still
    holds; the fallback is now the per-user cache log.
    """
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": "live log output"},
        installed_version="0.1.0",
        latest_version="0.2.0",
        mktemp_fails=True,
    )

    assert proc.returncode == 0
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        "live log output\n"
    )
    script = SCRIPT.read_text(encoding="utf-8")
    assert "/tmp/deepagents-install.$$" not in script
    assert "/tmp/deepagents-ripgrep-setup.$$" not in script


def test_install_script_live_log_is_not_world_readable(tmp_path: Path) -> None:
    """No group or world access on the log — uv's stderr can carry index URLs.

    See `setup_live_install_log` for why the explicit `umask 077` is needed.
    """
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": "secret index url"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    log_path = tmp_path / "home/.cache/deepagents-code/install.log"
    assert stat.S_IMODE(log_path.stat().st_mode) & 0o077 == 0


def test_install_script_symlinked_log_warns_and_offers_no_pointer(
    tmp_path: Path,
) -> None:
    """A symlink at the log path is durable state the user can act on.

    It disables the log feature entirely — no live tail *and* no `Full log:`
    pointer — with no fallback to the staged publish, so staying silent would
    leave logging off on every future run with no way to discover why.
    """
    log_dir = tmp_path / "home/.cache/deepagents-code"
    log_dir.mkdir(parents=True)
    target = tmp_path / "elsewhere.log"
    target.write_text("pre-existing target\n")
    (log_dir / "install.log").symlink_to(target)

    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": "uv output"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "is a symlink" in proc.stderr
    assert "Remove it to re-enable install logging." in proc.stderr
    assert "Update log:" not in proc.stdout
    assert "Full log:" not in proc.stdout
    assert target.read_text() == "pre-existing target\n"


def test_install_script_warns_when_live_log_ends_up_empty(tmp_path: Path) -> None:
    """Truncating the prior log and writing nothing must not be silent.

    A live run replaces the previous log before uv starts. If uv then writes
    nothing, both `Full log:` pointers stay quiet (they require `-s`), so
    without this warning the loss of yesterday's diagnostics is invisible.
    """
    log_path = tmp_path / "home/.cache/deepagents-code/install.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("yesterday's traceback\n")

    proc, _ = _invoke(tmp_path, {}, installed_version="0.1.0", latest_version="0.2.0")

    assert proc.returncode == 0
    assert log_path.read_text() == ""
    assert "uv wrote no output" in proc.stderr
    assert "Full log:" not in proc.stdout


@pytest.mark.parametrize("stdin_is_tty", [True, False])
def test_prompt_yn_eof_on_an_open_terminal_declines(
    tmp_path: Path, *, stdin_is_tty: bool
) -> None:
    """Ctrl-D at the prompt is a human declining (1), not an unaskable prompt (2).

    Both branches of `prompt_yn` must agree: the terminal opened, so somebody
    was there to ask, and the printed default is N. Returning 2 here would make
    callers proceed — installing an update and removing extras that the user
    just refused. Parametrised over stdin because the `/dev/tty` branch (the
    `curl | bash` shape) is the one whose callers act on the distinction, and
    it was previously reachable only through stubs.
    """
    output = _run_prompt_yn_on_tty(tmp_path, b"\x04", stdin_is_tty=stdin_is_tty)

    assert "[rc=1]" in output
    assert "WARN No answer — declining prompt." in output


@pytest.mark.parametrize("stdin_is_tty", [True, False])
def test_prompt_yn_keeps_an_answer_submitted_without_a_newline(
    tmp_path: Path, *, stdin_is_tty: bool
) -> None:
    """An answer terminated by Ctrl-D rather than Enter is still the user's answer.

    `read` reports failure on an unterminated final line but has already
    assigned what it read, so keying on the status alone discards a real "y"
    or "n". Accepting (rc 0) is only reachable if the assigned reply survived
    the failed status, which is what separates this from the EOF case above.
    """
    output = _run_prompt_yn_on_tty(tmp_path, b"y\x04\x04", stdin_is_tty=stdin_is_tty)

    assert "[rc=0]" in output
    assert "declining prompt" not in output


def test_install_script_attached_tty_eof_preserves_extras(tmp_path: Path) -> None:
    """A real Ctrl-D at the extras prompt aborts before uv rebuilds the environment.

    Two prompts fire here, so the update is accepted first and the extras-loss
    interrupt is answered with EOF. Declining that second prompt must leave the
    install — and its extras — untouched.
    """
    _write_uv_receipt(tmp_path / "tools", ["anthropic", "openai"])

    code, output, args_path = _invoke_interactive(
        tmp_path,
        {},
        answer=["y", _CTRL_D],
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert code == 0
    assert "No answer — declining prompt." in output
    assert "Aborted. deepagents-code was left unchanged." in output
    assert not args_path.exists()


def test_can_prompt_false_without_usable_tty(tmp_path: Path) -> None:
    """No openable `/dev/tty` yields false even when `IS_INTERACTIVE` is true.

    Guards the load-bearing line: `can_prompt` must actually open `/dev/tty`
    rather than trusting `IS_INTERACTIVE` (which only access-checks the device).
    A regression that returned 0 right after the `IS_INTERACTIVE` check would
    wrongly report the unanswerable cron/systemd/CI case as promptable.
    """
    assert _eval_can_prompt(tmp_path, is_interactive=True, stdin_is_tty=False) is False


def _eval_version_at_least(tmp_path: Path, have: str, want: str) -> bool:
    """Run the real `version_at_least` from install.sh, reporting its verdict."""
    script = tmp_path / "version_harness.sh"
    script.write_text(
        f"{_extract_shell_function('version_at_least')}\n"
        f"if version_at_least {have!r} {want!r}; then echo YES; else echo NO; fi\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip() == "YES"


@pytest.mark.parametrize(
    ("have", "want", "expected"),
    [
        ("14.1.1", "12.0.0", True),  # current managed pin clears the floor
        ("12.0.0", "12.0.0", True),  # boundary: equal is acceptable
        ("12.0.1", "12.0.0", True),  # patch above
        ("13.0.0", "12.0.0", True),  # minor above
        ("11.9.9", "12.0.0", False),  # minor below
        ("0.10.0", "12.0.0", False),  # ancient distro package
        ("", "12.0.0", False),  # empty is never acceptable
        ("abc", "12.0.0", False),  # unparseable is never acceptable
        # `want` is validated too: the footer's upgrade check passes package
        # versions here, and a PEP 440 prerelease reaching the `-ge` compare
        # leaks "integer expression expected" to stderr.
        ("12.0.0", "", False),
        ("0.1.0", "0.1.0rc1", False),
    ],
)
def test_version_at_least(tmp_path: Path, have: str, want: str, expected: bool) -> None:
    """Dotted-version comparison gates the system ripgrep floor."""
    assert _eval_version_at_least(tmp_path, have, want) is expected


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


def test_install_script_same_version_with_dependency_updates_says_dependencies_updated(
    tmp_path: Path,
) -> None:
    """Unchanged app version + a uv dependency diff reports the deps were updated.

    The fake `dcode -v` reports the same version before and after install, so
    `PRE_VERSION == NEW_VERSION` and the same-version branch fires; the `± pkg==`
    diff in stderr must steer it away from the flat "already up to date" message.
    Also verifies the raw uv diff is persisted to the cache install log, which
    the shared `Full log:` line points the user at.
    """
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _DEPENDENCY_UPDATE_DIFF},
        installed_version="0.1.8",
        latest_version="0.1.20",
    )

    assert proc.returncode == 0
    assert (
        "deepagents-code 0.1.8 was already up to date; dependencies were updated."
    ) in proc.stdout
    assert "Full log: ~/.cache/deepagents-code/install.log" in proc.stdout
    assert "deepagents-code 0.1.8 already up to date" not in proc.stdout
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_DEPENDENCY_UPDATE_DIFF}\n"
    )
    assert "✔ Dependencies updated. Run: dcode" in proc.stdout
    assert "✔ Already installed. Run: dcode" not in proc.stdout


def test_install_script_dependency_update_without_writable_log_omits_log_pointer(
    tmp_path: Path,
) -> None:
    """When the log dir can't be created, no `Full log:` pointer is printed.

    Points `XDG_CACHE_HOME` under a regular file so `mkdir -p` fails, leaving
    `INSTALL_LOG` empty. The dependency-update message must still fire, just
    without a pointer to a log that was never written — guards against the
    pointer being printed unconditionally.
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
    assert "Full log:" not in proc.stdout
    assert not (blocker / "cache").exists()


def test_install_script_live_log_create_failure_keeps_prior_log(
    tmp_path: Path,
) -> None:
    """A failed create must not cost the user yesterday's diagnostics.

    `setup_live_install_log` creates `install.log.new` and renames it into
    place, so the previous log survives until this run holds a writable file.
    The older destroy-then-create shape removed it first, so a create that
    then failed left the user with the old log gone and nothing in its place —
    silently, since the create error was discarded.

    A directory planted at the pending path fails the create while leaving the
    log dir writable, which is what separates the two shapes: the older one
    would have unlinked `install.log` before discovering it could not create.
    `mktemp_fails` closes the staged fallback so nothing writes a replacement
    log, leaving the prior one as the only thing that could still be there.
    """
    log_dir = tmp_path / "home/.cache/deepagents-code"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "install.log"
    log_path.write_text("yesterday's traceback\n")
    (log_dir / "install.log.new").mkdir()

    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": "this run's output"},
        installed_version="0.1.0",
        latest_version="0.2.0",
        mktemp_fails=True,
    )

    assert proc.returncode != 0
    assert log_path.read_text() == "yesterday's traceback\n"
    assert "continuing without live logging" in proc.stderr
    assert "Update log:" not in proc.stdout


def test_install_script_interrupt_reports_replaced_live_log(tmp_path: Path) -> None:
    """Ctrl-C between the log swap and uv's first byte must not lose it silently.

    The post-install empty-log warning sits after uv returns, so the interrupt
    handler never reaches it. Without a warning there, a user who cancels a
    slow download is told only "Installation interrupted." while yesterday's
    traceback is already gone.
    """
    harness = tmp_path / "interrupt_live_log.sh"
    log_path = tmp_path / "install.log"
    log_path.write_text("")
    harness.write_text(
        "set -uo pipefail\n"
        'log_warn() { printf "%s\\n" "$*" >&2; }\n'
        f"{_extract_shell_function('warn_live_log_replaced')}\n"
        "UV_LIVE_LOG=true\n"
        f"INSTALL_LOG={str(log_path)!r}\n"
        "INSTALL_LOG_DISPLAY='~/.cache/deepagents-code/install.log'\n"
        "warn_live_log_replaced\n"
        # The flag makes the notice fire once, so a run that already warned
        # after uv exited does not repeat itself from the EXIT trap.
        "warn_live_log_replaced\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )

    assert proc.returncode == 0
    assert proc.stderr.count("any previous install log was replaced") == 1


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
    assert "Full log:" not in proc.stdout
    assert not (target / "install.log").exists()


def _write_uv_receipt(
    tools: Path,
    extras: list[str] | None,
    *,
    with_packages: dict[str, list[str]] | None = None,
) -> Path:
    """Stage a uv tool receipt recording the extras the install was built with.

    The install script reads `uv-receipt.toml` to detect extras that a bare
    re-run would drop; the fake `uv tool dir` points at `tmp_path / "tools"`,
    so the receipt belongs under `tools/deepagents-code/`. `parents=True`
    creates `tools` as a side effect, which is why `_write_fake_tools` tolerates
    a pre-existing `tools` — the two helpers may run in either order.

    `extras=None` omits the `extras` key entirely, which is what uv actually
    writes for an install that has none — the shape every ordinary user has.
    Passing `[]` writes an explicit `extras = []` instead; both are exercised,
    since a parser can easily handle one and not the other.

    `with_packages` maps supplemental `uv tool install --with` package names to
    their own extras. uv records these in the *same* `requirements` array as the
    tool itself, so a receipt with them is what distinguishes a parser scoped to
    the `deepagents-code` entry from one that grabs any `extras = [...]`.

    The `entrypoints` array is always emitted, and always includes the console
    script literally named `deepagents-code` that this package declares. That
    inline table's `name` matches the same pattern the requirements entry does
    but never carries extras, so omitting it from the fixture would hide a
    parser that matches it instead of the real requirement.

    Returns the receipt path so tests can mangle its permissions or contents.
    """
    receipt_dir = tools / "deepagents-code"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if extras is None:
        entries = ['{ name = "deepagents-code", specifier = "==0.1.0" }']
    else:
        quoted = ", ".join(f'"{extra}"' for extra in extras)
        entry = f'{{ name = "deepagents-code", extras = [{quoted}], '
        entries = [entry + 'specifier = "==0.1.0" }']
    for name, pkg_extras in (with_packages or {}).items():
        pkg_quoted = ", ".join(f'"{extra}"' for extra in pkg_extras)
        entries.append(f'{{ name = "{name}", extras = [{pkg_quoted}] }}')
    receipt = receipt_dir / "uv-receipt.toml"
    receipt.write_text(
        "[tool]\n"
        "requirements = [" + ", ".join(entries) + "]\n"
        'python = "3.13"\n'
        "entrypoints = [\n"
        '    { name = "dcode", install-path = "/h/bin/dcode",'
        ' from = "deepagents-code" },\n'
        '    { name = "deepagents-code",'
        ' install-path = "/h/bin/deepagents-code", from = "deepagents-code" },\n'
        "]\n"
    )
    return receipt


def test_install_script_upgrade_prints_full_log_pointer(tmp_path: Path) -> None:
    """A successful upgrade surfaces the persistent uv log, not just failures."""
    proc, _ = _invoke(
        tmp_path,
        {"FAKE_UV_INSTALL_STDERR": _UPGRADE_DIFF},
        installed_version="0.1.18",
        latest_version="0.1.19",
    )

    assert proc.returncode == 0
    assert "Full log: ~/.cache/deepagents-code/install.log" in proc.stdout
    assert "Update log: tail -f ~/.cache/deepagents-code/install.log" in proc.stdout


def test_install_script_receipt_extras_with_metacharacters_warns(
    tmp_path: Path,
) -> None:
    """Shell metacharacters in receipt extras must not reach the printed hint.

    The extras value is echoed inside a double-quoted `DEEPAGENTS_CODE_EXTRAS`
    suggestion the user may paste; a tampered receipt could otherwise smuggle
    `$(...)` into that paste. Anything outside `[A-Za-z0-9_,.-]` is treated as
    an unparseable receipt — the run warns and omits the paste-ready hint.
    """
    # Sentinel under tmp_path, not a fixed system path: a shared target would
    # make this test fail (or pass) for reasons that have nothing to do with
    # the script, and it would leave litter behind.
    sentinel = tmp_path / "pwned"
    payload = f"$(touch {sentinel})"
    receipt = _write_uv_receipt(tmp_path / "tools", [payload])
    # Keep the file plausible TOML but with the payload as the extra name.
    receipt.write_text(
        '[tool]\nrequirements = [{ name = "deepagents-code",'
        f' extras = ["{payload}"], specifier = "==0.1.0" }}]\n'
    )

    proc, _ = _invoke(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "Could not read" in proc.stderr
    assert "DEEPAGENTS_CODE_EXTRAS=" not in proc.stderr
    assert "$(touch" not in proc.stderr
    assert not sentinel.exists()


def test_install_script_warns_when_receipt_is_unparseable(tmp_path: Path) -> None:
    """A receipt uv wrapped across lines warns instead of degrading to silence.

    uv currently keeps each requirement's inline table on one line. If a future
    formatter wraps it, the extras are unreadable — but the rebuild still drops
    them, so "couldn't tell" must not be reported the same way as "none".

    The `entrypoints` array is the trap here, and it is why this receipt is
    written out in full rather than mangling only the requirements line. uv
    records one entry per console script, and this package declares one named
    `deepagents-code` — an inline table that matches the requirement pattern on
    a single line but carries no extras. A parser that isn't scoped to the
    `requirements` assignment matches it once the real requirement wraps, reads
    "no extras" off it, and drops the user's packages in total silence.
    """
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    receipt.write_text(
        "[tool]\n"
        'requirements = [\n    { name = "deepagents-code", extras = [\n'
        '        "anthropic",\n    ] },\n]\n'
        'python = "3.13"\n'
        "entrypoints = [\n"
        '    { name = "deepagents-code",'
        ' install-path = "/h/bin/deepagents-code", from = "deepagents-code" },\n'
        "]\n"
    )

    proc, _ = _invoke(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "Could not read" in proc.stderr
    assert "re-run with the same value" in proc.stderr


def test_install_script_warns_when_upgrade_drops_receipt_extras(tmp_path: Path) -> None:
    """A bare re-run warns when the existing install was built with extras.

    Without `DEEPAGENTS_CODE_EXTRAS`, `uv tool install -U deepagents-code`
    rebuilds the environment against the bare package and silently removes the
    extras' packages. The script must read the extras off uv's receipt and warn
    (with the re-run hint) before that rebuild happens.
    """
    _write_uv_receipt(tmp_path / "tools", ["anthropic", "openai"])

    proc, _ = _invoke(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert (
        "This install has extras that a bare re-run will remove: anthropic,openai"
        in proc.stderr
    )
    assert (
        'To keep them, re-run with: DEEPAGENTS_CODE_EXTRAS="anthropic,openai"'
        in proc.stderr
    )


def test_install_script_warns_when_receipt_read_races(tmp_path: Path) -> None:
    """A receipt removed after preflight checks does not abort the installer."""
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    env = _env(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )
    sed = tmp_path / "bin" / "sed"
    sed.write_text(
        f"""#!/usr/bin/env bash
for argument in "$@"; do
  if [ "$argument" = {str(receipt)!r} ]; then
    exit 1
  fi
done
exec /usr/bin/sed "$@"
"""
    )
    _make_executable(sed)

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
    assert "Could not read" in proc.stderr
    assert (tmp_path / "uv-args.txt").is_file()


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root bypasses directory permissions")
def test_install_script_warns_when_tool_dir_is_unsearchable(tmp_path: Path) -> None:
    """A tool dir left root-owned and 0700 by a prior sudo run must warn.

    This is the common shape of the sudo-then-user case (a root umask of 077
    produces it), and it defeats every test on the receipt file itself: `-f`
    reports false through an unsearchable parent, which is indistinguishable
    from an absent receipt and, without a directory check, from no extras.
    """
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    install_dir = receipt.parent
    install_dir.chmod(0o000)
    try:
        proc, _ = _invoke(
            tmp_path,
            {},
            installed_version="0.1.0",
            latest_version="0.2.0",
        )
    finally:
        install_dir.chmod(0o755)

    assert proc.returncode == 0
    assert "Could not read" in proc.stderr
    assert "re-run with the same value" in proc.stderr


def test_install_script_extras_without_tty_proceeds_with_install(
    tmp_path: Path,
) -> None:
    """With no TTY the extras warning is printed and the install still runs.

    `prompt_yn` returns non-zero whenever it cannot ask, so a guard that lost
    its `can_prompt` half would read "nobody can answer" as "the user said no"
    and turn every CI job, Dockerfile, and piped `curl | bash` upgrade of an
    extras install into a silent no-op that still exits 0. Asserting the exit
    code alone cannot see that: abort and proceed both return 0. Assert uv was
    actually invoked.
    """
    _write_uv_receipt(tmp_path / "tools", ["anthropic"])

    proc, args_path = _invoke(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "extras that a bare re-run will remove: anthropic" in proc.stderr
    assert "Aborted." not in proc.stdout
    assert args_path.read_text().splitlines()[:3] == ["tool", "install", "-U"]


def test_install_script_warns_when_receipt_is_symlinked(tmp_path: Path) -> None:
    """A symlinked receipt is refused, but the refusal is announced.

    Reading through the symlink is the security problem; staying quiet about it
    is the usability one — the run still rebuilds and still drops the extras.
    """
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    target = tmp_path / "elsewhere-receipt.toml"
    target.write_text(receipt.read_text())
    receipt.unlink()
    receipt.symlink_to(target)

    proc, _ = _invoke(
        tmp_path,
        {},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "Could not read" in proc.stderr
    assert "extras that a bare re-run will remove" not in proc.stderr


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root bypasses file permissions")
def test_install_script_warns_when_receipt_is_unreadable(tmp_path: Path) -> None:
    """An unreadable receipt (e.g. written by a prior `sudo` run) warns."""
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    receipt.chmod(0o000)

    try:
        proc, _ = _invoke(
            tmp_path,
            {},
            installed_version="0.1.0",
            latest_version="0.2.0",
        )
    finally:
        receipt.chmod(0o644)

    assert proc.returncode == 0
    assert "Could not read" in proc.stderr


def test_install_script_extras_interrupt_decline_aborts_before_uv(
    tmp_path: Path,
) -> None:
    """Answering 'n' to the extras prompt exits before uv runs.

    Two prompts fire on this path — the update prompt, then the extras-loss
    interrupt — so both answers are fed. Declining the second must abort
    cleanly (exit 0, no uv invocation, no environment rebuild).
    """
    _write_uv_receipt(tmp_path / "tools", ["anthropic", "openai"])

    code, output, args_path = _invoke_interactive(
        tmp_path,
        {},
        answer=["y", "n"],
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert code == 0
    assert "Continue anyway and remove them?" in output
    assert "Aborted. deepagents-code was left unchanged." in output
    assert not args_path.exists()


def test_install_script_extras_assume_yes_skips_interrupt_with_tty(
    tmp_path: Path,
) -> None:
    """`DEEPAGENTS_CODE_YES=1` skips the extras prompt even on an attached TTY.

    The non-pty sibling below can't prove this: without a TTY `can_prompt` is
    already false, so it passes whether or not the `ASSUME_YES` half of the
    guard exists. A terminal *and* a pre-answered yes (tmux, `docker run -t`, a
    CI runner that allocates a pty) is the combination where dropping that half
    would hang the run forever. No answers are fed, so a prompt would block
    until `_invoke_interactive`'s timeout.
    """
    _write_uv_receipt(tmp_path / "tools", ["anthropic", "openai"])

    code, output, args_path = _invoke_interactive(
        tmp_path,
        {"DEEPAGENTS_CODE_YES": "1"},
        answer=[],
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert code == 0
    assert "extras that a bare re-run will remove" in output
    assert "Continue anyway and remove them?" not in output
    assert args_path.read_text().splitlines()[:3] == ["tool", "install", "-U"]


def test_install_script_extras_assume_yes_skips_interrupt(tmp_path: Path) -> None:
    """`DEEPAGENTS_CODE_YES=1` proceeds past the extras interrupt unattended."""
    _write_uv_receipt(tmp_path / "tools", ["anthropic", "openai"])

    proc, args_path = _invoke(
        tmp_path,
        {"DEEPAGENTS_CODE_YES": "1"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0
    assert "extras that a bare re-run will remove" in proc.stderr
    assert "Aborted. deepagents-code was left unchanged." not in proc.stderr
    assert args_path.read_text().splitlines()[:3] == ["tool", "install", "-U"]


def test_install_script_unreadable_receipt_interrupt_decline_aborts_before_uv(
    tmp_path: Path,
) -> None:
    """Answering 'n' to the unreadable-receipt prompt exits before uv runs.

    An unreadable receipt can still hide extras, so the unknown-extras case must
    offer the same abort prompt as the known-extras one: the user is told their
    extras may be removed and gets the chance to stop before the rebuild drops
    them. Two prompts fire on this path — the update prompt, then the
    unreadable-receipt interrupt — so both answers are fed.
    """
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    receipt.chmod(0o000)

    try:
        code, output, args_path = _invoke_interactive(
            tmp_path,
            {},
            answer=["y", "n"],
            installed_version="0.1.0",
            latest_version="0.2.0",
        )
    finally:
        receipt.chmod(0o644)

    assert code == 0
    assert "Continue anyway?" in output
    assert "Aborted. deepagents-code was left unchanged." in output
    assert not args_path.exists()


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root bypasses file permissions")
def test_install_script_unreadable_receipt_assume_yes_skips_interrupt(
    tmp_path: Path,
) -> None:
    """`DEEPAGENTS_CODE_YES=1` proceeds past the unreadable-receipt prompt."""
    receipt = _write_uv_receipt(tmp_path / "tools", ["anthropic"])
    receipt.chmod(0o000)

    try:
        proc, args_path = _invoke(
            tmp_path,
            {"DEEPAGENTS_CODE_YES": "1"},
            installed_version="0.1.0",
            latest_version="0.2.0",
        )
    finally:
        receipt.chmod(0o644)

    assert proc.returncode == 0
    assert "Could not read" in proc.stderr
    assert "Aborted. deepagents-code was left unchanged." not in proc.stderr
    assert args_path.read_text().splitlines()[:3] == ["tool", "install", "-U"]


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
    assert "Full log:" not in proc.stdout
    assert target.read_text() == "keep me\n"


def _run_copy_install_log(
    tmp_path: Path,
    *,
    race_hook: str = "",
    log_dir: Path | None = None,
    live: bool = False,
) -> tuple[int, Path, Path]:
    """Run the real `copy_install_log` from `install.sh` in isolation.

    Whole-script runs can only ever fail the publish step, and never observe
    the staged file mid-flight. Driving the function directly lets a test stand
    in the race window instead.

    `race_hook` is arbitrary shell evaluated before the call — in practice a
    function override (`cp`, `mv`, `rm`, `mktemp`, `id`, `cat`) that acts
    inside the window. An override shadowing a command the publish depends on
    must delegate with `command …` if the publish is expected to get that far.

    Every exit path of `copy_install_log` is supposed to remove its staging
    directory, so this asserts that centrally: the staged file holds uv's full
    captured stderr, and an orphan is both litter and a disclosure that
    repeated failures would accumulate. Staging lives in `/tmp` (see the
    function's own comment), so the check is a before/after diff of that
    directory rather than a glob of the log dir — it assumes the suite is not
    running copies of itself concurrently.

    `live=True` drives the live-log branch instead: uv streamed straight to
    `INSTALL_LOG`, so there is nothing to stage and the function only
    re-validates the path it is about to advertise.

    Returns the function's exit status, the log dir, and the publish path.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    install_log_dir = log_dir if log_dir is not None else home / "cache"
    install_log_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "uv-stderr.txt"
    source.write_text("captured uv stderr\n")

    harness = tmp_path / "copy_install_log_harness.sh"
    harness.write_text(
        "set -uo pipefail\n"
        "TEMP_FILES=()\n"
        "TEMP_DIRS=()\n"
        'register_temp() { TEMP_FILES+=("$1"); }\n'
        'register_temp_dir() { TEMP_DIRS+=("$1"); }\n'
        'log_warn() { printf "%s\\n" "$*" >&2; }\n'
        f"{_extract_shell_function('path_is_under_home')}\n"
        f"{_extract_shell_function('copy_install_log')}\n"
        f"HOME={str(home)!r}\n"
        f"install_log_dir={str(install_log_dir)!r}\n"
        # `${install_log_dir}` below is shell, not a Python f-string.
        'INSTALL_LOG="${install_log_dir}/install.log"\n'  # noqa: RUF027
        'INSTALL_LOG_DISPLAY="$INSTALL_LOG"\n'
        f"UV_LIVE_LOG={'true' if live else 'false'}\n"
        f"uv_stderr={str(source)!r}\n"
        f"{race_hook}\n"
        "copy_install_log\n"
        'printf "rc=%s\\n" "$?"\n'
    )
    before = set(_STAGE_ROOT.glob(_STAGE_GLOB))
    proc = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
    )
    leaked = set(_STAGE_ROOT.glob(_STAGE_GLOB)) - before
    assert leaked == set(), f"copy_install_log left staging behind: {sorted(leaked)}"
    match = re.search(r"rc=(\d+)", proc.stdout)
    assert match is not None, f"harness produced no status: {proc.stdout!r}"
    return int(match.group(1)), install_log_dir, install_log_dir / "install.log"


def test_copy_install_log_live_rejects_symlink_planted_after_uv(
    tmp_path: Path,
) -> None:
    """A symlink swapped in after uv exited must not be advertised.

    The descriptor pinned the inode uv wrote to, so only the *name* is at
    risk: a process able to write the cache dir can replace `install.log`
    between uv exiting and the `Full log:` pointer being printed. Returning 0
    here would send the user to a file of someone else's choosing; returning 1
    would drop the pointer but say nothing about real data loss, so this
    reports 2 — the code the caller warns on.
    """
    log_dir = tmp_path / "home/cache"
    log_dir.mkdir(parents=True)
    target = tmp_path / "attacker.log"
    target.write_text("attacker content\n")
    (log_dir / "install.log").symlink_to(target)

    rc, _log_dir, _published = _run_copy_install_log(tmp_path, live=True)

    assert rc == 2
    assert target.read_text() == "attacker content\n"


def test_copy_install_log_stages_outside_user_writable_log_dir(
    tmp_path: Path,
) -> None:
    """The copied log is staged in sticky `/tmp`, not below the cache path.

    A root installer cannot safely reopen a staging directory whose parent is
    user-writable: that user can rename the directory and replace it with a
    symlink before `cp` runs. Pin the `mktemp` template so the staging parent
    remains independent of the cache directory.
    """
    mktemp_args = tmp_path / "mktemp-args.txt"
    rc, _log_dir, published = _run_copy_install_log(
        tmp_path,
        race_hook=(
            "mktemp() {\n"
            f"  printf '%s\\n' \"$@\" > {str(mktemp_args)!r}\n"
            '  command mktemp "$@"\n'
            "}\n"
        ),
    )

    assert rc == 0
    template = mktemp_args.read_text().splitlines()
    assert template[0] == "-d"
    # The invariant is the location, not the exact template: staging must not
    # sit under the log dir, whose parent the target user may control.
    assert not template[1].startswith(str(tmp_path))
    assert template[1].startswith(f"{_STAGE_ROOT}/")
    assert published.read_text() == "captured uv stderr\n"


def test_copy_install_log_reports_a_failure_to_create_staging(tmp_path: Path) -> None:
    """`mktemp -d` failing is operational (2), so the caller warns about it.

    A full or read-only `/tmp` is something the user can act on, and it leaves
    the run with no log at all — exactly the case the rc-2 warning exists for.
    """
    rc, _log_dir, published = _run_copy_install_log(
        tmp_path, race_hook="mktemp() { return 1; }\n"
    )

    assert rc == 2
    assert not published.exists()


def test_copy_install_log_replaces_symlink_planted_during_the_race(
    tmp_path: Path,
) -> None:
    """A symlink planted at the publish path is replaced, never written through.

    The `-L` guard runs before staging, so it cannot cover a link planted after
    it. This runs unprivileged, where publication is a `mv`: that replaces the
    symlink's directory entry rather than writing through it, so the
    attacker's target remains untouched. (The privileged branch reaches the
    same result through `rm` plus a noclobber create; it is covered by the
    tests below that stub `id`.)
    """
    outside = tmp_path / "outside.txt"
    outside.write_text("do not clobber\n")
    rc, _log_dir, published = _run_copy_install_log(
        tmp_path,
        race_hook=(
            f'cp() {{\n  command cp "$@"\n  ln -s {str(outside)!r} "$INSTALL_LOG"\n}}\n'
        ),
    )

    assert rc == 0
    assert outside.read_text() == "do not clobber\n"
    assert not published.is_symlink()
    assert published.read_text() == "captured uv stderr\n"


def test_copy_install_log_refuses_publish_when_log_dir_is_swapped(
    tmp_path: Path,
) -> None:
    """Swapping the log dir for a symlink mid-run must not publish through it."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rc, log_dir, _published = _run_copy_install_log(
        tmp_path,
        race_hook=(
            "cp() {\n"
            '  command cp "$@"\n'
            '  mv "$install_log_dir" "${install_log_dir}.moved"\n'
            f'  ln -s {str(elsewhere)!r} "$install_log_dir"\n'
            "}\n"
        ),
    )

    assert rc != 0
    assert not (elsewhere / "install.log").exists()
    assert log_dir.is_symlink()


def test_copy_install_log_pins_parent_during_privileged_publication(
    tmp_path: Path,
) -> None:
    """A parent swap after validation cannot redirect root's rm or create."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rc, log_dir, _published = _run_copy_install_log(
        tmp_path,
        race_hook=(
            'id() { printf "0\\n"; }\n'
            "rm() {\n"
            '  if [ "${1:-}" = "-f" ] && [ "${2:-}" = "install.log" ]; then\n'
            '    mv "$install_log_dir" "${install_log_dir}.moved"\n'
            f'    command ln -s {str(elsewhere)!r} "$install_log_dir"\n'
            "  fi\n"
            '  command rm "$@"\n'
            "}\n"
        ),
    )

    assert rc == 0
    assert log_dir.is_symlink()
    assert not (elsewhere / "install.log").exists()
    assert (log_dir.with_name(f"{log_dir.name}.moved") / "install.log").read_text() == (
        "captured uv stderr\n"
    )


def test_copy_install_log_rejects_directory_created_during_publication(
    tmp_path: Path,
) -> None:
    """A directory created after removal makes privileged publication fail.

    `rm -f` does not remove directories. The noclobber redirection must reject
    one that appears after removal rather than treating it as a destination and
    publishing `install.log/install.log`.
    """
    rc, _log_dir, published = _run_copy_install_log(
        tmp_path,
        race_hook=(
            'id() { printf "0\\n"; }\n'
            "rm() {\n"
            '  if [ "${1:-}" = "-f" ] && [ "${2:-}" = "install.log" ]; then\n'
            '    command mkdir "install.log"\n'
            "  fi\n"
            '  command rm "$@"\n'
            "}\n"
        ),
    )

    assert rc == 2
    assert published.is_dir()
    assert not (published / "install.log").exists()


def test_copy_install_log_rejects_directory_created_before_nonroot_move(
    tmp_path: Path,
) -> None:
    """A non-root `mv` into a raced directory must not report success.

    `mv` moves the file *into* a directory destination and exits 0, so the
    status alone would read as a published log. The post-move check catches
    that, and the captured stderr must not be left sitting inside the
    attacker's directory afterwards.
    """
    rc, _log_dir, published = _run_copy_install_log(
        tmp_path,
        race_hook=('mv() {\n  command mkdir "$INSTALL_LOG"\n  command mv "$@"\n}\n'),
    )

    assert rc != 0
    assert published.is_dir()
    assert not (published / "install.log").exists()


def test_copy_install_log_cleans_up_staged_file_when_publish_fails(
    tmp_path: Path,
) -> None:
    """A failed publish leaves neither a half-written target nor its staging.

    The staging half is asserted by `_run_copy_install_log` for every case;
    this one drives the branch where cleanup has the most to do — the target
    was already created before the write failed, so both files exist when the
    failure path runs.
    """
    # Fail the root-safe publication copy after noclobber creates the target,
    # leaving both files in place for the cleanup to find.
    rc, _log_dir, published = _run_copy_install_log(
        tmp_path, race_hook='id() { printf "0\\n"; }\ncat() { return 1; }\n'
    )

    assert rc == 2
    assert not published.exists()


def test_copy_install_log_reports_operational_failure_distinctly(
    tmp_path: Path,
) -> None:
    """Failures the user can act on return 2; rejected paths return 1.

    The caller only warns on 2. Collapsing the two would either go silent on a
    full disk and a root-owned cache dir — leaving a `curl | bash` user with no
    log and no reason why — or cry wolf on every hostile path it correctly
    refused.
    """
    # A read-only log dir: staging succeeds in /tmp, but publishing into the
    # log dir fails. Operational → 2.
    readonly_dir = tmp_path / "home" / "readonly"
    readonly_dir.mkdir(parents=True)
    readonly_dir.chmod(0o500)
    try:
        rc_operational, _, _ = _run_copy_install_log(tmp_path, log_dir=readonly_dir)
    finally:
        readonly_dir.chmod(0o755)

    # A symlinked log dir is refused outright, and the user cannot act on it.
    rc_rejected, _, _ = _run_copy_install_log(
        tmp_path,
        race_hook=(
            "cp() {\n"
            '  command cp "$@"\n'
            '  rm -f "$1" "$2"\n'
            '  rmdir "$install_log_dir"\n'
            f'  ln -s {str(tmp_path / "swap")!r} "$install_log_dir"\n'
            "}\n"
        ),
    )

    assert rc_operational == 2
    assert rc_rejected == 1


def test_install_script_log_path_outside_home_stays_absolute(tmp_path: Path) -> None:
    """A log path outside `$HOME` is shown verbatim, not tilde-collapsed.

    The `~` collapse only fires for paths under `$HOME`; an `XDG_CACHE_HOME`
    elsewhere must surface the absolute path in the `Full log:` pointer.
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
    assert f"Full log: {expected_log}" in proc.stdout
    assert "Full log: ~/" not in proc.stdout
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
    assert "Full log: ~/.cache/deepagents-code/install.log" in proc.stderr
    assert (tmp_path / "home/.cache/deepagents-code/install.log").read_text() == (
        f"{_DEPENDENCY_UPDATE_DIFF}\n"
    )


# A PID above every platform's pid_max, so `kill -0` always reports it dead.
_DEAD_PID = "2147483647"


def _eval_install_lock_is_stale(
    tmp_path: Path,
    *,
    pid: str | None,
    started_at: str | None,
    stale_after: int = 600,
    make_dir: bool = True,
) -> bool:
    """Run the real `install_lock_is_stale` against a synthetic lock directory.

    Returns True when the function reports the lock as stale (exit 0). Threshold
    extremes (0 / huge) let the age comparison be exercised without depending on
    wall-clock timing.
    """
    lock_dir = tmp_path / "install.lock.d"
    if make_dir:
        lock_dir.mkdir()
        if pid is not None:
            (lock_dir / "pid").write_text(f"{pid}\n")
        if started_at is not None:
            (lock_dir / "started_at").write_text(f"{started_at}\n")

    script = tmp_path / "stale_harness.sh"
    script.write_text(
        f"INSTALL_LOCK_DIR={str(lock_dir)!r}\n"
        f"INSTALL_LOCK_STALE_AFTER_SECS={stale_after}\n"
        f"{_extract_shell_function('lock_dir_mtime')}\n"
        f"{_extract_shell_function('install_lock_identity')}\n"
        f"{_extract_shell_function('install_lock_is_stale')}\n"
        "install_lock_is_stale\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def test_install_lock_live_owner_is_never_stale(tmp_path: Path) -> None:
    """A lock whose PID is still running is never reclaimed, regardless of age."""
    assert not _eval_install_lock_is_stale(
        tmp_path, pid=str(os.getpid()), started_at="1"
    )


def test_install_lock_dead_owner_old_timestamp_is_stale(tmp_path: Path) -> None:
    """A dead owner past the staleness window is reclaimable."""
    assert _eval_install_lock_is_stale(tmp_path, pid=_DEAD_PID, started_at="1")


def test_install_lock_dead_owner_within_window_is_not_stale(tmp_path: Path) -> None:
    """A dead owner still inside the staleness window is left alone."""
    # Threshold must exceed the current epoch (~1.8e9) so `now - 1` stays inside
    # the window; 1e10 comfortably clears it without depending on wall-clock now.
    assert not _eval_install_lock_is_stale(
        tmp_path, pid=_DEAD_PID, started_at="1", stale_after=10**10
    )


def test_install_lock_fresh_lock_without_metadata_is_not_stale(
    tmp_path: Path,
) -> None:
    """A just-created lock (pid/timestamp not yet written) is respected.

    Guards the mkdir-race fix: the window between `mkdir` winning and the owner
    writing its metadata must not read as "stale", or a racing installer would
    delete a lock that was just acquired. The dir mtime (≈ now) keeps it fresh.
    """
    assert not _eval_install_lock_is_stale(tmp_path, pid=None, started_at=None)


def test_install_lock_missing_dir_is_not_stale(tmp_path: Path) -> None:
    """No lock directory means nothing to reclaim."""
    assert not _eval_install_lock_is_stale(
        tmp_path, pid=None, started_at=None, make_dir=False
    )


def test_install_script_lock_is_independent_of_deepagents_home(tmp_path: Path) -> None:
    """Installer serialization follows the uv tool environment, not a profile."""
    bin_dir, home, uv = _write_fake_tools(
        tmp_path, installed_version="0.0.1", latest_version="0.1.0"
    )
    configured = tmp_path / "custom-home"
    env = {
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_HOME": str(configured),
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

    assert proc.returncode == 0, proc.stderr
    assert not configured.exists()
    assert not (home / ".deepagents").exists()
    assert (tmp_path / "tools/.deepagents-code.deepagents-code-locks").is_dir()


def test_install_script_ignores_symlinked_legacy_lock_file(tmp_path: Path) -> None:
    """A symlinked legacy `install.lock` is not followed when flock is available.

    Guards the root-install symlink hardening: a non-root
    user who can write `~/.deepagents` could pre-create `install.lock` as a
    symlink to a root-writable path. The installer must use the directory lock
    instead of opening `install.lock`, so the target is never truncated by the
    shell's `>` redirect.

    macOS lacks `flock`, so a fake `flock` shim is staged on `PATH` to force
    the regression case where flock would otherwise be available.
    """
    bin_dir, home, uv = _write_fake_tools(
        tmp_path, installed_version="0.0.1", latest_version="0.1.0"
    )
    # Stage a fake `flock` so the flock path is taken even on macOS.
    flock = bin_dir / "flock"
    flock.write_text("#!/usr/bin/env bash\nexit 0\n")
    _make_executable(flock)

    deepagents = home / ".deepagents"
    deepagents.mkdir()
    target = tmp_path / "secret.txt"
    target.write_text("precious")
    (deepagents / "install.lock").symlink_to(target)
    env = {
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
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

    assert proc.returncode == 0
    # The symlink target must not have been truncated by the `>` redirect.
    assert target.read_text() == "precious"
    # The legacy lock path is ignored rather than replaced.
    lock_file = deepagents / "install.lock"
    assert lock_file.is_symlink()
    assert lock_file.resolve() == target


def test_install_script_does_not_redirect_to_legacy_lock_file() -> None:
    """Pin the TOCTOU fix: post-open symlink checks are too late."""
    script = SCRIPT.read_text(encoding="utf-8")

    assert "INSTALL_LOCK_FILE" not in script
    assert '>"$lock_root/install.lock"' not in script
    assert '>"$HOME/.deepagents/install.lock"' not in script


class TestInstallLockRootGuessWarning:
    """The guess warning must be reachable.

    `resolve_installation_root` is substituted by its caller, so a flag it
    assigns to a global dies in the subshell and the warning can never print.
    Concurrent installs then fail to serialize with no diagnostic at all.
    """

    def _run(self, tmp_path: Path, *, uv_bin: str) -> subprocess.CompletedProcess[str]:
        script = tmp_path / "lock_guess_harness.sh"
        script.write_text(
            "set -euo pipefail\n"
            f"HOME={str(tmp_path)!r}\n"
            # `resolve_installation_root` falls back to XDG_DATA_HOME. Pin it
            # inside tmp_path: inheriting the real one would create — and hold
            # — a lock in the developer's actual uv tool directory, blocking
            # every later test that acquires the install lock.
            f"XDG_DATA_HOME={str(tmp_path / 'xdg')!r}\n"
            f"UV_BIN={uv_bin!r}\n"
            "UV_TOOL_DIR_ENV=''\n"
            "INSTALL_LOCK_KIND=''\n"
            "INSTALL_LOCK_DIR=''\n"
            "INSTALL_LOCK_STALE_AFTER_SECS=600\n"
            "fix_file_owner() { return 0; }\n"
            "wait_for_install_lock_reclaim_guard() { return 0; }\n"
            "log_warn() { printf 'WARN %s\\n' \"$*\"; }\n"
            "log_error() { printf '%s\\n' \"$*\" >&2; }\n"
            f"{_extract_shell_function('normalize_absolute_path')}\n"
            f"{_extract_shell_function('resolve_installation_root')}\n"
            f"{_extract_shell_function('install_lock_identity')}\n"
            f"{_extract_shell_function('install_lock_is_stale')}\n"
            f"{_extract_shell_function('acquire_install_lock')}\n"
            "acquire_install_lock\n",
            encoding="utf-8",
        )
        return subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_warns_when_the_lock_root_is_a_guess(self, tmp_path: Path) -> None:
        """With uv unavailable the root is a guess and the user is told."""
        result = self._run(tmp_path, uv_bin="")

        assert result.returncode == 0, result.stderr
        assert "guessing the installer lock root" in result.stdout
        assert "may not serialize" in result.stdout

    def test_stays_quiet_when_uv_answers(self, tmp_path: Path) -> None:
        """An authoritative `uv tool dir` answer must not warn."""
        uv_bin = tmp_path / "uv"
        uv_bin.write_text(
            f'#!/usr/bin/env bash\nprintf "%s" {str(tmp_path / "tools")!r}\n',
            encoding="utf-8",
        )
        uv_bin.chmod(0o755)

        result = self._run(tmp_path, uv_bin=str(uv_bin))

        assert result.returncode == 0, result.stderr
        assert "guessing the installer lock root" not in result.stdout


def test_install_lock_fails_when_lock_root_cannot_create_child(
    tmp_path: Path,
) -> None:
    """An unwritable existing lock root fails instead of waiting forever."""
    installation_root = tmp_path / "tools" / "deepagents-code"
    lock_root = tmp_path / "tools" / ".deepagents-code.deepagents-code-locks"
    lock_root.mkdir(parents=True)
    script = tmp_path / "unwritable_lock_harness.sh"
    script.write_text(
        f"HOME={str(tmp_path)!r}\n"
        "INSTALL_LOCK_KIND=''\n"
        "INSTALL_LOCK_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_TOKEN=''\n"
        "INSTALL_LOCK_STALE_AFTER_SECS=600\n"
        "fix_file_owner() { return 0; }\n"
        f"resolve_installation_root() {{ printf '%s' {str(installation_root)!r}; }}\n"
        "log_warn() { return 0; }\n"
        "log_error() { printf '%s\\n' \"$*\" >&2; }\n"
        f"{_extract_shell_function('lock_dir_mtime')}\n"
        f"{_extract_shell_function('install_lock_identity')}\n"
        f"{_extract_shell_function('install_lock_is_stale')}\n"
        f"{_extract_shell_function('install_lock_reclaim_guard_is_stale')}\n"
        f"{_extract_shell_function('wait_for_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('release_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock')}\n"
        "mkdir() {\n"
        '  if [ "$1" = "$INSTALL_LOCK_DIR" ]; then\n'
        "    return 1\n"
        "  fi\n"
        '  command mkdir "$@"\n'
        "}\n"
        "sleep() { return 97; }\n"
        "acquire_install_lock\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )

    assert proc.returncode == 1
    assert "Cannot create installer lock" in proc.stderr
    assert "writable" in proc.stderr


def test_install_script_reclaim_skips_new_lock_after_stale_check(
    tmp_path: Path,
) -> None:
    """The reclaim re-check skips `mv` when the lock changed after stale detection.

    Simulates a peer reclaimer that clears the stale lock between this process's
    staleness check and its own identity re-check. `install_lock_identity` is
    stubbed to report the inspected fingerprint on its first call (so the lock
    reads as stale and that fingerprint is captured) and then, on the re-check,
    to remove the lock dir and report a different (empty) fingerprint. The
    mismatch must abort the rename so this process never moves a lock it did not
    inspect aside; it then acquires the now-free path cleanly and `mv` is never
    called. A filesystem marker sequences the two calls: each runs in a `$(...)`
    subshell, so a shell-variable counter would not carry across them.
    """
    installation_root = tmp_path / "deepagents-code"
    lock_root = tmp_path / ".deepagents-code.deepagents-code-locks"
    lock_dir = lock_root / "install.lock.d"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(f"{_DEAD_PID}\n")
    (lock_dir / "started_at").write_text("1\n")
    marker = tmp_path / "mv-called"
    checked = tmp_path / "identity-checked"
    script = tmp_path / "reclaim_race_harness.sh"
    script.write_text(
        f"HOME={str(tmp_path)!r}\n"
        "INSTALL_LOCK_KIND=''\n"
        "INSTALL_LOCK_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_TOKEN=''\n"
        "INSTALL_LOCK_STALE_AFTER_SECS=600\n"
        "fix_file_owner() { return 0; }\n"
        f"resolve_installation_root() {{ printf '%s' {str(installation_root)!r}; }}\n"
        "log_warn() { return 0; }\n"
        "log_error() { printf '%s\\n' \"$*\" >&2; }\n"
        f"{_extract_shell_function('lock_dir_mtime')}\n"
        # First call (inside install_lock_is_stale) reports the inspected
        # fingerprint so the lock reads as stale. The re-check call then clears
        # the lock — as a racing reclaimer would — and reports a different
        # (empty) fingerprint, which must make acquire_install_lock skip the mv.
        "install_lock_identity() {\n"
        f"  if [ ! -f {str(checked)!r} ]; then\n"
        f"    : >{str(checked)!r}\n"
        "    printf 'stale-fingerprint'\n"
        "    return 0\n"
        "  fi\n"
        '  rm -rf "$INSTALL_LOCK_DIR"\n'
        "  return 1\n"
        "}\n"
        f"{_extract_shell_function('install_lock_is_stale')}\n"
        f"{_extract_shell_function('install_lock_reclaim_guard_is_stale')}\n"
        f"{_extract_shell_function('wait_for_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('release_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock')}\n"
        f"{_extract_shell_function('release_install_lock')}\n"
        "mv() {\n"
        f"  printf 'called\\n' >{str(marker)!r}\n"
        "  return 1\n"
        "}\n"
        "acquire_install_lock\n"
        "release_install_lock\n"
        f"test ! -f {str(marker)!r}\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr


def test_install_script_reclaim_holds_guard_while_renaming_stale_lock(
    tmp_path: Path,
) -> None:
    """Stale reclaim renames the canonical lock only while peers are guarded."""
    installation_root = tmp_path / "deepagents-code"
    lock_root = tmp_path / ".deepagents-code.deepagents-code-locks"
    lock_dir = lock_root / "install.lock.d"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(f"{_DEAD_PID}\n")
    (lock_dir / "started_at").write_text("1\n")
    missing_guard = tmp_path / "missing-guard"
    script = tmp_path / "reclaim_guard_harness.sh"
    script.write_text(
        f"HOME={str(tmp_path)!r}\n"
        "INSTALL_LOCK_KIND=''\n"
        "INSTALL_LOCK_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_TOKEN=''\n"
        "INSTALL_LOCK_STALE_AFTER_SECS=600\n"
        "fix_file_owner() { return 0; }\n"
        f"resolve_installation_root() {{ printf '%s' {str(installation_root)!r}; }}\n"
        "log_warn() { return 0; }\n"
        "log_error() { printf '%s\\n' \"$*\" >&2; }\n"
        f"{_extract_shell_function('lock_dir_mtime')}\n"
        f"{_extract_shell_function('install_lock_identity')}\n"
        f"{_extract_shell_function('install_lock_is_stale')}\n"
        f"{_extract_shell_function('install_lock_reclaim_guard_is_stale')}\n"
        f"{_extract_shell_function('wait_for_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('release_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock')}\n"
        f"{_extract_shell_function('release_install_lock')}\n"
        "mv() {\n"
        '  if [ "$1" = "$INSTALL_LOCK_DIR" ] && \\\n'
        '    [ ! -d "$INSTALL_LOCK_RECLAIM_DIR" ]; then\n'
        f"    printf 'missing\\n' >{str(missing_guard)!r}\n"
        "    return 1\n"
        "  fi\n"
        '  command mv "$@"\n'
        "}\n"
        "acquire_install_lock\n"
        "release_install_lock\n"
        f"test ! -f {str(missing_guard)!r}\n"
        f"test ! -d {str(lock_root / 'install.lock.reclaim.d')!r}\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("our_token", "on_disk_token", "expected_removed"),
    [
        # We still hold the lock: the on-disk token matches ours -> remove it.
        ("mine", "mine", True),
        # A reclaimer took over the canonical path (different token) -> keep it.
        ("mine", "other", False),
        # We never recorded a token, so ownership is unprovable -> keep it.
        ("", "mine", False),
    ],
)
def test_install_script_release_removes_lock_only_when_token_matches(
    tmp_path: Path, our_token: str, on_disk_token: str, expected_removed: bool
) -> None:
    """release_install_lock removes the lock dir iff the on-disk token is ours.

    Guards the release ownership check: a regression to an unconditional
    `rm -rf "$INSTALL_LOCK_DIR"` would let a slow installer delete a lock a fresh
    owner now holds. The reclaim guard is left untouched here
    (INSTALL_LOCK_RECLAIM_TOKEN empty), so only the canonical lock is exercised.
    """
    lock_root = tmp_path / ".deepagents-code.deepagents-code-locks"
    lock_dir = lock_root / "install.lock.d"
    lock_dir.mkdir(parents=True)
    (lock_dir / "token").write_text(f"{on_disk_token}\n")
    script = tmp_path / "release_harness.sh"
    script.write_text(
        f"INSTALL_LOCK_DIR={str(lock_dir)!r}\n"
        f"INSTALL_LOCK_RECLAIM_DIR={str(lock_root / 'install.lock.reclaim.d')!r}\n"
        "INSTALL_LOCK_KIND='mkdir'\n"
        f"INSTALL_LOCK_TOKEN={our_token!r}\n"
        "INSTALL_LOCK_RECLAIM_TOKEN=''\n"
        f"{_extract_shell_function('release_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('release_install_lock')}\n"
        "release_install_lock\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert (not lock_dir.exists()) == expected_removed


def test_install_script_aborts_when_lock_token_cannot_be_written(
    tmp_path: Path,
) -> None:
    """A failed lock-token write aborts loudly and removes the half-made lock.

    After winning `mkdir`, the metadata write must succeed or the acquire has to
    `exit 1` and clean up. Here a stubbed `mkdir` plants a *directory* named
    `token` inside the fresh lock so `>"$INSTALL_LOCK_DIR/token"` fails. Guards
    against a regression that drops either the cleanup (orphan lock nobody can
    release) or the `exit` (install proceeds tokenless, so release never matches
    and the lock leaks permanently).
    """
    installation_root = tmp_path / "deepagents-code"
    lock_root = tmp_path / ".deepagents-code.deepagents-code-locks"
    lock_root.mkdir(parents=True)
    script = tmp_path / "token_write_harness.sh"
    script.write_text(
        f"HOME={str(tmp_path)!r}\n"
        "INSTALL_LOCK_KIND=''\n"
        "INSTALL_LOCK_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_DIR=''\n"
        "INSTALL_LOCK_RECLAIM_TOKEN=''\n"
        "INSTALL_LOCK_STALE_AFTER_SECS=600\n"
        "fix_file_owner() { return 0; }\n"
        f"resolve_installation_root() {{ printf '%s' {str(installation_root)!r}; }}\n"
        "log_warn() { return 0; }\n"
        "log_error() { printf '%s\\n' \"$*\" >&2; }\n"
        f"{_extract_shell_function('lock_dir_mtime')}\n"
        f"{_extract_shell_function('install_lock_identity')}\n"
        f"{_extract_shell_function('install_lock_is_stale')}\n"
        f"{_extract_shell_function('install_lock_reclaim_guard_is_stale')}\n"
        f"{_extract_shell_function('wait_for_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('release_install_lock_reclaim_guard')}\n"
        f"{_extract_shell_function('acquire_install_lock')}\n"
        # Win the mkdir, but plant a directory named `token` inside the lock so
        # the metadata write `>"$INSTALL_LOCK_DIR/token"` fails.
        "mkdir() {\n"
        '  if [ "$1" = "$INSTALL_LOCK_DIR" ]; then\n'
        '    command mkdir "$INSTALL_LOCK_DIR" || return 1\n'
        '    command mkdir "$INSTALL_LOCK_DIR/token"\n'
        "    return 0\n"
        "  fi\n"
        '  command mkdir "$@"\n'
        "}\n"
        "acquire_install_lock\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 1, proc.stderr
    assert "Cannot write installer lock metadata" in proc.stderr
    assert not (lock_root / "install.lock.d").exists()


def test_install_script_reclaims_stale_mkdir_lock(tmp_path: Path) -> None:
    """A stale mkdir lock left by a dead owner is reclaimed, and install proceeds.

    Drives the full `acquire_install_lock` mkdir path (not just the
    `install_lock_is_stale` predicate): a pre-existing `install.lock.d` with a
    dead PID and an old `started_at` must be renamed aside, removed, and the
    install allowed to continue. The concurrent-replacement cases are covered
    separately by test_install_script_reclaim_skips_new_lock_after_stale_check
    (identity changed before reclaim) and
    test_install_script_reclaim_holds_guard_while_renaming_stale_lock (peers held
    by the reclaim guard during the rename).
    """
    bin_dir, home, uv = _write_fake_tools(
        tmp_path, installed_version="0.0.1", latest_version="0.1.0"
    )
    lock_root = tmp_path / "tools/.deepagents-code.deepagents-code-locks"
    lock_dir = lock_root / "install.lock.d"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(f"{_DEAD_PID}\n")
    (lock_dir / "started_at").write_text("1\n")  # 1970 => well past the window
    env = {
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
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
        timeout=60,  # a reclaim regression could busy-loop; fail fast instead
    )

    assert proc.returncode == 0, proc.stderr
    assert "Removing stale installer lock" in proc.stderr
    # The install ran (lock acquired) rather than aborting on the stale lock.
    assert (tmp_path / "uv-args.txt").is_file()
    # The lock is released on exit, leaving no lock dir and no reclaim leftovers.
    assert not (lock_root / "install.lock.d").exists()
    assert not list(lock_root.glob("install.lock.d.reclaim.*"))
    assert not (lock_root / "install.lock.reclaim.d").exists()


@pytest.mark.skipif(
    _RUNNING_AS_ROOT, reason="root bypasses the directory permission bits"
)
def test_install_script_aborts_on_unremovable_stale_lock(tmp_path: Path) -> None:
    """An unremovable stale lock aborts loudly instead of spinning forever.

    When the stale `install.lock.d` can be neither renamed nor removed (here,
    its parent is read-only), the reclaim must `exit 1` with an actionable
    message. Regression guard for the busy-loop: `continue` skips the retry
    `sleep`, so a silently swallowed `rm` failure would spin on `mkdir` and
    spam the warning indefinitely. The `timeout` turns that hang into a
    failure rather than letting the test run wedge.
    """
    bin_dir, home, uv = _write_fake_tools(
        tmp_path, installed_version="0.0.1", latest_version="0.1.0"
    )
    lock_root = tmp_path / "tools/.deepagents-code.deepagents-code-locks"
    lock_dir = lock_root / "install.lock.d"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(f"{_DEAD_PID}\n")
    (lock_dir / "started_at").write_text("1\n")
    # Read+execute only: entries inside cannot be renamed or unlinked, so both
    # the `mv` and the fallback `rm -rf` fail with EACCES.
    lock_root.chmod(0o555)
    env = {
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
    }
    try:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            timeout=60,
        )
    finally:
        # Restore write access so tmp_path teardown can remove the tree.
        lock_root.chmod(0o755)

    assert proc.returncode == 1, proc.stderr
    assert "Cannot reclaim stale installer lock" in proc.stderr


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
    fail_if_lockf_called: bool = False,
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
    if fail_if_lockf_called:
        lockf = bin_dir / "lockf"
        lockf.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'lockf must not be used for installer locking\\n' >&2\n"
            "exit 64\n"
        )
        _make_executable(lockf)

    env = {
        **_clean_environ(),
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
    download_failures_before_success: int = 0,
    use_wget: bool = False,
    busybox_wget: bool = False,
    truncated: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the real `install_uv` from `install.sh` against a fake uv installer.

    A fake downloader (``curl`` by default, or ``wget`` when ``use_wget`` is set)
    writes a trivial "installer" to the file named by its output flag (``-o`` for
    curl, ``-O`` for wget); the harness runs it via ``sh``, so the noise lands in
    the captured output. When ``fails`` is set, that installer also exits
    non-zero, exercising the surface-output-on-failure branch. When ``no_shebang``
    is set, the installer content starts with an HTML tag instead of a shell
    shebang, exercising the shebang-verification rejection. When ``truncated`` is
    set, the content keeps a valid shebang but is cut mid-``if``, which only the
    parse check can catch. When ``download_fails``
    is set, the fake downloader writes an error to stderr and exits non-zero
    *without* creating the file, exercising the download-failure branch and
    proving the downloader's own error is surfaced. Returns the completed process
    so callers can assert on whether the noise reached the terminal and on the
    exit code.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    first_line = "<html>error</html>" if no_shebang else "#!/bin/sh"
    installer = first_line + "\n"
    if truncated:
        # A body cut mid-`if`: the shebang is intact, so only a parse check
        # rejects this. Exactly the shape a dropped connection leaves behind.
        installer += "if [ 1 = 1 ]; then\n"
    installer += "echo UV_INSTALLER_NOISE\n"
    if fails:
        installer += "exit 3\n"

    # The fake installer is served as a real file (not shell-quoted inline)
    # so the downloader can copy it to the output path install_uv passes.
    installer_fixture = tmp_path / "fake-uv-installer.sh"
    installer_fixture.write_text(installer, encoding="utf-8")

    # The fake downloader must handle its output flag (curl ``-o`` / wget
    # ``-O``) and copy the fixture there. With ``download_fails`` it instead
    # emits an error to stderr and exits non-zero without creating the file,
    # so install_uv sees a failed download.
    downloader_name = "wget" if use_wget else "curl"
    out_flag = "-O" if use_wget else "-o"

    if download_fails:
        write_body = (
            "printf 'DOWNLOADER_ERROR: could not resolve host\\n' >&2\nexit 7\n"
        )
    elif download_failures_before_success:
        attempts = tmp_path / "uv-download-attempts.txt"
        write_body = (
            "count=0\n"
            f"if [ -f {str(attempts)!r} ]; then read -r count < {str(attempts)!r}; fi\n"
            "count=$((count + 1))\n"
            f"printf '%s\\n' \"$count\" > {str(attempts)!r}\n"
            f'if [ "$count" -le {download_failures_before_success} ]; then\n'
            "  printf 'DOWNLOADER_ERROR: transient failure\\n' >&2\n"
            "  exit 7\n"
            "fi\n"
            f'cat {str(installer_fixture)!r} >"${{out:-/dev/stdout}}"\n'
        )
    else:
        write_body = f'cat {str(installer_fixture)!r} >"${{out:-/dev/stdout}}"\n'
    downloader = bin_dir / downloader_name
    # `wget_download` probes `wget --help` to decide which hardening flags this
    # wget implements. Answer those probes *before* any counting or output, so
    # capability probes are never mistaken for download attempts.
    #
    # BusyBox advertises none of the long options this function probes for and
    # errors out if one is passed anyway; GNU advertises all of them. Modelling
    # both is what lets the BusyBox test prove the flags were actually withheld.
    #
    # BusyBox also *exits 1* from `--help` (it prints usage and returns
    # EXIT_FAILURE). That matters: the probe runs under `set -o pipefail`, so a
    # piped `grep` would inherit the failure and report every option as
    # unsupported — silently disabling `-S` and the redirect audit with it.
    # Modelling the real exit code is what keeps that from regressing.
    if busybox_wget:
        help_text = "BusyBox v1.36.0 wget"
        help_rc = "1"
        reject_unsupported = (
            'for arg in "$@"; do\n'
            '  case "$arg" in\n'
            "    --max-redirect=*|--https-only|-S)\n"
            "      printf '%s\\n' \"wget: unrecognized option $arg\" >&2\n"
            "      exit 1\n"
            "      ;;\n"
            "  esac\n"
            "done\n"
        )
    else:
        help_text = "--header --max-redirect --https-only -S"
        help_rc = "0"
        reject_unsupported = ""
    help_handler = (
        'if [ "${1:-}" = "--help" ]; then\n'
        f"  printf '%s\\n' {help_text!r}\n"
        f"  exit {help_rc}\n"
        "fi\n"
    ) + reject_unsupported
    # Record the argv of every real (non-probe) invocation so tests can assert
    # which hardening flags were actually passed. Without this the fakes ignore
    # argv entirely and the flags could be deleted with no test failing.
    argv_log = tmp_path / "downloader-argv.txt"
    record_argv = f"printf '%s\\n' \"$*\" >> {str(argv_log)!r}\n"
    downloader.write_text(
        "#!/usr/bin/env bash\n" + help_handler + record_argv + "out=''\n"
        # Capture the full argv *before* the parsing loop shifts it away — the
        # body routes on the URL, which is gone from `$*` once the loop runs.
        'all_args="$*"\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        f'    {out_flag}) out="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n" + write_body
    )
    _make_executable(downloader)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    _make_executable(sleep)
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
        f"{_extract_shell_function('wget_supports_option')}\n"
        f"{_extract_shell_function('wget_download')}\n"
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


def test_uv_cache_snapshot_precedes_install_lock() -> None:
    """Lock creation cannot hide a newly created uv data tree from ownership repair."""
    source = SCRIPT.read_text(encoding="utf-8")
    bootstrap = source.split("if ! resolve_uv_bin; then", maxsplit=1)[1].split(
        "resolve_tool_bin_dir()", maxsplit=1
    )[0]

    assert bootstrap.index("snapshot_missing_uv_cache_paths") < bootstrap.index(
        "acquire_install_lock"
    )


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


def _eval_wget_supports_option(
    tmp_path: Path, *, option: str, help_text: str, help_rc: int
) -> bool:
    """Return `wget_supports_option`'s verdict against a fake `wget --help`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wget = bin_dir / "wget"
    wget.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' {help_text!r}\nexit {help_rc}\n",
        encoding="utf-8",
    )
    _make_executable(wget)
    script = tmp_path / "wget_probe_harness.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"{_extract_shell_function('wget_supports_option')}\n"
        f"if wget_supports_option {option!r}; then echo YES; else echo NO; fi\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip() == "YES"


def _run_wget_download(
    tmp_path: Path,
    *,
    url: str,
    response_headers: str = "",
) -> subprocess.CompletedProcess[str]:
    """Drive the real `wget_download` against a fake wget.

    ``response_headers`` is echoed to stderr the way ``wget -S`` reports a
    response, which is what the downgrade audit inspects.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wget = bin_dir / "wget"
    wget.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "--help" ]; then\n'
        "  printf '%s\\n' '--header --max-redirect --https-only -S'\n"
        "  exit 0\n"
        "fi\n"
        # %b, not %s: the audit's grep anchors Location: at the start of a
        # line, so the \n escapes have to become real newlines.
        f"printf '%b' {response_headers!r} >&2\n"
        "out=''\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    -O|-qO) out="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        # `-` means stdout to real wget; writing it literally would drop a file
        # named "-" into the working directory.
        'if [ -z "$out" ] || [ "$out" = "-" ]; then out=/dev/stdout; fi\n'
        "printf 'DOWNLOADED_BODY\\n' >\"$out\"\n",
        encoding="utf-8",
    )
    _make_executable(wget)
    script = tmp_path / "wget_download_harness.sh"
    script.write_text(
        "set -euo pipefail\n"
        'log_warn() { printf "%s\\n" "$*" >&2; }\n'
        "register_temp() { :; }\n"
        f"{_extract_shell_function('wget_supports_option')}\n"
        f"{_extract_shell_function('wget_download')}\n"
        f"rc=0\nwget_download - {url!r} '' true || rc=$?\n"
        'printf "RC=%s\\n" "$rc"\n',
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


def test_install_uv_curl_pins_https_for_request_and_redirects(
    tmp_path: Path,
) -> None:
    """Scheme pinning and a bounded redirect chain are passed to curl.

    `--proto-redir` is the only thing stopping a 3xx from downgrading the
    fetch to plaintext, and nothing else in the suite would notice if the
    flags were dropped.
    """
    proc = _run_install_uv(tmp_path, verbose=False)

    assert proc.returncode == 0
    argv = (tmp_path / "downloader-argv.txt").read_text()
    assert "--proto =https" in argv
    assert "--proto-redir =https" in argv
    assert "--max-redirs 3" in argv


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


def test_install_uv_rejects_truncated_download(tmp_path: Path) -> None:
    """A download cut mid-statement is rejected by the parse check before exec.

    The shebang is intact, so the shebang check passes: only reading the whole
    file without running it catches the unbalanced `if`. The payload must never
    execute.
    """
    proc = _run_install_uv(tmp_path, verbose=False, truncated=True)

    assert proc.returncode != 0
    assert "failed a shell syntax check" in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stderr
    assert "UV_INSTALLER_NOISE" not in proc.stdout


def test_install_uv_wget_passes_hardening_flags_when_supported(
    tmp_path: Path,
) -> None:
    """A GNU-shaped wget receives the redirect controls it advertises.

    The BusyBox test proves the flags are *withheld* when unsupported; this is
    the mirror that proves they are sent when they are supported, so the
    capability probe can't silently start answering "no" to everything.
    """
    proc = _run_install_uv(tmp_path, verbose=True, use_wget=True)

    assert proc.returncode == 0
    argv = (tmp_path / "downloader-argv.txt").read_text()
    assert "--max-redirect=3" in argv
    assert "--https-only" in argv
    assert "-S" in argv


def test_wget_download_refuses_non_https_url(tmp_path: Path) -> None:
    """Plaintext request URLs are refused, since wget has no `--proto`.

    The request URL must be HTTPS; a plaintext URL is refused before wget runs.
    """
    proc = _run_wget_download(tmp_path, url="http://plain.example/uv.sh")

    assert "RC=1" in proc.stdout
    assert "Refusing to download over a non-HTTPS URL" in proc.stderr
    assert "DOWNLOADED_BODY" not in proc.stdout


def test_wget_download_refuses_plaintext_redirect(tmp_path: Path) -> None:
    """A 3xx pointing at http:// fails closed.

    `--https-only` only applies in recursive mode, so it does not stop a
    one-shot fetch from being downgraded. Auditing the response headers for a
    plaintext Location is the only wget-side defense, and it must abort rather
    than warn and continue.
    """
    proc = _run_wget_download(
        tmp_path,
        url="https://ok.example/uv.sh",
        response_headers="  HTTP/1.1 302 Found\n  Location: http://plain.example/uv.sh\n",
    )

    assert "RC=1" in proc.stdout
    assert "Refusing a plaintext HTTP redirect" in proc.stderr


def test_wget_supports_option_ignores_help_exit_status(tmp_path: Path) -> None:
    """An advertised option is detected even when `--help` exits non-zero.

    BusyBox prints its usage and returns EXIT_FAILURE. Piping `wget --help`
    straight into `grep` under `set -o pipefail` makes that exit status the
    pipeline's, so every option reports as unsupported — which silently drops
    `-S` and disables the redirect-downgrade audit on exactly the minimal
    systems the capability probe exists to support. Failing *open* like that is
    invisible without this test.
    """
    assert _eval_wget_supports_option(
        tmp_path, option="-S", help_text="-S --max-redirect --https-only", help_rc=1
    )


@pytest.mark.parametrize("use_wget", [False, True])
def test_install_uv_retries_transient_download_failure(
    tmp_path: Path, *, use_wget: bool
) -> None:
    """The uv bootstrap retries two transient failures before succeeding."""
    proc = _run_install_uv(
        tmp_path,
        verbose=False,
        download_failures_before_success=2,
        use_wget=use_wget,
    )

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "uv-download-attempts.txt").read_text().strip() == "3"


def _run_signal_traps(
    tmp_path: Path, *, interrupt: bool, temp_dir: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Wire the real EXIT + INT/TERM traps from `install.sh` and trip one.

    Extracts the shipped `cleanup_on_signal`/`cleanup_on_interrupt` handlers and
    installs them exactly as the script does. With `interrupt=True` the process
    sends itself SIGINT (the Ctrl-C path); otherwise it exits non-zero without a
    signal (the ordinary-failure path). Returns the completed process so callers
    can assert both the trap message the user sees and the exit code reported.
    """
    script = tmp_path / "signal_trap_harness.sh"
    body = "kill -INT $$\nsleep 5\n" if interrupt else "exit 2\n"
    setup = ""
    if temp_dir is not None:
        setup = (
            f"mkdir -p {str(temp_dir)!r}\n"
            f"printf 'stderr\\n' > {str(temp_dir / 'install.log')!r}\n"
            f"register_temp_dir {str(temp_dir)!r}\n"
        )
    script.write_text(
        "set -uo pipefail\n"
        'log_warn()  { printf "%s\\n" "$*" >&2; }\n'
        'log_error() { printf "%s\\n" "$*" >&2; }\n'
        # cleanup_on_signal now calls these unconditionally; extract the real
        # implementations so the harness exercises shipped code without emitting
        # "command not found" noise (which would otherwise pass tests by luck).
        "TEMP_FILES=()\n"
        "TEMP_DIRS=()\n"
        f"{_extract_shell_function('register_temp')}\n"
        f"{_extract_shell_function('register_temp_dir')}\n"
        f"{_extract_shell_function('cleanup_temp_files')}\n"
        f"{_extract_shell_function('cleanup_temp_dirs')}\n"
        f"{_extract_shell_function('is_linux_os')}\n"
        f"{_extract_shell_function('restore_terminal_after_signal')}\n"
        f"{_extract_shell_function('log_signal_failure_hint')}\n"
        f"{_extract_shell_function('warn_live_log_replaced')}\n"
        f"{_extract_shell_function('cleanup_on_signal')}\n"
        f"{_extract_shell_function('cleanup_on_interrupt')}\n"
        "trap cleanup_on_signal EXIT\n"
        # Wire the traps exactly as install.sh does, passing the signal number
        # so the harness exercises the real 128+signo exit path.
        "trap 'cleanup_on_interrupt 2' INT\n"
        "trap 'cleanup_on_interrupt 15' TERM\n"
        "trap 'cleanup_on_interrupt 1' HUP\n"
        f"{setup}"
        f"{body}",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        start_new_session=True,
    )


def test_interrupt_exits_with_128_plus_signal(tmp_path: Path) -> None:
    """Ctrl-C exits 130 (128+SIGINT), not a generic 1.

    CI and wrapper scripts use the 128+signo convention to tell an interrupted
    install apart from an ordinary failure.
    """
    proc = _run_signal_traps(tmp_path, interrupt=True)

    assert proc.returncode == 130


def test_interrupt_shows_notice_without_failure_message(tmp_path: Path) -> None:
    """Ctrl-C prints only the interrupt notice, not the EXIT trap's failure line.

    `cleanup_on_interrupt` disarms the EXIT trap (`trap - EXIT`) before exiting,
    so the friendly "Installation interrupted." message isn't followed by a
    contradictory "Installation failed (exit code 1)". Guards against dropping
    that disarm, which would surface both messages on a single Ctrl-C.
    """
    stderr = _run_signal_traps(tmp_path, interrupt=True).stderr

    assert "Installation interrupted." in stderr
    assert "Installation failed" not in stderr
    # The harness must define every helper cleanup_on_signal calls; a missing
    # one would still pass the asserts above but corrupt the exercised path.
    assert "command not found" not in stderr


def test_interrupt_removes_registered_staging_directory(tmp_path: Path) -> None:
    """An interrupted installer removes its registered log staging directory."""
    stage_dir = tmp_path / "deepagents-code-install-log"

    proc = _run_signal_traps(tmp_path, interrupt=True, temp_dir=stage_dir)

    assert proc.returncode == 130
    assert not stage_dir.exists()


def test_exit_trap_removes_registered_staging_directory(tmp_path: Path) -> None:
    """An ordinary failure exit clears the staging directory too.

    `cleanup_on_signal` runs on EXIT as well as on a signal, and the staged
    file holds uv's full captured stderr — a failed install that leaves it
    behind is the same disclosure an interrupted one would be.
    """
    stage_dir = tmp_path / "deepagents-code-install-log"

    proc = _run_signal_traps(tmp_path, interrupt=False, temp_dir=stage_dir)

    assert proc.returncode == 2
    assert not stage_dir.exists()


def test_exit_trap_reports_failure_on_ordinary_error(tmp_path: Path) -> None:
    """A non-signal, non-zero exit still fires the EXIT trap's failure message.

    The interrupt handler's `trap - EXIT` must be scoped to the interrupt path
    only: an ordinary failure exit still needs `cleanup_on_signal` to tell the
    user the install failed and where to get help.
    """
    stderr = _run_signal_traps(tmp_path, interrupt=False).stderr

    assert "Installation failed (exit code 2)." in stderr
    assert "Installation interrupted." not in stderr
    assert "command not found" not in stderr


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


def test_install_script_macos_does_not_use_lockf(tmp_path: Path) -> None:
    """The macOS `lockf` is command-scoped, not a file-descriptor lock."""
    proc, uv_args = _invoke_with_os(
        tmp_path,
        uname_os="Darwin",
        xcode_select_rc=0,
        installed_version="0.0.1",
        latest_version="0.2.0",
        fail_if_lockf_called=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "lockf must not be used" not in proc.stderr
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
    tmp_path: Path,
    *,
    env_file_content: str | None = None,
    extra_env: dict[str, str] | None = None,
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
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{path_without_uv}",
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


def test_install_script_custom_bin_from_sourced_uv_persists_path(
    tmp_path: Path,
) -> None:
    """Sourcing uv's env cannot hide that its custom tool bin needs PATH setup."""
    tool_bin = tmp_path / "home/custom-bin"
    proc, uv_args = _invoke_with_local_uv_not_on_path(
        tmp_path,
        env_file_content='export PATH="$HOME/.local/bin:$PATH"\n',
        extra_env={
            "FAKE_UV_CREATE_LOCAL_DCODE": "1",
            "FAKE_UV_TOOL_BIN_DIR": str(tool_bin),
            "SHELL": "/bin/zsh",
        },
    )

    assert proc.returncode == 0, proc.stderr
    assert uv_args.exists()
    installed = tool_bin / "dcode"
    exposed = tmp_path / "home/.local/bin/dcode"
    assert installed.is_file()
    assert exposed.is_symlink()
    assert exposed.resolve() == installed.resolve()
    profile = tmp_path / "home/.zshrc"
    assert 'export PATH="$HOME/.local/bin:$PATH"' in profile.read_text()
    assert "Added ~/.local/bin to PATH" in proc.stdout


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


def test_install_script_honors_uv_tool_bin_dir(tmp_path: Path) -> None:
    """A custom uv tool bin is found, verified, and exposed on `PATH`."""
    tool_bin = tmp_path / "home" / "custom-bin"
    extra_env = {
        "UV_TOOL_BIN_DIR": str(tool_bin),
        "FAKE_UV_TOOL_BIN_DIR": str(tool_bin),
        "FAKE_UV_CREATE_LOCAL_DCODE": "1",
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{_path_without_dcode()}",
        "SHELL": "/bin/zsh",
    }

    proc, uv_args = _invoke(tmp_path, extra_env, installed_version=None)

    assert proc.returncode == 0, proc.stderr
    assert uv_args.exists()
    installed = tool_bin / "dcode"
    exposed = tmp_path / "home/.local/bin/dcode"
    assert installed.is_file()
    assert exposed.is_symlink()
    assert exposed.resolve() == installed.resolve()
    assert "deepagents-code 0.2.0 installed" in proc.stdout
    assert "command not found in PATH" not in proc.stderr


def test_install_script_old_uv_ignores_unsupported_tool_bin_override(
    tmp_path: Path,
) -> None:
    """An old uv falls back to its legacy bin instead of a newer-only override."""
    custom_bin = tmp_path / "home" / "custom-bin"
    legacy_bin = tmp_path / "home/.local/bin"
    proc, uv_args = _invoke(
        tmp_path,
        {
            "UV_TOOL_BIN_DIR": str(custom_bin),
            "XDG_BIN_HOME": "",
            "XDG_DATA_HOME": "",
            "FAKE_UV_TOOL_BIN_DIR": str(legacy_bin),
            "FAKE_UV_TOOL_DIR_BIN_UNSUPPORTED": "1",
            "FAKE_UV_CREATE_LOCAL_DCODE": "1",
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{_path_without_dcode()}",
            "SHELL": "/bin/zsh",
        },
        installed_version=None,
    )

    assert proc.returncode == 0, proc.stderr
    assert uv_args.exists()
    assert (legacy_bin / "dcode").is_file()
    assert not (legacy_bin / "dcode").is_symlink()
    assert not custom_bin.exists()


def test_install_script_does_not_replace_tool_bin_path_alias_with_symlink(
    tmp_path: Path,
) -> None:
    """Equivalent uv bin spellings cannot turn `dcode` into a symlink loop."""
    home = tmp_path / "home"
    alias_bin = home / ".local/share/../bin"
    proc, _ = _invoke(
        tmp_path,
        {
            "FAKE_UV_TOOL_BIN_DIR": str(alias_bin),
            "FAKE_UV_CREATE_LOCAL_DCODE": "1",
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{_path_without_dcode()}",
            "SHELL": "/bin/zsh",
        },
        installed_version=None,
    )

    installed = home / ".local/bin/dcode"
    assert proc.returncode == 0, proc.stderr
    assert installed.is_file()
    assert not installed.is_symlink()
    assert "deepagents-code 0.2.0 installed" in proc.stdout


def test_install_script_root_custom_bin_leaves_path_to_mdm(tmp_path: Path) -> None:
    """A root custom-bin install does not write through user-controlled PATH files."""
    home = tmp_path / "home"
    tool_bin = home / "custom-bin"
    tool_bin.mkdir(parents=True)
    dcode = tool_bin / "dcode"
    dcode.write_text("#!/usr/bin/env bash\nexit 0\n")
    _make_executable(dcode)
    harness = tmp_path / "root_path_setup.sh"
    harness.write_text(
        f"HOME={str(home)!r}\n"
        f"TOOL_BIN_DIR_DISPLAY={str(tool_bin)!r}\n"
        "VERBOSE=0\n"
        "id() { printf '0\\n'; }\n"
        "log_warn() { printf '%s\\n' \"$*\" >&2; }\n"
        f"{_extract_shell_function('paths_are_same_file')}\n"
        f"{_extract_shell_function('ensure_path_setup')}\n"
        "set +e\n"
        f"ensure_path_setup dcode {str(dcode)!r}\n"
        "rc=$?\n"
        "printf '%s\\n' \"$rc\"\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == "3"
    assert "MDM policy" in proc.stderr
    assert not (home / ".local").exists()
    assert not (home / ".zshrc").exists()


def test_install_script_root_does_not_execute_existing_dcode_before_install(
    tmp_path: Path,
) -> None:
    """A root install does not run a user-controlled pre-install executable."""
    env = _env(
        tmp_path,
        {"FAKE_UV_CREATE_LOCAL_DCODE": "1", "SUDO_USER": "target"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "pre-install-dcode-ran"
    dcode = bin_dir / "dcode"
    dcode.write_text(
        f"#!/usr/bin/env bash\nprintf 'ran\\n' > {str(marker)!r}\nexit 0\n"
    )
    _make_executable(dcode)
    for name, body in {
        "id": "printf '0\\n'\n",
        "uname": "printf 'Linux\\n'\n",
        "chown": "exit 0\n",
    }.items():
        tool = bin_dir / name
        tool.write_text(f"#!/usr/bin/env bash\n{body}")
        _make_executable(tool)

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "uv-args.txt").exists()
    assert not marker.exists()


def _invoke_with_local_dcode_not_on_path(
    tmp_path: Path,
    *,
    create_env_file: bool = False,
    shell: str = "/bin/zsh",
    extra_env: dict[str, str] | None = None,
    seed_home: Callable[[Path], None] | None = None,
    answer: str | None = None,
    uname_os: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run with a working `dcode` in ~/.local/bin but outside the original PATH.

    `seed_home` runs after the fake HOME is created but before the installer,
    for tests that need pre-existing dotfiles in place.

    `answer` attaches a pty to stdin and feeds that reply to the PATH prompt.
    Without it the run is non-interactive and PATH setup auto-adds instead of
    asking, so it is the only way to reach the prompt (and the decline branch).

    `uname_os` pins the detected OS, which the candidate builder branches on
    (macOS skips creating a ~/.bashrc). Without it the result would depend on
    the host running the suite. A Darwin pin also bypasses the Xcode CLT
    preflight: these runs exercise profile selection, not that gate, and on
    the Linux CI runner `xcode-select` is absent so the script would exit
    there first.
    """
    bin_dir, home, uv = _write_fake_tools(tmp_path, installed_version=None)
    if uname_os is not None:
        uname = bin_dir / "uname"
        uname.write_text(f"#!/usr/bin/env bash\necho {uname_os}\n")
        _make_executable(uname)
    if seed_home is not None:
        seed_home(home)

    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    dcode = local_bin / "dcode"
    dcode.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-v" ]; then printf "deepagents-code 0.1.0\\n"; exit 0; fi\n'
        "exit 0\n"
    )
    _make_executable(dcode)
    # This dcode sits in uv's tool bin dir, so the script sees a uv-managed
    # install and expects a receipt for it. These tests are about PATH and
    # profile selection; without the receipt every one of them would divert
    # into the "couldn't tell which extras this has" warning and its prompt.
    _write_uv_receipt(tmp_path / "tools", None)
    if create_env_file:
        (local_bin / "env").write_text('export PATH="$HOME/.local/bin:$PATH"\n')

    env = {
        **_clean_environ(),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": f"{bin_dir}{os.pathsep}{_path_without_dcode()}",
        "UV_BIN": str(uv),
        "DEEPAGENTS_CODE_SKIP_OPTIONAL": "1",
        "SHELL": shell,
        **({"DEEPAGENTS_CODE_SKIP_XCODE_CHECK": "1"} if uname_os == "Darwin" else {}),
        **(extra_env or {}),
    }
    if answer is not None:
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
        try:
            os.write(primary, f"{answer}\n".encode())
            assert proc.stdout is not None
            output = proc.stdout.read()
            proc.wait(timeout=30)
        finally:
            # See `_invoke_interactive`: the timeout path must not leak the
            # pipe and pty or orphan the child.
            if proc.stdout is not None:
                proc.stdout.close()
            os.close(primary)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
        return subprocess.CompletedProcess(
            args=["bash", str(SCRIPT)],
            returncode=proc.returncode,
            stdout=clean,
            stderr="",
        )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_root_install_chowns_only_exact_managed_leaves_with_optional_setup(
    tmp_path: Path,
) -> None:
    """Root optional setup never recursively takes ownership of broad roots."""
    env = _env(
        tmp_path,
        {"SUDO_USER": "target"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    # A profile cannot *be* the home directory, so point it at a subdirectory.
    # `home` itself must still never be a chown target.
    profile = home / ".deepagents"
    # `dcode tools install` falls back to the profile bin dir when the shared
    # installation one is unwritable. It sits inside the target user's home, so
    # a root install must hand it back or the user's next non-root run fails on
    # root-owned files.
    fallback_bin = profile / "bin"
    chown_log = tmp_path / "chown-invocations.txt"
    for name, body in {
        "id": "printf '0\\n'\n",
        "uname": "printf 'Linux\\n'\n",
        "chown": 'printf \'%s\\n\' "$*" >>"$CHOWN_LOG"\n',
    }.items():
        tool = bin_dir / name
        tool.write_text(f"#!/usr/bin/env bash\n{body}")
        _make_executable(tool)
    env.update(
        {
            "CHOWN_LOG": str(chown_log),
            "DEEPAGENTS_CODE_SKIP_OPTIONAL": "0",
            "DEEPAGENTS_HOME": str(profile),
            "FAKE_MANAGED_BIN_DIR": str(fallback_bin),
        }
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

    assert proc.returncode == 0, proc.stderr
    invocations = chown_log.read_text().splitlines()
    assert invocations
    # `-R` stays banned: ownership repair walks only trees this run created,
    # via `find -xdev -exec chown -h`, never a recursive chown of a broad root.
    assert all("-R" not in invocation.split() for invocation in invocations)
    # `find -exec ... +` batches several paths into one invocation, so collect
    # every argument rather than just the last token.
    targets = {
        arg
        for invocation in invocations
        for arg in invocation.split()
        if arg.startswith("/")
    }
    assert str(fallback_bin) in targets
    assert str(home) not in targets
    assert str(home / ".local") not in targets
    assert "/" not in targets
    assert (tmp_path / "dcode-tools.txt").exists()


def test_root_install_skips_a_managed_bin_dir_outside_home(tmp_path: Path) -> None:
    """A tool dir outside $HOME is never handed to the target user.

    `uv tool dir` is user-configurable (`UV_TOOL_DIR`, `uv.toml`, `--tool-dir`)
    and can name a system path such as `/opt/uv-tools`. Chowning that to a
    non-root user would be a privilege handover, so `fix_tree_owner`'s
    `path_is_under_home` gate must apply here too.
    """
    env = _env(
        tmp_path,
        {"SUDO_USER": "target"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    profile = home / ".deepagents"
    outside_bin = tmp_path / "opt/uv-tools/deepagents-code/share/deepagents-code/bin"
    chown_log = tmp_path / "chown-invocations.txt"
    for name, body in {
        "id": "printf '0\\n'\n",
        "uname": "printf 'Linux\\n'\n",
        "chown": 'printf \'%s\\n\' "$*" >>"$CHOWN_LOG"\n',
    }.items():
        tool = bin_dir / name
        tool.write_text(f"#!/usr/bin/env bash\n{body}")
        _make_executable(tool)
    env.update(
        {
            "CHOWN_LOG": str(chown_log),
            "DEEPAGENTS_CODE_SKIP_OPTIONAL": "0",
            "DEEPAGENTS_HOME": str(profile),
            "FAKE_MANAGED_BIN_DIR": str(outside_bin),
        }
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

    assert proc.returncode == 0, proc.stderr
    targets = {
        arg
        for invocation in chown_log.read_text().splitlines()
        for arg in invocation.split()
        if arg.startswith("/")
    }
    assert not any(target.startswith(str(tmp_path / "opt")) for target in targets)


def test_root_upgrade_still_chowns_a_preexisting_managed_bin_dir(
    tmp_path: Path,
) -> None:
    """The `sudo` upgrade path is the common one, and it must not be skipped.

    `dcode tools install` writes into these trees as root on every run. Gating
    the repair on "did this run create the directory?" skipped it whenever the
    directory already existed — i.e. on every upgrade — leaving the freshly
    written `rg` root-owned inside the target user's home.
    """
    env = _env(
        tmp_path,
        {"SUDO_USER": "target"},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    profile = home / ".deepagents"
    fallback_bin = profile / "bin"
    # The upgrade case: both managed bin directories are already on disk with a
    # binary in them before the installer runs.
    fallback_bin.mkdir(parents=True)
    (fallback_bin / "rg").write_text("stale binary\n")
    chown_log = tmp_path / "chown-invocations.txt"
    for name, body in {
        "id": "printf '0\n'\n",
        "uname": "printf 'Linux\n'\n",
        "chown": 'printf \'%s\\n\' "$*" >>"$CHOWN_LOG"\n',
    }.items():
        tool = bin_dir / name
        tool.write_text(f"#!/usr/bin/env bash\n{body}")
        _make_executable(tool)
    env.update(
        {
            "CHOWN_LOG": str(chown_log),
            "DEEPAGENTS_CODE_SKIP_OPTIONAL": "0",
            "DEEPAGENTS_HOME": str(profile),
            "FAKE_MANAGED_BIN_DIR": str(fallback_bin),
        }
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

    assert proc.returncode == 0, proc.stderr
    targets = {
        arg
        for invocation in chown_log.read_text().splitlines()
        for arg in invocation.split()
        if arg.startswith("/")
    }
    assert str(fallback_bin) in targets
    # The repair still walks only the managed tree, never the home or profile.
    assert str(home) not in targets
    assert str(profile) not in targets


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
    # SHELL is zsh and HOME starts empty, so the candidate set is exactly
    # ~/.zshrc. Naming it beats `any(...)` over three files, which would still
    # pass if the block landed in the wrong one.
    zshrc = (tmp_path / "home/.zshrc").read_text()
    assert "# >>> deepagents-code installer >>>" in zshrc
    assert 'export PATH="$HOME/.local/bin:$PATH"' in zshrc
    assert "# <<< deepagents-code installer <<<" in zshrc
    assert "source ~/.local/bin/env" not in combined


def test_install_script_tilde_display_has_no_literal_backslash(
    tmp_path: Path,
) -> None:
    r"""Paths render as `~/.zshrc`, not `\~/.zshrc`.

    bash 3.2 — still the default /bin/bash on macOS, and this script is run as
    `curl ... | bash` — does not unescape `\~` in a pattern-substitution
    replacement, so the old form leaked a backslash into user-facing messages.
    """
    proc = _invoke_with_local_dcode_not_on_path(tmp_path)

    combined = proc.stdout + proc.stderr
    assert "\\~" not in combined


def test_install_script_rewrites_fish_block_instead_of_duplicating(
    tmp_path: Path,
) -> None:
    """A stale managed block in the fish conf.d file is rewritten, not appended.

    The conf.d file is ours alone, but our block can still collide with itself:
    appending unconditionally would leave another dead block behind on every
    change to the fish export line.
    """

    def seed(home: Path) -> None:
        conf = home / ".config/fish/conf.d/deepagents-code.fish"
        conf.parent.mkdir(parents=True, exist_ok=True)
        conf.write_text(
            "\n# >>> deepagents-code installer >>>\n"
            "set -gx PATH /some/stale/entry $PATH\n"
            "# <<< deepagents-code installer <<<\n"
        )

    proc = _invoke_with_local_dcode_not_on_path(
        tmp_path, shell="/usr/local/bin/fish", seed_home=seed
    )

    assert proc.returncode == 0, proc.stderr
    fish_conf = tmp_path / "home/.config/fish/conf.d/deepagents-code.fish"
    assert fish_conf.read_text().count("# >>> deepagents-code installer >>>") == 1
    assert "/some/stale/entry" not in fish_conf.read_text()


def test_install_script_preserves_symlinked_profile(tmp_path: Path) -> None:
    """A symlinked ~/.zshrc is edited through, not replaced by a regular file.

    Dotfile managers (chezmoi, stow, home-manager) symlink startup files into a
    repo. Replacing the link with a regular file meant the next `apply` silently
    reverted the PATH entry after we reported success.
    """

    def seed(home: Path) -> None:
        dotfiles = home / "dotfiles"
        dotfiles.mkdir()
        target = dotfiles / "zshrc"
        # A stale managed block forces the rewrite path, which is the one
        # that used to `mv` over the symlink.
        target.write_text(
            "# managed by dotfiles\n"
            "# >>> deepagents-code installer >>>\n"
            "export PATH=/stale/entry:$PATH\n"
            "# <<< deepagents-code installer <<<\n"
        )
        (home / ".zshrc").symlink_to(target)

    proc = _invoke_with_local_dcode_not_on_path(tmp_path, seed_home=seed)

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    real_zshrc = home / "dotfiles/zshrc"
    assert (home / ".zshrc").is_symlink()
    assert "/stale/entry" not in real_zshrc.read_text()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in real_zshrc.read_text()


def test_rewrite_managed_path_block_fixes_resolved_symlink_target_owner(
    tmp_path: Path,
) -> None:
    """A root rewrite restores ownership of the replaced dotfile target."""
    target = tmp_path / "dotfiles/zshrc"
    target.parent.mkdir()
    target.write_text(
        "# >>> deepagents-code installer >>>\n"
        "export PATH=/stale/entry:$PATH\n"
        "# <<< deepagents-code installer <<<\n"
    )
    profile = tmp_path / ".zshrc"
    profile.symlink_to(target)
    chown_log = tmp_path / "chown-paths.txt"
    harness = tmp_path / "rewrite_profile.sh"
    harness.write_text(
        "log_warn() { :; }\n"
        "register_temp() { :; }\n"
        'fix_file_owner() { printf \'%s\\n\' "$1" >>"$CHOWN_LOG"; }\n'
        f"{_extract_shell_function('resolve_link_target')}\n"
        f"{_extract_shell_function('rewrite_managed_path_block')}\n"
        "rewrite_managed_path_block "
        f"{str(profile)!r} 'export PATH=/fresh/entry:$PATH'\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["bash", str(harness)],
        env={**_clean_environ(), "CHOWN_LOG": str(chown_log)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert profile.is_symlink()
    assert chown_log.read_text().splitlines() == [str(target)]
    assert "export PATH=/fresh/entry:$PATH" in target.read_text()


def test_install_script_follows_symlink_chain_to_final_target(tmp_path: Path) -> None:
    """A symlink chain (~/.zshrc -> links/zshrc -> ../dotfiles/zshrc) is followed.

    Resolving only the first hop would `mv` over the intermediate link with a
    regular file: the real dotfile-manager source stays stale, and the next
    restow recreates the link and reverts the PATH entry. The rewrite must
    land on the final regular file, leaving both links intact.
    """

    def seed(home: Path) -> None:
        dotfiles = home / "dotfiles"
        links = home / "links"
        dotfiles.mkdir()
        links.mkdir()
        target = dotfiles / "zshrc"
        target.write_text(
            "# managed by dotfiles\n"
            "# >>> deepagents-code installer >>>\n"
            "export PATH=/stale/entry:$PATH\n"
            "# <<< deepagents-code installer <<<\n"
        )
        (links / "zshrc").symlink_to("../dotfiles/zshrc")
        (home / ".zshrc").symlink_to("links/zshrc")

    proc = _invoke_with_local_dcode_not_on_path(tmp_path, seed_home=seed)

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    real_zshrc = home / "dotfiles/zshrc"
    assert (home / ".zshrc").is_symlink()
    assert (home / "links/zshrc").is_symlink()
    assert "/stale/entry" not in real_zshrc.read_text()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in real_zshrc.read_text()


def test_install_script_uses_zdotdir_pointing_at_missing_dir(
    tmp_path: Path,
) -> None:
    """A ~/.zshenv ZDOTDIR naming a nonexistent directory is still honored.

    ZDOTDIR is authoritative for zsh even when the directory doesn't exist
    yet: zsh reads only ${ZDOTDIR}/.zshrc, so writing ~/.zshrc instead would
    leave `dcode` off PATH. The installer creates the missing directory and
    writes the PATH block into the zshrc at the ZDOTDIR location.
    """

    def seed(home: Path) -> None:
        (home / ".zshenv").write_text("ZDOTDIR=$HOME/.config/zsh\n")

    proc = _invoke_with_local_dcode_not_on_path(tmp_path, seed_home=seed)

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    zshrc = home / ".config/zsh/.zshrc"
    assert zshrc.exists()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in zshrc.read_text()
    assert not (home / ".zshrc").exists()


def _eval_resolve_zdotdir(tmp_path: Path, zshenv: str | None) -> str:
    """Return `resolve_zdotdir`'s output for a given ~/.zshenv body.

    Drives the shipped parser directly. `None` means no ~/.zshenv at all.
    """
    home = tmp_path / "home"
    home.mkdir()
    if zshenv is not None:
        (home / ".zshenv").write_text(zshenv)
    script = tmp_path / "resolve_zdotdir_harness.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"HOME={str(home)!r}\n"
        "unset ZDOTDIR\n"
        'log_warn() { printf "%s\\n" "$*" >&2; }\n'
        f"{_extract_shell_function('shell_block_delta')}\n"
        f"{_extract_shell_function('resolve_zdotdir')}\n"
        "resolve_zdotdir\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.parametrize(
    ("zshenv", "expected"),
    [
        pytest.param(None, "", id="no-zshenv"),
        pytest.param("", "", id="empty"),
        pytest.param("ZDOTDIR=$HOME/.config/zsh\n", "{home}/.config/zsh", id="bare"),
        pytest.param(
            'export ZDOTDIR="$HOME/.config/zsh"\n',
            "{home}/.config/zsh",
            id="export-double-quoted",
        ),
        pytest.param(
            "export ZDOTDIR='$HOME/.config/zsh'\n",
            "{home}/.config/zsh",
            id="export-single-quoted",
        ),
        pytest.param("ZDOTDIR=~/.config/zsh\n", "{home}/.config/zsh", id="tilde"),
        pytest.param(
            "ZDOTDIR=${HOME}/.config/zsh\n", "{home}/.config/zsh", id="braced-home"
        ),
        pytest.param(
            "ZDOTDIR=$HOME/.config/zsh;\n",
            "{home}/.config/zsh",
            id="trailing-semicolon",
        ),
        pytest.param(
            "ZDOTDIR=$HOME/.config/zsh # relocate\n",
            "{home}/.config/zsh",
            id="trailing-comment",
        ),
        pytest.param(
            "ZDOTDIR=$HOME/first\nZDOTDIR=$HOME/second\n",
            "{home}/second",
            id="last-assignment-wins",
        ),
        # A relative ZDOTDIR is meaningless to zsh, which resolves it against
        # whatever cwd the shell happened to start in.
        pytest.param("ZDOTDIR=relative/path\n", "", id="rejects-relative"),
        pytest.param("# ZDOTDIR=$HOME/.config/zsh\n", "", id="commented-out"),
        pytest.param(
            'if [ -d "$HOME/x" ]; then\n  export ZDOTDIR="$HOME/x"\nfi\n',
            "",
            id="skips-if-block",
        ),
        pytest.param(
            'case "$OSTYPE" in\n  darwin*) ZDOTDIR=$HOME/x ;;\nesac\n',
            "",
            id="skips-case-block",
        ),
        pytest.param(
            'if [ -d "$HOME/x" ]; then export ZDOTDIR="$HOME/x"; fi\n'
            "ZDOTDIR=$HOME/after\n",
            "{home}/after",
            id="one-line-block-does-not-swallow-later",
        ),
    ],
)
def test_resolve_zdotdir_parses_zshenv(
    tmp_path: Path, zshenv: str | None, expected: str
) -> None:
    """The hand-rolled ~/.zshenv parser handles the spellings people write.

    This runs over user-authored files, so the surface is wide: quoting,
    `export`, trailing punctuation, `~`/`$HOME` expansion, last-wins, and the
    conditional blocks that must not be treated as authoritative.
    """
    result = _eval_resolve_zdotdir(tmp_path, zshenv)

    assert result == expected.format(home=tmp_path / "home")


def test_install_script_ignores_conditional_zdotdir_assignment(
    tmp_path: Path,
) -> None:
    """A ZDOTDIR guarded by `if [ -d ... ]` is not treated as authoritative.

    The portable-dotfiles idiom relocates zsh only when the directory exists.
    While it doesn't, zsh reads ~/.zshrc. Honoring the assignment anyway would
    make the installer *create* that directory, which flips the guard true and
    strands every later shell in a config dir holding nothing but our PATH
    block — the user's aliases, prompt and plugins silently stop loading.
    """

    def seed(home: Path) -> None:
        (home / ".zshenv").write_text(
            'if [ -d "$HOME/.config/zsh" ]; then\n'
            '  export ZDOTDIR="$HOME/.config/zsh"\n'
            "fi\n"
        )
        (home / ".zshrc").write_text("alias ll='ls -la'\n")

    proc = _invoke_with_local_dcode_not_on_path(tmp_path, seed_home=seed)

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    # The directory must not be created: creating it is what activates the guard.
    assert not (home / ".config/zsh").exists()
    zshrc = (home / ".zshrc").read_text()
    assert "alias ll='ls -la'" in zshrc
    assert 'export PATH="$HOME/.local/bin:$PATH"' in zshrc


def test_install_script_honors_zdotdir_after_a_guarded_block(
    tmp_path: Path,
) -> None:
    """Skipping guarded assignments must not swallow a later top-level one.

    Guards the nesting counter: if depth never returned to zero after the `fi`,
    every subsequent assignment would be ignored and relocated zsh setups would
    silently regress to ~/.zshrc.
    """

    def seed(home: Path) -> None:
        (home / ".zshenv").write_text(
            'if [ -d "$HOME/nope" ]; then\n'
            '  export ZDOTDIR="$HOME/nope"\n'
            "fi\n"
            'export ZDOTDIR="$HOME/.config/zsh"\n'
        )

    proc = _invoke_with_local_dcode_not_on_path(tmp_path, seed_home=seed)

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    assert (
        'export PATH="$HOME/.local/bin:$PATH"'
        in (home / ".config/zsh/.zshrc").read_text()
    )
    assert not (home / "nope").exists()


def test_install_script_writes_both_zdotdir_and_legacy_zshrc(
    tmp_path: Path,
) -> None:
    """A ZDOTDIR zshrc and a legacy ~/.zshrc both get the PATH block.

    The shell itself reads only the ZDOTDIR zshrc, but a stale ~/.zshrc is a
    trap for any later unset of ZDOTDIR, so it is updated too.
    """

    def seed(home: Path) -> None:
        zdot = home / ".config/zsh"
        zdot.mkdir(parents=True)
        (zdot / ".zshrc").write_text("# zdotdir config\n")
        (home / ".zshrc").write_text("# legacy config\n")

    proc = _invoke_with_local_dcode_not_on_path(
        tmp_path,
        extra_env={"ZDOTDIR": str(tmp_path / "home/.config/zsh")},
        seed_home=seed,
    )

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    assert (
        'export PATH="$HOME/.local/bin:$PATH"'
        in (home / ".config/zsh/.zshrc").read_text()
    )
    assert 'export PATH="$HOME/.local/bin:$PATH"' in (home / ".zshrc").read_text()


def test_install_script_preserves_relative_symlinked_profile(tmp_path: Path) -> None:
    """A relative symlink (~/.zshrc -> dotfiles/zshrc) is rewritten atomically.

    The rewrite resolves the link target relative to the link's directory, so
    the temp file lands in the target's directory and the `mv` over it is
    atomic — an interrupted install can't leave the dotfile-manager source
    truncated, as writing through the link in place would.
    """

    def seed(home: Path) -> None:
        dotfiles = home / "dotfiles"
        dotfiles.mkdir()
        target = dotfiles / "zshrc"
        target.write_text(
            "# managed by dotfiles\n"
            "# >>> deepagents-code installer >>>\n"
            "export PATH=/stale/entry:$PATH\n"
            "# <<< deepagents-code installer <<<\n"
        )
        (home / ".zshrc").symlink_to("dotfiles/zshrc")

    proc = _invoke_with_local_dcode_not_on_path(tmp_path, seed_home=seed)

    assert proc.returncode == 0, proc.stderr
    home = tmp_path / "home"
    real_zshrc = home / "dotfiles/zshrc"
    assert (home / ".zshrc").is_symlink()
    assert "/stale/entry" not in real_zshrc.read_text()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in real_zshrc.read_text()


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


def test_install_script_rewrites_existing_managed_path_block(tmp_path: Path) -> None:
    """An old installer-owned PATH block is rewritten in place."""
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

    zshrc = home / ".zshrc"
    zshrc.write_text(
        "before\n"
        "# >>> deepagents-code installer >>>\n"
        'export PATH="$HOME/old-bin:$PATH"\n'
        "# <<< deepagents-code installer <<<\n"
        "after\n"
    )

    env = {
        **_clean_environ(),
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
    profile = zshrc.read_text()
    assert profile.count("# >>> deepagents-code installer >>>") == 1
    assert 'export PATH="$HOME/.local/bin:$PATH"' in profile
    assert "$HOME/old-bin" not in profile
    assert profile.startswith("before\n")
    assert profile.endswith("after\n")


def test_install_script_current_shadow_does_not_skip_uv_install(tmp_path: Path) -> None:
    """A current non-uv `dcode` cannot suppress installation into uv's bin."""
    proc, uv_args = _invoke(
        tmp_path,
        {
            "FAKE_UV_TOOL_BIN_DIR": str(tmp_path / "home/.local/bin"),
            "FAKE_UV_CREATE_LOCAL_DCODE": "1",
            "FAKE_LOCAL_DCODE_VERSION": "0.2.0",
        },
        installed_version="0.2.0",
        latest_version="0.2.0",
    )

    assert proc.returncode == 0, proc.stderr
    assert uv_args.exists()
    assert "outside uv's configured tool bin" in proc.stdout
    assert "Already up to date" not in proc.stdout


def test_install_script_current_uv_tool_repairs_shadowed_path(tmp_path: Path) -> None:
    """A current uv tool still continues when another binary wins on `PATH`."""
    tool_bin = tmp_path / "home/.local/bin"
    env = _env(
        tmp_path,
        {"FAKE_UV_TOOL_BIN_DIR": str(tool_bin)},
        installed_version="0.1.0",
        latest_version="0.2.0",
    )
    tool_bin.mkdir(parents=True)
    dcode = tool_bin / "dcode"
    dcode.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-v" ]; then printf "deepagents-code 0.2.0\\n"; fi\n'
    )
    _make_executable(dcode)

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "uv-args.txt").exists()
    assert "not selected on PATH" in proc.stdout
    assert "Detected existing dcode" in proc.stderr


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


_NORMALIZE_PARITY_CASES = [
    "/a//b/",
    "/a/./b",
    "/a/b/..",
    "/a/b/../../..",
    "/..",
    "/",
    "/tmp/../..",
    "/x/./../y",
    # POSIX leaves a leading exactly-double slash implementation-defined and
    # `os.path.normpath` preserves it. The installer must agree, or it exports
    # one directory while every later launch resolves another.
    "//",
    "//.",
    "//..",
    "///a",
    "//srv/profile",
    "//a/../b",
    "//a/b/../..",
    "//a/b/../c",
]


@pytest.mark.parametrize(
    "tool_dir",
    [
        "/opt/uv/tools",
        "/Users/someone/.local/share/uv/tools",
        "/tmp/a b/tools",
        "/var//lib/uv/tools",
    ],
)
def test_shell_lock_root_matches_python_installation_paths(tool_dir: str) -> None:
    """The installer and dcode must derive the same install/update lock root.

    `install.sh` says it mirrors `_paths._installation_paths`, but the existing
    checks compare against hardcoded strings, so the two could drift while the
    claim still read as verified. If they drift, a `curl | bash` install and a
    dcode self-upgrade stop sharing a lock and nothing fails.
    """
    from deepagents_code._paths import _normalize_absolute

    installation_root = _normalize_absolute(Path(tool_dir) / "deepagents-code")
    expected = (
        installation_root.parent / f".{installation_root.name}.deepagents-code-locks"
    )

    # The three lines `acquire_install_lock` uses to build `lock_root`.
    script = (
        'installation_root="$1"\n'
        'installation_parent="${installation_root%/*}"\n'
        'installation_name="${installation_root##*/}"\n'
        "printf '%s' "
        '"${installation_parent}/.${installation_name}.deepagents-code-locks"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script, "bash", str(installation_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == str(expected)


def test_shell_lock_root_formula_is_the_one_the_installer_uses() -> None:
    """Pin the formula above to the real `acquire_install_lock` source.

    The test above reproduces three lines rather than running the function,
    which needs uv. Fail if those lines change.
    """
    source = _extract_shell_function("acquire_install_lock")

    assert 'installation_parent="${installation_root%/*}"' in source
    assert 'installation_name="${installation_root##*/}"' in source
    assert (
        'lock_root="${installation_parent}/.${installation_name}'
        '.deepagents-code-locks"' in source
    )


@pytest.mark.parametrize(
    ("target", "expected_message"),
    [("home", "home directory itself"), ("root", "filesystem root")],
)
def test_install_script_rejects_symlinked_deepagents_home_alias(
    tmp_path: Path, target: str, expected_message: str
) -> None:
    """Installer validation uses directory identity, matching app startup."""
    home = tmp_path / "home"
    link = tmp_path / "profile-link"
    destination = home if target == "home" else Path("/")
    link.symlink_to(destination, target_is_directory=True)

    proc, uv_args = _invoke(tmp_path, {"DEEPAGENTS_HOME": str(link)})

    assert proc.returncode == 1
    assert expected_message in proc.stderr
    assert not uv_args.exists()


@pytest.mark.parametrize("target", ["missing", "profile-link"])
def test_install_script_rejects_unresolved_deepagents_home_symlink(
    tmp_path: Path, target: str
) -> None:
    """Dangling links and symlink loops fail before installation starts."""
    link = tmp_path / "profile-link"
    link.symlink_to(target, target_is_directory=True)

    proc, uv_args = _invoke(tmp_path, {"DEEPAGENTS_HOME": str(link)})

    assert proc.returncode == 1
    assert "symlink whose target is missing or cannot be resolved" in proc.stderr
    assert not uv_args.exists()


def test_install_script_rejects_non_directory_deepagents_home(tmp_path: Path) -> None:
    """An existing regular file cannot hold a profile."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")

    proc, uv_args = _invoke(tmp_path, {"DEEPAGENTS_HOME": str(blocker)})

    assert proc.returncode == 1
    assert "not a directory" in proc.stderr
    assert not uv_args.exists()


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root bypasses permission bits")
@pytest.mark.parametrize("mode", [0o300, 0o400], ids=["unreadable", "unsearchable"])
def test_install_script_rejects_unusable_deepagents_home(
    tmp_path: Path, mode: int
) -> None:
    """The installer rejects a profile that app startup cannot traverse."""
    profile = tmp_path / "profile"
    profile.mkdir(mode=mode)
    try:
        proc, uv_args = _invoke(tmp_path, {"DEEPAGENTS_HOME": str(profile)})
    finally:
        profile.chmod(0o700)

    assert proc.returncode == 1
    assert "cannot be read or searched" in proc.stderr
    assert not uv_args.exists()


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root bypasses permission bits")
def test_install_script_rejects_profile_behind_unsearchable_ancestor(
    tmp_path: Path,
) -> None:
    """An inaccessible parent must not make the profile look merely absent."""
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o600)
    profile = blocked / "profile"
    try:
        proc, uv_args = _invoke(tmp_path, {"DEEPAGENTS_HOME": str(profile)})
    finally:
        blocked.chmod(0o700)

    assert proc.returncode == 1
    assert "parent directory cannot be searched" in proc.stderr
    assert not uv_args.exists()
