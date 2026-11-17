"""Locate, probe and run the MXC command-line executor.

The executor is a native binary bundled with the `@microsoft/mxc-sdk` npm
package (`wxc-exec.exe` on Windows, `mxc-exec-mac` on macOS, `lxc-exec` on
Linux). It takes an MXC request JSON document, applies the policy with the
operating system's own sandboxing mechanism, runs the workload and passes the
workload's stdout, stderr and exit code back.

Everything this module assumes about the executor's behaviour was observed by
running `@microsoft/mxc-sdk` 1.0.1 on Windows, macOS and Linux. There are three
kinds of result:

1. the workload ran: exit code and streams are the workload's own;
2. the request was rejected before a sandbox existed: exit 1, empty stdout,
   plain-text stderr starting with `Request error` or `error: the '...'
   backend is not available`;
3. the executor failed at runtime (timeout, spawn failure): exit 255 and the
   last stderr line is a JSON envelope `{"error": {"code": ..., "message": ...}}`.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import platform as _platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ENV_EXECUTOR_PATH = "MXC_EXECUTOR_PATH"
"""Environment variable naming the executor binary directly."""

ENV_BIN_DIR = "MXC_BIN_DIR"
"""Environment variable naming a `bin/` directory with `<arch>/<executor>` inside.

The Node SDK honours the same variable, so one setting serves both.
"""

SDK_PACKAGE = "@microsoft/mxc-sdk"

EXECUTOR_NAMES: dict[str, str] = {
    "win32": "wxc-exec.exe",
    "darwin": "mxc-exec-mac",
    "linux": "lxc-exec",
}

DEFAULT_GRACE_SECONDS = 5.0
"""Extra time the host waits past the request timeout before killing the tree."""

PROBE_TIMEOUT_SECONDS = 30.0

_EXECUTOR_RUNTIME_FAILURE_EXIT = 255
_NTSTATUS_ERROR_BASE = 0xC0000000
_REQUEST_REJECTED_PREFIXES = ("Request error", "error: the '")


class MXCExecutorNotFoundError(RuntimeError):
    """Raised when no MXC executor binary can be located."""


class MXCUnsupportedPlatformError(RuntimeError):
    """Raised on an operating system MXC does not support."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def executor_name(platform: str | None = None) -> str:
    """Return the executor file name for `platform` (defaults to this host)."""
    plat = platform or sys.platform
    try:
        return EXECUTOR_NAMES[plat]
    except KeyError:
        msg = f"MXC supports Windows, macOS and Linux; this host reports sys.platform={plat!r}."
        raise MXCUnsupportedPlatformError(msg) from None


def sdk_arch(machine: str | None = None) -> str:
    """Return the `bin/<arch>` folder name used by the npm package."""
    m = (machine or _platform.machine()).lower()
    return "arm64" if m in {"arm64", "aarch64"} else "x64"


def _node_global_roots(platform: str, environ: Mapping[str, str], home: Path) -> list[Path]:
    """Common global `node_modules` locations, without running npm."""
    roots: list[Path] = []
    if platform == "win32":
        roots.extend(Path(environ[var]) / "npm" / "node_modules" for var in ("APPDATA", "LOCALAPPDATA") if environ.get(var))
        roots.extend(
            Path(environ[var]) / "nodejs" / "node_modules" for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432") if environ.get(var)
        )
        roots.append(home / "scoop" / "apps" / "nodejs" / "current" / "node_modules")
        return roots

    roots.extend(
        [
            Path("/usr/local/lib/node_modules"),
            Path("/opt/homebrew/lib/node_modules"),
            Path("/usr/lib/node_modules"),
            home / ".npm-global" / "lib" / "node_modules",
            home / ".local" / "lib" / "node_modules",
            home / ".local" / "share" / "npm" / "lib" / "node_modules",
        ]
    )
    if environ.get("NVM_DIR"):
        roots.extend(
            Path(p) for p in sorted(glob.glob(str(Path(environ["NVM_DIR"]) / "versions" / "node" / "*" / "lib" / "node_modules")), reverse=True)
        )
    roots.extend(Path(p) for p in sorted(glob.glob(str(home / ".nvm" / "versions" / "node" / "*" / "lib" / "node_modules")), reverse=True))
    for fnm_root in (home / ".local" / "share" / "fnm", home / ".fnm", home / "Library" / "Application Support" / "fnm"):
        roots.extend(
            Path(p) for p in sorted(glob.glob(str(fnm_root / "node-versions" / "*" / "installation" / "lib" / "node_modules")), reverse=True)
        )
    roots.append(home / ".volta" / "tools" / "image" / "packages" / SDK_PACKAGE / "lib" / "node_modules")
    return roots


def candidate_executor_paths(
    *,
    platform: str | None = None,
    machine: str | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> list[Path]:
    """List the places searched for the executor, in search order.

    Order: `MXC_EXECUTOR_PATH`, `MXC_BIN_DIR`, `node_modules/@microsoft/mxc-sdk`
    in the working directory and every parent, well-known global
    `node_modules` folders, then `PATH`.
    """
    plat = platform or sys.platform
    env = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else Path(cwd)
    home = Path.home() if home is None else Path(home)
    name = executor_name(plat)
    arch = sdk_arch(machine)
    rel = Path("node_modules") / SDK_PACKAGE / "bin" / arch / name

    candidates: list[Path] = []
    if env.get(ENV_EXECUTOR_PATH):
        candidates.append(Path(env[ENV_EXECUTOR_PATH]).expanduser())
    if env.get(ENV_BIN_DIR):
        candidates.append(Path(env[ENV_BIN_DIR]).expanduser() / arch / name)
    candidates.extend(directory / rel for directory in (cwd, *cwd.parents))
    candidates.extend(root / SDK_PACKAGE / "bin" / arch / name for root in _node_global_roots(plat, env, home))
    found_on_path = shutil.which(name)
    if found_on_path:
        candidates.append(Path(found_on_path))
    return candidates


def _npm_global_root(timeout: float = 15.0) -> Path | None:
    """Ask npm for its global root. Slow (hundreds of ms), so used last."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv; `npm` comes from `shutil.which`
            [npm, "root", "-g"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    root = proc.stdout.strip()
    return Path(root) if proc.returncode == 0 and root else None


def install_instructions(platform: str | None = None, machine: str | None = None) -> str:
    """Human-readable instructions for getting the executor onto this host."""
    plat = platform or sys.platform
    name = EXECUTOR_NAMES.get(plat, "the MXC executor")
    arch = sdk_arch(machine)
    return (
        f"MXCSandbox needs the native MXC executor ({name}, {arch}) that ships inside the "
        f"{SDK_PACKAGE} npm package. Install it with Node.js 24 or newer:\n\n"
        f"    npm install {SDK_PACKAGE}        # into ./node_modules of your project\n"
        f"    npm install -g {SDK_PACKAGE}     # or globally\n\n"
        "Then run from a directory that contains node_modules/@microsoft/mxc-sdk, or set "
        f"{ENV_EXECUTOR_PATH}=<full path to {name}>, or pass executor_path=... to MXCSandbox. "
        "Host requirements: Windows 11 24H2 (build 26100.9278+), macOS 15+, or Linux with "
        "bubblewrap 0.5+ and unprivileged user namespaces."
    )


def find_executor(
    explicit: str | os.PathLike[str] | None = None,
    *,
    platform: str | None = None,
    machine: str | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
    use_npm: bool = True,
) -> Path:
    """Locate the executor binary.

    Args:
        explicit: A path supplied by the caller. If given it must exist; no
            other location is tried.
        platform: `sys.platform` override (tests).
        machine: `platform.machine()` override (tests).
        environ: Environment mapping override (tests).
        cwd: Working directory override (tests).
        home: Home directory override (tests).
        use_npm: Whether to fall back to `npm root -g` when nothing else matches.

    Returns:
        Path to an existing executor file.

    Raises:
        MXCExecutorNotFoundError: With installation instructions when no
            executor is found.
        MXCUnsupportedPlatformError: On an unsupported operating system.
    """
    plat = platform or sys.platform
    name = executor_name(plat)
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        msg = f"executor_path {str(path)!r} does not exist or is not a file.\n\n{install_instructions(plat, machine)}"
        raise MXCExecutorNotFoundError(msg)

    candidates = candidate_executor_paths(platform=plat, machine=machine, environ=environ, cwd=cwd, home=home)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if use_npm:
        root = _npm_global_root()
        if root is not None:
            candidate = root / SDK_PACKAGE / "bin" / sdk_arch(machine) / name
            candidates.append(candidate)
            if candidate.is_file():
                return candidate

    searched = "\n".join(f"  - {c}" for c in candidates)
    msg = f"Could not find {name}.\n\n{install_instructions(plat, machine)}\n\nSearched:\n{searched}"
    raise MXCExecutorNotFoundError(msg)


# ---------------------------------------------------------------------------
# Running requests
# ---------------------------------------------------------------------------


@dataclass
class ExecutorResult:
    """Raw outcome of one executor invocation."""

    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool = False
    """True when the host-side wait expired and the process tree was killed."""

    @property
    def stdout_text(self) -> str:
        """Stdout decoded as UTF-8 with replacement, with Windows line endings normalised."""
        return self.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")

    @property
    def stderr_text(self) -> str:
        """Stderr decoded as UTF-8 with replacement, with Windows line endings normalised."""
        return self.stderr.decode("utf-8", errors="replace").replace("\r\n", "\n")


ExecutorCommand = str | os.PathLike[str] | Sequence[str]
"""Either the executor path, or an argv prefix (tests use `[sys.executable, script]`)."""


def _argv_prefix(executor: ExecutorCommand) -> list[str]:
    if isinstance(executor, (str, os.PathLike)):
        return [os.fspath(executor)]
    return [os.fspath(part) for part in executor]


def _posix_descendants(root_pid: int) -> list[int]:
    """Return every descendant pid of `root_pid` using `ps` (stdlib only)."""
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],  # noqa: S607 - system `ps`, resolved through PATH like a shell would
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:  # noqa: PLR2004
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    found: list[int] = []
    queue = [root_pid]
    while queue:
        current = queue.pop()
        for child in children.get(current, []):
            if child not in found:
                found.append(child)
                queue.append(child)
    return found


def kill_process_tree(pid: int) -> None:
    """Kill `pid` and every descendant, including their process groups.

    The executor starts the workload shell in a process group of its own, so
    killing the executor alone leaves the workload running (verified on macOS
    and Linux).
    """
    if sys.platform == "win32":
        subprocess.run(  # noqa: S603 - fixed argv; the pid is an int
            ["taskkill", "/F", "/T", "/PID", str(pid)],  # noqa: S607 - system `taskkill`, resolved through PATH
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        return

    own_pgid = os.getpgid(0)
    targets = [pid, *_posix_descendants(pid)]
    for target in targets:
        try:
            pgid = os.getpgid(target)
        except ProcessLookupError:
            continue
        if pgid not in {own_pgid, 0, 1}:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
    for target in targets:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(target, signal.SIGKILL)


def run_request(
    executor: ExecutorCommand,
    request: Mapping[str, Any],
    *,
    timeout: float | None,
    grace: float = DEFAULT_GRACE_SECONDS,
    config_dir: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
) -> ExecutorResult:
    """Run one MXC request through the executor and collect the result.

    The request is written to a temporary JSON file passed with `--config`
    (supported by all three executors, and immune to command-line length
    limits). stdin is `/dev/null` so the workload sees EOF instead of
    inheriting the caller's terminal.

    Args:
        executor: Executor path, or an argv prefix.
        request: The MXC request document (already including `process.timeout`).
        timeout: Seconds to wait on the host before killing the process tree,
            not counting `grace`. `None` waits forever.
        grace: Seconds added to `timeout` so the executor's own timeout can
            fire first and clean up the tree itself.
        config_dir: Directory for the temporary config file.
        extra_args: Extra executor flags, for example `["--dry-run"]`.

    Returns:
        The raw result.

    Raises:
        OSError: If the executor cannot be started.
    """
    argv = _argv_prefix(executor)
    fd, config_path = tempfile.mkstemp(prefix="mxc-request-", suffix=".json", dir=None if config_dir is None else os.fspath(config_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(request, handle)
        argv += ["--config", config_path, *extra_args]

        windows = sys.platform == "win32"
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0) if windows else 0

        proc = subprocess.Popen(  # noqa: S603 - running the MXC executor is this function's purpose
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            start_new_session=not windows,
        )
        timed_out = False
        wait_for = None if timeout is None else max(timeout, 0) + max(grace, 0)
        try:
            stdout, stderr = proc.communicate(timeout=wait_for)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=max(grace, 1.0))
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = b"", b""
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
        return ExecutorResult(
            stdout=stdout or b"",
            stderr=stderr or b"",
            returncode=proc.returncode,
            timed_out=timed_out,
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------

OutcomeKind = Literal[
    "workload",
    "host_timeout",
    "timeout",
    "request_rejected",
    "backend_unavailable",
    "backend_error",
]


@dataclass(frozen=True)
class Outcome:
    """What the executor's exit code and streams mean."""

    kind: OutcomeKind
    message: str = ""
    """Executor error text (empty for `workload`)."""
    code: str | None = None
    """Error code from a JSON envelope, if any."""
    workload_stdout: str = ""
    workload_stderr: str = ""
    """Workload streams with any executor envelope removed."""


_ENVELOPE_MARKER = '{"error":'


def _parse_envelope(line: str) -> tuple[str | None, str] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    if not isinstance(payload, dict) or "error" not in payload:
        return None
    error = payload["error"]
    if isinstance(error, dict):
        return (str(error.get("code")) if error.get("code") is not None else None, str(error.get("message", "")))
    return (None, str(error))


def classify_result(result: ExecutorResult) -> Outcome:  # noqa: C901, PLR0911 - one flat branch per observed executor outcome
    """Classify an executor result per the observed contract described in the module docstring."""
    stdout = result.stdout_text
    stderr = result.stderr_text

    if result.timed_out:
        return Outcome(
            kind="host_timeout", message="the executor did not exit in time and was killed", workload_stdout=stdout, workload_stderr=stderr
        )

    stripped_err = stderr.lstrip()
    if result.returncode == 1 and not stdout.strip() and stripped_err.startswith(_REQUEST_REJECTED_PREFIXES):
        lines = [line for line in stripped_err.splitlines() if line.strip()]
        if stripped_err.startswith("Request error"):
            detail = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
            detail = detail.removeprefix("Configuration parse error: ").strip()
            return Outcome(kind="request_rejected", message=detail, workload_stderr="")
        first = lines[0].removeprefix("error: ").strip()
        return Outcome(kind="backend_unavailable", message=first, workload_stderr="")

    # The executor reports its own runtime failures as a JSON envelope on the
    # last line of stderr. macOS and Linux exit 255 with the envelope on a line
    # of its own. Observed on Windows (wxc-exec 1.0.1): when the workload cannot
    # start, the exit code is the NTSTATUS (for example 0xC0000142) and the
    # envelope follows the plain-text message on the same line. Other exit
    # codes are left alone so a workload that prints its own JSON error to
    # stderr is not mistaken for an executor failure.
    if result.returncode == _EXECUTOR_RUNTIME_FAILURE_EXIT or (result.returncode or 0) >= _NTSTATUS_ERROR_BASE:
        err_lines = stderr.rstrip("\r\n").splitlines()
        if err_lines:
            last = err_lines[-1]
            start = 0 if last.lstrip().startswith("{") else last.find(_ENVELOPE_MARKER)
            envelope = _parse_envelope(last[start:]) if start >= 0 else None
            if envelope is not None:
                code, message = envelope
                kept = err_lines[:-1]
                prefix = last[:start].strip()
                if prefix and prefix != message.strip():
                    kept.append(prefix)
                remaining = "\n".join(kept)
                if remaining:
                    remaining += "\n"
                kind: OutcomeKind = "timeout" if "timed out" in message.lower() else "backend_error"
                return Outcome(kind=kind, message=message, code=code, workload_stdout=stdout, workload_stderr=remaining)

    if result.returncode == 1 and not stdout.strip() and stripped_err.startswith("Error: ") and "config" in stripped_err.lower():
        return Outcome(kind="backend_error", message=stripped_err.splitlines()[0].removeprefix("Error: ").strip())

    # Observed on Linux: when bubblewrap cannot build the sandbox (for example
    # in a container that forbids mounting /proc) the executor exits with
    # bwrap's code and bwrap's message, and the workload never starts.
    if result.returncode not in (0, None) and not stdout.strip() and stripped_err.startswith("bwrap: "):
        return Outcome(kind="backend_error", message=stripped_err.splitlines()[0], code="sandbox_start_failed")

    return Outcome(kind="workload", workload_stdout=stdout, workload_stderr=stderr)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Whether MXC can enforce the policy on this host."""

    ok: bool
    reason: str = ""
    details: dict[str, str] = field(default_factory=dict)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


_DRY_RUN_FAILED = "validation failed"


def _dry_run_failure_reason(result: ExecutorResult) -> str:
    """Extract the reason from a failed `--dry-run`.

    Observed forms: a policy the backend cannot
    enforce exits 255 with `Dry run completed. Result: validation failed: <reason>`
    on **stdout**; a malformed request or a missing backend exits 1 with the
    reason on stderr.
    """
    for line in result.stdout_text.splitlines():
        if _DRY_RUN_FAILED in line:
            _, _, reason = line.partition(_DRY_RUN_FAILED)
            return reason.lstrip(": ").strip() or line.strip()
    outcome = classify_result(result)
    if outcome.message:
        return outcome.message
    stderr_line = _first_line(result.stderr_text)
    if stderr_line:
        return stderr_line
    stdout_lines = [line.strip() for line in result.stdout_text.splitlines() if line.strip()]
    if stdout_lines:
        return stdout_lines[-1]
    return f"exit code {result.returncode}"


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def probe_executor(  # noqa: C901, PLR0911, PLR0912, PLR0915 - a linear checklist; each step returns its own failure
    executor: ExecutorCommand,
    build_request: Callable[[str], Mapping[str, Any]],
    *,
    allowed_paths: Iterable[str | os.PathLike[str]] = (),
    platform: str | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    config_dir: str | os.PathLike[str] | None = None,
) -> ProbeResult:
    """Check that the executor accepts the policy and actually enforces it.

    Four steps:

    1. `--dry-run` with the real policy. Catches schema problems and a backend
       that is missing on this host (for example bubblewrap not installed).
    2. A liveness check: `echo <token>` must run inside the sandbox and
       return the token, so a sandbox that cannot start is reported as such.
    3. An enforcement canary: a file is created in a fresh temporary directory
       outside every allowed path, and the sandbox is asked to print it. The
       probe fails if the content comes back, which would mean the policy is
       not being enforced.
    4. On Windows `--probe` and on Linux `--available-backends` are run for
       diagnostics only; their output is stored in `details` and does not
       affect the verdict.

    Args:
        executor: Executor path or argv prefix.
        build_request: Builds a full request for a shell command string.
        allowed_paths: Paths the policy grants, used to make sure the canary is
            placed outside them.
        platform: `sys.platform` override.
        timeout: Per-step timeout in seconds.
        config_dir: Directory for temporary config files.

    Returns:
        The verdict with a reason suitable for showing to a model or a user.
    """
    plat = platform or sys.platform
    details: dict[str, str] = {}

    # 1. validation
    try:
        dry = run_request(executor, build_request("exit 0"), timeout=timeout, config_dir=config_dir, extra_args=["--dry-run"])
    except OSError as exc:
        return ProbeResult(ok=False, reason=f"could not start the MXC executor: {exc}", details=details)
    details["dry_run_stdout"] = dry.stdout_text[-2000:]
    details["dry_run_stderr"] = dry.stderr_text[-2000:]
    if dry.returncode != 0 or "validation passed" not in dry.stdout_text:
        reason = _dry_run_failure_reason(dry)
        return ProbeResult(ok=False, reason=f"MXC rejected the sandbox policy on this host: {reason}", details=details)

    # 2. liveness: a trivial command must actually run and return its output.
    # Without this, a sandbox that cannot start at all would look like one
    # that blocks everything, and the canary below would pass.
    alive_token = f"mxc-alive-{uuid.uuid4().hex}"
    try:
        alive = run_request(executor, build_request(f"echo {alive_token}"), timeout=timeout, config_dir=config_dir)
    except OSError as exc:
        return ProbeResult(ok=False, reason=f"could not start the MXC executor: {exc}", details=details)
    details["alive_stdout"] = alive.stdout_text[-500:]
    details["alive_stderr"] = alive.stderr_text[-500:]
    alive_outcome = classify_result(alive)
    if alive_outcome.kind != "workload" or alive.returncode != 0 or alive_token not in alive.stdout_text:
        why = alive_outcome.message or _first_line(alive.stderr_text) or f"exit code {alive.returncode}"
        return ProbeResult(ok=False, reason=f"MXC could not run a sandboxed command on this host: {why}", details=details)

    # 3. enforcement canary
    canary_dir = Path(tempfile.mkdtemp(prefix="langchain-mxc-canary-")).resolve()
    try:
        allowed = [Path(p).resolve() for p in allowed_paths]
        if any(_is_under(canary_dir, a) for a in allowed):
            details["canary"] = "skipped: the temporary directory is inside an allowed path"
        else:
            token = f"mxc-canary-{uuid.uuid4().hex}"
            canary_file = canary_dir / "canary.txt"
            canary_file.write_text(token, encoding="utf-8")
            command = f'type "{canary_file}"' if plat == "win32" else f"cat {shlex.quote(str(canary_file))}"
            try:
                run = run_request(executor, build_request(command), timeout=timeout, config_dir=config_dir)
            except OSError as exc:
                return ProbeResult(ok=False, reason=f"could not start the MXC executor: {exc}", details=details)
            outcome = classify_result(run)
            details["canary_stdout"] = run.stdout_text[-500:]
            details["canary_stderr"] = run.stderr_text[-500:]
            if outcome.kind != "workload":
                return ProbeResult(
                    ok=False, reason=f"MXC could not run a sandboxed command on this host ({outcome.kind}): {outcome.message}", details=details
                )
            if token in run.stdout_text:
                return ProbeResult(
                    ok=False,
                    reason="MXC did not enforce the filesystem policy: a file outside the allowed folders was readable from inside the sandbox",
                    details=details,
                )
            if run.returncode == 0:
                return ProbeResult(
                    ok=False,
                    reason="MXC did not enforce the filesystem policy: reading a file outside the allowed folders unexpectedly succeeded",
                    details=details,
                )
            details["canary"] = "blocked as expected"
    finally:
        shutil.rmtree(canary_dir, ignore_errors=True)

    # 4. informational host probe
    info_args: list[str] | None = None
    if plat == "win32":
        info_args = ["--probe"]
    elif plat == "linux":
        info_args = ["--available-backends"]
    if info_args:
        try:
            info = run_request(executor, build_request("exit 0"), timeout=timeout, config_dir=config_dir, extra_args=info_args)
            details["host_probe"] = (info.stdout_text or info.stderr_text)[-4000:]
        except OSError as exc:
            details["host_probe"] = f"unavailable: {exc}"

    return ProbeResult(ok=True, details=details)


__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "ENV_BIN_DIR",
    "ENV_EXECUTOR_PATH",
    "EXECUTOR_NAMES",
    "ExecutorResult",
    "MXCExecutorNotFoundError",
    "MXCUnsupportedPlatformError",
    "Outcome",
    "ProbeResult",
    "candidate_executor_paths",
    "classify_result",
    "executor_name",
    "find_executor",
    "install_instructions",
    "kill_process_tree",
    "probe_executor",
    "run_request",
    "sdk_arch",
]
