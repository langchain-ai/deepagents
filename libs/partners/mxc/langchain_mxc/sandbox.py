"""`MXCSandbox`: Deep Agents backend that runs shell commands inside MXC.

This is `deepagents.backends.local_shell.LocalShellBackend` with the
`subprocess.run(shell=True)` call replaced by the MXC executor. File tools
(`ls`, `read`, `write`, `edit`, `glob`, `grep`) are inherited from
`FilesystemBackend` in virtual mode and run in the host process, limited to
`root_dir` by Deep Agents' path checks. Only `execute` is OS-enforced.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from langchain_mxc._executor import (
    DEFAULT_GRACE_SECONDS,
    ExecutorResult,
    Outcome,
    ProbeResult,
    classify_result,
    find_executor,
    probe_executor,
    run_request,
)
from langchain_mxc._policy import HostInfo, SandboxPolicy

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""

TIMEOUT_EXIT_CODE = 124
"""Exit code reported for a timed-out command (same as `LocalShellBackend`)."""

BLOCKED_HINT = "Blocked by sandbox policy: {what}. Ask the user to grant access if this is needed."

# Observed denial text. The first form is what
# shells and coreutils print (`cat: /path: Operation not permitted`), the
# second is Python's (`PermissionError: [Errno 1] Operation not permitted: '/path'`).
_DENIAL_WORDS = r"(?:Operation not permitted|Permission denied|Access is denied)"
_DENIAL_PATH_BEFORE = re.compile(rf"(?P<path>(?:/|[A-Za-z]:\\)[^:'\"\n]*?)['\"]?: {_DENIAL_WORDS}")
_DENIAL_PATH_AFTER = re.compile(rf"{_DENIAL_WORDS}: ['\"](?P<path>[^'\"\n]+)['\"]")
_DENIAL_ANY = re.compile(_DENIAL_WORDS)
# Observed on Linux: bubblewrap does not mount what the policy leaves out, so
# a blocked path reads as missing (`ls: cannot access '/p': No such file or directory`).
# dash words a blocked redirect as `cannot create /p: Directory nonexistent`.
_MISSING_WORDS = r"(?:No such file or directory|Directory nonexistent)"
_MISSING_PATH_BEFORE = re.compile(rf"(?P<path>/[^:'\"\n]*?)['\"]?: {_MISSING_WORDS}")
_MISSING_PATH_AFTER = re.compile(rf"{_MISSING_WORDS}: ['\"](?P<path>/[^'\"\n]+)['\"]")
_NETWORK_DENIAL = re.compile(
    r"Could not resolve host|nodename nor servname provided|Temporary failure in name resolution"
    r"|Name or service not known|No address associated with hostname|getaddrinfo failed"
    r"|Network is unreachable|Network unreachable|EAI_AGAIN|ENETUNREACH|ENOTFOUND"
    r"|(?:socket|bind|connect): Operation not permitted|Errno 1\] Operation not permitted.*(?:socket|connect)",
    re.IGNORECASE,
)


# Observed on Windows (wxc-exec 1.0.1, build 28000): cmd's `del`/`erase` and
# PowerShell's `Remove-Item` are denied even inside read-write folders, while
# Python, Node and git deletes, `rmdir /s` and `move /y` work.
_WINDOWS_DELETE_COMMAND = re.compile(r"(?:^|[\s&|(;\"'])(?:del|erase|Remove-Item|ri)\s", re.IGNORECASE)
_WINDOWS_DELETE_DENIED = re.compile(r"Access is denied")
WINDOWS_DELETE_HINT = (
    "Note: on Windows the sandbox denies `del`, `erase` and PowerShell's `Remove-Item`, even inside the allowed folders. "
    "Delete files with `python -c \"import os; os.remove('FILE')\"` and folders with `rmdir /s /q FOLDER` instead."
)
_UI_BLOCKED = re.compile(r"STATUS_DLL_INIT_FAILED|0xC0000142|Win32k", re.IGNORECASE)
_UI_CRASH_EXIT_CODES = frozenset({0xC0000142, 0xC06D007E})  # DLL init failed; delay-loaded module not found
_UI_CRASH_TEXT = re.compile(r"DLL\) initialization routine failed|Windows fatal exception: code 0xc06d007e", re.IGNORECASE)


class MXCSandbox(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem backend whose shell commands run inside an MXC sandbox.

    Every `execute` call starts one MXC process sandbox for that command and
    tears it down when the command exits. The operating system enforces the
    policy: writes are limited to `root_dir` and a scratch directory, reads to
    those plus the detected toolchain and `read_only_paths`, and the network
    is off unless `allow_network=True`. Files persist between commands because
    the sandbox runs over the real `root_dir`.

    The agent loop and model calls run on the host as usual. File tools run
    in the host process and are limited to `root_dir` by Deep Agents' path
    checks, which are not OS-enforced; only `execute` is.

    Example:
        ```python
        from deepagents import create_deep_agent
        from langchain_mxc import MXCSandbox

        agent = create_deep_agent(model="...", backend=MXCSandbox(root_dir="./my-project"))
        ```
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        *,
        read_only_paths: Iterable[str | os.PathLike[str]] = (),
        read_write_paths: Iterable[str | os.PathLike[str]] = (),
        allow_network: bool = False,
        allow_ui: bool = False,
        clipboard: str = "none",
        allow_input_injection: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        executor_path: str | os.PathLike[str] | None = None,
    ) -> None:
        """Create the backend.

        Args:
            root_dir: The project folder. Commands run with this as their
                working directory and may read and write anything inside it.
                File tools are limited to it as well.
            read_only_paths: Extra directories or files the sandbox may read.
                Nonexistent entries are ignored. `~` is expanded.
            read_write_paths: Extra directories or files the sandbox may write.
            allow_network: Allow outbound network access from commands.
                Off by default.
            allow_ui: Let commands create windows and use the GUI. Off by
                default (MXC's own default). Needed for GUI app tests or a
                headed browser. **On Windows, most toolchains need it:**
                `git`, `node`, PowerShell and `pytest` cannot start while it
                is off, and the response says so when that happens. Clipboard
                and input injection stay off unless set separately.
            clipboard: Clipboard access for commands: `"none"` (default),
                `"read"` (commands can read the user's clipboard), `"write"`
                (commands can replace it) or `"all"`.
            allow_input_injection: Let commands send synthetic keyboard and
                mouse input to other applications. Off by default.
                These three UI settings are enforced on Windows and macOS;
                Linux bubblewrap ignores them. On Windows, `clipboard` and
                `allow_input_injection` only take effect when `allow_ui=True`.
            env: Environment variables layered on top of the minimal sandbox
                environment (`PATH`, `HOME`/`TEMP` pointing at scratch, and on
                Windows `SYSTEMROOT` and `LOCALAPPDATA`).
            timeout: Default per-command timeout in seconds. Must be positive.
            max_output_bytes: Output beyond this many bytes is truncated.
            executor_path: Path to the MXC executor. When `None`, the
                `MXC_EXECUTOR_PATH` environment variable and the usual
                `node_modules/@microsoft/mxc-sdk/bin/<arch>/` locations are searched.

        Raises:
            ValueError: If `timeout` is not positive or `clipboard` is not a
                recognised level.
            MXCExecutorNotFoundError: If no executor can be found; the message
                explains how to install MXC.
            MXCUnsupportedPlatformError: On an unsupported operating system.
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        super().__init__(root_dir=os.fspath(root_dir), virtual_mode=True, max_file_size_mb=10)

        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._allow_network = bool(allow_network)
        self._sandbox_id = f"mxc-{uuid.uuid4().hex[:8]}"

        self._executor_path = find_executor(executor_path)

        scratch = Path(tempfile.mkdtemp(prefix=f"langchain-mxc-{self._sandbox_id}-")).resolve()
        self._scratch_dir = scratch
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(scratch), True)  # noqa: FBT003 - `ignore_errors`; `finalize` forwards only positionals

        self._host = HostInfo.current()
        self._policy = SandboxPolicy.create(
            root_dir=str(self.cwd),
            scratch_dir=str(scratch),
            read_only_paths=[os.fspath(p) for p in read_only_paths],
            read_write_paths=[os.fspath(p) for p in read_write_paths],
            allow_network=self._allow_network,
            env=dict(env) if env else None,
            host=self._host,
            allow_ui=allow_ui,
            clipboard=clipboard,
            allow_input_injection=allow_input_injection,
        )
        if self._policy.dropped_paths:
            logger.warning("MXCSandbox %s ignored nonexistent paths: %s", self._sandbox_id, ", ".join(self._policy.dropped_paths))

        self._probe_lock = threading.Lock()
        self._probe_result: ProbeResult | None = None

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> str:
        """Unique identifier for this backend instance (`mxc-<hex>`)."""
        return self._sandbox_id

    @property
    def executor_path(self) -> Path:
        """The MXC executor binary in use."""
        return self._executor_path

    @property
    def scratch_dir(self) -> Path:
        """Per-instance writable scratch directory (`HOME`/`TEMP` inside the sandbox)."""
        return self._scratch_dir

    @property
    def policy(self) -> SandboxPolicy:
        """The resolved sandbox policy."""
        return self._policy

    def build_request(self, command: str, timeout: float | None = None) -> dict:
        """Return the MXC request that `execute` would send for `command`."""
        return self._policy.build_request(command, timeout)

    # -- probe -------------------------------------------------------------

    def _ensure_probed(self) -> ProbeResult:
        """Run the host probe once and cache the verdict."""
        with self._probe_lock:
            if self._probe_result is None:
                self._probe_result = probe_executor(
                    self._executor_path,
                    lambda cmd: self._policy.build_request(cmd, 30),
                    allowed_paths=self._policy.allowed_paths,
                    platform=self._host.platform,
                    config_dir=self._scratch_dir,
                )
                if self._probe_result.ok:
                    logger.info("MXCSandbox %s: probe passed (%s)", self._sandbox_id, self._executor_path)
                else:
                    logger.error("MXCSandbox %s: probe failed: %s", self._sandbox_id, self._probe_result.reason)
            return self._probe_result

    # -- execute -----------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""Run a shell command inside an MXC sandbox.

        The command is wrapped as `/bin/sh -c <command>` (or `cmd.exe /d /s /c
        "<command>"` on Windows) and started by the MXC executor with the
        instance's policy. stdout and stderr are combined; stderr lines are
        prefixed with `[stderr]`. A non-zero exit code is appended as
        `Exit code: N`, and output longer than `max_output_bytes` is
        truncated. A timed-out command returns exit code 124.

        If the one-time host probe finds that MXC cannot enforce the policy,
        every call returns an error response and nothing is run: the backend
        never falls back to running the command unsandboxed.

        When the sandbox blocks a path or the network, one extra line is
        appended that the model can act on, for example
        `Blocked by sandbox policy: /Users/me/.ssh. Ask the user to grant access if this is needed.`

        Args:
            command: Shell command string.
            timeout: Per-command timeout in seconds; overrides the default.

        Returns:
            `ExecuteResponse` with `output`, `exit_code` and `truncated`.

        Raises:
            ValueError: If `timeout` is not positive.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1, truncated=False)

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        probe = self._ensure_probed()
        if not probe.ok:
            return ExecuteResponse(
                output=(
                    f"Error: MXC sandbox is not available on this host, so the command was not run. {probe.reason}. "
                    "Commands are never run outside the sandbox."
                ),
                exit_code=1,
                truncated=False,
            )

        request = self._policy.build_request(command, effective_timeout)
        try:
            result = run_request(
                self._executor_path,
                request,
                timeout=effective_timeout,
                grace=DEFAULT_GRACE_SECONDS,
                config_dir=self._scratch_dir,
            )
        except Exception as e:  # noqa: BLE001 - mirror LocalShellBackend: always return a response
            return ExecuteResponse(output=f"Error executing command ({type(e).__name__}): {e}", exit_code=1, truncated=False)

        return self._format_response(result, effective_timeout=effective_timeout, custom_timeout=timeout is not None, command=command)

    # -- response formatting ------------------------------------------------

    def _format_response(  # noqa: C901 - one branch per executor outcome
        self, result: ExecutorResult, *, effective_timeout: float, custom_timeout: bool, command: str = ""
    ) -> ExecuteResponse:
        outcome = classify_result(result)

        if outcome.kind in {"timeout", "host_timeout"}:
            if custom_timeout:
                msg = f"Error: Command timed out after {effective_timeout} seconds (custom timeout). The command may be stuck or require more time."
            else:
                msg = f"Error: Command timed out after {effective_timeout} seconds. For long-running commands, re-run using the timeout parameter."
            partial = self._combine(outcome.workload_stdout, outcome.workload_stderr)
            if partial != "<no output>":
                msg = f"{partial.rstrip()}\n\n{msg}"
            return ExecuteResponse(output=msg, exit_code=TIMEOUT_EXIT_CODE, truncated=False)

        if outcome.kind == "request_rejected":
            output = f"Error: MXC rejected the sandbox policy, so the command was not run: {outcome.message}\n" + BLOCKED_HINT.format(
                what="the request itself was rejected by MXC; check the MXCSandbox configuration"
            )
            return ExecuteResponse(output=output, exit_code=1, truncated=False)

        if outcome.kind == "backend_unavailable":
            output = f"Error: MXC cannot sandbox commands on this host, so the command was not run: {outcome.message}. Commands are never run outside the sandbox."  # noqa: E501
            return ExecuteResponse(output=output, exit_code=1, truncated=False)

        if outcome.kind == "backend_error":
            partial = self._combine(outcome.workload_stdout, outcome.workload_stderr)
            output = f"Error: MXC failed while running the command ({outcome.code or 'error'}): {outcome.message}"
            if not self._policy.allow_ui and _UI_BLOCKED.search(outcome.message):
                # Observed on Windows: programs that load the window system
                # (git.exe among them) cannot start while the UI is locked.
                output += "\n" + BLOCKED_HINT.format(
                    what="window-system access, which this program needs to start (MXCSandbox was created with allow_ui=False)"
                )
            if partial != "<no output>":
                output = f"{partial.rstrip()}\n\n{output}"
            return ExecuteResponse(output=output, exit_code=1, truncated=False)

        # workload ran
        exit_code = result.returncode if result.returncode is not None else 1
        output = self._combine(outcome.workload_stdout, outcome.workload_stderr)

        truncated = False
        output_bytes = output.encode("utf-8")
        if len(output_bytes) > self._max_output_bytes:
            output = output_bytes[: self._max_output_bytes].decode("utf-8", errors="ignore")
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        if exit_code != 0:
            output = f"{output.rstrip()}\n\nExit code: {exit_code}"

        hint = self._denial_hint(outcome, exit_code=exit_code, command=command)
        if hint:
            output = f"{output.rstrip()}\n{hint}"

        return ExecuteResponse(output=output, exit_code=exit_code, truncated=truncated)

    @staticmethod
    def _combine(stdout: str, stderr: str) -> str:
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.extend(f"[stderr] {line}" for line in stderr.strip().split("\n"))
        return "\n".join(parts) if parts else "<no output>"

    def _hidden_by_sandbox(self, path: str) -> bool:
        """Whether a path the command saw as missing exists on the host but is outside the policy."""
        if self._policy.is_allowed(path):
            return False
        return os.path.lexists(path) or os.path.isdir(os.path.dirname(path))

    def _denial_hint(self, outcome: Outcome, *, exit_code: int, command: str = "") -> str | None:  # noqa: C901 - one check per observed denial shape
        """One line naming what the sandbox blocked, or None."""
        text = f"{outcome.workload_stderr}\n{outcome.workload_stdout}"
        if (
            self._host.platform == "win32"
            and exit_code != 0
            and _WINDOWS_DELETE_COMMAND.search(command)
            and _WINDOWS_DELETE_DENIED.search(text)
            and not any(m.group("path").strip() for m in _DENIAL_PATH_BEFORE.finditer(text))
        ):
            return WINDOWS_DELETE_HINT
        blocked: list[str] = []

        candidates = [match.group("path").strip() for pattern in (_DENIAL_PATH_BEFORE, _DENIAL_PATH_AFTER) for match in pattern.finditer(text)]
        for path in candidates:
            if path and not self._policy.is_allowed(path) and path not in blocked:
                blocked.append(path)

        if self._host.platform == "linux" and exit_code != 0:
            for pattern in (_MISSING_PATH_BEFORE, _MISSING_PATH_AFTER):
                for match in pattern.finditer(text):
                    path = match.group("path").strip()
                    if path and path not in blocked and self._hidden_by_sandbox(path):
                        blocked.append(path)

        network_hit = not self._allow_network and _NETWORK_DENIAL.search(text) is not None

        if not blocked and not network_hit and exit_code != 0 and _DENIAL_ANY.search(outcome.workload_stderr) and not candidates:
            # EPERM with no path in the message: a file outside the policy or,
            # when the network is off, a socket call (Python prints the same
            # `PermissionError: [Errno 1] Operation not permitted` for both).
            blocked.append("a path outside the allowed folders" + (" or the network, which is off" if not self._allow_network else ""))

        if network_hit:
            blocked.append("network")

        # Observed on Windows: with the window system locked, a program that loads
        # it part-way through dies with one of these instead of the executor's own
        # start-up error (Python's `platform` and `ctypes` do).
        ui_crash = exit_code in _UI_CRASH_EXIT_CODES or _UI_CRASH_TEXT.search(outcome.workload_stderr) is not None
        if not blocked and self._host.platform == "win32" and not self._policy.allow_ui and ui_crash:
            blocked.append("window-system access, which this program appears to need (MXCSandbox was created with allow_ui=False)")

        if not blocked:
            return None
        shown = blocked[:3]
        if len(blocked) > len(shown):
            shown.append(f"and {len(blocked) - len(shown)} more")
        return BLOCKED_HINT.format(what=", ".join(shown))


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "TIMEOUT_EXIT_CODE", "MXCSandbox"]
