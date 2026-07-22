"""LangGraph server lifecycle management for the app.

Handles starting/stopping a `langgraph dev` server process and generating the
required `langgraph.json` configuration file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess  # noqa: S404
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code.config import _INHERITED_PYTHONPATH_ENV

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"

_EPHEMERAL_PORT = 0
"""Sentinel port meaning "let `start()` pick a free ephemeral port".

The server is internal and ephemeral — callers reach it via `ServerProcess.url`,
never a typed-in address — so it deliberately avoids binding the well-known
`langgraph dev` default (2024). Leaving 2024 free lets users run their own
`langgraph dev` projects alongside `deepagents-code` without a port collision.
"""

_HEALTH_POLL_INTERVAL_LOCAL = 0.1

_HEALTH_POLL_INTERVAL_REMOTE = 0.3

_HEALTH_TIMEOUT = 60

_SHUTDOWN_TIMEOUT = 3
"""Seconds to wait for a graceful SIGTERM exit before escalating to SIGKILL."""

_SIGKILL_TIMEOUT = 2
"""Seconds to wait for the group/process to exit after SIGKILL."""

_PROCESS_GROUP_POLL_INTERVAL = 0.05

_LOG_TAIL_CHARS = 3000
"""Max chars of subprocess log appended to the early-exit `RuntimeError` message.

Enough to carry a Python traceback without flooding the TUI banner when it
surfaces via `ServerStartFailed`.
"""

_STARTUP_ERROR_MARKER = "DEEPAGENTS_STARTUP_ERROR:"
"""Machine-readable prefix emitted by the server subprocess for known startup errors."""

_SERVER_ENV_DENYLIST = frozenset(
    {
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "GIT_ASKPASS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "PYTHONEXECUTABLE",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SSH_ASKPASS",
    }
)
"""Inherited env keys that can alter subprocess startup behavior.

`PYTHONPATH` is stripped here so an inherited launch value cannot land on the
server interpreter's `sys.path` during startup, where a path inside an untrusted
project could shadow a stdlib/third-party module and run before any approval
gate. A user who launched with `PYTHONPATH` still wants it for their agent
`execute` commands, so `_build_server_env` relays the value via
`config._INHERITED_PYTHONPATH_ENV` and `agent._apply_inherited_pythonpath`
re-applies it only to the approval-gated shell backend.
"""


def _port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use.

    Args:
        host: Host to check.
        port: Port to check.

    Returns:
        `True` if the port is in use.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return True
        else:
            return False


def _find_free_port(host: str) -> int:
    """Find a free port on the given host.

    Args:
        host: Host to bind to.

    Returns:
        An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def get_server_url(host: str = _DEFAULT_HOST, port: int = _EPHEMERAL_PORT) -> str:
    """Build the server base URL.

    Args:
        host: Server host.
        port: Server port.

    Returns:
        Base URL string.
    """
    return f"http://{host}:{port}"


def _extract_startup_error_marker(output: str) -> str | None:
    """Extract a marked startup error from subprocess output.

    Args:
        output: Combined stdout/stderr captured from the server subprocess.

    Returns:
        The marked startup error message, or `None` if no marker was emitted.
    """
    for line in reversed(output.splitlines()):
        if _STARTUP_ERROR_MARKER in line:
            _, summary = line.rsplit(_STARTUP_ERROR_MARKER, 1)
            return summary.strip() or None
    return None


def generate_langgraph_json(
    output_dir: str | Path,
    *,
    graph_ref: str = "deepagents_code.server_graph:make_graph",
    env_file: str | None = None,
    checkpointer_path: str | None = None,
) -> Path:
    """Generate a `langgraph.json` config file for `langgraph dev`.

    Args:
        output_dir: Directory to write the config file.
        graph_ref: Python "module:attribute" reference to the graph, where the
            attribute is a graph factory (e.g. `make_graph`) or a graph object.
        env_file: Optional path to an env file.
        checkpointer_path: Import path to an async context manager that yields a
            `BaseCheckpointSaver`. When set, the server persists checkpoint data
            to disk instead of in-memory.

    Returns:
        Path to the generated config file.
    """
    config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {"agent": graph_ref},
    }
    if env_file:
        config["env"] = env_file
    if checkpointer_path:
        config["checkpointer"] = {"path": checkpointer_path}

    output_path = Path(output_dir) / "langgraph.json"
    output_path.write_text(json.dumps(config, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Scoped env-var management
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _scoped_env_overrides(
    overrides: dict[str, str],
) -> Iterator[None]:
    """Apply env-var overrides, rolling back only on exception.

    Separates the concern of temporary `os.environ` mutations from subprocess
    management, making both independently testable.

    On normal exit the overrides are left in place (the caller "keeps"
    them). On exception the previous values are restored so the next attempt
    starts from a known-good state.

    Args:
        overrides: Key/value pairs to set in `os.environ`.

    Yields:
        Control to the caller.
    """
    prev: dict[str, str | None] = {}
    for key, val in overrides.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        yield
    except Exception:
        for key, old_val in prev.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
        raise


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------


async def wait_for_server_healthy(
    url: str,
    *,
    timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    process: subprocess.Popen | None = None,
    read_log: Callable[[], str] | None = None,
    local: bool = False,
) -> None:
    """Poll a LangGraph server health endpoint until it responds.

    Args:
        url: Server base URL (health endpoint is `{url}/ok`).
        timeout: Max seconds to wait.
        process: Optional subprocess handle; if the process exits early
            we fail fast instead of waiting for the timeout.
        read_log: Optional callable returning log file contents (for
            error messages on early exit).
        local: Use a shorter poll interval for local servers.

    Raises:
        RuntimeError: If the server doesn't become healthy in time.
    """
    import httpx

    poll_interval = (
        _HEALTH_POLL_INTERVAL_LOCAL if local else _HEALTH_POLL_INTERVAL_REMOTE
    )
    health_url = f"{url}/ok"
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    last_exc: Exception | None = None

    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                output = read_log() if read_log else ""
                msg = f"Server process exited with code {process.returncode}"
                if output:
                    summary = _extract_startup_error_marker(output)
                    if summary:
                        msg += f": {summary}"
                    msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                raise RuntimeError(msg)

            try:
                resp = await client.get(health_url, timeout=2)
                if resp.status_code == 200:  # noqa: PLR2004
                    logger.info("Server is healthy at %s", url)
                    return
                last_status = resp.status_code
                logger.debug("Health check returned status %d", resp.status_code)
            except (httpx.TransportError, OSError) as exc:
                logger.debug("Health check attempt failed: %s", exc)
                last_exc = exc

            await asyncio.sleep(poll_interval)

    msg = f"Server did not become healthy within {timeout}s"
    if last_status is not None:
        msg += f" (last status: {last_status})"
    elif last_exc is not None:
        msg += f" (last error: {last_exc})"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Server command / env construction
# ---------------------------------------------------------------------------


def _build_server_cmd(config_path: Path, *, host: str, port: int) -> list[str]:
    """Build the `langgraph dev` command line.

    Args:
        config_path: Path to the `langgraph.json` config file.
        host: Host to bind.
        port: Port to bind.

    Returns:
        Command argv list.
    """
    return [
        sys.executable,
        "-m",
        "langgraph_cli",
        "dev",
        "--host",
        host,
        "--port",
        str(port),
        "--no-browser",
        "--no-reload",
        "--config",
        str(config_path),
    ]


def _build_server_env() -> dict[str, str]:
    """Build the environment dict for the server subprocess.

    Copies `os.environ`, sets required flags, and strips variables that are not
    needed or can alter subprocess startup behavior.

    A launch-time `PYTHONPATH` is captured into `config._INHERITED_PYTHONPATH_ENV`
    before being stripped, so the value never reaches the server interpreter's
    `sys.path` but can still be re-applied to agent `execute` commands downstream.

    Returns:
        Environment dict for `subprocess.Popen`.
    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LANGGRAPH_AUTH_TYPE"] = "noop"

    # Capture a launch-time PYTHONPATH before stripping it. Never trust an
    # inherited carrier var: pop it first, then set it only from the real value.
    env.pop(_INHERITED_PYTHONPATH_ENV, None)
    inherited_pythonpath = os.environ.get("PYTHONPATH")

    for key in (
        "LANGGRAPH_AUTH",
        "LANGGRAPH_CLOUD_LICENSE_KEY",
        "LANGSMITH_CONTROL_PLANE_API_KEY",
        "LANGSMITH_TENANT_ID",
        *_SERVER_ENV_DENYLIST,
    ):
        env.pop(key, None)

    if inherited_pythonpath is not None:
        env[_INHERITED_PYTHONPATH_ENV] = inherited_pythonpath
    return env


# ---------------------------------------------------------------------------
# Process-group teardown
# ---------------------------------------------------------------------------


def _server_process_group(pid: int) -> int | None:
    """Return the server's own process group id to signal, or `None`.

    The server is spawned with `start_new_session=True` on POSIX, so it leads
    its own session and process group (its pgid equals its pid). Signaling that
    group reaches the whole `langgraph dev` process tree, so descendants receive
    the same shutdown signals as the root rather than being left running when
    only the root is signaled.

    Returns `None` — meaning "signal only the root process" — on Windows (no
    POSIX process groups) and whenever the server is not the leader of its own
    dedicated group. As a defensive check, the `pgid == os.getpgid(0)` clause
    also refuses to return dcode's own group, so the group handed back can never
    be the one whose termination would take down the TUI.

    Args:
        pid: Process id of the server subprocess.

    Returns:
        The server's dedicated process group id, or `None` to fall back to
        signaling just the root process.
    """
    if sys.platform == "win32":
        return None
    try:
        pgid = os.getpgid(pid)
        own_pgid = os.getpgid(0)
    except ProcessLookupError:
        # The process already exited; there is no group left to signal.
        return None
    except OSError:
        # Resolving the group failed unexpectedly (getpgid on an owned child
        # should not). Fall back to root-only signaling, but surface it so a
        # silently orphaned descendant tree is diagnosable rather than invisible.
        logger.warning(
            "Could not resolve process group for pid=%d; "
            "falling back to root-only signaling",
            pid,
            exc_info=True,
        )
        return None
    if pgid != pid or pgid == own_pgid:
        return None
    return pgid


def _wait_for_process_group_exit(
    process: subprocess.Popen[Any], pgid: int, timeout: float
) -> bool:
    """Wait until every process in a POSIX process group has exited.

    `Popen.wait()` only observes the group leader. Poll it on every pass so an
    exited leader does not remain a zombie and keep the group probe alive, then
    continue probing because descendants may remain after the leader exits.

    Args:
        process: The group leader, reaped as soon as it exits.
        pgid: Process group id to probe.
        timeout: Maximum seconds to wait for the whole group.

    Returns:
        `True` when the group is gone, or `False` on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        # `poll()` reaps an exited leader without blocking. Until that happens,
        # its zombie entry keeps `killpg(..., 0)` reporting the group as alive.
        process.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            process.wait()
            return True
        except PermissionError:
            # The group still exists even if the probe is not permitted.
            pass

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PROCESS_GROUP_POLL_INTERVAL, remaining))


def _terminate_server_process(process: subprocess.Popen[Any]) -> None:
    """Terminate the `langgraph dev` server and its descendants.

    Sends SIGTERM, waits `_SHUTDOWN_TIMEOUT` for a graceful exit, then escalates
    to SIGKILL. On POSIX the whole detached process group is signaled via
    `os.killpg`, and teardown waits for the entire group to exit — not just the
    root — so a child that outlives the `langgraph dev` root is still escalated
    to SIGKILL rather than orphaned. On Windows (or if the server is not its own
    group leader) only the root process is signaled. `_server_process_group`
    guarantees dcode's own process group is never targeted.

    Args:
        process: The running server subprocess to terminate.
    """
    pid = process.pid
    pgid = _server_process_group(pid)
    scope = "process group" if pgid is not None else "process"

    logger.info("Stopping langgraph dev server (pid=%d)", pid)
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
            stopped = _wait_for_process_group_exit(process, pgid, _SHUTDOWN_TIMEOUT)
        else:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                stopped = False
            else:
                stopped = True
    except ProcessLookupError:
        # The server exited before the SIGTERM landed; nothing left to reap.
        logger.debug("Server %s pid=%d already exited before SIGTERM", scope, pid)
        return
    except OSError:
        # SIGTERM could not be delivered (e.g. EPERM). We never reach the SIGKILL
        # escalation, so the server is left running — report it with the same
        # fidelity as a failed SIGKILL rather than a bare "error stopping".
        logger.exception(
            "Failed to signal server %s pid=%d; it may be orphaned", scope, pid
        )
        return

    if stopped:
        return

    logger.warning("Server did not stop gracefully, killing %s", scope)
    # Guard escalation explicitly: `ProcessLookupError` means the group exited
    # just before SIGKILL, while any other `OSError` means it may be orphaned.
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
            if not _wait_for_process_group_exit(process, pgid, _SIGKILL_TIMEOUT):
                logger.warning(
                    "Server %s pid=%d did not exit after SIGKILL", scope, pid
                )
        else:
            process.kill()
            try:
                process.wait(timeout=_SIGKILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server %s pid=%d did not exit after SIGKILL", scope, pid
                )
    except ProcessLookupError:
        logger.debug("Server %s pid=%d already exited before SIGKILL", scope, pid)
    except OSError:
        logger.exception(
            "Failed to SIGKILL server %s pid=%d; it may be orphaned",
            scope,
            pid,
        )


# ---------------------------------------------------------------------------
# ServerProcess
# ---------------------------------------------------------------------------


class ServerProcess:
    """Manages a `langgraph dev` server subprocess.

    Focuses on subprocess lifecycle (start, stop, restart) and health checking.
    Env-var management for restarts (e.g. configuration changes requiring a full
    restart) is handled by `_scoped_env_overrides`, keeping this class focused
    on process management.
    """

    def __init__(
        self,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _EPHEMERAL_PORT,
        config_dir: str | Path | None = None,
        owns_config_dir: bool = False,
        scaffold: Callable[[Path], None] | None = None,
    ) -> None:
        """Initialize server process manager.

        Args:
            host: Host to bind the server to.
            port: Initial port to bind the server to. Defaults to
                `_EPHEMERAL_PORT` (0), so `start()` picks a free port and avoids
                squatting the well-known `langgraph dev` default (2024).

                An explicit port is honored, but `start()` still falls back to a
                free port if it is already in use.
            config_dir: Directory containing `langgraph.json`.
            owns_config_dir: When `True`, the server will delete `config_dir`
                on `stop()`.
            scaffold: Optional callable that (re)generates the working
                directory's `langgraph.json` and supporting files. When the
                config is missing at `start()` (e.g. the temp dir was purged
                between the initial boot and a later `/restart`), it is invoked
                to rebuild the workspace instead of failing.
        """
        self.host = host
        self.port = port
        self.config_dir = Path(config_dir) if config_dir else None
        self._owns_config_dir = owns_config_dir
        self._scaffold = scaffold
        self._process: subprocess.Popen | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._log_file: tempfile.NamedTemporaryFile | None = None  # ty: ignore[invalid-type-form]
        self._env_overrides: dict[str, str] = {}
        self._persistent_env_overrides: dict[str, str] = {}
        # Async lifecycle calls must be serialized by task, not by OS thread:
        # every coroutine on an event loop runs on the same thread, so an
        # RLock would let unrelated tasks enter while another task is awaiting.
        self._lifecycle_lock = asyncio.Lock()
        # Synchronous shutdown can also run from a fallback worker thread. Keep
        # its critical sections short and never hold this lock across an await.
        self._state_lock = threading.Lock()
        self._stopped = False
        self._stop_generation = 0

    @property
    def url(self) -> str:
        """Server base URL."""
        return get_server_url(self.host, self.port)

    @property
    def running(self) -> bool:
        """Whether the server process is running."""
        with self._state_lock:
            return self._running_locked()

    def _running_locked(self) -> bool:
        """Return whether the process is running while `_state_lock` is held."""
        return self._process is not None and self._process.poll() is None

    def _read_log_file(self) -> str:
        """Read the server log file contents.

        Returns:
            Log file contents as a string (may be empty).
        """
        with self._state_lock:
            if self._log_file is None:
                return ""
            try:
                self._log_file.flush()
                return Path(self._log_file.name).read_text(
                    encoding="utf-8", errors="replace"
                )
            # `ValueError` covers a flush/read on a closed handle ("I/O
            # operation on closed file"): re-checking `is None` under
            # `_state_lock` makes a closed-but-non-None handle nearly
            # unreachable, so this is defensive. `read_text(errors="replace")`
            # cannot raise `ValueError` for decoding, so the catch stays narrow
            # in practice.
            except (OSError, ValueError):
                logger.warning(
                    "Failed to read server log file %s",
                    self._log_file.name,
                    exc_info=True,
                )
                return ""

    async def start(
        self,
        *,
        timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    ) -> None:
        """Start the `langgraph dev` server and wait for it to be healthy.

        Args:
            timeout: Max seconds to wait for the server to become healthy.

        Raises:
            RuntimeError: If the server fails to start or become healthy.
        """  # noqa: DOC502  # `RuntimeError` propagates from `_start`
        async with self._lifecycle_lock:
            await self._start(timeout=timeout)

    async def _start(
        self,
        *,
        timeout: float,  # noqa: ASYNC109
        expected_stop_generation: int | None = None,
    ) -> None:
        """Start while the caller owns `_lifecycle_lock`.

        Args:
            timeout: Max seconds to wait for the server to become healthy.
            expected_stop_generation: Generation captured before a restart's
                stop phase. If synchronous terminal shutdown ran meanwhile,
                abort instead of resurrecting the subprocess.
        """
        process = self._spawn_process(
            expected_stop_generation=expected_stop_generation,
        )
        if process is None:
            return
        started = False
        try:
            await wait_for_server_healthy(
                self.url,
                timeout=timeout,
                process=process,
                read_log=self._read_log_file,
                local=True,
            )
            started = True
        finally:
            if not started:
                # Reap the subprocess we just spawned if startup did not
                # complete — including cancellation (e.g. Ctrl+D / SIGINT before
                # the health check returns). A `finally` rather than `except
                # Exception` is deliberate: `asyncio.CancelledError` is a
                # `BaseException`, so an `except Exception` guard would skip this
                # and orphan the process. Offload `stop()` because terminating a
                # subprocess can block for several seconds; restart cancellation
                # runs this path on Textual's event loop. The inner guard stops a
                # `stop()` error from masking the exception already propagating;
                # `stop()` is effectively non-raising today, so if it does fire it
                # signals an unexpected leak — hence `error`, not `warning`.
                try:
                    await asyncio.to_thread(self.stop)
                except Exception:
                    logger.exception(
                        "Error stopping server during startup cleanup",
                    )

    def _spawn_process(
        self,
        *,
        expected_stop_generation: int | None,
    ) -> subprocess.Popen | None:
        """Synchronously prepare and spawn the subprocess under `_state_lock`.

        Args:
            expected_stop_generation: Optional shutdown generation required by
                a restart.

        Returns:
            The new process, or `None` when one is already running.

        Raises:
            asyncio.CancelledError: If terminal shutdown preempted a restart.
            RuntimeError: If the server workspace cannot be prepared.
        """
        with self._state_lock:
            if (
                expected_stop_generation is not None
                and self._stop_generation != expected_stop_generation
            ):
                raise asyncio.CancelledError
            if self._running_locked():
                return None
            self._stopped = False

            work_dir = self.config_dir
            if work_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(
                    prefix="deepagents_server_"
                )
                work_dir = Path(self._temp_dir.name)

            config_path = work_dir / "langgraph.json"
            if not config_path.exists() and self._scaffold is not None:
                logger.info("langgraph.json missing in %s; rescaffolding", work_dir)
                try:
                    work_dir.mkdir(parents=True, exist_ok=True)
                    self._scaffold(work_dir)
                except OSError as exc:
                    msg = f"Failed to rescaffold server workspace at {work_dir}: {exc}"
                    raise RuntimeError(msg) from exc
            if not config_path.exists():
                if self._scaffold is not None:
                    contents = sorted(p.name for p in work_dir.iterdir())
                    msg = (
                        f"Rescaffolding {work_dir} did not produce langgraph.json "
                        f"(directory contents: {contents})."
                    )
                else:
                    msg = (
                        f"langgraph.json not found in {work_dir}. "
                        "Call generate_langgraph_json() first."
                    )
                raise RuntimeError(msg)

            if self.port == _EPHEMERAL_PORT:
                self.port = _find_free_port(self.host)
                logger.info(
                    "Using ephemeral port %d for langgraph dev server", self.port
                )
            elif _port_in_use(self.host, self.port):
                self.port = _find_free_port(self.host)
                logger.info("Requested port in use, using port %d instead", self.port)

            cmd = _build_server_cmd(config_path, host=self.host, port=self.port)
            env = _build_server_env()
            env.update(self._persistent_env_overrides)
            env.update(self._env_overrides)

            logger.info("Starting langgraph dev server: %s", " ".join(cmd))
            self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                prefix="deepagents_server_log_",
                suffix=".txt",
                delete=False,
                mode="w",
                encoding="utf-8",
            )
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(work_dir),
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=(sys.platform != "win32"),
            )
            return self._process

    async def wait_for_graph_ready(
        self,
        graph_name: str = "agent",
        *,
        timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    ) -> None:
        """Resolve the served graph once so lazy startup failures surface early.

        Args:
            graph_name: Registered graph name from `langgraph.json`.
            timeout: Max seconds to wait for the graph readiness request.

        Raises:
            RuntimeError: If the server process exits or the graph endpoint
                does not return a successful response.
        """
        import httpx

        if self._process is None:
            msg = "Server process is not running"
            raise RuntimeError(msg)

        graph_url = f"{self.url}/assistants/{quote(graph_name, safe='')}/graph"
        deadline = time.monotonic() + timeout

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    msg = f"Server process exited with code {self._process.returncode}"
                    output = self._read_log_file()
                    if output:
                        summary = _extract_startup_error_marker(output)
                        if summary:
                            msg += f": {summary}"
                        msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                    raise RuntimeError(msg)

                remaining = max(0.1, deadline - time.monotonic())
                try:
                    resp = await client.get(graph_url, timeout=remaining)
                except (httpx.TransportError, httpx.TimeoutException, OSError) as exc:
                    output = self._read_log_file()
                    summary = _extract_startup_error_marker(output)
                    if self._process.poll() is not None:
                        msg = (
                            f"Server process exited with code "
                            f"{self._process.returncode}"
                        )
                    else:
                        msg = (
                            f"Server graph '{graph_name}' did not initialize within "
                            f"{timeout}s"
                        )
                    if summary:
                        msg += f": {summary}"
                    if output:
                        msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                    raise RuntimeError(msg) from exc

                if resp.status_code == 200:  # noqa: PLR2004
                    logger.info("Server graph %s is ready at %s", graph_name, self.url)
                    return

                output = self._read_log_file()
                msg = (
                    f"Server graph '{graph_name}' failed readiness check "
                    f"(status: {resp.status_code})"
                )
                summary = _extract_startup_error_marker(output)
                if summary:
                    msg += f": {summary}"
                if output:
                    msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                raise RuntimeError(msg)

        msg = f"Server graph '{graph_name}' did not initialize within {timeout}s"
        raise RuntimeError(msg)

    def _stop_process(self) -> None:
        """Stop only the server subprocess and its log file.

        Unlike `stop()`, this does NOT clean up the config directory or temp
        directory, so the server can be restarted with the same config.
        """
        with self._state_lock:
            self._stop_process_locked()

    def _stop_process_locked(self) -> None:
        """Stop the subprocess while `_state_lock` is held."""
        if self._process is None:
            return

        if self._process.poll() is None:
            _terminate_server_process(self._process)
            # `_terminate_server_process` is best-effort. If the process is still
            # alive here (e.g. SIGKILL failed with EPERM), then once we drop the
            # handle below we can no longer observe or reap this pid, so surface
            # the still-running process rather than clearing state as if shutdown
            # succeeded.
            if self._process.poll() is None:
                logger.warning(
                    "Dropping handle to server pid=%d that is still running; "
                    "it may be orphaned",
                    self._process.pid,
                )

        self._process = None

        if self._log_file is not None:
            log_path = Path(self._log_file.name)
            try:
                self._log_file.close()
            except OSError:
                logger.debug("Failed to close log file", exc_info=True)

            from deepagents_code._env_vars import DEBUG, is_env_truthy

            if is_env_truthy(DEBUG):
                print(  # noqa: T201
                    f"Server log preserved at: {log_path}",
                    file=sys.stderr,
                )
            else:
                try:
                    log_path.unlink()
                except OSError:
                    logger.debug("Failed to clean up log file", exc_info=True)
            self._log_file = None

    def stop(self) -> None:
        """Stop the server process and clean up all resources.

        Idempotent and safe to call concurrently. The synchronous state lock
        prevents process teardown and resource cleanup from interleaving, while
        the generation counter prevents an in-flight async restart from
        spawning a replacement after this terminal stop wins the race.
        """
        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_generation += 1

            self._stop_process_locked()

            if self._temp_dir is not None:
                try:
                    self._temp_dir.cleanup()
                except OSError:
                    # Debug, not warning (unlike the config dir below): a
                    # directory under the OS temp root is eventually reclaimed
                    # by the OS temp reaper (systemd-tmpfiles / tmpwatch / the
                    # macOS periodic cleanup), so a failure here is not the
                    # unrecoverable, never-retried leak the config dir would be.
                    # (`TemporaryDirectory.cleanup()` detaches its finalizer
                    # before removal, so a failed explicit cleanup is not
                    # retried at GC — the OS reaper is what reclaims it.)
                    logger.debug("Failed to clean up temp dir", exc_info=True)
                self._temp_dir = None

            if self._owns_config_dir and self.config_dir is not None:
                import shutil

                try:
                    shutil.rmtree(self.config_dir)
                except OSError:
                    # Warning, not debug: cleanup runs exactly once and is never
                    # retried, and this is a process-owned dir that may hold
                    # session state, so a persistent failure (a real leak) must
                    # be visible to an operator.
                    logger.warning(
                        "Failed to clean up config dir %s",
                        self.config_dir,
                        exc_info=True,
                    )
                self._owns_config_dir = False

    def update_env(self, **overrides: str) -> None:
        """Stage env var overrides to apply on the next `restart()`.

        These are applied to `os.environ` immediately before the subprocess
        starts, keeping mutation scoped to the restart call.

        Args:
            **overrides: Key/value env var pairs
                (e.g., `DEEPAGENTS_CODE_SERVER_MODEL="anthropic:claude-sonnet-4-6"`).
        """
        self._env_overrides.update(overrides)

    def persist_env(self, **overrides: str) -> None:
        """Persist env var overrides for every future subprocess start.

        Args:
            **overrides: Key/value env var pairs that should be passed to all
                future server subprocesses.

        Raises:
            ValueError: If an override is not an app-owned server env var.
        """
        invalid = [key for key in overrides if not key.startswith(SERVER_ENV_PREFIX)]
        if invalid:
            msg = (
                "persistent server env overrides must use the "
                f"{SERVER_ENV_PREFIX!r} prefix"
            )
            raise ValueError(msg)
        self._persistent_env_overrides.update(overrides)

    async def restart(self, *, timeout: float = _HEALTH_TIMEOUT) -> None:  # noqa: ASYNC109
        """Restart the server process, reusing the existing config directory.

        Stops the subprocess, then starts a new one. Any env overrides staged
        via `update_env()` are applied within a `_scoped_env_overrides` context
        manager so that failures automatically roll back the environment to the
        last known-good state.

        Args:
            timeout: Max seconds to wait for the server to become healthy.

        Raises:
            asyncio.CancelledError: Either if the restart task is cancelled
                (the blocking subprocess cleanup is awaited to completion
                first), or if a terminal `stop()` bumped the stop generation
                during cleanup — in which case `_start` aborts rather than
                resurrecting the subprocess the terminal stop just tore down.
            RuntimeError: If workspace preparation, process startup, or the
                server health check fails.
        """  # noqa: DOC502  # RuntimeError propagates from _start().
        logger.info("Restarting langgraph dev server")
        async with self._lifecycle_lock:
            with self._state_lock:
                stop_generation = self._stop_generation
            # Offload the synchronous subprocess shutdown (it blocks up to
            # `_SHUTDOWN_TIMEOUT` + SIGKILL grace waiting on `process.wait`) so
            # the caller's event loop — the Textual reactor for `/restart` —
            # keeps processing input instead of freezing the TUI. Shield the
            # thread and await it after cancellation so no later lifecycle call
            # can mutate process state while cleanup is still running.
            stop_task = asyncio.create_task(asyncio.to_thread(self._stop_process))
            try:
                await asyncio.shield(stop_task)
            except asyncio.CancelledError:
                await stop_task
                raise

            with _scoped_env_overrides(self._env_overrides):
                await self._start(
                    timeout=timeout,
                    expected_stop_generation=stop_generation,
                )

            self._env_overrides.clear()

    async def __aenter__(self) -> Self:
        """Async context manager entry.

        Returns:
            The server process instance.
        """
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        self.stop()
