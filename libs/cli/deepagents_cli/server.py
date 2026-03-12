"""LangGraph server lifecycle management for the CLI.

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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 2024
_HEALTH_POLL_INTERVAL = 0.3
_HEALTH_TIMEOUT = 60
_SHUTDOWN_TIMEOUT = 5


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


def get_server_url(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> str:
    """Build the server base URL.

    Args:
        host: Server host.
        port: Server port.

    Returns:
        Base URL string.
    """
    return f"http://{host}:{port}"


def generate_langgraph_json(
    output_dir: str | Path,
    *,
    graph_ref: str = "./server_graph.py:graph",
    env_file: str | None = None,
    checkpointer_path: str | None = None,
) -> Path:
    """Generate a `langgraph.json` config file for `langgraph dev`.

    Args:
        output_dir: Directory to write the config file.
        graph_ref: Python module:variable reference to the graph.
        env_file: Optional path to an env file.
        checkpointer_path: Import path to an async context manager that yields a
            `BaseCheckpointSaver`. When set, the server persists checkpoint data
            to disk instead of in-memory.

    Returns:
        Path to the generated config file.
    """
    config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {
            "agent": graph_ref,
        },
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
    """Context manager that applies env-var overrides and rolls them back.

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
) -> None:
    """Poll a LangGraph server health endpoint until it responds.

    Args:
        url: Server base URL (health endpoint is `{url}/ok`).
        timeout: Max seconds to wait.
        process: Optional subprocess handle; if the process exits early
            we fail fast instead of waiting for the timeout.
        read_log: Optional callable returning log file contents (for
            error messages on early exit).

    Raises:
        RuntimeError: If the server doesn't become healthy in time.
    """
    import httpx

    health_url = f"{url}/ok"
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            output = read_log() if read_log else ""
            msg = f"Server process exited with code {process.returncode}"
            if output:
                msg += f"\n{output[-3000:]}"
            raise RuntimeError(msg)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(health_url, timeout=2)
                if resp.status_code == 200:  # noqa: PLR2004
                    logger.info("Server is healthy at %s", url)
                    return
                last_status = resp.status_code
                logger.debug("Health check returned status %d", resp.status_code)
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            logger.debug("Health check attempt failed: %s", exc)
            last_exc = exc

        await asyncio.sleep(_HEALTH_POLL_INTERVAL)

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

    Copies `os.environ`, sets required flags, and strips auth-related variables
    that are not needed (and could interfere) for the local dev server.

    Returns:
        Environment dict for `subprocess.Popen`.
    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LANGGRAPH_AUTH_TYPE"] = "noop"
    for key in (
        "LANGGRAPH_AUTH",
        "LANGGRAPH_CLOUD_LICENSE_KEY",
        "LANGSMITH_CONTROL_PLANE_API_KEY",
        "LANGSMITH_TENANT_ID",
    ):
        env.pop(key, None)
    return env


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
        port: int = _DEFAULT_PORT,
        config_dir: str | Path | None = None,
        owns_config_dir: bool = False,
    ) -> None:
        """Initialize server process manager.

        Args:
            host: Host to bind the server to.
            port: Initial port to bind the server to.

                May be reassigned automatically by `start()` if the port is
                already in use.
            config_dir: Directory containing `langgraph.json`.
            owns_config_dir: When `True`, the server will delete `config_dir`
                on `stop()`.
        """
        self.host = host
        self.port = port
        self.config_dir = Path(config_dir) if config_dir else None
        self._owns_config_dir = owns_config_dir
        self._process: subprocess.Popen | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._log_file: tempfile.NamedTemporaryFile | None = None  # type: ignore[type-arg]
        self._env_overrides: dict[str, str] = {}

    @property
    def url(self) -> str:
        """Server base URL."""
        return get_server_url(self.host, self.port)

    @property
    def running(self) -> bool:
        """Whether the server process is running."""
        return self._process is not None and self._process.poll() is None

    def _read_log_file(self) -> str:
        """Read the server log file contents.

        Returns:
            Log file contents as a string (may be empty).
        """
        if self._log_file is None:
            return ""
        try:
            self._log_file.flush()
            return Path(self._log_file.name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
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
        """
        if self.running:
            return

        work_dir = self.config_dir
        if work_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="deepagents_server_")
            work_dir = Path(self._temp_dir.name)

        config_path = work_dir / "langgraph.json"
        if not config_path.exists():
            msg = (
                f"langgraph.json not found in {work_dir}. "
                "Call generate_langgraph_json() first."
            )
            raise RuntimeError(msg)

        if _port_in_use(self.host, self.port):
            self.port = _find_free_port(self.host)
            logger.info("Default port in use, using port %d instead", self.port)

        cmd = _build_server_cmd(config_path, host=self.host, port=self.port)
        env = _build_server_env()

        logger.info("Starting langgraph dev server: %s", " ".join(cmd))
        self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="deepagents_server_log_",
            suffix=".txt",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        self._process = subprocess.Popen(  # noqa: S603, ASYNC220
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            await wait_for_server_healthy(
                self.url,
                timeout=timeout,
                process=self._process,
                read_log=self._read_log_file,
            )
        except Exception:
            self.stop()
            raise

    def _stop_process(self) -> None:
        """Stop only the server subprocess and its log file.

        Unlike `stop()`, this does NOT clean up the config directory or temp
        directory, so the server can be restarted with the same config.
        """
        if self._process is None:
            return

        if self._process.poll() is None:
            logger.info("Stopping langgraph dev server (pid=%d)", self._process.pid)
            try:
                self._process.send_signal(signal.SIGTERM)
                self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, killing")
                self._process.kill()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Server process pid=%d did not exit after SIGKILL",
                        self._process.pid,
                    )
            except OSError:
                logger.warning("Error stopping server", exc_info=True)

        self._process = None

        if self._log_file is not None:
            try:
                self._log_file.close()
                Path(self._log_file.name).unlink()
            except OSError:
                logger.debug("Failed to clean up log file", exc_info=True)
            self._log_file = None

    def stop(self) -> None:
        """Stop the server process and clean up all resources."""
        self._stop_process()

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                logger.debug("Failed to clean up temp dir", exc_info=True)
            self._temp_dir = None

        if self._owns_config_dir and self.config_dir is not None:
            import shutil

            try:
                shutil.rmtree(self.config_dir, ignore_errors=True)
            except OSError:
                logger.debug(
                    "Failed to clean up config dir %s", self.config_dir, exc_info=True
                )
            self._owns_config_dir = False

    def update_env(self, **overrides: str) -> None:
        """Stage env var overrides to apply on the next `restart()`.

        These are applied to `os.environ` immediately before the subprocess
        starts, keeping mutation scoped to the restart call.

        Args:
            **overrides: Key/value env var pairs
                (e.g., `DA_SERVER_MODEL="anthropic:claude-sonnet-4-6"`).
        """
        self._env_overrides.update(overrides)

    async def restart(self, *, timeout: float = _HEALTH_TIMEOUT) -> None:  # noqa: ASYNC109
        """Restart the server process, reusing the existing config directory.

        Stops the subprocess, then starts a new one. Any env overrides staged
        via `update_env()` are applied within a `_scoped_env_overrides` context
        manager so that failures automatically roll back the environment to the
        last known-good state.

        Args:
            timeout: Max seconds to wait for the server to become healthy.
        """
        logger.info("Restarting langgraph dev server")
        self._stop_process()

        with _scoped_env_overrides(self._env_overrides):
            await self.start(timeout=timeout)

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
