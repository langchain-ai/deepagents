"""LangGraph server lifecycle management for the CLI.

Handles starting/stopping a `langgraph dev` server process and generating
the required `langgraph.json` configuration file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess  # noqa: S404
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Self

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


def _read_process_output(proc: subprocess.Popen) -> str:
    """Read combined stdout and stderr from a finished subprocess.

    Args:
        proc: Completed subprocess.

    Returns:
        Combined output string (may be empty).
    """
    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout.read().decode(errors="replace"))
    if proc.stderr:
        parts.append(proc.stderr.read().decode(errors="replace"))
    return "\n".join(p for p in parts if p.strip())


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
        checkpointer_path: Import path to an async context manager
            that yields a `BaseCheckpointSaver`. When set, the server
            persists checkpoint data to disk instead of in-memory.

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


class ServerProcess:
    """Manages a `langgraph dev` server subprocess.

    Starts the server, waits for it to become healthy, and provides
    clean shutdown.
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
            port: Port to bind the server to.
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

        cmd = [
            sys.executable,
            "-m",
            "langgraph_cli",
            "dev",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--no-browser",
            "--no-reload",
            "--config",
            str(config_path),
        ]

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

        await self._wait_for_healthy(timeout)

    async def _wait_for_healthy(self, timeout: float) -> None:  # noqa: ASYNC109
        """Poll the server health endpoint until it responds.

        Args:
            timeout: Max seconds to wait.

        Raises:
            RuntimeError: If the server doesn't become healthy in time.
        """
        import httpx

        url = f"{self.url}/ok"
        deadline = time.monotonic() + timeout
        last_status: int | None = None

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                output = self._read_log_file()
                msg = f"Server process exited with code {self._process.returncode}"
                if output:
                    msg += f"\n{output[-3000:]}"
                raise RuntimeError(msg)

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=2)
                    if resp.status_code == 200:  # noqa: PLR2004
                        logger.info("Server is healthy at %s", self.url)
                        return
                    last_status = resp.status_code
                    logger.debug("Health check returned status %d", resp.status_code)
            except (httpx.ConnectError, httpx.TimeoutException, OSError):
                pass

            await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        msg = f"Server did not become healthy within {timeout}s"
        if last_status is not None:
            msg += f" (last status: {last_status})"
        raise RuntimeError(msg)

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
                self._process.wait(timeout=2)
            except OSError:
                logger.debug("Error stopping server", exc_info=True)

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
        via `update_env()` are applied to `os.environ` before the new process
        starts. On failure, env overrides are rolled back.

        Args:
            timeout: Max seconds to wait for the server to become healthy.
        """
        logger.info("Restarting langgraph dev server")
        self._stop_process()

        # Apply env overrides, saving old values for rollback.
        prev_env: dict[str, str | None] = {}
        for key, val in self._env_overrides.items():
            prev_env[key] = os.environ.get(key)
            os.environ[key] = val

        try:
            await self.start(timeout=timeout)
        except Exception:
            # Roll back env so the next restart attempt (or error display)
            # reflects the state of the last successful server.
            for key, old_val in prev_env.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val
            raise

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
