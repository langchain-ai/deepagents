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
) -> Path:
    """Generate a `langgraph.json` config file for `langgraph dev`.

    Args:
        output_dir: Directory to write the config file.
        graph_ref: Python module:variable reference to the graph.
        env_file: Optional path to an env file.

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
    ) -> None:
        """Initialize server process manager.

        Args:
            host: Host to bind the server to.
            port: Port to bind the server to.
            config_dir: Directory containing `langgraph.json`.
        """
        self.host = host
        self.port = port
        self.config_dir = Path(config_dir) if config_dir else None
        self._process: subprocess.Popen | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    @property
    def url(self) -> str:
        """Server base URL."""
        return get_server_url(self.host, self.port)

    @property
    def running(self) -> bool:
        """Whether the server process is running."""
        return self._process is not None and self._process.poll() is None

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
        env["LANGSMITH_TRACING"] = "false"
        for key in (
            "LANGGRAPH_AUTH",
            "LANGGRAPH_CLOUD_LICENSE_KEY",
            "LANGSMITH_API_KEY",
            "LANGSMITH_CONTROL_PLANE_API_KEY",
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_TENANT_ID",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_ENDPOINT",
            "LANGCHAIN_TRACING_V2",
        ):
            env.pop(key, None)

        logger.info("Starting langgraph dev server: %s", " ".join(cmd))
        self._process = subprocess.Popen(  # noqa: S603, ASYNC220
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                output = _read_process_output(self._process)
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
            except (httpx.ConnectError, httpx.TimeoutException, OSError):
                pass

            await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        msg = f"Server did not become healthy within {timeout}s"
        raise RuntimeError(msg)

    def stop(self) -> None:
        """Stop the server process gracefully."""
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

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                logger.debug("Failed to clean up temp dir", exc_info=True)
            self._temp_dir = None

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
