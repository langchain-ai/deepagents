"""Utilities for running MCP servers as subprocesses in tests.

This module provides context managers and utilities for spawning MCP servers
as subprocesses, following the pattern used by langchain-mcp-adapters.
"""

import socket
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from multiprocessing import Process

from mcp.server.fastmcp import FastMCP


def _run_streamable_http_server(server_factory: Callable[[], FastMCP], port: int) -> None:
    """Run a FastMCP server with streamable HTTP transport.

    This function is intended to run in a subprocess.

    Args:
        server_factory: Factory function that creates a FastMCP server
        port: Port to run the server on
    """
    import uvicorn

    server = server_factory()
    app = server.streamable_http_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def _wait_for_server(port: int, timeout: float, poll_interval: float) -> None:
    """Wait for a server to become available on the given port.

    Args:
        port: Port to check
        timeout: Maximum time to wait in seconds
        poll_interval: Time between connection attempts

    Raises:
        RuntimeError: If the server does not start within the timeout
    """
    max_attempts = int(timeout / poll_interval)
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(poll_interval)
                s.connect(("127.0.0.1", port))
                return
        except (ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(poll_interval)

    msg = f"Server did not start within {timeout} seconds on port {port}"
    raise RuntimeError(msg)


@contextmanager
def run_mcp_server(
    server_factory: Callable[[], FastMCP],
    port: int,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
) -> Generator[str, None, None]:
    """Context manager that spawns an MCP server subprocess.

    Args:
        server_factory: Factory function that creates a FastMCP server
        port: Port to run the server on
        timeout: Maximum time to wait for server startup
        poll_interval: Time between connection attempts

    Yields:
        The server URL (e.g., "http://localhost:8181/mcp")

    Example:
        with run_mcp_server(create_math_server, 8181) as url:
            middleware = MCPMiddleware(servers=[{"name": "math", "url": url}])
            await middleware.connect()
    """
    process = Process(
        target=_run_streamable_http_server,
        args=(server_factory, port),
        daemon=True,
    )
    process.start()

    try:
        _wait_for_server(port, timeout, poll_interval)
        yield f"http://localhost:{port}/mcp"
    finally:
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def get_available_port() -> int:
    """Get an available port for testing.

    Returns:
        An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
