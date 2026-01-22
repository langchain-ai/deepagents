"""Pytest fixtures for MCP middleware tests."""

from collections.abc import Generator
from typing import Any

import pytest

from tests.unit_tests.middleware.mcp_test_servers import (
    create_math_server,
    create_weather_server,
)
from tests.unit_tests.middleware.mcp_test_utils import get_available_port, run_mcp_server


@pytest.fixture
def math_server() -> Generator[dict[str, Any], None, None]:
    """Run a math MCP server and yield its URL and port.

    Yields:
        Dict with "url" and "port" keys
    """
    port = get_available_port()
    with run_mcp_server(create_math_server, port) as url:
        yield {"url": url, "port": port}


@pytest.fixture
def weather_server() -> Generator[dict[str, Any], None, None]:
    """Run a weather MCP server and yield its URL and port.

    Yields:
        Dict with "url" and "port" keys
    """
    port = get_available_port()
    with run_mcp_server(create_weather_server, port) as url:
        yield {"url": url, "port": port}
