"""Test MCP servers for unit tests.

This module provides simple FastMCP servers that can be spawned as subprocesses
for testing the MCPMiddleware against real MCP server connections.
"""

from mcp.server.fastmcp import FastMCP


def create_math_server() -> FastMCP:
    """Create a simple math MCP server for testing.

    Returns:
        A FastMCP server with add, multiply, and divide tools.
    """
    mcp = FastMCP("math")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            The sum of a and b
        """
        return a + b

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            The product of a and b
        """
        return a * b

    @mcp.tool()
    def divide(a: int, b: int) -> float:
        """Divide two numbers.

        Args:
            a: Numerator
            b: Denominator

        Returns:
            The quotient of a divided by b

        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            msg = "Cannot divide by zero"
            raise ValueError(msg)
        return a / b

    return mcp


def create_weather_server() -> FastMCP:
    """Create a simple weather MCP server for testing multi-server scenarios.

    Returns:
        A FastMCP server with a get_weather tool.
    """
    mcp = FastMCP("weather")

    @mcp.tool()
    def get_weather(city: str) -> str:
        """Get the weather for a city.

        Args:
            city: Name of the city

        Returns:
            A string describing the weather
        """
        return f"The weather in {city} is sunny, 72F"

    return mcp
