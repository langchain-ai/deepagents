"""Entry point for running the ACP server as a module."""

import asyncio

from deepagents_acp.server import _serve_test_agent


def main() -> None:
    """Run the test ACP agent server."""
    asyncio.run(_serve_test_agent())


if __name__ == "__main__":
    main()
