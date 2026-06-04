"""Command line entry point for the Talon runtime host."""

from __future__ import annotations

import argparse
import asyncio
import logging

from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.runtime import EchoAgentRuntime


def main() -> None:
    """Run the Talon host with the placeholder runtime."""
    parser = argparse.ArgumentParser(description="Run the DeepAgents Talon host.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Start and stop immediately after bootstrapping the host.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = TalonConfig.from_env()
    host = TalonHost(config=config, agent=EchoAgentRuntime())

    if args.once:
        asyncio.run(_run_once(host))
        return

    asyncio.run(host.run_until_stopped())


async def _run_once(host: TalonHost) -> None:
    await host.start()
    await host.stop()


if __name__ == "__main__":
    main()
