"""Command line entry point for the Talon runtime host."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import TYPE_CHECKING

from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.runtime import EchoAgentRuntime

if TYPE_CHECKING:
    from deepagents_talon.interfaces import ChannelAdapter


def main() -> None:
    """Run the Talon host with the placeholder runtime."""
    parser = argparse.ArgumentParser(description="Run the DeepAgents Talon host.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Start and stop immediately after bootstrapping the host.",
    )
    parser.add_argument(
        "--whatsapp",
        action="store_true",
        help="Attach the WhatsApp channel adapter.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = TalonConfig.from_env()
    host = TalonHost(
        config=config,
        agent=EchoAgentRuntime(),
        channels=_channels(config, enabled=args.whatsapp),
    )

    if args.once:
        asyncio.run(_run_once(host))
        return

    asyncio.run(host.run_until_stopped())


async def _run_once(host: TalonHost) -> None:
    await host.start()
    await host.stop()


def _channels(config: TalonConfig, *, enabled: bool) -> tuple[ChannelAdapter, ...]:
    if not enabled and config.env.get("DEEPAGENTS_TALON_WHATSAPP_ENABLED", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return ()
    return (WhatsAppChannel(WhatsAppChannelConfig.from_talon_config(config)),)


if __name__ == "__main__":
    main()
