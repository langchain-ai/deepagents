"""Command line entry point for the Talon runtime host.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.channels.telegram import TelegramChannel, TelegramChannelConfig
from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, PersistentCronScheduler
from deepagents_talon.data_lifecycle import cleanup_sensitive_state
from deepagents_talon.fleet_import import (
    FleetImportError,
    format_import_stdout,
    import_fleet_zip,
)
from deepagents_talon.host import TalonHost
from deepagents_talon.mcp import load_mcp_tools, print_mcp_config_paths
from deepagents_talon.runtime import (
    DeepAgentRuntime,
    EchoAgentRuntime,
    interrupt_on_with_env_overlay,
)
from deepagents_talon.speech import build_voice_transcriber

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from deepagents_talon.cron import CronJob
    from deepagents_talon.interfaces import ChannelAdapter

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Talon host with the placeholder runtime."""
    parser = argparse.ArgumentParser(description="Run the Deep Agents Talon host.")
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
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Attach the Telegram channel adapter.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_import_fleet_parser(subparsers)
    _add_mcp_parsers(subparsers)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = TalonConfig.from_env()
    if args.command == "import-fleet":
        sys.exit(_run_import_fleet_command(args, config))
    if args.command == "mcp":
        sys.exit(asyncio.run(_run_mcp_command(args, config)))

    cron_factory = CronJobStore
    cron_store = cron_factory(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    config.ensure_home()
    cleanup_sensitive_state(config=config, cron_store=cron_store)

    channels = _channels(config, whatsapp=args.whatsapp, telegram=args.telegram)
    host = TalonHost(
        config=config,
        agent=asyncio.run(_agent_runtime(config, cron_store)),
        channels=channels,
        voice_transcriber=build_voice_transcriber(config),
    )
    if channels:
        host.scheduler = PersistentCronScheduler(
            store=cron_store,
            run_job=host.run_scheduled_job,
            deliver_result=lambda job, text: _deliver_cron_result(host, channels, job, text),
        )

    if args.once:
        asyncio.run(_run_once(host))
        return

    asyncio.run(host.run_until_stopped())


def _add_import_fleet_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    importer = subparsers.add_parser(
        "import-fleet",
        help="Import a Fleet zip export into a Talon local agent directory",
        description=(
            "Import a Fleet zip export into a Talon local agent directory. By default, "
            "the target directory is the selected assistant manifest directory."
        ),
        epilog=(
            "Usage: deepagents-talon import-fleet <fleet-export.zip> "
            "[--assistant-id <id>] [--target-dir <dir>]\n\n"
            ".mcp.json is generated as the runtime MCP config file; "
            ".mcp.json.setup is a human-readable setup handoff for operators. "
            "Fleet config.json is "
            "ignored, Fleet tools.json is import input only, and old Fleet direct-run "
            "environment variables are unsupported. Use import-fleet before running "
            "the Talon host."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    importer.add_argument("fleet_export", type=Path, help="Fleet zip export to import")
    importer.add_argument(
        "--assistant-id",
        help="Assistant id used for default target directory resolution",
    )
    importer.add_argument(
        "--target-dir",
        type=Path,
        help="Directory to receive materialized Talon agent files",
    )


def _add_mcp_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    mcp = subparsers.add_parser("mcp", help="Manage MCP servers")
    mcp_sub = mcp.add_subparsers(dest="mcp_command")

    mcp_sub.add_parser("config", help="Show MCP config discovery paths")

    login = mcp_sub.add_parser("login", help="Run OAuth login for an MCP server")
    login.add_argument("server", help="Server name from mcpServers")
    login.add_argument("--mcp-config", dest="config_path", default=None)


def _run_import_fleet_command(args: argparse.Namespace, config: TalonConfig) -> int:
    target_dir = args.target_dir
    if target_dir is None:
        target_config = config
        if args.assistant_id:
            target_config = TalonConfig.from_env(
                {
                    **config.env,
                    "DEEPAGENTS_TALON_ASSISTANT_ID": args.assistant_id,
                },
                base_home=config.home.parent,
            )
        target_dir = target_config.manifest_dir

    try:
        result = import_fleet_zip(args.fleet_export, target_dir=target_dir)
    except FleetImportError as exc:
        print(f"import-fleet: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print(format_import_stdout(result), end="")  # noqa: T201
    return 0


async def _agent_runtime(
    config: TalonConfig,
    cron_store: CronJobStore,
) -> EchoAgentRuntime | DeepAgentRuntime:
    env = _runtime_env(config)
    if config.model is None:
        return EchoAgentRuntime()

    mcp = await load_mcp_tools(config)
    for server in mcp.servers:
        if server.error is not None:
            logger.warning("MCP server %s failed: %s", server.name, server.error)
        else:
            logger.info("MCP server %s loaded %d tool(s)", server.name, len(server.tools))
    return DeepAgentRuntime(
        model=config.model,
        tools=mcp.tools,
        assistant_dir=config.manifest_dir,
        cron_store=cron_store,
        interrupt_on=interrupt_on_with_env_overlay(None, env),
        env=env,
    )


async def _run_mcp_command(args: argparse.Namespace, config: TalonConfig) -> int:
    if args.mcp_command == "config":
        print_mcp_config_paths(config)
        return 0
    if args.mcp_command == "login":
        return await _run_mcp_login(args)
    print("Specify an MCP command: config or login", file=sys.stderr)  # noqa: T201
    return 2


async def _run_mcp_login(args: argparse.Namespace) -> int:
    try:
        module = importlib.import_module("deepagents_code.mcp_commands")
    except ImportError:
        print(  # noqa: T201
            "MCP login requires deepagents-code to be installed in this environment.",
            file=sys.stderr,
        )
        return 1
    run_mcp_login = module.run_mcp_login
    return await run_mcp_login(server=args.server, config_path=args.config_path)


async def _run_once(host: TalonHost) -> None:
    await host.start()
    await host.stop()


def _channels(
    config: TalonConfig,
    *,
    whatsapp: bool = False,
    telegram: bool = False,
) -> tuple[ChannelAdapter, ...]:
    channels: list[ChannelAdapter] = []
    if whatsapp or _env_enabled(config.env, "DEEPAGENTS_TALON_WHATSAPP_ENABLED"):
        channels.append(WhatsAppChannel(WhatsAppChannelConfig.from_talon_config(config)))
    if telegram or _env_enabled(config.env, "DEEPAGENTS_TALON_TELEGRAM_ENABLED"):
        channels.append(TelegramChannel(TelegramChannelConfig.from_talon_config(config)))
    return tuple(channels)


def _env_enabled(env: Mapping[str, str], key: str) -> bool:
    """Check whether a boolean environment flag is truthy.

    Args:
        env: Environment variable mapping.
        key: Environment variable name.

    Returns:
        `True` when the value is one of ``1``, ``true``, or ``yes``.
    """
    return env.get(key, "").lower() in {"1", "true", "yes"}


def _runtime_env(config: TalonConfig) -> dict[str, str]:
    values = dict(os.environ)
    values.update(config.env)
    return values


async def _deliver_cron_result(
    host: TalonHost,
    channels: Sequence[ChannelAdapter],
    job: CronJob,
    text: str,
) -> None:
    for channel in channels:
        if job.origin.channel is None or (await channel.status()).provider == job.origin.channel:
            await host.deliver_scheduled_result(channel, job, text)
            return


if __name__ == "__main__":
    main()
