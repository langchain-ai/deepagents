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
from typing import TYPE_CHECKING

from deepagents_talon.channels.telegram import TelegramChannel, TelegramChannelConfig
from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, PersistentCronScheduler
from deepagents_talon.data_lifecycle import cleanup_sensitive_state
from deepagents_talon.fleet import FleetAgentComponents, load_fleet_agent_components
from deepagents_talon.fleet_manifest import ChannelSelection, refresh_fleet_run_manifest
from deepagents_talon.host import TalonHost
from deepagents_talon.mcp import load_mcp_tools, print_mcp_config_paths
from deepagents_talon.runtime import (
    DeepAgentRuntime,
    EchoAgentRuntime,
    RuntimeAgentComponents,
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
    _add_mcp_parsers(subparsers)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = TalonConfig.from_env()
    if args.command == "mcp":
        sys.exit(asyncio.run(_run_mcp_command(args, config)))

    cron_factory = CronJobStore
    cron_store = cron_factory(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    config.ensure_home()
    cleanup_sensitive_state(config=config, cron_store=cron_store)

    channel_selection = _channel_selection(config, whatsapp=args.whatsapp, telegram=args.telegram)
    channels = _channels(config, whatsapp=args.whatsapp, telegram=args.telegram)
    host = TalonHost(
        config=config,
        agent=asyncio.run(_agent_runtime(config, cron_store, selected_channel=channel_selection)),
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


def _add_mcp_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    mcp = subparsers.add_parser("mcp", help="Manage MCP servers")
    mcp_sub = mcp.add_subparsers(dest="mcp_command")

    mcp_sub.add_parser("config", help="Show MCP config discovery paths")

    login = mcp_sub.add_parser("login", help="Run OAuth login for an MCP server")
    login.add_argument("server", help="Server name from mcpServers")
    login.add_argument("--mcp-config", dest="config_path", default=None)


async def _agent_runtime(
    config: TalonConfig,
    cron_store: CronJobStore,
    *,
    selected_channel: ChannelSelection | None = None,
) -> EchoAgentRuntime | DeepAgentRuntime:
    env = _runtime_env(config)
    if config.fleet_dir is not None:
        fleet_dir = config.fleet_dir
        components = await load_fleet_agent_components(fleet_dir, env=env)
        refresh_fleet_run_manifest(config, components, selected_channel=selected_channel)
        runtime_components = _runtime_components_from_fleet(config, components, env=env)

        async def reload_fleet_components() -> RuntimeAgentComponents:
            refreshed = await load_fleet_agent_components(fleet_dir, env=env)
            refresh_fleet_run_manifest(config, refreshed, selected_channel=selected_channel)
            return _runtime_components_from_fleet(config, refreshed, env=env)

        return DeepAgentRuntime(
            model=runtime_components.model,
            tools=runtime_components.tools,
            system_prompt=runtime_components.system_prompt,
            subagents=runtime_components.subagents,
            skills=runtime_components.skills,
            middleware=runtime_components.middleware,
            interrupt_on=runtime_components.interrupt_on,
            cron_store=cron_store,
            env=env,
            reload_agent_components=reload_fleet_components,
        )

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


def _runtime_components_from_fleet(
    config: TalonConfig,
    components: FleetAgentComponents,
    *,
    env: Mapping[str, str],
) -> RuntimeAgentComponents:
    return RuntimeAgentComponents(
        model=config.model or components.model,
        tools=components.tools,
        system_prompt=components.system_prompt,
        subagents=components.subagents,
        skills=components.skills,
        middleware=components.middleware,
        interrupt_on=interrupt_on_with_env_overlay(components.interrupt_on, env),
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


def _channel_selection(
    config: TalonConfig,
    *,
    whatsapp: bool = False,
    telegram: bool = False,
) -> ChannelSelection | None:
    selections: list[tuple[str, str]] = []
    if whatsapp:
        selections.append(("whatsapp", "cli"))
    elif _env_enabled(config.env, "DEEPAGENTS_TALON_WHATSAPP_ENABLED"):
        selections.append(("whatsapp", "environment"))

    if telegram:
        selections.append(("telegram", "cli"))
    elif _env_enabled(config.env, "DEEPAGENTS_TALON_TELEGRAM_ENABLED"):
        selections.append(("telegram", "environment"))

    if not selections:
        return None

    if len(selections) == 1:
        provider, source = selections[0]
        return ChannelSelection(provider=provider, source=source)

    return ChannelSelection(
        provider="multiple",
        source="mixed",
        metadata={
            "providers": ",".join(provider for provider, _source in selections),
            "sources": ",".join(source for _provider, source in selections),
        },
    )


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
