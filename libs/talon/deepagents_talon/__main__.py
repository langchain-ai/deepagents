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
from deepagents_talon.fleet import FleetAgentComponents, load_fleet_agent_components
from deepagents_talon.host import TalonHost
from deepagents_talon.import_fleet import (
    FleetImportError,
    FleetImportSummary,
    import_fleet_manifest,
)
from deepagents_talon.mcp import MCPTools, load_mcp_tools, print_mcp_config_paths
from deepagents_talon.runtime import (
    DeepAgentRuntime,
    EchoAgentRuntime,
    RuntimeAgentComponents,
    interrupt_on_with_env_overlay,
)
from deepagents_talon.speech import build_voice_transcriber

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from langchain_core.tools import BaseTool

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
    _add_import_fleet_parser(subparsers)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.command == "import-fleet":
        sys.exit(_run_import_fleet_command(args))
    config = TalonConfig.from_env()
    if args.command == "mcp":
        sys.exit(asyncio.run(_run_mcp_command(args, config)))

    _run_host(
        config,
        once=args.once,
        whatsapp=args.whatsapp,
        telegram=args.telegram,
    )


def _run_host(
    config: TalonConfig,
    *,
    once: bool,
    whatsapp: bool,
    telegram: bool,
) -> None:
    cron_factory = CronJobStore
    cron_store = cron_factory(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    config.ensure_home()
    cleanup_sensitive_state(config=config, cron_store=cron_store)

    channels = _channels(
        config,
        whatsapp=whatsapp,
        telegram=telegram,
    )
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

    if once:
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


def _add_import_fleet_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "import-fleet",
        help="Materialize a Talon assistant directory from a Fleet export",
    )
    parser.add_argument("fleet_dir", type=Path, help="Fleet export zip file or directory")
    parser.add_argument("--assistant-id", required=True, help="Assistant id for Talon local state")
    parser.add_argument(
        "--channel",
        choices=("telegram", "whatsapp"),
        default=None,
        help="Accepted for compatibility; channels are selected by the Talon runtime environment.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Accepted for compatibility; import-fleet does not prompt.",
    )


async def _agent_runtime(
    config: TalonConfig,
    cron_store: CronJobStore,
) -> EchoAgentRuntime | DeepAgentRuntime:
    env = _runtime_env(config)
    if config.fleet_dir is not None:
        fleet_dir = config.fleet_dir
        components = await load_fleet_agent_components(fleet_dir, env=env)
        mcp = await load_mcp_tools(config, allow_empty=True)
        _log_mcp_servers(mcp)
        runtime_components = _runtime_components_from_fleet(
            config,
            components,
            env=env,
            local_tools=mcp.tools,
        )

        async def reload_fleet_components() -> RuntimeAgentComponents:
            refreshed = await load_fleet_agent_components(fleet_dir, env=env)
            reloaded = await load_mcp_tools(config, allow_empty=True)
            _log_mcp_servers(reloaded)
            return _runtime_components_from_fleet(
                config,
                refreshed,
                env=env,
                local_tools=reloaded.tools,
            )

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

    mcp = await load_mcp_tools(config, allow_empty=True)
    _log_mcp_servers(mcp)
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
    local_tools: Sequence[BaseTool | Callable[..., object]] = (),
) -> RuntimeAgentComponents:
    return RuntimeAgentComponents(
        model=config.model or components.model,
        tools=_merge_tools(components.tools, local_tools),
        system_prompt=components.system_prompt,
        subagents=components.subagents,
        skills=components.skills,
        middleware=components.middleware,
        interrupt_on=interrupt_on_with_env_overlay(components.interrupt_on, env),
    )


def _log_mcp_servers(mcp: MCPTools) -> None:
    for server in mcp.servers:
        if server.error is not None:
            logger.warning("MCP server %s failed: %s", server.name, server.error)
        else:
            logger.info("MCP server %s loaded %d tool(s)", server.name, len(server.tools))


def _merge_tools(
    primary: Sequence[BaseTool | Callable[..., object]],
    extra: Sequence[BaseTool | Callable[..., object]],
) -> tuple[BaseTool | Callable[..., object], ...]:
    tools: list[BaseTool | Callable[..., object]] = list(primary)
    seen = {_tool_name(tool) for tool in primary}
    for tool in extra:
        name = _tool_name(tool)
        if name is not None and name in seen:
            continue
        tools.append(tool)
        if name is not None:
            seen.add(name)
    return tuple(tools)


def _tool_name(tool: BaseTool | Callable[..., object]) -> str | None:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    fallback = getattr(tool, "__name__", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    return None


async def _run_mcp_command(args: argparse.Namespace, config: TalonConfig) -> int:
    if args.mcp_command == "config":
        print_mcp_config_paths(config)
        return 0
    if args.mcp_command == "login":
        return await _run_mcp_login(args)
    print("Specify an MCP command: config or login", file=sys.stderr)  # noqa: T201
    return 2


def _run_import_fleet_command(args: argparse.Namespace) -> int:
    try:
        del args.channel, args.non_interactive
        summary = import_fleet_manifest(
            args.fleet_dir,
            assistant_id=args.assistant_id,
        )
    except (FleetImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    _print_import_summary(summary)
    return 0


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


def _print_import_summary(summary: FleetImportSummary) -> None:
    print("Imported Fleet export for Talon.")  # noqa: T201
    print(f"  fleet_source: {summary.fleet_source}")  # noqa: T201
    print(f"  assistant_id: {summary.assistant_id}")  # noqa: T201
    print(f"  agent_dir: {summary.agent_dir}")  # noqa: T201
    print(f"  root_mcp_config: {summary.mcp_config_path}")  # noqa: T201
    print(f"  tools_summarized: {summary.tool_count}")  # noqa: T201
    print(f"  mcp_servers: {summary.server_count}")  # noqa: T201
    print(f"  interrupt_tools: {summary.interrupt_tool_count}")  # noqa: T201
    if not summary.mcp_server_notes:
        print("MCP configuration notes: none")  # noqa: T201
        return

    print("MCP configuration notes:")  # noqa: T201
    for note in summary.mcp_server_notes:
        print(f"  - {note.scope}: {note.server_name} ({note.endpoint})")  # noqa: T201
        print(f"    tools: {_join_names(note.tool_names)}")  # noqa: T201
        print(f"    interrupt_on: {_join_names(note.interrupt_tools)}")  # noqa: T201
    interrupt_tools = sorted(
        {tool for note in summary.mcp_server_notes for tool in note.interrupt_tools}
    )
    if interrupt_tools:
        print(  # noqa: T201
            f"Recommended human-in-the-loop tools: {_join_names(tuple(interrupt_tools))}",
        )


def _join_names(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


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
