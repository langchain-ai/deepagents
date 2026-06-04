"""Command line entry point for the Talon runtime host."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, PersistentCronScheduler
from deepagents_talon.data_lifecycle import cleanup_sensitive_state
from deepagents_talon.host import TalonHost
from deepagents_talon.mcp import load_mcp_tools, print_mcp_config_paths, write_mcp_server_config
from deepagents_talon.runtime import DeepAgentRuntime, EchoAgentRuntime
from deepagents_talon.speech import build_voice_transcriber

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_talon.cron import CronJob
    from deepagents_talon.interfaces import ChannelAdapter

logger = logging.getLogger(__name__)


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

    channels = _channels(config, enabled=args.whatsapp)
    host = TalonHost(
        config=config,
        agent=asyncio.run(_agent_runtime(config)),
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


def _add_mcp_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    mcp = subparsers.add_parser("mcp", help="Manage MCP servers")
    mcp_sub = mcp.add_subparsers(dest="mcp_command")

    mcp_sub.add_parser("config", help="Show MCP config discovery paths")

    login = mcp_sub.add_parser("login", help="Run OAuth login for an MCP server")
    login.add_argument("server", help="Server name from mcpServers")
    login.add_argument("--mcp-config", dest="config_path", default=None)

    add = mcp_sub.add_parser("add", help="Add an MCP server to a Talon config file")
    add.add_argument("name", help="Server name to add")
    add.add_argument("--mcp-config", dest="config_path", default=None)
    add.add_argument("--transport", choices=["stdio", "sse", "http"], default=None)
    add.add_argument("--url", default=None, help="Remote MCP server URL")
    add.add_argument("--command", dest="server_command", default=None, help="Stdio server command")
    add.add_argument("--arg", action="append", default=[], help="Command argument for stdio")
    add.add_argument("--header", action="append", default=[], help="Header as Name=Value")
    add.add_argument("--env", action="append", default=[], help="Environment entry as Name=Value")
    add.add_argument("--allow", action="append", default=[], help="Tool name or glob to allow")
    add.add_argument("--disable", action="append", default=[], help="Tool name or glob to disable")
    add.add_argument("--oauth", action="store_true", help="Mark remote server as OAuth-backed")
    add.add_argument("--overwrite", action="store_true", help="Replace an existing server entry")


async def _agent_runtime(config: TalonConfig) -> EchoAgentRuntime | DeepAgentRuntime:
    if config.model is None:
        return EchoAgentRuntime()

    mcp = await load_mcp_tools(config)
    for server in mcp.servers:
        if server.error is not None:
            logger.warning("MCP server %s failed: %s", server.name, server.error)
        else:
            logger.info("MCP server %s loaded %d tool(s)", server.name, server.tool_count)
    return DeepAgentRuntime(
        model=config.model,
        tools=mcp.tools,
        system_prompt=_system_prompt(config),
    )


async def _run_mcp_command(args: argparse.Namespace, config: TalonConfig) -> int:
    if args.mcp_command == "config":
        print_mcp_config_paths(config)
        return 0
    if args.mcp_command == "add":
        return _run_mcp_add(args, config)
    if args.mcp_command == "login":
        return await _run_mcp_login(args)
    print("Specify an MCP command: config, add, or login", file=sys.stderr)  # noqa: T201
    return 2


def _run_mcp_add(args: argparse.Namespace, config: TalonConfig) -> int:
    try:
        path = _mcp_config_write_path(args.config_path, config)
        server = _server_config_from_args(args)
        write_mcp_server_config(
            path=path,
            name=args.name,
            server=server,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"MCP add failed: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"Added MCP server {args.name!r} to {path}")  # noqa: T201
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


def _server_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.allow and args.disable:
        msg = "--allow and --disable are mutually exclusive"
        raise ValueError(msg)

    transport = args.transport or ("http" if args.url else "stdio")
    server: dict[str, object] = {"type": transport}
    if transport in {"http", "sse"}:
        if not args.url:
            msg = "--url is required for remote MCP servers"
            raise ValueError(msg)
        server["url"] = args.url
        if args.oauth:
            server["auth"] = "oauth"
    else:
        if not args.server_command:
            msg = "--command is required for stdio MCP servers"
            raise ValueError(msg)
        server["command"] = args.server_command
        if args.arg:
            server["args"] = args.arg

    _add_optional_server_fields(server, args)
    return server


def _add_optional_server_fields(server: dict[str, object], args: argparse.Namespace) -> None:
    headers = _pairs(args.header, "--header")
    if headers:
        server["headers"] = headers
    env = _pairs(args.env, "--env")
    if env:
        server["env"] = env
    if args.allow:
        server["allowedTools"] = args.allow
    if args.disable:
        server["disabledTools"] = args.disable


def _pairs(values: Sequence[str], label: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            msg = f"{label} values must use Name=Value"
            raise ValueError(msg)
        pairs[key] = item
    return pairs


def _mcp_config_write_path(config_path: str | None, config: TalonConfig) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()
    return config.manifest_dir / "tools.json"


def _system_prompt(config: TalonConfig) -> str | None:
    path = config.manifest_dir / "AGENTS.md"
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read Talon system prompt from %s", path, exc_info=True)
    return None


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
