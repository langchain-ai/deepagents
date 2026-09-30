"""Command line entry point for the Talon runtime host.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.async_subagents import load_async_subagents
from deepagents_talon.channels.discord import DiscordChannel, DiscordChannelConfig
from deepagents_talon.channels.slack import SlackChannel, SlackChannelConfig
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
from deepagents_talon.mcp import MCPToolProvider, login_mcp_server, print_mcp_config_paths
from deepagents_talon.mcp_middleware import talon_mcp_middleware
from deepagents_talon.pairing import (
    PAIRING_CHANNELS,
    PAIRING_FILENAME,
    PairedSender,
    PairingStore,
    SenderPairing,
    approve_code,
    env_sender_ids,
    format_listing,
    pause_jobs,
    revoke_sender,
    sender_jobs,
)
from deepagents_talon.sandbox import SandboxStartupError, open_sandbox
from deepagents_talon.speech import build_voice_transcriber

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langgraph.types import Checkpointer

    from deepagents_talon.cron import CronJob
    from deepagents_talon.interfaces import AgentRuntime, ChannelAdapter
    from deepagents_talon.sandbox import SandboxSession

logger = logging.getLogger(__name__)

_DCODE_DEBUG_ENV = "DEEPAGENTS_CODE_DEBUG"
_DCODE_LOG_LEVEL_ENV = "DEEPAGENTS_CODE_LOG_LEVEL"
_DCODE_DEBUG_VALUES = frozenset({"1", "true", "yes", "on"})
_DCODE_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_CHANNEL_LOGGER_NAME = "deepagents_talon.channels"


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
    parser.add_argument(
        "--discord",
        action="store_true",
        help="Attach the Discord channel adapter.",
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Attach the Slack channel adapter.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_import_fleet_parser(subparsers)
    _add_mcp_parsers(subparsers)
    _add_pairing_parsers(subparsers)
    args = parser.parse_args()

    _configure_logging(os.environ)

    config = TalonConfig.from_env()
    if args.command == "import-fleet":
        sys.exit(_run_import_fleet_command(args, config))
    if args.command == "mcp":
        sys.exit(asyncio.run(_run_mcp_command(args, config)))
    if args.command == "pairing":
        config.ensure_home()
        sys.exit(_run_pairing_command(args, config))

    cron_factory = CronJobStore
    cron_store = cron_factory(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    config.ensure_home()
    cleanup_sensitive_state(config=config, cron_store=cron_store)

    channels = _channels(
        config,
        whatsapp=args.whatsapp,
        telegram=args.telegram,
        discord=args.discord,
        slack=args.slack,
    )
    try:
        asyncio.run(_run_host(args, config, cron_store, channels))
    except SandboxStartupError as exc:
        print(f"talon: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


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
            "Fleet config.json and tools.json are ignored, and old Fleet direct-run "
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


def _add_pairing_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    pairing = subparsers.add_parser("pairing", help="Manage paired DM senders")
    pairing_sub = pairing.add_subparsers(dest="pairing_command", required=True)
    listing = pairing_sub.add_parser("list", help="Show pending requests and paired senders")
    listing.add_argument("channel", nargs="?", choices=PAIRING_CHANNELS)
    approve = pairing_sub.add_parser("approve", help="Approve a pending request by code")
    approve.add_argument("channel", choices=PAIRING_CHANNELS)
    approve.add_argument("code")
    revoke = pairing_sub.add_parser("revoke", help="Revoke a paired sender")
    revoke.add_argument("channel", choices=PAIRING_CHANNELS)
    revoke.add_argument("sender_id")
    pause = pairing_sub.add_parser(
        "pause-jobs",
        help="Pause cron jobs a sender created. Run only while Talon is stopped: "
        "the running host is the cron store's only writer.",
    )
    pause.add_argument("channel", choices=PAIRING_CHANNELS)
    pause.add_argument("sender_id")


def _run_pairing_command(args: argparse.Namespace, config: TalonConfig) -> int:
    channels = [args.channel] if args.channel else list(PAIRING_CHANNELS)
    if args.pairing_command == "list":
        listings = (format_listing(_cli_pairing(config, channel)) for channel in channels)
        print("\n".join(listings))  # noqa: T201
        return 0
    if args.pairing_command == "pause-jobs":
        jobs = _cli_jobs(config, args.channel, args.sender_id)
        paused = pause_jobs(_cron_store(config), jobs)
        print(f"Paused {paused} scheduled job(s).")  # noqa: T201
        return 0
    pairing = _cli_pairing(config, args.channel)
    if args.pairing_command == "approve":
        result = approve_code(pairing, args.code)
        print(result.reply)  # noqa: T201
        return 0 if result.approved is not None else 1
    result = revoke_sender(pairing, args.sender_id)
    print(result.reply)  # noqa: T201
    if result.revoked is None:
        return 1
    _report_revoked_jobs(config, args.channel, result.revoked)
    return 0


def _cli_pairing(config: TalonConfig, channel: str) -> SenderPairing:
    prefix = f"DEEPAGENTS_TALON_{channel.upper()}"
    return SenderPairing(
        store=PairingStore(config.home / PAIRING_FILENAME),
        provider=channel,
        env_sender_ids=env_sender_ids(config.env, prefix),
    )


def _cron_store(config: TalonConfig) -> CronJobStore:
    return CronJobStore(assistant_id=config.assistant_id, cron_dir=config.cron_dir)


def _cli_jobs(config: TalonConfig, channel: str, sender_id: str) -> list[CronJob]:
    return sender_jobs(_cron_store(config), channel, sender_id)


def _report_revoked_jobs(config: TalonConfig, channel: str, revoked: PairedSender) -> None:
    jobs = _cli_jobs(config, channel, revoked.sender_id)
    enabled = [job for job in jobs if job.enabled]
    if not enabled:
        return
    names = ", ".join(f"{job.id} ({job.name})" for job in enabled)
    print(  # noqa: T201
        f"Still enabled, created by them: {names}. Stop Talon and run: "
        f"deepagents-talon pairing pause-jobs {channel} {revoked.sender_id}"
    )


def _run_import_fleet_command(args: argparse.Namespace, config: TalonConfig) -> int:
    target_dir = args.target_dir
    assistant_home = None
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
        elif not _has_configured_assistant_id(config.env):
            target_config = TalonConfig.from_env(
                {
                    **config.env,
                    "DEEPAGENTS_TALON_ASSISTANT_ID": args.fleet_export.stem,
                },
                base_home=config.home.parent,
            )
        target_dir = target_config.manifest_dir
        assistant_home = target_config.home

    try:
        result = import_fleet_zip(
            args.fleet_export,
            target_dir=target_dir,
            assistant_home=assistant_home,
        )
    except FleetImportError as exc:
        print(f"import-fleet: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print(format_import_stdout(result), end="")  # noqa: T201
    return 0


def _has_configured_assistant_id(env: Mapping[str, str]) -> bool:
    return "DEEPAGENTS_TALON_ASSISTANT_ID" in env or "AGENT_ASSISTANT_ID" in env


async def _run_host(
    args: argparse.Namespace,
    config: TalonConfig,
    cron_store: CronJobStore,
    channels: Sequence[ChannelAdapter],
) -> None:
    if config.model is None:
        await _run_host_with_agent(args, config, cron_store, channels, await _agent_runtime(config))
        return
    async with open_sandbox(config) as sandbox:
        await _run_model_host(args, config, cron_store, channels, sandbox=sandbox)


async def _run_model_host(
    args: argparse.Namespace,
    config: TalonConfig,
    cron_store: CronJobStore,
    channels: Sequence[ChannelAdapter],
    *,
    sandbox: SandboxSession | None,
) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415

    from deepagents_talon.archive_saver import ConversationSaver  # noqa: PLC0415
    from deepagents_talon.history_backends import open_history  # noqa: PLC0415

    async with (
        AsyncSqliteSaver.from_conn_string(str(config.checkpoint_path)) as sqlite_checkpointer,
        open_history(config) as archive,
    ):
        await sqlite_checkpointer.setup()
        agent = await _agent_runtime(
            config,
            cron_store=cron_store,
            checkpointer=ConversationSaver(sqlite_checkpointer, archive=archive),
            sandbox=sandbox,
        )
        await _run_host_with_agent(args, config, cron_store, channels, agent)


async def _run_host_with_agent(
    args: argparse.Namespace,
    config: TalonConfig,
    cron_store: CronJobStore,
    channels: Sequence[ChannelAdapter],
    agent: AgentRuntime,
) -> None:
    host = TalonHost(
        config=config,
        agent=agent,
        channels=channels,
        voice_transcriber=build_voice_transcriber(config),
    )
    if channels:
        host.scheduler = PersistentCronScheduler(
            store=cron_store,
            run_job=host.run_scheduled_job,
            deliver_result=lambda job, text: _deliver_cron_result(host, job, text),
        )
    if args.once:
        await _run_once(host)
    else:
        await host.run_until_stopped()


async def _agent_runtime(
    config: TalonConfig,
    cron_store: CronJobStore | None = None,
    checkpointer: Checkpointer | None = None,
    sandbox: SandboxSession | None = None,
) -> AgentRuntime:
    from deepagents_talon.runtime import (  # noqa: PLC0415
        DeepAgentRuntime,
        EchoAgentRuntime,
    )

    env = _runtime_env(config)
    if config.model is None:
        return EchoAgentRuntime()

    mcp_provider = MCPToolProvider(config)
    mcp = await mcp_provider.load()
    for server in mcp.servers:
        if server.error is not None:
            logger.warning("MCP server %s failed: %s", server.name, server.error)
        else:
            logger.info("MCP server %s loaded %d tool(s)", server.name, len(server.tools))
    return DeepAgentRuntime(
        model=config.model,
        tools=mcp.tools,
        refresh_tools=mcp_provider.refresh_if_needed,
        reload_tools=mcp_provider.reload,
        assistant_dir=config.manifest_dir,
        load_subagents=load_async_subagents,
        cron_store=cron_store,
        checkpointer=checkpointer,
        middleware=(talon_mcp_middleware(),),
        env=env,
        backend=sandbox.backend if sandbox is not None else None,
        sandbox_working_dir=sandbox.working_dir if sandbox is not None else None,
    )


async def _run_mcp_command(args: argparse.Namespace, config: TalonConfig) -> int:
    if args.mcp_command == "config":
        print_mcp_config_paths(config)
        return 0
    if args.mcp_command == "login":
        return await login_mcp_server(config, args.server, args.config_path)
    print("Specify an MCP command: config or login", file=sys.stderr)  # noqa: T201
    return 2


async def _run_once(host: TalonHost) -> None:
    await host.start()
    await host.stop()


def _channels(
    config: TalonConfig,
    *,
    whatsapp: bool = False,
    telegram: bool = False,
    discord: bool = False,
    slack: bool = False,
) -> tuple[ChannelAdapter, ...]:
    channels: list[ChannelAdapter] = []
    if whatsapp or _env_enabled(config.env, "DEEPAGENTS_TALON_WHATSAPP_ENABLED"):
        channels.append(WhatsAppChannel(WhatsAppChannelConfig.from_talon_config(config)))
    if telegram or _env_enabled(config.env, "DEEPAGENTS_TALON_TELEGRAM_ENABLED"):
        channels.append(TelegramChannel(TelegramChannelConfig.from_talon_config(config)))
    if discord or _env_enabled(config.env, "DEEPAGENTS_TALON_DISCORD_ENABLED"):
        channels.append(DiscordChannel(DiscordChannelConfig.from_talon_config(config)))
    if slack or _env_enabled(config.env, "DEEPAGENTS_TALON_SLACK_ENABLED"):
        channels.append(SlackChannel(SlackChannelConfig.from_talon_config(config)))
    return tuple(channels)


def _configure_logging(env: Mapping[str, str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger(_CHANNEL_LOGGER_NAME).setLevel(_channel_log_level(env))


def _channel_log_level(env: Mapping[str, str]) -> int:
    debug_enabled = env.get(_DCODE_DEBUG_ENV, "").strip().lower() in _DCODE_DEBUG_VALUES
    fallback = logging.DEBUG if debug_enabled else logging.INFO
    raw_level = env.get(_DCODE_LOG_LEVEL_ENV, "").strip().upper()
    if not raw_level:
        return fallback
    if level := _DCODE_LOG_LEVELS.get(raw_level):
        return level
    logger.warning(
        "Ignoring invalid %s; expected DEBUG, INFO, WARNING, ERROR, or CRITICAL",
        _DCODE_LOG_LEVEL_ENV,
    )
    return fallback


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


async def _deliver_cron_result(host: TalonHost, job: CronJob, text: str) -> None:
    channel = await host.origin_channel(job.origin)
    if channel is None:
        logger.warning("No channel serves cron job %s; dropping its result", job.id)
        return
    await host.deliver_scheduled_result(channel, job, text)


if __name__ == "__main__":
    main()
