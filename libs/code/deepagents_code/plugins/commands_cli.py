"""CLI helpers for plugin management."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from deepagents_code.plugins import (
    add_marketplace_source,
    disable_plugin,
    enable_plugin,
    install_plugin,
    list_available_plugins,
    remove_marketplace,
    uninstall_plugin,
)
from deepagents_code.plugins.marketplace import MarketplaceError
from deepagents_code.plugins.store import load_marketplace_records


def setup_plugin_parser(
    subparsers: Any,  # noqa: ANN401  # argparse subparsers uses dynamic typing
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
    add_output_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """Set up the `plugin` CLI parser.

    Args:
        subparsers: Parent argparse subparsers object.
        make_help_action: Factory for parser-specific help actions.
        add_output_args: Optional callback that adds output-format flags.

    Returns:
        Plugin command parser.
    """

    def _help() -> None:
        from deepagents_code.ui import show_plugins_help

        show_plugins_help()

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-h", "--help", action=make_help_action(_help))
    parser = subparsers.add_parser(
        "plugin",
        aliases=["plugins"],
        help="Manage plugins",
        add_help=False,
        parents=[parent],
    )
    if add_output_args is not None:
        add_output_args(parser)
    plugin_sub = parser.add_subparsers(dest="plugin_command")

    list_parser = plugin_sub.add_parser("list", aliases=["ls"], help="List plugins")
    if add_output_args is not None:
        add_output_args(list_parser)

    install_parser = plugin_sub.add_parser("install", help="Install a plugin")
    install_parser.add_argument("plugin_id")
    install_parser.add_argument(
        "--scope",
        choices=("user", "project", "local"),
        default="user",
        help="Install scope (default: user)",
    )
    uninstall_parser = plugin_sub.add_parser("uninstall", help="Uninstall a plugin")
    uninstall_parser.add_argument("plugin_id")
    uninstall_parser.add_argument(
        "--scope",
        choices=("user", "project", "local"),
        default=None,
        help="Scope to uninstall (default: all scopes)",
    )

    enable_parser = plugin_sub.add_parser("enable", help="Enable a plugin")
    enable_parser.add_argument("plugin_id")
    disable_parser = plugin_sub.add_parser("disable", help="Disable a plugin")
    disable_parser.add_argument("plugin_id")

    marketplace_parser = plugin_sub.add_parser(
        "marketplace", help="Manage plugin marketplaces"
    )
    marketplace_sub = marketplace_parser.add_subparsers(dest="marketplace_command")
    marketplace_list = marketplace_sub.add_parser(
        "list", aliases=["ls"], help="List marketplaces"
    )
    if add_output_args is not None:
        add_output_args(marketplace_list)
    marketplace_add = marketplace_sub.add_parser("add", help="Add a marketplace")
    marketplace_add.add_argument("source")
    marketplace_add.add_argument("--enable-all", action="store_true")
    marketplace_remove = marketplace_sub.add_parser(
        "remove", help="Remove a marketplace and uninstall its plugins"
    )
    marketplace_remove.add_argument("name")
    return parser


def _rows_json() -> list[dict[str, object]]:
    return [
        {"id": plugin_id, "description": description, "enabled": enabled}
        for plugin_id, description, enabled in list_available_plugins()
    ]


def execute_plugin_command(args: argparse.Namespace) -> str | None:
    """Execute a plugin management command.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Text output for slash-command callers, or `None` when output was written.
    """
    output_format = getattr(args, "output_format", "text")
    command = getattr(args, "plugin_command", None)
    if command is None:
        from deepagents_code.ui import show_plugins_help

        show_plugins_help()
        return None
    if command in {"list", "ls"}:
        rows = _rows_json()
        if output_format == "json":
            from deepagents_code.output import write_json

            write_json("plugin list", rows)
            return None
        if not rows:
            text = "No plugin marketplaces configured."
        else:
            lines = []
            for row in rows:
                status = "enabled" if row["enabled"] else "disabled"
                lines.append(f"{status} {row['id']} {row['description']}".rstrip())
            text = "\n".join(lines)
        print(text)  # noqa: T201
        return text
    if command == "install":
        try:
            instance = install_plugin(args.plugin_id, scope=args.scope)
        except (MarketplaceError, FileNotFoundError, OSError, ValueError) as exc:
            text = f"Failed to install {args.plugin_id}: {exc}"
            print(text)  # noqa: T201
            return text
        details = f"scope: {args.scope}"
        if instance.version is not None:
            details += f", version: {instance.version}"
        text = (
            f"Installed plugin {instance.plugin_id} ({details}). "
            "Run /reload-plugins to activate."
        )
        print(text)  # noqa: T201
        return text
    if command == "uninstall":
        uninstall_plugin(args.plugin_id, scope=args.scope)
        text = f"Uninstalled plugin {args.plugin_id}."
        print(text)  # noqa: T201
        return text
    if command == "enable":
        enable_plugin(args.plugin_id)
        text = f"Enabled plugin {args.plugin_id}. Run /reload-plugins to activate."
        print(text)  # noqa: T201
        return text
    if command == "disable":
        disable_plugin(args.plugin_id)
        text = f"Disabled plugin {args.plugin_id}. Run /reload-plugins to unload."
        print(text)  # noqa: T201
        return text
    if command == "marketplace":
        marketplace_command = getattr(args, "marketplace_command", None)
        if marketplace_command in {"list", "ls"}:
            records = load_marketplace_records()
            rows = [
                {
                    "name": record.name,
                    "source_type": record.source_type,
                    "source": record.source,
                    "install_location": record.install_location,
                }
                for record in records.values()
            ]
            if output_format == "json":
                from deepagents_code.output import write_json

                write_json("plugin marketplace list", rows)
                return None
            text = (
                "No plugin marketplaces configured."
                if not rows
                else "\n".join(f"{row['name']} {row['source']}" for row in rows)
            )
            print(text)  # noqa: T201
            return text
        if marketplace_command == "add":
            marketplace = add_marketplace_source(args.source)
            installed: list[str] = []
            failed: list[str] = []
            if args.enable_all:
                for plugin in marketplace.plugins:
                    plugin_id = f"{plugin.name}@{marketplace.name}"
                    try:
                        install_plugin(plugin_id, scope="user")
                    except (
                        MarketplaceError,
                        FileNotFoundError,
                        OSError,
                        ValueError,
                    ) as exc:
                        failed.append(f"{plugin_id}: {exc}")
                    else:
                        installed.append(plugin_id)
            text = (
                f"Added marketplace {marketplace.name} "
                f"({len(marketplace.plugins)} plugin(s))."
            )
            if installed:
                text += f" Installed: {', '.join(installed)}."
            if failed:
                text += f" Failed to install: {'; '.join(failed)}."
            print(text)  # noqa: T201
            return text
        if marketplace_command == "remove":
            removed = remove_marketplace(args.name)
            text = (
                f"Removed marketplace {args.name} and its installed plugins."
                if removed
                else f"Marketplace {args.name} is not configured."
            )
            print(text)  # noqa: T201
            return text
    text = "Usage: plugin {list,install,uninstall,enable,disable,marketplace}"
    print(text)  # noqa: T201
    return text
