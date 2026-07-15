"""Pure plugin manager content builders."""

from typing import Literal

from textual.content import Content
from textual.widgets.option_list import Option

from deepagents_code.config import get_glyphs
from deepagents_code.tui.modals.plugin_manager.models import _MarketplaceRow, _PluginRow


def _plugin_options(
    rows: tuple[_PluginRow, ...],
    *,
    action: Literal["detail", "installed"],
    status: str | None,
) -> list[Option]:
    options: list[Option] = []
    for index, row in enumerate(rows):
        if index > 0:
            options.append(Option(" ", id=f"spacer:{index}", disabled=True))
        options.append(
            Option(_plugin_prompt(row, status=status), id=f"{action}:{row.plugin_id}")
        )
    return options


def _plugin_prompt(row: _PluginRow, *, status: str | None) -> Content:
    glyphs = get_glyphs()
    plugin_name, _, marketplace = row.plugin_id.partition("@")
    meta_parts = ["Plugin", marketplace]
    if row.enabled:
        meta_parts.append(f"{glyphs.checkmark} enabled")
    if row.skill_count:
        meta_parts.append(
            f"{row.skill_count} {'skill' if row.skill_count == 1 else 'skills'}"
        )
    if row.mcp_connected is True:
        meta_parts.append(f"{glyphs.checkmark} connected")
    elif row.mcp_connected is False:
        meta_parts.append("restart to connect")
    if status:
        meta_parts.append(status)
    return Content.assemble(
        plugin_name,
        Content.styled(f" · {' · '.join(meta_parts)}", "dim"),
        "\n  ",
        Content.styled(row.description or "No description provided.", "dim"),
    )


def _install_details_options() -> list[Option]:
    return [
        Option("Install", id="action:install"),
        Option("Back to plugin list", id="details-back"),
    ]


def _installed_details_options(row: _PluginRow) -> list[Option]:
    return [
        Option(
            "Disable plugin" if row.enabled else "Enable plugin",
            id="action:toggle-enabled",
        ),
        Option(Content.styled("Uninstall", "bold"), id="action:uninstall"),
        Option("Back to plugin list", id="details-back"),
    ]


def _plugin_details_content(row: _PluginRow) -> Content:
    plugin_name, _, marketplace = row.plugin_id.partition("@")
    parts: list[Content | str] = [
        Content.styled("Plugin details", "bold"),
        "\n\n",
        Content.styled(plugin_name, "bold"),
        "\n",
        Content.styled(f"from {marketplace}", "dim"),
    ]
    if row.version:
        parts.extend(["\n", Content.styled(f"Version: {row.version}", "dim")])
    if row.description:
        parts.extend(["\n\n", row.description])
    if row.author:
        parts.extend(["\n\n", Content.styled(f"By: {row.author}", "dim")])
    parts.extend(
        [
            "\n\n",
            Content.styled("Will install:", "bold"),
            "\n  ",
            Content.styled("Components will be discovered at installation.", "dim"),
            "\n\n",
            Content.styled(
                "Make sure you trust a plugin before installing, updating, "
                "or using it.",
                "dim",
            ),
        ]
    )
    return Content.assemble(*parts)


def _installed_plugin_details_content(row: _PluginRow) -> Content:
    glyphs = get_glyphs()
    plugin_name, _, marketplace = row.plugin_id.partition("@")
    parts: list[Content | str] = [
        Content.styled(f"{plugin_name} @ {marketplace}", "bold")
    ]
    if row.version:
        parts.extend(["\n", Content.styled(f"Version: {row.version}", "dim")])
    if row.description:
        parts.extend(["\n\n", row.description])
    if row.author:
        parts.extend(["\n\n", Content.styled(f"Author: {row.author}", "dim")])
    status = f"{glyphs.checkmark} Enabled" if row.enabled else "Disabled"
    parts.extend(
        [
            "\n\n",
            Content.styled(f"Status: {status}", "dim"),
            "\n\n",
            Content.styled("Installed components:", "bold"),
        ]
    )
    lines = (
        [f"Skills: {', '.join(row.skill_names)}"]
        if row.skill_names
        else ([f"Skills: {row.skill_count}"] if row.skill_count else [])
    )
    if row.mcp_server_names:
        lines.append(f"MCP: {', '.join(row.mcp_server_names)}")
    for line in lines or ["No components discovered."]:
        parts.extend(["\n  ", Content.styled(line, "dim")])
    return Content.assemble(*parts)


def _marketplace_label(row: _MarketplaceRow, bullet: str) -> str:
    count = "failed" if row.plugin_count is None else f"{row.plugin_count} available"
    return f"{row.name} {bullet} {row.source} {bullet} {count}"


def _marketplace_details_options() -> list[Option]:
    return [
        Option(
            Content.styled("Remove marketplace", "bold"), id="action:remove-marketplace"
        ),
        Option("Back to marketplace list", id="details-back"),
    ]


def _confirm_marketplace_removal_options(row: _MarketplaceRow) -> list[Option]:
    label = "installed plugin" if row.installed_count == 1 else "installed plugins"
    return [
        Option(
            Content.styled(
                f"Remove marketplace and {row.installed_count} {label}", "bold"
            ),
            id="action:confirm-remove-marketplace",
        ),
        Option("Cancel", id="details-back"),
    ]


def _marketplace_details_content(row: _MarketplaceRow) -> Content:
    available = (
        "Unavailable" if row.plugin_count is None else f"{row.plugin_count} available"
    )
    return Content.assemble(
        Content.styled(row.name, "bold"),
        "\n",
        Content.styled(f"Source: {row.source}", "dim"),
        "\n",
        Content.styled(f"Plugins: {available}", "dim"),
        "\n",
        Content.styled(f"Installed: {row.installed_count}", "dim"),
    )


def _marketplace_removal_content(row: _MarketplaceRow) -> Content:
    suffix = "s" if row.installed_count != 1 else ""
    warning = (
        f"This also uninstalls {row.installed_count} plugin{suffix} from this "
        "marketplace. "
        if row.installed_count
        else ""
    )
    return Content.assemble(
        Content.styled(f"Remove marketplace {row.name}?", "bold"),
        "\n\n",
        Content.styled(
            f"{warning}The marketplace record and managed caches will be removed. "
            "Local source directories are not deleted.",
            "dim",
        ),
    )
