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


def _load_state_label(row: _PluginRow) -> str | None:
    glyphs = get_glyphs()
    if row.load_state == "error":
        return "error"
    if row.load_state == "pending_reload":
        return "pending /reload"
    if row.load_state == "enabled":
        return f"{glyphs.checkmark} enabled"
    return None


def _plugin_prompt(row: _PluginRow, *, status: str | None) -> Content:
    glyphs = get_glyphs()
    _, _, marketplace = row.plugin_id.partition("@")
    meta_parts = [Content.styled("Plugin", "dim"), Content.styled(marketplace, "dim")]
    load_label = _load_state_label(row)
    if load_label:
        if row.load_state == "enabled":
            meta_parts.append(Content.styled(load_label, "bold"))
        elif row.load_state == "error":
            meta_parts.append(Content.styled(load_label, "bold $error"))
        else:
            meta_parts.append(Content.styled(load_label, "dim"))
    if row.skill_count:
        skill_label = "skill" if row.skill_count == 1 else "skills"
        meta_parts.append(Content.styled(f"{row.skill_count} {skill_label}", "dim"))
    if row.load_state == "enabled":
        if row.mcp_connected is True:
            meta_parts.append(Content.styled(f"{glyphs.checkmark} connected", "dim"))
        elif row.mcp_connected is False:
            meta_parts.append(Content.styled("restart to connect", "bold $warning"))
    elif (
        row.load_state == "pending_reload"
        and row.session_loaded
        and row.mcp_connected is True
    ):
        meta_parts.append(Content.styled(f"{glyphs.checkmark} connected", "dim"))
    if status:
        meta_parts.append(Content.styled(status, "dim"))
    separator = Content.styled(" · ", "dim")
    return Content.assemble(
        row.label,
        separator,
        separator.join(meta_parts),
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


def _component_summary_lines(row: _PluginRow) -> list[str]:
    lines: list[str] = []
    if row.skill_names:
        lines.append(f"Skills: {', '.join(row.skill_names)}")
    elif row.skill_count:
        lines.append(f"Skills: {row.skill_count}")
    if row.mcp_server_names:
        lines.append(f"MCP: {', '.join(row.mcp_server_names)}")
    if row.unsupported_components:
        names = ", ".join(f"{name}/" for name in row.unsupported_components)
        lines.append(f"Unsupported (not loaded): {names}")
    return lines


def _unsupported_summary(components: tuple[str, ...]) -> str:
    return f"Found unsupported: {', '.join(f'{name}/' for name in components)}."


def _will_install_lines(row: _PluginRow) -> list[str]:
    lines = _component_summary_lines(row)
    if row.skill_names or row.mcp_server_names or row.skill_count:
        return lines
    if row.unsupported_components:
        return [
            "No supported components (skills/MCP).",
            _unsupported_summary(row.unsupported_components),
        ]
    if row.skill_count is None:
        return [
            (
                "Skills and MCP servers if present "
                "(agents/, commands/, and hooks/ are not loaded)."
            )
        ]
    return [
        "No supported components (skills/MCP).",
        "agents/, commands/, and hooks/ are not loaded by deepagents-code.",
    ]


def _installed_component_lines(row: _PluginRow) -> list[str]:
    lines = _component_summary_lines(row)
    if lines:
        has_supported = bool(row.skill_names or row.mcp_server_names)
        if not has_supported and row.unsupported_components:
            return [
                "No supported components (skills/MCP).",
                _unsupported_summary(row.unsupported_components),
            ]
        return lines
    return ["No components found."]


def _status_lines(row: _PluginRow) -> list[str]:
    glyphs = get_glyphs()
    if row.load_state == "error":
        detail = row.load_error or "Plugin failed to load."
        return [f"Status: Error — {detail}", "Fix the error, then run /reload."]
    if row.load_state == "disabled":
        return ["Status: Disabled", "Enable the plugin, then run /reload to load it."]
    if row.load_state == "pending_reload":
        if row.enabled:
            return [
                "Status: Installed · pending /reload",
                "Run /reload to load this plugin into the current session.",
            ]
        return [
            "Status: Disabled · pending /reload",
            "Run /reload to unload this plugin from the current session.",
        ]
    lines = [f"Status: {glyphs.checkmark} Enabled"]
    if row.mcp_connected is False:
        lines.append("MCP servers need a server restart (/reload) to connect.")
    return lines


def _plugin_details_content(row: _PluginRow) -> Content:
    _, _, marketplace = row.plugin_id.partition("@")
    parts: list[Content | str] = [
        Content.styled("Plugin details", "bold"),
        "\n\n",
        Content.styled(row.label, "bold"),
        "\n",
        Content.styled(f"from {marketplace}", "dim"),
    ]
    if row.version:
        parts.extend(["\n", Content.styled(f"Version: {row.version}", "dim")])
    if row.description:
        parts.extend(["\n\n", row.description])
    if row.author:
        parts.extend(["\n\n", Content.styled(f"By: {row.author}", "dim")])
    parts.extend(["\n\n", Content.styled("Will install:", "bold")])
    for line in _will_install_lines(row):
        parts.extend(["\n  ", Content.styled(line, "dim")])
    parts.extend(
        [
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
    _, _, marketplace = row.plugin_id.partition("@")
    parts: list[Content | str] = [
        Content.styled(f"{row.label} @ {marketplace}", "bold")
    ]
    if row.version:
        parts.extend(["\n", Content.styled(f"Version: {row.version}", "dim")])
    if row.description:
        parts.extend(["\n\n", row.description])
    if row.author:
        parts.extend(["\n\n", Content.styled(f"Author: {row.author}", "dim")])
    parts.extend(["\n\n"])
    for index, line in enumerate(_status_lines(row)):
        if index:
            parts.append("\n")
        parts.append(Content.styled(line, "dim"))
    parts.extend(["\n\n", Content.styled("Installed components:", "bold")])
    for line in _installed_component_lines(row):
        parts.extend(["\n  ", Content.styled(line, "dim")])
    return Content.assemble(*parts)


def _marketplace_label(row: _MarketplaceRow) -> Content:
    glyphs = get_glyphs()
    prefix = f"{row.name} {glyphs.bullet} {row.source} {glyphs.bullet} "
    if row.has_error:
        return Content.assemble(
            prefix,
            Content.styled(f"{glyphs.error} Error", "$error"),
        )
    return Content.assemble(prefix, f"{row.plugin_count} available")


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
    available = "Unavailable" if row.has_error else f"{row.plugin_count} available"
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
    if row.installed_count:
        detail = (
            f"This removes the marketplace and uninstalls {row.installed_count} "
            f"plugin{suffix} from it."
        )
    else:
        detail = "This removes the marketplace from your installed list."
    return Content.assemble(
        Content.styled(f"Remove marketplace {row.name}?", "bold"),
        "\n\n",
        Content.styled(detail, "dim"),
    )
