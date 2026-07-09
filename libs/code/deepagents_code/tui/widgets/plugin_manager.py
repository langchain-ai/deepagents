"""Interactive plugin manager screen."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from textual.app import ComposeResult

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.plugins.models import InstallScope, PluginInstance

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.plugins import (
    add_marketplace_source,
    disable_plugin,
    discover_plugins,
    enable_plugin_with_scope,
    install_plugin,
    remove_marketplace,
    uninstall_plugin,
)
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    load_marketplace_location,
)
from deepagents_code.plugins.store import (
    get_primary_install_entry,
    load_enabled_plugins,
    load_favorite_plugins,
    load_installed_plugins,
    load_marketplace_records,
    set_plugin_favorite,
)

logger = logging.getLogger(__name__)

PluginTab = Literal["discover", "installed", "marketplaces", "errors"]
ViewMode = Literal[
    "list",
    "add_marketplace",
    "plugin_details",
    "installed_details",
    "mcp_details",
    "marketplace_details",
    "confirm_remove_marketplace",
]
InstalledGroup = Literal["favorites", "project", "local", "user"]


@dataclass(frozen=True, slots=True)
class _PluginRow:
    plugin_id: str
    description: str
    enabled: bool
    version: str | None
    author: str | None
    scope: InstallScope | None
    favorite: bool = False
    skill_count: int | None = None
    """Discovered skill count, or `None` when not computed (e.g. not enabled)."""
    skill_names: tuple[str, ...] = ()
    mcp_connected: bool | None = None
    """MCP connection state, or `None` when the plugin declares no MCP servers."""
    mcp_server_names: tuple[str, ...] = ()
    command_count: int = 0
    agent_count: int = 0
    hook_count: int = 0


@dataclass(frozen=True, slots=True)
class _McpRow:
    name: str
    transport: str
    status: str
    tool_count: int
    tool_names: tuple[str, ...]
    scope: InstallScope
    favorite: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _MarketplaceRow:
    name: str
    source: str
    plugin_count: int | None
    installed_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ManagerState:
    available_plugins: tuple[_PluginRow, ...]
    installed_plugins: tuple[_PluginRow, ...]
    mcp_servers: tuple[_McpRow, ...]
    marketplaces: tuple[_MarketplaceRow, ...]
    errors: tuple[str, ...]
    favorites: frozenset[str] = field(default_factory=frozenset)


def _list_plugin_skill_names(instance: PluginInstance) -> tuple[str, ...]:
    """Return namespaced skill names contributed by a plugin instance."""
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.skills import (
        _list_skills as list_sdk_skills,  # noqa: PLC2701
    )

    from deepagents_code.plugins.adapters.skills import plugin_skill_sources

    names: list[str] = []
    for path, _label, prefix in plugin_skill_sources((instance,)):
        try:
            backend = FilesystemBackend(root_dir=path, virtual_mode=False)
            names.extend(
                f"{prefix}{skill['name']}" for skill in list_sdk_skills(backend, ".")
            )
        except Exception:
            logger.warning(
                "Could not list skills for plugin %s",
                instance.plugin_id,
                exc_info=True,
            )
    return tuple(dict.fromkeys(names))


def _plugin_mcp_server_names(instance: PluginInstance) -> tuple[str, ...]:
    """Return scoped MCP server names declared by a plugin."""
    from deepagents_code.plugins.adapters.mcp import plugin_mcp_configs

    names: list[str] = []
    for config in plugin_mcp_configs((instance,)):
        servers = config.get("mcpServers")
        if isinstance(servers, dict):
            names.extend(key for key in servers if isinstance(key, str))
    return tuple(dict.fromkeys(names))


def _plugin_mcp_connected(
    instance: PluginInstance, mcp_server_info: Sequence[MCPServerInfo]
) -> bool | None:
    """Return whether a plugin's MCP servers are connected in this session.

    Args:
        instance: Discovered plugin instance.
        mcp_server_info: Live MCP server metadata from the running session.

    Returns:
        `True` when every scoped MCP server the plugin declares is connected,
        `False` when at least one declared server isn't connected yet (e.g.
        before `/restart`), or `None` when the plugin declares no MCP servers.
    """
    expected = frozenset(_plugin_mcp_server_names(instance))
    if not expected:
        return None
    connected = {info.name for info in mcp_server_info if info.status == "ok"}
    return expected <= connected


def _mcp_scope_for_server(
    server_name: str, plugin_mcp_scopes: dict[str, InstallScope]
) -> InstallScope:
    """Infer an Installed-tab scope for an MCP server name.

    Returns:
        Scope used for grouping: owning plugin scope when known, otherwise `user`.
    """
    return plugin_mcp_scopes.get(server_name, "user")


def _status_label(status: str, glyphs: object) -> str:
    checkmark = getattr(glyphs, "checkmark", "✓")
    if status == "ok":
        return f"{checkmark} connected"
    if status == "disabled":
        return "disabled"
    if status == "unauthenticated":
        return "needs login"
    if status == "awaiting_reconnect":
        return "restart to connect"
    if status == "error":
        return "error"
    return status


def _instance_for_manager_row(
    plugin_id: str,
    *,
    discovered: dict[str, PluginInstance],
    is_installed: bool,
) -> PluginInstance | None:
    """Return a plugin instance for manager metadata.

    Prefers the live discovery result (enabled + cached). For installed but
    disabled plugins, loads inventory from the install cache path so the
    Installed tab can still show component counts.
    """
    instance = discovered.get(plugin_id)
    if instance is not None:
        return instance
    if not is_installed:
        return None
    entry = get_primary_install_entry(plugin_id, project_path=str(Path.cwd().resolve()))
    if entry is None:
        return None
    root = Path(entry.install_path)
    if not root.is_dir():
        return None
    from deepagents_code.plugins.discovery import _plugin_from_install_path

    plugin_name = plugin_id.rsplit("@", 1)[0]
    marketplace_name = plugin_id.rsplit("@", 1)[-1]
    loaded, _warnings = _plugin_from_install_path(
        plugin_id=plugin_id,
        root=root,
        marketplace_name=marketplace_name,
        fallback_name=plugin_name,
        version=entry.version,
    )
    return loaded


def _load_manager_state(
    mcp_server_info: Sequence[MCPServerInfo] = (),
) -> _ManagerState:
    records = load_marketplace_records()
    enabled = load_enabled_plugins()
    installed = load_installed_plugins()
    favorites = load_favorite_plugins()
    discovered = {
        instance.plugin_id: instance for instance in discover_plugins().plugins
    }
    available_plugins: list[_PluginRow] = []
    installed_plugins: list[_PluginRow] = []
    marketplaces: list[_MarketplaceRow] = []
    errors: list[str] = []
    plugin_mcp_scopes: dict[str, InstallScope] = {}

    for name, record in sorted(records.items()):
        try:
            marketplace = load_marketplace_location(Path(record.install_location))
        except MarketplaceError as exc:
            message = f"{name}: {exc}"
            errors.append(message)
            marketplaces.append(
                _MarketplaceRow(
                    name=name,
                    source=record.source,
                    plugin_count=None,
                    installed_count=sum(
                        plugin_id.endswith(f"@{name}") for plugin_id in installed
                    ),
                    error=str(exc),
                )
            )
            continue

        marketplaces.append(
            _MarketplaceRow(
                name=marketplace.name,
                source=record.source,
                plugin_count=len(marketplace.plugins),
                installed_count=sum(
                    plugin_id.endswith(f"@{marketplace.name}")
                    for plugin_id in installed
                ),
            )
        )
        for plugin in marketplace.plugins:
            plugin_id = f"{plugin.name}@{marketplace.name}"
            is_enabled = enabled.get(plugin_id, False)
            is_installed = plugin_id in installed
            instance = _instance_for_manager_row(
                plugin_id, discovered=discovered, is_installed=is_installed
            )
            scope: InstallScope | None = None
            if is_installed:
                primary = get_primary_install_entry(
                    plugin_id, project_path=str(Path.cwd().resolve())
                )
                scope = _scope_value(primary.scope if primary else None)
            skill_names = _list_plugin_skill_names(instance) if instance else ()
            mcp_names = _plugin_mcp_server_names(instance) if instance else ()
            if scope is not None:
                plugin_mcp_scopes.update(dict.fromkeys(mcp_names, scope))
            row = _PluginRow(
                plugin_id=plugin_id,
                description=plugin.description or "",
                enabled=is_enabled,
                version=(instance.version if instance is not None else plugin.version),
                author=_author_display(plugin.manifest_fields.get("author")),
                scope=scope,
                favorite=plugin_id in favorites,
                skill_count=len(skill_names) if instance else None,
                skill_names=skill_names,
                mcp_connected=(
                    _plugin_mcp_connected(instance, mcp_server_info)
                    if instance and is_enabled
                    else None
                ),
                mcp_server_names=mcp_names,
                command_count=(
                    len(instance.inventory.commands)
                    + (
                        len(instance.manifest.inline_commands)
                        if instance and instance.manifest
                        else 0
                    )
                    if instance
                    else 0
                ),
                agent_count=len(instance.inventory.agents) if instance else 0,
                hook_count=(
                    (
                        len(instance.inventory.hooks_files)
                        + (
                            len(instance.manifest.inline_hooks)
                            if instance.manifest
                            else 0
                        )
                    )
                    if instance
                    else 0
                ),
            )
            if is_installed:
                installed_plugins.append(row)
            else:
                available_plugins.append(row)

    mcp_servers: list[_McpRow] = []
    for info in mcp_server_info:
        if info.name.startswith("<config:"):
            continue
        tool_names = tuple(tool.name for tool in info.tools)
        mcp_servers.append(
            _McpRow(
                name=info.name,
                transport=info.transport,
                status=info.status,
                tool_count=len(info.tools),
                tool_names=tool_names,
                scope=_mcp_scope_for_server(info.name, plugin_mcp_scopes),
                favorite=info.name in favorites,
                error=info.error,
            )
        )

    return _ManagerState(
        available_plugins=tuple(available_plugins),
        installed_plugins=tuple(installed_plugins),
        mcp_servers=tuple(mcp_servers),
        marketplaces=tuple(marketplaces),
        errors=tuple(errors),
        favorites=frozenset(favorites),
    )


def _author_display(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            return name
    return None


def _scope_value(value: str | None) -> InstallScope | None:
    if value == "user":
        return "user"
    if value == "project":
        return "project"
    if value == "local":
        return "local"
    return None


class PluginManagerScreen(ModalScreen[None]):
    """Arrow-key navigable plugin manager for `/plugins`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
        Binding("left", "previous_tab", "Previous tab", show=False, priority=True),
        Binding("right", "next_tab", "Next tab", show=False, priority=True),
        Binding("tab", "next_tab", "Next tab", show=False, priority=True),
        Binding("shift+tab", "previous_tab", "Previous tab", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("d", "remove_marketplace", "Remove marketplace", show=False),
    ]

    CSS = """
    PluginManagerScreen {
        align: center middle;
        background: transparent;
    }

    PluginManagerScreen > Vertical {
        width: 88;
        max-width: 94%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    PluginManagerScreen .plugin-manager-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    PluginManagerScreen .plugin-manager-tabs {
        color: $text-muted;
        margin-bottom: 0;
    }

    PluginManagerScreen .plugin-manager-divider {
        color: $text-muted;
        margin: 0 0 1 0;
        height: 1;
    }

    PluginManagerScreen .plugin-manager-status {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    PluginManagerScreen .plugin-manager-error {
        height: auto;
        color: $error;
        margin-bottom: 1;
    }

    PluginManagerScreen #plugin-manager-options {
        height: 1fr;
        min-height: 5;
        background: $background;
    }

    PluginManagerScreen #plugin-marketplace-source {
        margin-top: 1;
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    PluginManagerScreen #plugin-marketplace-source:focus {
        border: solid $primary;
    }

    PluginManagerScreen .plugin-manager-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    _tabs: ClassVar[tuple[PluginTab, ...]] = (
        "discover",
        "installed",
        "marketplaces",
        "errors",
    )

    _SCOPE_ORDER: ClassVar[tuple[InstallScope, ...]] = ("project", "local", "user")
    _GROUP_ORDER: ClassVar[tuple[InstalledGroup, ...]] = (
        "favorites",
        "project",
        "local",
        "user",
    )
    _GROUP_LABELS: ClassVar[dict[InstalledGroup, str]] = {
        "favorites": "Favorites",
        "project": "Project",
        "local": "Local",
        "user": "User",
    }

    def __init__(
        self,
        *,
        mcp_server_info: Sequence[MCPServerInfo] = (),
        on_toggle_mcp_disable: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            mcp_server_info: Live MCP server metadata from the running session,
                used to show connection status for plugins that declare MCP
                servers and to list MCP rows in Installed.
            on_toggle_mcp_disable: Async callback to flip an MCP server's
                persistent disabled state (same path as `/mcp` F2).
        """
        super().__init__()
        self._tab: PluginTab = "discover"
        self._mode: ViewMode = "list"
        self._mcp_server_info = mcp_server_info
        self._on_toggle_mcp_disable = on_toggle_mcp_disable
        self._state = _load_manager_state(mcp_server_info)
        self._status: str | None = None
        self._error: str | None = None
        self._selected_plugin: _PluginRow | None = None
        self._selected_mcp: _McpRow | None = None
        self._selected_marketplace: _MarketplaceRow | None = None

    @override
    def render(self) -> str:
        """Render an empty modal background.

        Returns:
            Blank renderable content for the modal screen itself.
        """
        return ""

    def compose(self) -> ComposeResult:
        """Compose the manager screen.

        Yields:
            Widgets for the plugin manager UI.
        """
        with Vertical():
            yield Static("Plugins", classes="plugin-manager-title")
            yield Static(self._tabs_text(), id="plugin-manager-tabs")
            yield Rule(
                line_style="heavy" if not is_ascii_mode() else "ascii",
                classes="plugin-manager-divider",
            )
            yield Static(
                "",
                id="plugin-manager-status",
                classes="plugin-manager-status",
                markup=False,
            )
            yield Static(
                "",
                id="plugin-manager-error",
                classes="plugin-manager-error",
                markup=False,
            )
            yield OptionList(id="plugin-manager-options")
            yield Input(
                placeholder=(
                    "owner/repo, git@github.com:owner/repo.git, "
                    "https://.../marketplace.json, or ./path"
                ),
                id="plugin-marketplace-source",
            )
            yield Static("", id="plugin-manager-help", classes="plugin-manager-help")

    def on_mount(self) -> None:
        """Apply initial render and focus."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._refresh_view()

    def _tabs_text(self) -> str:
        labels = {
            "discover": "Plugins",
            "installed": "Installed",
            "marketplaces": "Marketplaces",
            "errors": "Errors",
        }
        parts = []
        for tab in self._tabs:
            label = labels[tab]
            parts.append(f"> {label} <" if tab == self._tab else f"  {label}  ")
        return " ".join(parts)

    def _current_options(self) -> list[Option]:
        glyphs = get_glyphs()
        if self._tab == "discover":
            if not self._state.marketplaces:
                return [
                    Option(
                        "No plugins available. Add a marketplace first.",
                        id="empty",
                    )
                ]
            if not self._state.available_plugins:
                return [Option("All available plugins are installed.", id="empty")]
            return self._plugin_options(
                self._state.available_plugins,
                action="detail",
                status=None,
            )
        if self._tab == "installed":
            if not self._state.installed_plugins and not self._state.mcp_servers:
                return [Option("No plugins installed.", id="empty")]
            return self._grouped_installed_options()
        if self._tab == "marketplaces":
            options = [Option("+ Add Marketplace", id="add-marketplace")]
            options.extend(
                Option(
                    Content(self._marketplace_label(row, glyphs.bullet)),
                    id=f"marketplace:{row.name}",
                )
                for row in self._state.marketplaces
            )
            return options
        if not self._state.errors:
            return [Option("No plugin errors.", id="empty")]
        return [Option(Content(error), id="empty") for error in self._state.errors]

    @staticmethod
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
                Option(
                    PluginManagerScreen._plugin_prompt(row, status=status),
                    id=f"{action}:{row.plugin_id}",
                )
            )
        return options

    def _grouped_installed_options(self) -> list[Option]:
        """Group installed plugins and MCP servers by Favorites then scope.

        Returns:
            Options with disabled section headers and selectable plugin/MCP rows.
        """
        groups: dict[InstalledGroup, list[_PluginRow | _McpRow]] = {
            group: [] for group in self._GROUP_ORDER
        }
        for row in self._state.installed_plugins:
            if row.favorite:
                groups["favorites"].append(row)
            else:
                groups[row.scope or "user"].append(row)
        for row in self._state.mcp_servers:
            if row.favorite:
                groups["favorites"].append(row)
            else:
                groups[row.scope].append(row)

        options: list[Option] = []
        for group in self._GROUP_ORDER:
            rows = groups[group]
            if not rows:
                continue
            if options:
                options.append(Option(" ", id=f"spacer:{group}", disabled=True))
            options.append(
                Option(
                    Content.styled(self._GROUP_LABELS[group], "bold"),
                    id=f"header:{group}",
                    disabled=True,
                )
            )
            for row in rows:
                if isinstance(row, _PluginRow):
                    options.append(
                        Option(
                            self._plugin_prompt(row, status=None),
                            id=f"installed:{row.plugin_id}",
                        )
                    )
                else:
                    options.append(
                        Option(
                            self._mcp_prompt(row),
                            id=f"mcp:{row.name}",
                        )
                    )
        return options

    @staticmethod
    def _plugin_prompt(row: _PluginRow, *, status: str | None) -> Content:
        glyphs = get_glyphs()
        plugin_name, _, marketplace = row.plugin_id.partition("@")
        meta_parts = ["Plugin", marketplace]
        if row.enabled:
            meta_parts.append(f"{glyphs.checkmark} enabled")
        if row.skill_count:
            unit = "skill" if row.skill_count == 1 else "skills"
            meta_parts.append(f"{row.skill_count} {unit}")
        if row.command_count:
            unit = "command" if row.command_count == 1 else "commands"
            meta_parts.append(f"{row.command_count} {unit}")
        if row.agent_count:
            unit = "agent" if row.agent_count == 1 else "agents"
            meta_parts.append(f"{row.agent_count} {unit}")
        if row.hook_count:
            unit = "hook" if row.hook_count == 1 else "hooks"
            meta_parts.append(f"{row.hook_count} {unit}")
        if row.mcp_connected is True:
            meta_parts.append(f"{glyphs.checkmark} connected")
        elif row.mcp_connected is False:
            meta_parts.append("restart to connect")
        if status:
            meta_parts.append(status)
        meta = " · ".join(meta_parts)
        description = row.description or "No description provided."
        return Content.assemble(
            plugin_name,
            Content.styled(f" · {meta}", "dim"),
            "\n  ",
            Content.styled(description, "dim"),
        )

    @staticmethod
    def _mcp_prompt(row: _McpRow) -> Content:
        glyphs = get_glyphs()
        meta_parts = ["MCP", row.transport, _status_label(row.status, glyphs)]
        if row.tool_count:
            unit = "tool" if row.tool_count == 1 else "tools"
            meta_parts.append(f"{row.tool_count} {unit}")
        meta = " · ".join(meta_parts)
        return Content.assemble(
            row.name,
            Content.styled(f" · {meta}", "dim"),
        )

    @staticmethod
    def _install_details_options() -> list[Option]:
        return [
            Option("Install for you (user scope)", id="install:user"),
            Option(
                "Install for you in this project (project scope)",
                id="install:project",
            ),
            Option(
                "Install for you, in this repo only (local scope)",
                id="install:local",
            ),
            Option("Back to plugin list", id="details-back"),
        ]

    @staticmethod
    def _installed_details_options(row: _PluginRow) -> list[Option]:
        favorite_label = "Remove from favorites" if row.favorite else "Add to favorites"
        return [
            Option(
                "Disable plugin" if row.enabled else "Enable plugin",
                id="action:toggle-enabled",
            ),
            Option(favorite_label, id="action:toggle-favorite"),
            Option(
                Content.styled("Uninstall", "bold"),
                id="action:uninstall",
            ),
            Option("Back to plugin list", id="details-back"),
        ]

    @staticmethod
    def _mcp_details_options(row: _McpRow) -> list[Option]:
        toggle_label = "Enable server" if row.status == "disabled" else "Disable server"
        favorite_label = "Remove from favorites" if row.favorite else "Add to favorites"
        return [
            Option(toggle_label, id="action:toggle-mcp"),
            Option(favorite_label, id="action:toggle-favorite"),
            Option("Back to plugin list", id="details-back"),
        ]

    @staticmethod
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
                    "Make sure you trust a plugin before installing, updating, or "
                    "using it.",
                    "dim",
                ),
            ]
        )
        return Content.assemble(*parts)

    @staticmethod
    def _installed_plugin_details_content(row: _PluginRow) -> Content:
        glyphs = get_glyphs()
        plugin_name, _, marketplace = row.plugin_id.partition("@")
        parts: list[Content | str] = [
            Content.styled(f"{plugin_name} @ {marketplace}", "bold"),
        ]
        if row.scope:
            parts.extend(["\n", Content.styled(f"Scope: {row.scope}", "dim")])
        if row.version:
            parts.extend(["\n", Content.styled(f"Version: {row.version}", "dim")])
        if row.description:
            parts.extend(["\n\n", row.description])
        if row.author:
            parts.extend(["\n\n", Content.styled(f"Author: {row.author}", "dim")])
        status = f"{glyphs.checkmark} Enabled" if row.enabled else "Disabled"
        parts.extend(["\n\n", Content.styled(f"Status: {status}", "dim")])
        parts.extend(["\n\n", Content.styled("Installed components:", "bold")])
        component_lines: list[str] = []
        if row.skill_names:
            component_lines.append(f"Skills: {', '.join(row.skill_names)}")
        elif row.skill_count:
            component_lines.append(f"Skills: {row.skill_count}")
        if row.command_count:
            unit = "command" if row.command_count == 1 else "commands"
            component_lines.append(f"Commands: {row.command_count} {unit}")
        if row.agent_count:
            unit = "agent" if row.agent_count == 1 else "agents"
            component_lines.append(f"Agents: {row.agent_count} {unit}")
        if row.hook_count:
            unit = "hook" if row.hook_count == 1 else "hooks"
            component_lines.append(f"Hooks: {row.hook_count} {unit}")
        if row.mcp_server_names:
            component_lines.append(f"MCP: {', '.join(row.mcp_server_names)}")
        if not component_lines:
            component_lines.append("No components discovered.")
        for line in component_lines:
            parts.extend(["\n  ", Content.styled(line, "dim")])
        return Content.assemble(*parts)

    @staticmethod
    def _mcp_details_content(row: _McpRow) -> Content:
        glyphs = get_glyphs()
        parts: list[Content | str] = [
            Content.styled(row.name, "bold"),
            "\n",
            Content.styled("MCP server", "dim"),
            "\n",
            Content.styled(f"Transport: {row.transport}", "dim"),
            "\n",
            Content.styled(
                f"Status: {_status_label(row.status, glyphs)}",
                "dim",
            ),
            "\n",
            Content.styled(f"Scope: {row.scope}", "dim"),
        ]
        if row.error and row.status != "ok":
            parts.extend(["\n\n", Content.styled(row.error, "dim")])
        if row.tool_names:
            parts.extend(["\n\n", Content.styled("Tools:", "bold")])
            for name in row.tool_names:
                parts.extend(["\n  ", Content.styled(name, "dim")])
        elif row.status == "ok":
            parts.extend(
                ["\n\n", Content.styled("No tools exposed by this server.", "dim")]
            )
        return Content.assemble(*parts)

    @staticmethod
    def _nearest_enabled_index(options: OptionList, candidate: int) -> int | None:
        """Return the nearest selectable option index to `candidate`.

        Scope-group header rows (and spacers) are disabled options, so a
        highlighted index carried over from a previous tab/refresh can land
        on one after the option count or grouping changes. Scans forward
        first, then backward, so the cursor always rests on a real row.

        Args:
            options: Option list to scan.
            candidate: Preferred index (already clamped to bounds).

        Returns:
            `candidate` if selectable, the nearest selectable index otherwise,
            or `None` if every option (or the list itself) is disabled/empty.
        """
        if not options.option_count:
            return None
        if not options.get_option_at_index(candidate).disabled:
            return candidate
        for index in range(candidate + 1, options.option_count):
            if not options.get_option_at_index(index).disabled:
                return index
        for index in range(candidate - 1, -1, -1):
            if not options.get_option_at_index(index).disabled:
                return index
        return None

    @staticmethod
    def _marketplace_label(row: _MarketplaceRow, bullet: str) -> str:
        count = (
            "failed" if row.plugin_count is None else f"{row.plugin_count} available"
        )
        return f"{row.name} {bullet} {row.source} {bullet} {count}"

    @staticmethod
    def _marketplace_details_options() -> list[Option]:
        return [
            Option(
                Content.styled("Remove marketplace", "bold"),
                id="action:remove-marketplace",
            ),
            Option("Back to marketplace list", id="details-back"),
        ]

    @staticmethod
    def _confirm_marketplace_removal_options(
        row: _MarketplaceRow,
    ) -> list[Option]:
        plugin_label = (
            "installed plugin" if row.installed_count == 1 else "installed plugins"
        )
        return [
            Option(
                Content.styled(
                    f"Remove marketplace and {row.installed_count} {plugin_label}",
                    "bold",
                ),
                id="action:confirm-remove-marketplace",
            ),
            Option("Cancel", id="details-back"),
        ]

    @staticmethod
    def _marketplace_details_content(row: _MarketplaceRow) -> Content:
        available = (
            "Unavailable"
            if row.plugin_count is None
            else f"{row.plugin_count} available"
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

    @staticmethod
    def _marketplace_removal_content(row: _MarketplaceRow) -> Content:
        warning = (
            f"This also uninstalls {row.installed_count} plugin"
            f"{'s' if row.installed_count != 1 else ''} from this marketplace. "
            if row.installed_count
            else ""
        )
        return Content.assemble(
            Content.styled(f"Remove marketplace {row.name}?", "bold"),
            "\n\n",
            Content.styled(
                f"{warning}The marketplace record and managed caches will be "
                "removed. Local source directories are not deleted.",
                "dim",
            ),
        )

    def _details_mode_active(self) -> bool:
        return self._mode in {
            "plugin_details",
            "installed_details",
            "mcp_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }

    def _refresh_view(self) -> None:
        self.query_one("#plugin-manager-tabs", Static).update(self._tabs_text())
        status_widget = self.query_one("#plugin-manager-status", Static)
        if self._mode == "plugin_details" and self._selected_plugin is not None:
            status_widget.update(self._plugin_details_content(self._selected_plugin))
        elif self._mode == "installed_details" and self._selected_plugin is not None:
            status_widget.update(
                self._installed_plugin_details_content(self._selected_plugin)
            )
        elif self._mode == "mcp_details" and self._selected_mcp is not None:
            status_widget.update(self._mcp_details_content(self._selected_mcp))
        elif (
            self._mode == "marketplace_details"
            and self._selected_marketplace is not None
        ):
            status_widget.update(
                self._marketplace_details_content(self._selected_marketplace)
            )
        elif (
            self._mode == "confirm_remove_marketplace"
            and self._selected_marketplace is not None
        ):
            status_widget.update(
                self._marketplace_removal_content(self._selected_marketplace)
            )
        else:
            status_widget.update(self._status or "")
        error = self._error or ""
        self.query_one("#plugin-manager-error", Static).update(error)

        options = self.query_one("#plugin-manager-options", OptionList)
        source_input = self.query_one("#plugin-marketplace-source", Input)
        help_text = self.query_one("#plugin-manager-help", Static)
        glyphs = get_glyphs()

        if self._mode == "add_marketplace":
            options.display = False
            source_input.display = True
            source_input.focus()
            help_text.update(
                f"Enter to add {glyphs.bullet} Esc cancel "
                f"{glyphs.bullet} paste a marketplace source"
            )
            return

        if self._details_mode_active():
            source_input.display = False
            options.display = True
            options.clear_options()
            detail_options = self._active_details_options()
            for option in detail_options:
                options.add_option(option)
            if options.option_count:
                options.highlighted = self._nearest_enabled_index(options, 0)
            options.focus()
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter choose {glyphs.bullet} Esc back"
            )
            return

        source_input.display = False
        options.display = True
        highlighted = options.highlighted
        options.clear_options()
        for option in self._current_options():
            options.add_option(option)
        if options.option_count:
            candidate = (
                0 if highlighted is None else min(highlighted, options.option_count - 1)
            )
            options.highlighted = self._nearest_enabled_index(options, candidate)
        options.focus()

        if self._tab == "marketplaces":
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter add/view {glyphs.bullet} D remove {glyphs.bullet} "
                f"Left/Right tabs {glyphs.bullet} Esc close"
            )
        elif self._tab in {"discover", "installed"}:
            action = "view" if self._tab == "installed" else "install"
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter {action} {glyphs.bullet} Left/Right tabs "
                f"{glyphs.bullet} Esc close"
            )
        else:
            help_text.update(f"Left/Right tabs {glyphs.bullet} Esc close")

    def _active_details_options(self) -> list[Option]:
        if self._mode == "plugin_details":
            return self._install_details_options()
        if self._mode == "installed_details" and self._selected_plugin is not None:
            return self._installed_details_options(self._selected_plugin)
        if self._mode == "mcp_details" and self._selected_mcp is not None:
            return self._mcp_details_options(self._selected_mcp)
        if (
            self._mode == "marketplace_details"
            and self._selected_marketplace is not None
        ):
            return self._marketplace_details_options()
        if (
            self._mode == "confirm_remove_marketplace"
            and self._selected_marketplace is not None
        ):
            return self._confirm_marketplace_removal_options(self._selected_marketplace)
        return [Option("Back to plugin list", id="details-back")]

    def _refresh_state(self) -> None:
        self._state = _load_manager_state(self._mcp_server_info)
        if self._selected_plugin is not None:
            refreshed = self._find_installed_plugin(self._selected_plugin.plugin_id)
            if refreshed is None:
                refreshed = self._find_available_plugin(self._selected_plugin.plugin_id)
            self._selected_plugin = refreshed
        if self._selected_mcp is not None:
            self._selected_mcp = self._find_mcp(self._selected_mcp.name)
        if self._selected_marketplace is not None:
            self._selected_marketplace = self._find_marketplace(
                self._selected_marketplace.name
            )
        self._refresh_view()

    def action_cancel(self) -> None:
        """Close or leave the add-marketplace / details prompt."""
        if self._mode == "add_marketplace":
            self._mode = "list"
            self._error = None
            self._refresh_view()
            return
        if self._details_mode_active():
            if self._mode == "confirm_remove_marketplace":
                self._mode = "marketplace_details"
                self._error = None
                self._refresh_view()
                return
            self._mode = "list"
            self._selected_plugin = None
            self._selected_mcp = None
            self._selected_marketplace = None
            self._error = None
            self._refresh_view()
            return
        self.dismiss(None)

    def action_next_tab(self) -> None:
        """Switch to the next tab."""
        if self._mode != "list":
            return
        index = self._tabs.index(self._tab)
        self._tab = self._tabs[(index + 1) % len(self._tabs)]
        self._error = None
        self._refresh_view()

    def action_previous_tab(self) -> None:
        """Switch to the previous tab."""
        if self._mode != "list":
            return
        index = self._tabs.index(self._tab)
        self._tab = self._tabs[(index - 1) % len(self._tabs)]
        self._error = None
        self._refresh_view()

    def action_cursor_down(self) -> None:
        """Move the option-list cursor down."""
        if self._mode in {
            "list",
            "plugin_details",
            "installed_details",
            "mcp_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }:
            self.query_one("#plugin-manager-options", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option-list cursor up."""
        if self._mode in {
            "list",
            "plugin_details",
            "installed_details",
            "mcp_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }:
            self.query_one("#plugin-manager-options", OptionList).action_cursor_up()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Handle row activation."""
        option_id = event.option.id
        if option_id is None or option_id == "empty":
            return
        if option_id == "add-marketplace":
            self._mode = "add_marketplace"
            self._status = None
            self._error = None
            self.query_one("#plugin-marketplace-source", Input).value = ""
            self._refresh_view()
            return
        if option_id.startswith("marketplace:"):
            name = option_id.removeprefix("marketplace:")
            row = self._find_marketplace(name)
            if row is None:
                return
            self._selected_marketplace = row
            self._selected_plugin = None
            self._selected_mcp = None
            self._mode = "marketplace_details"
            self._status = None
            self._error = None
            self._refresh_view()
            return
        if option_id.startswith("detail:"):
            plugin_id = option_id.removeprefix("detail:")
            row = self._find_available_plugin(plugin_id)
            if row is None:
                return
            self._selected_plugin = row
            self._selected_mcp = None
            self._mode = "plugin_details"
            self._error = None
            self._refresh_view()
            return
        if option_id.startswith("installed:"):
            plugin_id = option_id.removeprefix("installed:")
            row = self._find_installed_plugin(plugin_id)
            if row is None:
                return
            self._selected_plugin = row
            self._selected_mcp = None
            self._mode = "installed_details"
            self._error = None
            self._status = None
            self._refresh_view()
            return
        if option_id.startswith("mcp:"):
            name = option_id.removeprefix("mcp:")
            row = self._find_mcp(name)
            if row is None:
                return
            self._selected_mcp = row
            self._selected_plugin = None
            self._mode = "mcp_details"
            self._error = None
            self._status = None
            self._refresh_view()
            return
        if option_id.startswith("install:"):
            scope = option_id.removeprefix("install:")
            if scope == "user":
                await self._install_selected_plugin("user")
            elif scope == "project":
                await self._install_selected_plugin("project")
            elif scope == "local":
                await self._install_selected_plugin("local")
            return
        if option_id == "action:toggle-enabled":
            self._toggle_selected_plugin_enabled()
            return
        if option_id == "action:toggle-favorite":
            self._toggle_selected_favorite()
            return
        if option_id == "action:uninstall":
            await self._uninstall_selected_plugin()
            return
        if option_id == "action:toggle-mcp":
            await self._toggle_selected_mcp()
            return
        if option_id == "action:remove-marketplace":
            self._mode = "confirm_remove_marketplace"
            self._error = None
            self._refresh_view()
            return
        if option_id == "action:confirm-remove-marketplace":
            await self._remove_selected_marketplace()
            return
        if option_id == "details-back":
            self._mode = (
                "marketplace_details"
                if self._mode == "confirm_remove_marketplace"
                else "list"
            )
            self._selected_plugin = None
            self._selected_mcp = None
            if self._mode == "list":
                self._selected_marketplace = None
            self._refresh_view()

    def _find_available_plugin(self, plugin_id: str) -> _PluginRow | None:
        return next(
            (
                row
                for row in self._state.available_plugins
                if row.plugin_id == plugin_id
            ),
            None,
        )

    def _find_installed_plugin(self, plugin_id: str) -> _PluginRow | None:
        return next(
            (
                row
                for row in self._state.installed_plugins
                if row.plugin_id == plugin_id
            ),
            None,
        )

    def _find_mcp(self, name: str) -> _McpRow | None:
        return next(
            (row for row in self._state.mcp_servers if row.name == name),
            None,
        )

    def _find_marketplace(self, name: str) -> _MarketplaceRow | None:
        return next(
            (row for row in self._state.marketplaces if row.name == name),
            None,
        )

    async def _install_selected_plugin(self, scope: InstallScope) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            await asyncio.to_thread(
                install_plugin, row.plugin_id, scope=scope, trust=True
            )
        except (MarketplaceError, FileNotFoundError, OSError, ValueError) as exc:
            self._error = str(exc)
            self._status = None
            self._refresh_view()
            return
        scope_label = {
            "user": "user scope",
            "project": "project scope",
            "local": "local scope",
        }[scope]
        self._mode = "list"
        self._tab = "installed"
        self._selected_plugin = None
        self._status = (
            f"Installed {row.plugin_id} ({scope_label}). "
            "Run /reload-plugins to activate."
        )
        self._error = None
        self._refresh_state()

    def _toggle_selected_plugin_enabled(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        if row.enabled:
            disable_plugin(row.plugin_id)
            self._status = f"Disabled {row.plugin_id}. Run /reload-plugins to unload."
            self._mode = "list"
            self._selected_plugin = None
        else:
            enable_plugin_with_scope(row.plugin_id, row.scope or "user")
            self._status = f"Enabled {row.plugin_id}. Run /reload-plugins to activate."
            self._mode = "list"
            self._tab = "installed"
            self._selected_plugin = None
        self._error = None
        self._refresh_state()

    def _toggle_selected_favorite(self) -> None:
        if self._selected_plugin is not None:
            item_id = self._selected_plugin.plugin_id
            new_state = not self._selected_plugin.favorite
        elif self._selected_mcp is not None:
            item_id = self._selected_mcp.name
            new_state = not self._selected_mcp.favorite
        else:
            return
        set_plugin_favorite(item_id, new_state)
        self._status = (
            f"Added {item_id} to favorites."
            if new_state
            else f"Removed {item_id} from favorites."
        )
        self._mode = "list"
        self._tab = "installed"
        self._selected_plugin = None
        self._selected_mcp = None
        self._error = None
        self._refresh_state()

    async def _uninstall_selected_plugin(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        await asyncio.to_thread(uninstall_plugin, row.plugin_id)
        self._mode = "list"
        self._selected_plugin = None
        reload_hint = " Run /reload-plugins to unload." if row.enabled else ""
        self._status = f"Uninstalled {row.plugin_id}.{reload_hint}"
        self._error = None
        self._refresh_state()

    async def _remove_selected_marketplace(self) -> None:
        row = self._selected_marketplace
        if row is None:
            return
        self._status = f"Removing marketplace {row.name}..."
        self._error = None
        try:
            removed = await asyncio.to_thread(remove_marketplace, row.name)
        except OSError as exc:
            self._status = None
            self._error = f"Could not remove marketplace: {exc}"
            self._refresh_view()
            return
        if not removed:
            self._status = None
            self._error = f"Marketplace {row.name} is no longer configured."
            self._refresh_state()
            return
        plugin_label = "plugin" if row.installed_count == 1 else "plugins"
        self._mode = "list"
        self._tab = "marketplaces"
        self._selected_marketplace = None
        self._status = (
            f"Removed marketplace {row.name} and uninstalled "
            f"{row.installed_count} {plugin_label}."
        )
        self._error = None
        self._refresh_state()

    async def _toggle_selected_mcp(self) -> None:
        row = self._selected_mcp
        if row is None:
            return
        if self._on_toggle_mcp_disable is None:
            from deepagents_code.mcp_disabled import (
                is_server_disabled,
                set_server_disabled,
            )

            currently_disabled = await asyncio.to_thread(is_server_disabled, row.name)
            ok, detail = await asyncio.to_thread(
                set_server_disabled, row.name, not currently_disabled
            )
            if not ok:
                self._error = detail or f"Could not update {row.name}."
                self._refresh_view()
                return
            verb = "enabled" if currently_disabled else "disabled"
            self._status = f"{verb.capitalize()} MCP server {row.name}. Run /restart."
        else:
            await self._on_toggle_mcp_disable(row.name)
            self._status = f"Toggled MCP server {row.name}. Run /restart to apply."
            # Refresh live info from the app if available.
            mcp_info = getattr(self.app, "_mcp_server_info", None)
            if mcp_info is not None:
                self._mcp_server_info = mcp_info
        self._mode = "list"
        self._selected_mcp = None
        self._error = None
        self._refresh_state()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Add a marketplace from the source input."""
        if event.input.id != "plugin-marketplace-source":
            return
        source = event.value.strip()
        if not source:
            self._error = "Please enter a marketplace source."
            self._refresh_view()
            return
        self._status = "Adding marketplace..."
        self._error = None
        self._refresh_view()
        try:
            marketplace = await asyncio.to_thread(add_marketplace_source, source)
        except (MarketplaceError, OSError, RuntimeError) as exc:
            self._status = None
            self._error = f"Could not add marketplace: {exc}"
            self._refresh_view()
            return
        self._mode = "list"
        self._tab = "discover"
        self._status = (
            f"Added marketplace {marketplace.name} "
            f"({len(marketplace.plugins)} plugin(s))."
        )
        self._error = None
        self._refresh_state()

    def action_remove_marketplace(self) -> None:
        """Remove the highlighted marketplace from the Marketplaces tab."""
        if self._mode != "list" or self._tab != "marketplaces":
            return
        options = self.query_one("#plugin-manager-options", OptionList)
        highlighted = options.highlighted
        if highlighted is None:
            return
        option = options.get_option_at_index(highlighted)
        option_id = option.id
        if option_id is None or not option_id.startswith("marketplace:"):
            return
        name = option_id.removeprefix("marketplace:")
        row = self._find_marketplace(name)
        if row is None:
            return
        self._selected_marketplace = row
        self._mode = "confirm_remove_marketplace"
        self._status = None
        self._error = None
        self._refresh_view()
