"""Interactive plugin manager screen."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from collections.abc import Sequence

    from textual.app import ComposeResult

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.tui.modals.plugin_manager.models import (
        PluginManagerView,
        PluginTab,
        _MarketplaceRow,
        _PluginRow,
    )

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.plugins import (
    add_marketplace_source,
    install_plugin,
    remove_marketplace,
    set_installed_plugin_enabled,
    uninstall_plugin,
)
from deepagents_code.plugins.marketplace import MarketplaceError
from deepagents_code.tui.modals.plugin_manager.content import (
    _confirm_marketplace_removal_options,
    _install_details_options,
    _installed_details_options,
    _installed_plugin_details_content,
    _marketplace_details_content,
    _marketplace_details_options,
    _marketplace_label,
    _marketplace_removal_content,
    _plugin_details_content,
    _plugin_options,
)
from deepagents_code.tui.modals.plugin_manager.models import _ManagerState
from deepagents_code.tui.modals.plugin_manager.state import _load_manager_state


class PluginManagerScreen(ModalScreen[None]):  # noqa: RUF067
    """Arrow-key navigable plugin manager for `/plugins`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
        Binding("left", "previous_tab", "Previous tab", show=False, priority=True),
        Binding("right", "next_tab", "Next tab", show=False, priority=True),
        Binding("tab", "next_tab", "Next tab", show=False, priority=True),
        Binding("shift+tab", "previous_tab", "Previous tab", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
    ]

    CSS_PATH = "plugin_manager.tcss"

    _tabs: ClassVar[tuple[PluginTab, ...]] = (
        "discover",
        "installed",
        "marketplaces",
        "errors",
    )

    def __init__(
        self,
        *,
        mcp_server_info: Sequence[MCPServerInfo] = (),
    ) -> None:
        """Initialize the plugin manager.

        Args:
            mcp_server_info: Live MCP server metadata from the running session,
                used to show connection status for plugins that declare MCP
                servers.
        """
        super().__init__()
        self._tab: PluginTab = "discover"
        self._mode: PluginManagerView = "list"
        self._mcp_server_info = mcp_server_info
        self._state = _ManagerState((), (), (), ())
        self._status: str | None = None
        self._error: str | None = None
        self._selected_plugin: _PluginRow | None = None
        self._selected_marketplace: _MarketplaceRow | None = None

    def compose(self) -> ComposeResult:
        """Compose the manager screen.

        Yields:
            Widgets for the plugin manager UI.
        """
        with Vertical():
            yield Static(
                "Plugins", id="plugin-manager-title", classes="plugin-manager-title"
            )
            yield Static(self._tabs_text(), id="plugin-manager-tabs")
            yield Rule(
                line_style="heavy" if not is_ascii_mode() else "ascii",
                id="plugin-manager-divider",
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
                placeholder="",
                id="plugin-marketplace-source",
            )
            yield Static("", id="plugin-manager-help", classes="plugin-manager-help")

    async def on_mount(self) -> None:
        """Apply initial render, then load plugin state off the UI thread."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._status = "Loading plugins..."
        self._refresh_view()
        await self._refresh_state()
        if self._status == "Loading plugins...":
            self._status = None
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
            return _plugin_options(
                self._state.available_plugins,
                action="detail",
                status=None,
            )
        if self._tab == "installed":
            if not self._state.installed_plugins:
                return [Option("No plugins installed.", id="empty")]
            return _plugin_options(
                self._state.installed_plugins,
                action="installed",
                status=None,
            )
        if self._tab == "marketplaces":
            options = [Option("+ Add Marketplace", id="add-marketplace")]
            options.extend(
                Option(
                    Content(_marketplace_label(row, glyphs.bullet)),
                    id=f"marketplace:{row.name}",
                )
                for row in self._state.marketplaces
            )
            return options
        if not self._state.errors:
            return [Option("No plugin errors.", id="empty")]
        return [Option(Content(error), id="empty") for error in self._state.errors]

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

    def _details_mode_active(self) -> bool:
        return self._mode in {
            "plugin_details",
            "installed_details",
            "marketplace_details",
            "confirm_remove_marketplace",
        }

    def _refresh_view(self) -> None:
        title = self.query_one("#plugin-manager-title", Static)
        tabs = self.query_one("#plugin-manager-tabs", Static)
        divider = self.query_one("#plugin-manager-divider", Rule)
        tabs.update(self._tabs_text())
        status_widget = self.query_one("#plugin-manager-status", Static)
        if self._mode == "plugin_details" and self._selected_plugin is not None:
            status_widget.update(_plugin_details_content(self._selected_plugin))
        elif self._mode == "installed_details" and self._selected_plugin is not None:
            status_widget.update(
                _installed_plugin_details_content(self._selected_plugin)
            )
        elif (
            self._mode == "marketplace_details"
            and self._selected_marketplace is not None
        ):
            status_widget.update(
                _marketplace_details_content(self._selected_marketplace)
            )
        elif (
            self._mode == "confirm_remove_marketplace"
            and self._selected_marketplace is not None
        ):
            status_widget.update(
                _marketplace_removal_content(self._selected_marketplace)
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
            title.update("Add Marketplace")
            tabs.display = False
            divider.display = False
            if self._status is None:
                status_widget.update(
                    "Enter marketplace source:\n"
                    "\n"
                    "Examples:\n"
                    f"  {glyphs.bullet} owner/repo (GitHub)\n"
                    f"  {glyphs.bullet} git@github.com:owner/repo.git (SSH)\n"
                    f"  {glyphs.bullet} https://example.com/marketplace.json\n"
                    f"  {glyphs.bullet} ./path/to/marketplace"
                )
            options.display = False
            source_input.display = True
            source_input.focus()
            help_text.update(f"Enter to add {glyphs.bullet} Esc to cancel")
            return

        title.update("Plugins")
        tabs.display = True
        divider.display = True

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
                f"Enter add/view {glyphs.bullet} "
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
            return _install_details_options()
        if self._mode == "installed_details" and self._selected_plugin is not None:
            return _installed_details_options(self._selected_plugin)
        if (
            self._mode == "marketplace_details"
            and self._selected_marketplace is not None
        ):
            return _marketplace_details_options()
        if (
            self._mode == "confirm_remove_marketplace"
            and self._selected_marketplace is not None
        ):
            return _confirm_marketplace_removal_options(self._selected_marketplace)
        return [Option("Back to plugin list", id="details-back")]

    async def _refresh_state(self) -> None:
        self._state = await asyncio.to_thread(
            _load_manager_state, self._mcp_server_info
        )
        if self._selected_plugin is not None:
            refreshed = self._find_installed_plugin(self._selected_plugin.plugin_id)
            if refreshed is None:
                refreshed = self._find_available_plugin(self._selected_plugin.plugin_id)
            self._selected_plugin = refreshed
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
            self._mode = "installed_details"
            self._error = None
            self._status = None
            self._refresh_view()
            return
        if option_id == "action:install":
            await self._install_selected_plugin()
            return
        if option_id == "action:toggle-enabled":
            await self._toggle_selected_plugin_enabled()
            return
        if option_id == "action:uninstall":
            await self._uninstall_selected_plugin()
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

    def _find_marketplace(self, name: str) -> _MarketplaceRow | None:
        return next(
            (row for row in self._state.marketplaces if row.name == name),
            None,
        )

    async def _install_selected_plugin(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            await asyncio.to_thread(install_plugin, row.plugin_id)
        except (MarketplaceError, FileNotFoundError, OSError, ValueError) as exc:
            self._error = str(exc)
            self._status = None
            self._refresh_view()
            return
        self._mode = "list"
        self._tab = "installed"
        self._selected_plugin = None
        self._status = f"Installed {row.plugin_id}. Run /reload-plugins to activate."
        self._error = None
        await self._refresh_state()

    async def _toggle_selected_plugin_enabled(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            if row.enabled:
                set_installed_plugin_enabled(row.plugin_id, enabled=False)
                self._status = (
                    f"Disabled {row.plugin_id}. Run /reload-plugins to unload."
                )
                self._mode = "list"
                self._selected_plugin = None
            else:
                set_installed_plugin_enabled(row.plugin_id, enabled=True)
                self._status = (
                    f"Enabled {row.plugin_id}. Run /reload-plugins to activate."
                )
                self._mode = "list"
                self._tab = "installed"
                self._selected_plugin = None
        except OSError as exc:
            self._error = f"Could not update plugin state: {exc}"
            self._status = None
            self._refresh_view()
            return
        self._error = None
        await self._refresh_state()

    async def _uninstall_selected_plugin(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            await asyncio.to_thread(uninstall_plugin, row.plugin_id)
        except OSError as exc:
            self._error = f"Could not uninstall plugin: {exc}"
            self._status = None
            self._refresh_view()
            return
        self._mode = "list"
        self._selected_plugin = None
        reload_hint = " Run /reload-plugins to unload." if row.enabled else ""
        self._status = f"Uninstalled {row.plugin_id}.{reload_hint}"
        self._error = None
        await self._refresh_state()

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
            await self._refresh_state()
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
        await self._refresh_state()

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
        await self._refresh_state()
