"""Interactive plugin manager screen."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.events import Click, MouseMove
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from collections.abc import Sequence, Set as AbstractSet

    from textual.app import ComposeResult

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.tui.modals.plugin_manager.models import (
        PluginManagerView,
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
from deepagents_code.tui.modals.plugin_manager.models import (
    PluginTab,
    _ManagerState,
)
from deepagents_code.tui.modals.plugin_manager.state import _load_manager_state

_TAB_LABELS: dict[PluginTab, str] = {
    "discover": "Plugins",
    "installed": "Installed",
    "marketplaces": "Marketplaces",
    "errors": "Errors",
}


class PluginTabSelected(Message):
    """Posted when a plugin manager tab label is clicked."""

    def __init__(self, tab: PluginTab) -> None:
        """Initialize with the selected tab id.

        Args:
            tab: Tab to activate.
        """
        super().__init__()
        self.tab = tab


class PluginTabLabel(Static):
    """Mouse-clickable tab label in the plugin manager header."""

    def __init__(self, tab: PluginTab, label: str) -> None:
        """Create a tab label.

        Args:
            tab: Tab id this label activates.
            label: Display text for the tab.
        """
        super().__init__(
            f"  {label}  ",
            id=f"plugin-tab-{tab}",
            classes="plugin-manager-tab",
            markup=False,
        )
        self._tab = tab
        self._label = label
        self.can_focus = False

    def set_active(self, active: bool) -> None:
        """Update the active marker and style.

        Args:
            active: Whether this tab is the current tab.
        """
        self.update(f"> {self._label} <" if active else f"  {self._label}  ")
        self.set_class(active, "active")

    def on_click(self, event: Click) -> None:
        """Select this tab on click.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(PluginTabSelected(self._tab))


class PluginManagerScreen(ModalScreen[tuple[str, bool] | None]):  # noqa: RUF067
    """Arrow-key navigable plugin manager for `/plugins`.

    Dismisses with `(label, needs_mcp_login)` after an MCP-bearing plugin
    install so the app can offer reconnect + login guidance, otherwise
    `None`.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
        # Separate from tab/shift+tab so check_action can release arrows to a
        # non-empty Input for caret movement while keeping Tab as tab cycling.
        Binding(
            "left", "arrow_previous_tab", "Previous tab", show=False, priority=True
        ),
        Binding("right", "arrow_next_tab", "Next tab", show=False, priority=True),
        Binding("tab", "next_tab", "Next tab", show=False, priority=True),
        Binding("shift+tab", "previous_tab", "Previous tab", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("/", "focus_search", "Search", show=False, priority=True),
    ]

    CSS_PATH = "plugin_manager.tcss"
    # Prefer the option list over the search Input so Enter activates rows on
    # open; `/` still focuses search explicitly.
    AUTO_FOCUS = "#plugin-manager-options"

    # Divider width used before the options list has been laid out (e.g. in unit
    # tests that build options off-screen). At render time the divider is sized to
    # the measured options width instead so it never wraps on a narrower modal.
    _DIVIDER_FALLBACK_WIDTH: ClassVar[int] = 72

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
        loaded_plugin_ids: AbstractSet[str] | None = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            mcp_server_info: Live MCP server metadata from the running session,
                used to show connection status for plugins that declare MCP
                servers.
            loaded_plugin_ids: Plugin ids loaded into the current session.
                Plugins whose enabled state differs from this set (enabled but
                not loaded, or disabled but still loaded) are shown as pending
                reload.
        """
        super().__init__()
        self._tab: PluginTab = "discover"
        self._mode: PluginManagerView = "list"
        self._mcp_server_info = mcp_server_info
        self._loaded_plugin_ids: frozenset[str] = frozenset(loaded_plugin_ids or ())
        self._state = _ManagerState((), (), (), ())
        self._status: str | None = None
        self._error: str | None = None
        self._selected_plugin: _PluginRow | None = None
        self._selected_marketplace: _MarketplaceRow | None = None
        self._search_query = ""

    def compose(self) -> ComposeResult:
        """Compose the manager screen.

        Yields:
            Widgets for the plugin manager UI.
        """
        with Vertical():
            yield Static(
                "Plugins", id="plugin-manager-title", classes="plugin-manager-title"
            )
            with Horizontal(id="plugin-manager-tabs", classes="plugin-manager-tabs"):
                for tab in self._tabs:
                    yield PluginTabLabel(tab, _TAB_LABELS[tab])
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
            yield Input(
                placeholder="Search plugins...",
                id="plugin-manager-search",
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

    def on_resize(self) -> None:
        """Refit the width-sized marketplaces divider when the modal resizes.

        The divider is the only content sized to the options width, so restrict the
        rebuild to the marketplaces list view to avoid needless focus/highlight churn
        on unrelated tabs and while the add-marketplace input is focused.
        """
        if (
            self._mode == "list"
            and self._tab == "marketplaces"
            and self._state.marketplaces
        ):
            self._refresh_view()

    def _update_tab_labels(self) -> None:
        """Refresh active styling on each clickable tab label."""
        for tab in self._tabs:
            self.query_one(f"#plugin-tab-{tab}", PluginTabLabel).set_active(
                tab == self._tab
            )

    def _select_tab(self, tab: PluginTab) -> None:
        """Activate `tab`, exiting details into list mode when needed.

        Args:
            tab: Tab to show.
        """
        if self._mode == "add_marketplace":
            return
        if self._details_mode_active():
            self._mode = "list"
            self._selected_plugin = None
            self._selected_marketplace = None
        self._tab = tab
        self._error = None
        self._refresh_view()

    def _filtered_plugins(self, rows: Sequence[_PluginRow]) -> tuple[_PluginRow, ...]:
        query = self._search_query.strip().casefold()
        if not query:
            return tuple(rows)
        return tuple(
            row
            for row in rows
            if query in row.plugin_id.casefold()
            or query in row.label.casefold()
            or query in row.description.casefold()
        )

    def _current_options(self) -> list[Option]:
        glyphs = get_glyphs()
        if self._tab == "discover":
            if not self._state.marketplaces:
                return [
                    Option(
                        "No marketplaces installed. Add one to discover plugins.",
                        id="empty",
                        disabled=True,
                    ),
                    Option("+ Add marketplace", id="add-marketplace"),
                ]
            if not self._state.available_plugins:
                return [Option("All available plugins are installed.", id="empty")]
            rows = self._filtered_plugins(self._state.available_plugins)
            if not rows:
                return [Option("No plugins match your search.", id="empty")]
            return _plugin_options(rows, action="detail", status=None)
        if self._tab == "installed":
            if not self._state.installed_plugins:
                return [Option("No plugins installed.", id="empty")]
            rows = self._filtered_plugins(self._state.installed_plugins)
            if not rows:
                return [Option("No installed plugins match your search.", id="empty")]
            return _plugin_options(rows, action="installed", status=None)
        if self._tab == "marketplaces":
            options = [Option("+ Add marketplace", id="add-marketplace")]
            if self._state.marketplaces:
                options.append(
                    Option(
                        Content.styled(
                            glyphs.box_horizontal * self._divider_width(), "dim"
                        ),
                        id="marketplace-divider",
                        disabled=True,
                    )
                )
            options.extend(
                Option(
                    _marketplace_label(row),
                    id=f"marketplace:{row.name}",
                )
                for row in self._state.marketplaces
            )
            return options
        if not self._state.errors:
            return [Option("No plugin errors.", id="empty")]
        return [Option(Content(error), id="empty") for error in self._state.errors]

    def _divider_width(self) -> int:
        """Width for the marketplaces divider, sized to the options list.

        The options list respects the modal's `max-width`, so a fixed width wraps on
        terminals narrower than the full modal. Measure the laid-out content width when
        available and fall back to a constant before the first layout (e.g. in tests).

        Returns:
            The measured options content width, or `_DIVIDER_FALLBACK_WIDTH` if the
            options list is not mounted or has not been laid out yet.
        """
        try:
            width = self.query_one(
                "#plugin-manager-options", OptionList
            ).content_size.width
        except NoMatches:
            return self._DIVIDER_FALLBACK_WIDTH
        return width if width > 0 else self._DIVIDER_FALLBACK_WIDTH

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
        tabs = self.query_one("#plugin-manager-tabs", Horizontal)
        divider = self.query_one("#plugin-manager-divider", Rule)
        self._update_tab_labels()
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
        search_input = self.query_one("#plugin-manager-search", Input)
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
            search_input.display = False
            source_input.display = True
            source_input.focus()
            help_text.update(f"Enter to add {glyphs.bullet} Esc to cancel")
            return

        title.update("Plugins")
        tabs.display = True
        divider.display = True

        if self._details_mode_active():
            search_input.display = False
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
        search_input.display = self._tab in {"discover", "installed"}
        if search_input.display and search_input.value != self._search_query:
            search_input.value = self._search_query
        highlighted = options.highlighted
        options.clear_options()
        for option in self._current_options():
            options.add_option(option)
        if options.option_count:
            candidate = (
                0 if highlighted is None else min(highlighted, options.option_count - 1)
            )
            options.highlighted = self._nearest_enabled_index(options, candidate)
        if not search_input.has_focus:
            options.focus()

        if self._tab == "marketplaces":
            help_text.update(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} select {glyphs.bullet} "
                f"Enter add/view {glyphs.bullet} "
                f"Left/Right tabs {glyphs.bullet} Esc close"
            )
        elif self._tab in {"discover", "installed"}:
            if self._tab == "installed":
                action = "view"
            elif not self._state.marketplaces:
                action = "add marketplace"
            else:
                action = "install"
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
            _load_manager_state,
            self._mcp_server_info,
            loaded_plugin_ids=self._loaded_plugin_ids,
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

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],  # noqa: ARG002  # required by Textual's DOMNode.check_action override signature
    ) -> bool | None:
        """Gate priority bindings that would otherwise steal Input keystrokes.

        `/` is only enabled on searchable list views so it stays typeable in the
        Add Marketplace source field. Left/right release to a focused Input once
        it has at least one character, so caret movement works while empty-field
        arrows keep switching tabs.

        Returns:
            `False` to step a binding aside so the focused widget receives the
                key; `True` to allow the action.
        """
        if action in {"arrow_previous_tab", "arrow_next_tab"}:
            focused = self.focused
            return not (isinstance(focused, Input) and bool(focused.value))
        if action == "focus_search":
            return self._mode == "list" and self._tab in {"discover", "installed"}
        return True

    def on_plugin_tab_selected(self, event: PluginTabSelected) -> None:
        """Switch tabs from a mouse click on a tab label.

        Args:
            event: Tab selection message from `PluginTabLabel`.
        """
        self._select_tab(event.tab)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a pointer cursor over clickable tab labels.

        Args:
            event: Mouse move event.
        """
        self.styles.pointer = (
            "pointer" if isinstance(event.widget, PluginTabLabel) else "default"
        )

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the manager."""
        self.styles.pointer = "default"

    def action_cancel(self) -> None:
        """Clear search, close, or leave the active prompt."""
        search_input = self.query_one("#plugin-manager-search", Input)
        if search_input.has_focus:
            if self._search_query:
                self._search_query = ""
                search_input.value = ""
                self._refresh_view()
                search_input.focus()
            else:
                self.query_one("#plugin-manager-options", OptionList).focus()
            return
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

    def action_focus_search(self) -> None:
        """Focus the plugin filter on searchable tabs."""
        if self._mode == "list" and self._tab in {"discover", "installed"}:
            self.query_one("#plugin-manager-search", Input).focus()

    def _cycle_details_option(self, step: int) -> None:
        options = self.query_one("#plugin-manager-options", OptionList)
        enabled = [
            index
            for index in range(options.option_count)
            if not options.get_option_at_index(index).disabled
        ]
        if not enabled:
            return
        current = options.highlighted
        position = enabled.index(current) if current in enabled else 0
        options.highlighted = enabled[(position + step) % len(enabled)]
        options.focus()

    def action_arrow_next_tab(self) -> None:
        """Switch tabs via right arrow when the caret is not editing text."""
        self.action_next_tab()

    def action_arrow_previous_tab(self) -> None:
        """Switch tabs via left arrow when the caret is not editing text."""
        self.action_previous_tab()

    def action_next_tab(self) -> None:
        """Switch tabs or focus the next details option."""
        if self._details_mode_active():
            self._cycle_details_option(1)
            return
        if self._mode != "list":
            return
        index = self._tabs.index(self._tab)
        self._select_tab(self._tabs[(index + 1) % len(self._tabs)])

    def action_previous_tab(self) -> None:
        """Switch tabs or focus the previous details option."""
        if self._details_mode_active():
            self._cycle_details_option(-1)
            return
        if self._mode != "list":
            return
        index = self._tabs.index(self._tab)
        self._select_tab(self._tabs[(index - 1) % len(self._tabs)])

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
            # install_plugin returns the loaded instance; Discover rows do not
            # carry MCP metadata until install, so inspect the result here.
            instance = await asyncio.to_thread(install_plugin, row.plugin_id)
        except (MarketplaceError, FileNotFoundError, OSError, ValueError) as exc:
            self._error = str(exc)
            self._status = None
            self._refresh_view()
            return
        from deepagents_code.plugins.adapters.mcp import plugin_mcp_server_entries

        label = row.label
        entries = plugin_mcp_server_entries(instance)
        has_mcp = bool(entries)
        needs_login = any(needs for _label, _scoped, needs in entries)
        self.notify(f"Installed {label}", timeout=5, markup=False)
        self._mode = "list"
        self._tab = "installed"
        self._selected_plugin = None
        self._status = f"Installed {label}."
        self._error = None
        await self._refresh_state()
        if has_mcp:
            # Close the manager so the reconnect prompt is not buried under it.
            self.dismiss((label, needs_login))
            return

    async def _toggle_selected_plugin_enabled(self) -> None:
        row = self._selected_plugin
        if row is None:
            return
        try:
            if row.enabled:
                set_installed_plugin_enabled(row.plugin_id, enabled=False)
                self._status = f"Disabled {row.label}. Run /reload to unload."
                self._mode = "list"
                self._selected_plugin = None
            else:
                set_installed_plugin_enabled(row.plugin_id, enabled=True)
                self._status = f"Enabled {row.label}. Run /reload to activate."
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
        reload_hint = " Run /reload to unload." if row.enabled else ""
        self._status = f"Uninstalled {row.label}.{reload_hint}"
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

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the current plugin list as the search query changes."""
        if event.input.id != "plugin-manager-search":
            return
        self._search_query = event.value
        self._refresh_view()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Activate the highlighted row from search, or add a marketplace."""
        if event.input.id == "plugin-manager-search":
            event.stop()
            options = self.query_one("#plugin-manager-options", OptionList)
            options.focus()
            options.action_select()
            return
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
